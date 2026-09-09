# src/riskquant/sensitivity_pnl/kernels/taylor_validity.py
"""
Linear/quadratic split of the Taylor expansion — the raw material for
diagnosing when a second-order approximation is starting to extrapolate
rather than interpolate.

    PnL_m(s) = SUM_j  d1_mj * x_j(s)                      <- LINEAR_TERM
             + SUM_j  0.5 * d2_mj * x_j(s)^2               <- QUADRATIC_TERM
             + SUM_{i<j}  c_mij * x_i(s) * x_j(s)          <- QUADRATIC_TERM

`kernel_taylor_pnl` (`kernels/taylor.py`) sums these into one number because
that is what a PnL grid needs. This kernel keeps them apart, because their
RATIO is the diagnostic: a Taylor expansion truncated after the quadratic term
is trustworthy exactly when the quadratic piece is a small correction to the
linear one. When the quadratic term is comparable to or larger than the
linear term, the series is not behaving like a converging expansion any more
— it says nothing about whether the (unmodelled) cubic term is small, but it
is the only evidence available without a third-order coefficient or a full
reval, and its ABSENCE is the reassuring case: if even the captured curvature
is negligible, the odds that an uncaptured, higher-order term is somehow large
are low for any reasonably smooth payoff.

What this kernel does NOT claim
--------------------------------
It does not bound the truncation error. A payoff that is EXACTLY bilinear (see
`kernels/taylor.py`'s own docstring on quanto forwards) has a nonzero
quadratic term that is not an approximation at all — it is exact, because the
third derivative is genuinely zero. This kernel cannot distinguish "exact
curvature, fully captured" from "truncated curvature, more where that came
from" — both look like the same nonzero quadratic term. It is a necessary-
condition heuristic (small quadratic term is reassuring), not a sufficient one
(large quadratic term is not proof of anything beyond "this scenario is
exercising real curvature, worth a closer look").

Deliberately unblocked
-----------------------
`kernel_taylor_pnl` folds 4 diagonal entries and 2 cross entries into a single
accumulator pass for a measured 1.3x throughput gain, because it sits on the
hot path of every VaR calculation. This kernel does not: it is a periodic
model-validity check, not something run once per batch of every live pricing
call, and a straightforward per-entry loop halves the number of things that
could have a transcription bug in a brand-new kernel for a gain that does not
matter at this call frequency. Revisit only if usage patterns make this
kernel hot too — re-run `kernels/taylor.py`'s blocking argument first; it is
not free even there (a naive zero-seed lost to a branch-seeded accumulator by
about 7% when measured on that kernel).

Zero-seed, not first-entry-seed
---------------------------------
For the same reason: `kernel_taylor_pnl` seeds its accumulator from the first
CSR entry to save a redundant write, with four branch arms to cover the empty-
diagonal and empty-both cases. Here every accumulator row is unconditionally
zeroed before its position's entries are added, which handles an entirely
empty row (a declared position with no sensitivities) for free, with no special
case at all.

Bounds
------
Runs under `boundscheck=False`, like every kernel in this package. Every index
dereferenced here is range-checked in `CompiledSensitivityBook.__post_init__`
(CSR monotonicity and terminator, `factor_uix` and both cross index arrays
against Ka) or in `TaylorValidityEngine` (`_batch_contract.check_2d_batch_source`,
shared with `SensitivityPnlEngine` so the two engines cannot silently diverge
on caller-buffer bounds). This kernel adds no guards of its own.

Memory layout
-------------
- `displacement` is C-contiguous `(Ka, B_capacity)` f32 — read `NNZ * B` times
  per batch, same as in the PnL kernel; see `kernels/displacement.py` for why
  f32 costs nothing here relative to the Taylor truncation error itself.
- `linear_out` / `quadratic_out` are C-contiguous f32 `(M, S_total)` — row-slice
  views of caller-owned grids, not owned buffers, exactly like `PnlGrid`.
- `chunk_offsets` partitions the row axis for `prange`, reusing
  `scheduling.balanced_chunk_offsets` unchanged: the per-row cost here still
  scales with `nnz + 2*nnz_cross`, so the SAME relative balance across
  positions applies even though this kernel's absolute cost per entry differs
  (two accumulator streams and two scatters instead of one).
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]
from numba.core import types as nb_types

# See kernels/displacement.py for why these come from riskquant.jit rather
# than a local redefinition.
from riskquant.jit import _F32_2D_C, _F64_1D_C, _F64_2D_C, _I32_1D_C, parallel_njit


@parallel_njit(
    nb_types.void(
        _I32_1D_C,  # chunk_offsets          (C+1,)  work-balanced row ranges
        _I32_1D_C,  # position_offsets       (M+1,)
        _I32_1D_C,  # factor_uix             (NNZ,)
        _F64_1D_C,  # first_order            (NNZ,)
        _F64_1D_C,  # second_order           (NNZ,)
        _I32_1D_C,  # cross_offsets          (M+1,)
        _I32_1D_C,  # cross_factor_uix_i     (NNZX,)
        _I32_1D_C,  # cross_factor_uix_j     (NNZX,)
        _F64_1D_C,  # cross_second_order     (NNZX,)
        _F32_2D_C,  # displacement         (Ka, B_capacity)        f32 C-contig
        _F64_2D_C,  # linear_accumulator   (n_threads, Bp)         f64 C-contig
        _F64_2D_C,  # quadratic_accumulator(n_threads, Bp)         f64 C-contig
        nb_types.int64,  # batch_len
        nb_types.int64,  # scenario_offset
        _F32_2D_C,  # linear_out           (M, S_total)            f32 C-contig view
        _F32_2D_C,  # quadratic_out        (M, S_total)            f32 C-contig view
    ),
)
def kernel_taylor_validity_terms(
    chunk_offsets,
    position_offsets,
    factor_uix,
    first_order,
    second_order,
    cross_offsets,
    cross_factor_uix_i,
    cross_factor_uix_j,
    cross_second_order,
    displacement,
    linear_accumulator,
    quadratic_accumulator,
    batch_len,
    scenario_offset,
    linear_out,
    quadratic_out,
) -> None:
    """Evaluate one scenario batch's linear and quadratic Taylor terms."""
    for c in nb.prange(chunk_offsets.shape[0] - 1):
        # Hoisted: one thread owns the whole chunk, so both scratch rows are
        # resolved once per chunk rather than once per position.
        thread_id = nb.get_thread_id()  # type: ignore
        lin_acc = linear_accumulator[thread_id]
        quad_acc = quadratic_accumulator[thread_id]

        for m in range(chunk_offsets[c], chunk_offsets[c + 1]):
            # Zero-seed: both accumulators are reused across every row in the
            # chunk, so this must happen unconditionally, and it makes a row
            # with zero entries (a declared, unhedged-in-this-factor position)
            # correct for free rather than as a special case.
            for s in range(batch_len):
                lin_acc[s] = 0.0
                quad_acc[s] = 0.0

            # ---- Diagonal: split first-order from second-order ----
            for e in range(position_offsets[m], position_offsets[m + 1]):
                a = first_order[e]
                b = 0.5 * second_order[e]
                v = displacement[factor_uix[e]]
                for s in range(batch_len):
                    x = v[s]
                    lin_acc[s] += a * x
                    quad_acc[s] += b * x * x

            # ---- Cross: entirely second-order ----
            for e in range(cross_offsets[m], cross_offsets[m + 1]):
                coefficient = cross_second_order[e]
                vi = displacement[cross_factor_uix_i[e]]
                vj = displacement[cross_factor_uix_j[e]]
                for s in range(batch_len):
                    quad_acc[s] += coefficient * vi[s] * vj[s]

            # ---- Fused narrow-cast scatter, both rows still hot in L1 ----
            linear_row = linear_out[m]
            quadratic_row = quadratic_out[m]
            for s in range(batch_len):
                linear_row[scenario_offset + s] = nb_types.float32(lin_acc[s])
                quadratic_row[scenario_offset + s] = nb_types.float32(quad_acc[s])
