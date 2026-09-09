# src/riskquant/sensitivity_pnl/kernels/taylor.py
"""
Zero-allocation second-order Taylor PnL kernel.

    PnL_m(s) = SUM_j [ d1_mj * x_j(s) + 0.5 * d2_mj * x_j(s)^2 ]
             + SUM_{i<j} c_mij * x_i(s) * x_j(s)

Both sums are CSR row-slices of `m`.

Entry blocking (the load-bearing optimisation)
----------------------------------------------
The naive form makes ONE FULL PASS over the 500-wide accumulator PER CSR ENTRY:
an 8-byte load and an 8-byte store of `acc[s]` to buy 2 FLOPs. That is 16 bytes
of L1 traffic per 2 FLOPs — an arithmetic intensity of 0.125 FLOP/byte, which
pins the loop to the load/store units while the FMA ports idle.

Four entries are therefore folded into a single accumulator pass, cutting that
traffic 4x and quadrupling arithmetic intensity. Measured 1.32-1.33x on
realistic books (12 and 40 entries per position), 1.15x at 4 entries. A width
sweep over {1, 2, 4, 6, 8} put the knee at 4-6; 4 is taken because 8 broadcast
coefficients plus 4 loaded displacement vectors plus the accumulator already
sits near the 16-register AVX2 ceiling, and wider blocks buy nothing once the
loop is FMA-bound.

Each diagonal term is evaluated in Horner form `(d1 + 0.5*d2*x) * x` — two FMAs
on ONE load, rather than three multiplies and an add on two. The four terms are
then summed PAIRWISE, `(t0+t1) + (t2+t3)`, which is both a shorter dependency
chain for the FMA pipeline and strictly better conditioned than a sequential
chain when legs cancel.

Displacement is f32
-------------------
`displacement` is the only array read `NNZ * B` times per batch; everything else
is read `NNZ` times. Halving its width halves the dominant traffic for a
representation error three to five orders below the Taylor truncation error.
See `kernels/displacement.py`. Coefficients stay f64: their traffic is
negligible and `quote_base ** 2` folding can scale them by 1e8.

Accumulate f64, emit f32
------------------------
A position with hundreds of entries, or with large offsetting legs, loses its
small legs to f32 rounding. Accumulation is f64 and the narrow-cast is fused
into the tail of the same `prange` iteration while the accumulator row is still
hot in L1. The alternative — a f64 `(M, S)` master grid narrowed once at the end
— costs 8 bytes per position-scenario for the whole run instead of per BATCH,
and measured 2.05x slower because each batch touched a cold column-stripe of a
multi-GB array.

Work-balanced chunking
----------------------
`prange` ranges over CHUNKS, not positions. Numba partitions a `prange` into
contiguous, equal-COUNT slices, and equal count is not equal work on a
heavy-tailed book — measured scaling on a 12-core host swung between 0.88x and
2.13x across runs of identical code, decided purely by where the heavy rows
happened to sit. `chunk_offsets` cuts the row axis into contiguous ranges of
equal COST (see `scheduling.py`), so every thread receives a proportional share
by construction. Predicted ceiling on the SKEWED shape at 12 threads: 0.17
unbalanced, 1.00 chunked.

Sorting positions by nnz is the opposite of a fix — because chunks are
contiguous it piles every heavy row onto thread 0, and it measures at 0.17 even
on a book whose natural order was already fine at 0.93. Do not do it.

Per-thread scratch
------------------
`accumulator` is `(n_threads_max, B_capacity)`, indexed by `nb.get_thread_id()`,
not `(M, B)`. A `prange` iteration executes start-to-finish on one thread and
iterations sharing a thread run sequentially, so the row is exclusively owned;
the lookup is hoisted to once per chunk. Rows within a chunk run sequentially on
that thread and reuse the same scratch row, which is why the seed below must
WRITE rather than accumulate.
Sized at `NUMBA_NUM_THREADS` (the ceiling `set_num_threads` cannot exceed), so
it can never be over-indexed. At 16 threads and B=500 the whole scratch is 64 KB
and lives in L2.

    Layer-1 golden master asserts bit-identical output at 1 thread and at N
    threads. That test is the guard on this pattern — do not delete it.

Bounds
------
Runs under `boundscheck=False`. Every index dereferenced here is range-checked
in `CompiledSensitivityBook.__post_init__` (CSR monotonicity and terminator,
`factor_uix` and both cross index arrays against Ka) or in
`SensitivityPnlEngine` (`batch_len`, `scenario_offset`, levels-matrix height,
and `chunk_offsets` monotonicity and termination at M).
This kernel adds no guards of its own and must not be called with arrays that
did not come through those two gates.

Memory layout
-------------
- `displacement` is C-contiguous `(Ka, B_capacity)`; `displacement[j]` is a
  contiguous run of scenarios, so every inner loop reads unit-stride.
- `out` is C-contiguous f32 `(M, S_total)` — a row-slice VIEW of the shared PnL
  grid, not an owned buffer. `out[m]` is a contiguous run and chunks partition
  the row axis, so each `prange` iteration owns a disjoint set of whole rows and
  there is no cross-thread write aliasing.
- `scenario_offset` is a scalar, not a slice of `out`: a column slice is not
  C-contiguous and would break the pinned signature.

A register-blocked SCENARIO-tiled variant (accumulators in registers, outer loop
over scenario tiles) was built and measured at 0.21-0.30x this kernel — the
500-wide inner loop vectorises to AVX FMA over unit stride and the accumulator
never leaves L1, so the traffic scenario-tiling saves does not exist. A split
linear/quadratic CSR layout (gamma stored as an i==j quadratic entry) was also
built and measured at 0.58-0.90x: it turns one Horner pass into two passes and
costs two loads for what Horner does in one. Both are dead ends. Do not
reintroduce either without re-running
`scripts/demos/sensitivity_pnl/kernels/taylor_kernel_throughput.py`.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]
from numba.core import types as nb_types

# See kernels/displacement.py for why these come from riskquant.jit rather
# than a local redefinition.
from riskquant.jit import _F32_2D_C, _F64_1D_C, _F64_2D_C, _I32_1D_C, parallel_njit


@parallel_njit(
    nb_types.void(
        _I32_1D_C,  # chunk_offsets       (C+1,)  work-balanced row ranges
        _I32_1D_C,  # position_offsets    (M+1,)
        _I32_1D_C,  # factor_uix          (NNZ,)
        _F64_1D_C,  # first_order         (NNZ,)
        _F64_1D_C,  # second_order        (NNZ,)
        _I32_1D_C,  # cross_offsets       (M+1,)
        _I32_1D_C,  # cross_factor_uix_i  (NNZX,)
        _I32_1D_C,  # cross_factor_uix_j  (NNZX,)
        _F64_1D_C,  # cross_second_order  (NNZX,)
        _F32_2D_C,  # displacement      (Ka, B_capacity)        f32 C-contig
        _F64_2D_C,  # accumulator       (n_threads, B_capacity) f64 C-contig
        nb_types.int64,  # batch_len
        nb_types.int64,  # scenario_offset
        _F32_2D_C,  # out               (M, S_total)            f32 C-contig view
    ),
)
def kernel_taylor_pnl(
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
    accumulator,
    batch_len,
    scenario_offset,
    out,
) -> None:
    """Evaluate one scenario batch of Taylor PnL and scatter it into `out`."""
    for c in nb.prange(chunk_offsets.shape[0] - 1):
        # Hoisted: one thread owns the whole chunk, so the scratch row is
        # resolved once per chunk rather than once per position.
        acc = accumulator[nb.get_thread_id()] # type: ignore

        for m in range(chunk_offsets[c], chunk_offsets[c + 1]):
            lo = position_offsets[m]
            hi = position_offsets[m + 1]
            qlo = cross_offsets[m]
            qhi = cross_offsets[m + 1]

            # ---- Seed: WRITE the accumulator, never accumulate into it ----
            # `acc` is reused across every row in the chunk, so an unconditional
            # seed is mandatory — stale values are another position's PnL, not
            # the harmless garbage of a fresh np.empty row. All four arms sit
            # OUTSIDE the scenario loop and cost one branch per row per batch.
            if hi - lo >= 4:
                a0 = first_order[lo]
                b0 = 0.5 * second_order[lo]
                v0 = displacement[factor_uix[lo]]
                a1 = first_order[lo + 1]
                b1 = 0.5 * second_order[lo + 1]
                v1 = displacement[factor_uix[lo + 1]]
                a2 = first_order[lo + 2]
                b2 = 0.5 * second_order[lo + 2]
                v2 = displacement[factor_uix[lo + 2]]
                a3 = first_order[lo + 3]
                b3 = 0.5 * second_order[lo + 3]
                v3 = displacement[factor_uix[lo + 3]]
                for s in range(batch_len):
                    x0 = v0[s]
                    x1 = v1[s]
                    x2 = v2[s]
                    x3 = v3[s]
                    acc[s] = ((a0 + b0 * x0) * x0 + (a1 + b1 * x1) * x1) + (
                        (a2 + b2 * x2) * x2 + (a3 + b3 * x3) * x3
                    )
                lo += 4
            elif lo < hi:
                a0 = first_order[lo]
                b0 = 0.5 * second_order[lo]
                v0 = displacement[factor_uix[lo]]
                for s in range(batch_len):
                    x0 = v0[s]
                    acc[s] = (a0 + b0 * x0) * x0
                lo += 1
            elif qlo < qhi:
                c0 = cross_second_order[qlo]
                i0 = displacement[cross_factor_uix_i[qlo]]
                j0 = displacement[cross_factor_uix_j[qlo]]
                for s in range(batch_len):
                    acc[s] = c0 * i0[s] * j0[s]
                qlo += 1
            else:
                for s in range(batch_len):
                    acc[s] = 0.0

            # ---- Diagonal block, 4 entries per accumulator pass ----
            while lo + 4 <= hi:
                a0 = first_order[lo]
                b0 = 0.5 * second_order[lo]
                v0 = displacement[factor_uix[lo]]
                a1 = first_order[lo + 1]
                b1 = 0.5 * second_order[lo + 1]
                v1 = displacement[factor_uix[lo + 1]]
                a2 = first_order[lo + 2]
                b2 = 0.5 * second_order[lo + 2]
                v2 = displacement[factor_uix[lo + 2]]
                a3 = first_order[lo + 3]
                b3 = 0.5 * second_order[lo + 3]
                v3 = displacement[factor_uix[lo + 3]]
                for s in range(batch_len):
                    x0 = v0[s]
                    x1 = v1[s]
                    x2 = v2[s]
                    x3 = v3[s]
                    acc[s] += ((a0 + b0 * x0) * x0 + (a1 + b1 * x1) * x1) + (
                        (a2 + b2 * x2) * x2 + (a3 + b3 * x3) * x3
                    )
                lo += 4

            # ---- Diagonal remainder (< 4 entries) ----
            for e in range(lo, hi):
                a0 = first_order[e]
                b0 = 0.5 * second_order[e]
                v0 = displacement[factor_uix[e]]
                for s in range(batch_len):
                    x0 = v0[s]
                    acc[s] += (a0 + b0 * x0) * x0

            # ---- Cross block, 2 entries per accumulator pass ----
            # Two, not four: each cross entry costs two displacement rows, so the
            # register budget is reached at half the diagonal width.
            while qlo + 2 <= qhi:
                c0 = cross_second_order[qlo]
                i0 = displacement[cross_factor_uix_i[qlo]]
                j0 = displacement[cross_factor_uix_j[qlo]]
                c1 = cross_second_order[qlo + 1]
                i1 = displacement[cross_factor_uix_i[qlo + 1]]
                j1 = displacement[cross_factor_uix_j[qlo + 1]]
                for s in range(batch_len):
                    acc[s] += c0 * i0[s] * j0[s] + c1 * i1[s] * j1[s]
                qlo += 2

            # ---- Cross remainder ----
            for e in range(qlo, qhi):
                c0 = cross_second_order[e]
                i0 = displacement[cross_factor_uix_i[e]]
                j0 = displacement[cross_factor_uix_j[e]]
                for s in range(batch_len):
                    acc[s] += c0 * i0[s] * j0[s]

            # ---- Fused narrow-cast scatter (acc row still hot in L1) ----
            row = out[m]
            for s in range(batch_len):
                row[scenario_offset + s] = nb_types.float32(acc[s])
