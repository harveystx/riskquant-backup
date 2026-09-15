# src/riskquant/simlab/shocks/shock_kernel.py
"""
Zero-allocation apply-shocks kernel.

Translates `(B, K_unique)` shock columns into `(N_req, B)` shocked
levels via the per-required `(req_to_uix, shock_scale, shift_type)`
mapping and three branch-hoisted formulas (LOG_RETURN, SIMPLE_RETURN,
DIFFERENCE).

Memory layout
-------------
- `shocks` is F-contiguous `(B, K_unique)` — pulling `shocks[:, k]` is
  one contiguous run of B values per shock column.
- `out` is C-contiguous `(N_req, B)` — writing `out[n, b]` per scenario
  walks one cache line at a time.

Hot path discipline
-------------------
- Layout-pinned `_NB_VOID(...)` signature: every array dimension is
  stride-tagged so Numba emits straight-line SIMD with no dynamic
  stride checks at runtime.
- Branch ladder hoisted out of the B-loop: one `if shift_type` decision
  per required exposure, then a tight inner loop.
- `math.exp` (libm) instead of `np.exp` (ufunc) — direct call, no
  ufunc setup overhead.
- All validation lives in the wrapper and runs ONCE at runner-build
  time (see `validate_apply_shocks_buffers`). The hot-path wrapper
  trusts the contract; debug builds get cheap `assert`s.
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import (
    _F32_2D_C,
    _F32_2D_F,
    _F64_1D_C,
    _F64_2D_C,
    _I8_1D_C,
    _I32_1D_C,
    _NB_VOID,
    parallel_njit,
)
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
)

# Layout-pinned aliases.


# Two `shocks` layouts are accepted, deliberately (RQ-AUD-017, consumer side).
# The generators yield an F-contiguous `(B, K)` shocks buffer, which is the
# fast path this kernel is written for. But a trailing batch of exactly ONE
# scenario -- `(1, K)`, which happens whenever the scenario count is not a
# multiple of batch_size -- is reported by numpy as BOTH C- and F-contiguous
# (a single row has only one traversal order), and Numba's dispatcher
# resolves that ambiguity to 'C'. No amount of `asfortranarray` can remove
# the ambiguity, so a C overload is registered rather than forcing callers
# to copy. Column access in the loop body is identical either way; only the
# stride pattern differs.
@parallel_njit(
    [
        _NB_VOID(
            _F64_1D_C,  # base_levels        (N,)
            _F32_2D_F,  # shocks             (B, K)  F-contig
            _I32_1D_C,  # req_to_uix         (N,)
            _F64_1D_C,  # shock_scale        (N,)
            _I8_1D_C,  # shift_types        (N,)
            _F64_2D_C,  # out                (N, B)  C-contig
        ),
        _NB_VOID(
            _F64_1D_C,  # base_levels        (N,)
            _F32_2D_C,  # shocks             (B, K)  C-contig / single-row
            _I32_1D_C,  # req_to_uix         (N,)
            _F64_1D_C,  # shock_scale        (N,)
            _I8_1D_C,  # shift_types        (N,)
            _F64_2D_C,  # out                (N, B)  C-contig
        ),
    ],
)
def _kernel_apply_shocks(
    base_levels,
    shocks,
    req_to_uix,
    shock_scale,
    shift_types,
    out,
) -> None:
    """
    Inner JIT-compiled apply-shocks loop, parallelised over the N
    required exposures.
    """
    n_req = base_levels.shape[0]
    b_scn = shocks.shape[0]

    for n in nb.prange(n_req):
        base = base_levels[n]
        k = req_to_uix[n]
        scale = shock_scale[n]
        st = shift_types[n]

        # F-contig column view — unit-stride read down the scenario axis.
        shock_col = shocks[:, k]

        # C-contig row view — unit-stride write across the scenario axis.
        out_row = out[n]

        if st == SHIFT_LOG_RETURN_ID:
            for b in range(b_scn):
                out_row[b] = base * math.exp(shock_col[b] * scale)

        elif st == SHIFT_SIMPLE_RETURN_ID:
            for b in range(b_scn):
                out_row[b] = base * (1.0 + shock_col[b] * scale)

        elif st == SHIFT_DIFFERENCE_ID:
            for b in range(b_scn):
                out_row[b] = base + shock_col[b] * scale

        elif st == SHIFT_ABSOLUTE_BASE_RETURN_ID:
            # Inverse of r = (v_t - v_{t-1}) / |v_{t-1}|. The increment scales
            # with |base|, so a negative base (rate/spread) moves in the
            # direction of the shock rather than inverting. Equals SIMPLE when
            # base > 0; differs only for base < 0.
            abs_base = abs(base)
            for b in range(b_scn):
                out_row[b] = base + shock_col[b] * scale * abs_base

        elif st == SHIFT_NO_SHIFT_ID:
            # Held-fixed factor: level is the base level for every scenario.
            for b in range(b_scn):
                out_row[b] = base
