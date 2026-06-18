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
- Layout-pinned `nb.void(...)` signature: every array dimension is
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

from riskquant.numba import parallel_njit
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)

# Layout-pinned aliases.
_F64_1D = nb.float64[::1]
_F64_2D_C = nb.float64[:, ::1]
_F64_2D_F = nb.float64[::1, :]
_F32_2D_F = nb.float32[::1, :]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]


@parallel_njit(
    nb.void(
        _F64_1D,  # base_levels        (N,)
        _F32_2D_F,  # shocks             (B, K)  F-contig
        _I32_1D,  # req_to_uix         (N,)
        _F64_1D,  # shock_scale        (N,)
        _I8_1D,  # shift_types        (N,)
        _F64_2D_C,  # out                (N, B)  C-contig
    ),
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
