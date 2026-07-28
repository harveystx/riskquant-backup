# src/riskquant/sensitivity_pnl/kernels/displacement.py
"""
Zero-allocation levels -> displacement kernel.

Converts a `(N_req, B)` shocked-levels batch into the `(Ka, B)` FLOAT32
displacement matrix the Taylor kernel consumes, gathering only the factors the
book is actually sensitive to.

Why the output is float32
-------------------------
The displacement matrix is the single hottest read in the engine: the Taylor
kernel streams `NNZ * B` elements out of it per batch, against `NNZ` reads of
every other array. Halving its element width halves the dominant traffic and
measured a 1.35x end-to-end speedup.

The precision cost is nil in context. f32 carries ~6e-8 relative error on the
displacement; the second-order Taylor form itself truncates at O(x^3), which is
~1.7e-5 relative at a 1% shock and ~1e-3 at a 4-sigma shock. The representation
error is three to five orders of magnitude below the model error it feeds.
`riskquant.constants` already pins scenario shocks at float32 for exactly this
reason — a float64 displacement matrix was the outlier, not this.

The arithmetic is still done in f64 and narrowed on store: `L - L0` for a large
level and a small shock is a cancelling subtraction, and doing it in f32 would
lose significance BEFORE the narrowing rather than after.

Memory layout
-------------
- `shocked_levels` is C-contiguous f64 `(N_req, B)` — the identical buffer the
  full-reval runner already fills, so both PnL blocks are driven by one market
  state. Row `n` is a contiguous run of B levels.
- `out` is C-contiguous f32 `(Ka, B_capacity)`; row `k` is a contiguous run.
- `prange` over the active-factor axis: each thread owns whole rows of `out`.

The per-factor convention branch is hoisted above the scenario loop, so each
inner loop is straight-line SIMD over unit-stride memory with no branch inside.
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import parallel_njit
from riskquant.sensitivity_pnl.conventions import (
    DISPLACEMENT_ABSOLUTE_ID,
    DISPLACEMENT_RELATIVE_ID,
)

_F64_1D = nb.float64[::1]
_F64_2D_C = nb.float64[:, ::1]
_F32_2D_C = nb.float32[:, ::1]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]


@parallel_njit(
    nb.void(
        _F64_1D,  # base_levels        (N_req,)
        _F64_2D_C,  # shocked_levels   (N_req, S) f64 C-contig
        _I32_1D,  # active_level_uix   (Ka,)
        _I8_1D,  # convention_ids      (Ka,)
        nb.int64,  # batch_len
        nb.int64,  # levels_offset
        _F32_2D_C,  # out              (Ka, B_capacity) f32 C-contig
    ),
)
def kernel_build_displacement(
    base_levels,
    shocked_levels,
    active_level_uix,
    convention_ids,
    batch_len,
    levels_offset,
    out,
) -> None:
    """
    Displacement per active factor, computed f64 and narrowed on store.

    Unknown convention ids are rejected at compile time, so the terminal branch
    is the LOG arm rather than a guard.
    """
    num_active = active_level_uix.shape[0]

    for k in nb.prange(num_active):
        n = active_level_uix[k]
        base = base_levels[n]
        src = shocked_levels[n]
        dst = out[k]
        convention = convention_ids[k]

        if convention == DISPLACEMENT_ABSOLUTE_ID:
            for s in range(batch_len):
                dst[s] = nb.float32(src[levels_offset + s] - base)
        elif convention == DISPLACEMENT_RELATIVE_ID:
            inv_base = 1.0 / base
            for s in range(batch_len):
                dst[s] = nb.float32(src[levels_offset + s] * inv_base - 1.0)
        else:
            inv_base = 1.0 / base
            for s in range(batch_len):
                dst[s] = nb.float32(math.log(src[levels_offset + s] * inv_base))
