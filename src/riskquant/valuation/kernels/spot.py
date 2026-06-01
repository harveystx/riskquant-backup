# src/riskquant/valuation/kernels/spot.py
"""
Spot pricing kernel.

PV = qty * spot_price * fx_effective

`fx_spots_effective` rows are already the resolved transform (direct or
pre-inverted) — pricer kernels do no inversion math. Per-trade FX is one
gather + one multiply per scenario. The `fx_effective_uix < 0` sentinel
selects the "no FX" branch which is hoisted outside the scenario loop.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant._njit import parallel_njit

# ==============================================================================
# Numba layout-pinned aliases — force C-contiguous strides at compile time so
# the kernel emits straight-line SIMD loops with no dynamic stride checks.
# ==============================================================================

_F64_2D = nb.float64[:, ::1]
_F64_1D = nb.float64[::1]
_I32_1D = nb.int32[::1]


@parallel_njit(
    nb.void(
        _F64_2D,  # out
        _F64_2D,  # prices
        _I32_1D,  # price_uix
        _F64_1D,  # qty
        _F64_2D,  # fx_spots_effective
        _I32_1D,  # fx_spots_effective_uix
    ),
)
def _kernel_price_spot(
    out,
    prices,
    price_uix,
    qty,
    fx_spots_effective,
    fx_spots_effective_uix,
) -> None:
    """
    Fused spot pricing kernel: out[i, s] = prices[p, s] * qty[i] * fx[f, s].

    Zero allocations. Each row is a contiguous gather + scale + (gated)
    multiply streaming through C-contiguous memory.
    """
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        p_idx = price_uix[i]
        f_idx = fx_spots_effective_uix[i]
        q = qty[i]

        p_row = prices[p_idx]
        out_row = out[i]

        if f_idx < 0:
            # No FX leg: straight price * qty.
            for s in range(s_scenarios):
                out_row[s] = p_row[s] * q
        else:
            # FX leg: three-way fused multiply.
            f_row = fx_spots_effective[f_idx]
            for s in range(s_scenarios):
                out_row[s] = p_row[s] * q * f_row[s]
