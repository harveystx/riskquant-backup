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

from riskquant.numba import parallel_njit

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
        _F64_2D,  # spot_prices
        _F64_2D,  # fx_spots_effective
        _F64_1D,  # qty
        _I32_1D,  # spot_prices_uix
        _I32_1D,  # fx_spots_effective_uix
    ),
)
def _kernel_price_spot(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    spot_prices,  # (P, S)
    fx_spots_effective,  # (F, S)
    # BLOCK FIELDS
    qty,  # (N,)
    spot_price_uix,  # (N,)
    fx_spot_effective_uix,  # (N,)
) -> None:
    """
    Fused spot pricing kernel: out[i, s] = spot_prices[p, s] * qty[i] * fx[f, s].

    Zero allocations. Each row is a contiguous gather + scale + (gated)
    multiply streaming through C-contiguous memory.
    """
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        qty_i = qty[i]

        spot_price_id = spot_price_uix[i]
        fx_id = fx_spot_effective_uix[i]

        spot_prices_row = spot_prices[spot_price_id]
        fx_row = fx_spots_effective[fx_id]

        out_row = out[i]

        for s in range(s_scenarios):
            out_row[s] = qty_i * spot_prices_row[s] * fx_row[s]
