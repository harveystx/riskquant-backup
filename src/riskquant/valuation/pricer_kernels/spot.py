# src/riskquant/valuation/pricer_kernels/spot.py
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

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _I32_1D_C,
    parallel_njit,
)

# ==============================================================================
# Numba layout-pinned aliases — force C-contiguous strides at compile time so
# the kernel emits straight-line SIMD loops with no dynamic stride checks.
# ==============================================================================



@parallel_njit(
    nb.void(
        _F64_2D_C,  # out
        _F64_2D_C,  # effective_spot_prices
        _F64_1D_C,  # qty
        _I32_1D_C,  # effective_spot_price_uix
    ),
)
def _kernel_price_spot(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    effective_spot_prices,  # (P, S)
    # BLOCK FIELDS
    qty,  # (N,)
    effective_spot_price_uix,  # (N,)
) -> None:
    """
    Fused spot pricing kernel: out[i, s] = effective_spot_prices[p, s] * qty[i].

    Zero allocations. Each row is a contiguous gather + scale + (gated)
    multiply streaming through C-contiguous memory.
    """
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        qty_i = qty[i]

        eff_spot_price_id = effective_spot_price_uix[i]

        eff_spot_prices_row = effective_spot_prices[eff_spot_price_id]

        out_row = out[i]

        for s in range(s_scenarios):
            out_row[s] = qty_i * eff_spot_prices_row[s]
