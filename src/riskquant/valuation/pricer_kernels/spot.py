# src/riskquant/valuation/pricer_kernels/spot.py
"""
Spot pricing kernel.

    PV = qty * effective_spot_price

`ValuationState.effective_spot_prices` rows are already the fully-resolved
settlement-currency price: the market-data compiler has folded the asset-to-
settle FX rate (direct or pre-inverted) into the asset's spot price. The kernel
does no inversion, no branching and no sentinel handling -- one gather and one
multiply per scenario.

Spot carries no time value, so no discount factor is applied.

Preconditions (established by the block compiler, not here)
----------------------------------------------------------
* `effective_spot_price_uix[i]` is a valid dense row index into
  `effective_spot_prices`. The kernel runs under `boundscheck=False`.
* `effective_spot_prices.shape[1] >= out.shape[1]`.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import (
    _F64_1D_C_RO,
    _F64_2D_C,
    _F64_2D_C_RO,
    _I32_1D_C_RO,
    _NB_VOID,
    parallel_njit,
)


@parallel_njit(
    _NB_VOID(
        # OUTPUT
        _F64_2D_C,  # out
        # MARKET DATA
        _F64_2D_C_RO,  # effective_spot_prices
        # BLOCK FIELDS
        _F64_1D_C_RO,  # qty
        _I32_1D_C_RO,  # effective_spot_price_uix
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

    Zero allocations. Each row is a contiguous gather + scale streaming through
    C-contiguous memory.
    """
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        qty_i = qty[i]

        eff_spot_price_id = effective_spot_price_uix[i]

        eff_spot_prices_row = effective_spot_prices[eff_spot_price_id]

        out_row = out[i]

        for s in range(s_scenarios):
            out_row[s] = qty_i * eff_spot_prices_row[s]
