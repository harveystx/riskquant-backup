# src/riskquant/valuation/kernels/vanilla_forward.py
"""
Vanilla forward pricing kernel.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    _I32_1D,
    parallel_njit,
)


@parallel_njit(
    nb.void(
        # OUTPUT
        _F64_2D,  # out
        # MARKET DATA
        _F64_2D,  # effective_forward_prices
        _F64_2D,  # effective_discount_factors
        # BLOCK FIELDS
        _F64_1D,  # qty
        _F64_1D,  # strike
        _I32_1D,  # effective_forward_price_uix
        _I32_1D,  # effective_discount_factor_uix
    ),
)
def _kernel_price_vanilla_forward(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    effective_forward_prices,  # (P, S)
    effective_discount_factors,  # (D, S)
    # BLOCK FIELDS
    qty,  # (N,)
    strike,  # (N,)
    effective_forward_price_uix,  # (N,)
    effective_discount_factor_uix,  # (N,)
) -> None:

    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        qty_i = qty[i]
        strike_i = strike[i]

        eff_fwd_price_id = effective_forward_price_uix[i]
        eff_fwd_prices_row = effective_forward_prices[eff_fwd_price_id]

        eff_df_id = effective_discount_factor_uix[i]
        eff_dfs_row = effective_discount_factors[eff_df_id]

        out_row = out[i]

        for s in range(s_scenarios):
            pv_leg_u = eff_fwd_prices_row[s]
            pv_leg_k = strike_i * eff_dfs_row[s]

            out_row[s] = qty_i * (pv_leg_u - pv_leg_k)
