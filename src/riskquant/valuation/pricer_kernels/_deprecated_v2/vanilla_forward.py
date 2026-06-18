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
        _F64_2D,  # forward_prices
        _F64_2D,  # discount_factors
        _F64_2D,  # fx_spots_effective
        # BLOCK FIELDS
        _F64_1D,  # qty
        _F64_1D,  # strike
        _I32_1D,  # forward_price_uix
        _I32_1D,  # discount_factor_u_uix
        _I32_1D,  # discount_factor_k_uix
        _I32_1D,  # fx_spot_effective_u2s_uix
        _I32_1D,  # fx_spot_effective_k2s_uix
    ),
)
def _kernel_price_vanilla_forward(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    forward_prices,  # (P, S)
    discount_factors,  # (D, S)
    fx_spots_effective,  # (F, S)
    # BLOCK FIELDS
    qty,  # (N,)
    strike,  # (N,)
    forward_price_uix,  # (N,)
    discount_factor_u_uix,  # (N,)
    discount_factor_k_uix,  # (N,)
    fx_spot_effective_u2s_uix,  # (N,)
    fx_spot_effective_k2s_uix,  # (N,)
) -> None:

    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        qty_i = qty[i]
        strike_i = strike[i]

        fwd_price_id = forward_price_uix[i]
        df_u_id = discount_factor_u_uix[i]
        df_k_id = discount_factor_k_uix[i]
        fx_u_id = fx_spot_effective_u2s_uix[i]
        fx_k_id = fx_spot_effective_k2s_uix[i]

        fwd_price_row = forward_prices[fwd_price_id]

        df_u_row = discount_factors[df_u_id]
        df_k_row = discount_factors[df_k_id]

        fx_u_row = fx_spots_effective[fx_u_id]
        fx_k_row = fx_spots_effective[fx_k_id]

        out_row = out[i]

        for s in range(s_scenarios):
            leg_u = fwd_price_row[s] * df_u_row[s] * fx_u_row[s]
            leg_k = strike_i * df_k_row[s] * fx_k_row[s]

            out_row[s] = qty_i * (leg_u - leg_k)
