# src/riskquant/valuation/kernels/vanilla_european_option.py
"""Pricing kernels for vanilla European options."""

from __future__ import annotations

import numba as nb

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    _I8_1D,
    _I32_1D,
    parallel_njit,
)
from riskquant.valuation.pricing_models.black76.model import (
    _black76_fv_kernel_primitive,
)


@parallel_njit(
    nb.void(
        # OUTPUT
        _F64_2D,  # out
        # MARKET DATA
        _F64_2D,  # forward_prices_strike_ccy
        _F64_2D,  # option_log_forward_prices_strike_ccy
        _F64_2D,  # vols
        _F64_2D,  # effective_discount_factors
        # BLOCK FIELDS (TRADE SPECIFICS)
        _F64_1D,  # qty
        _F64_1D,  # strike
        _F64_1D,  # log_strike
        _I8_1D,  # cp_mult
        _F64_1D,  # sqrt_tte
        _F64_1D,  # half_tte
        # BLOCK FIELDS (LINKAGES / UIX)
        _I32_1D,  # forward_price_strike_ccy_uix
        _I32_1D,  # option_log_forward_price_strike_ccy_uix
        _I32_1D,  # vol_uix
        _I32_1D,  # effective_discount_factor_uix
    ),
)
def _kernel_price_vanilla_european_option(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    forward_prices_strike_ccy,  # (P, S)
    option_log_forward_prices_strike_ccy,  # (P, S)
    vols,  # (V, S)
    effective_discount_factors,  # (D, S)
    # BLOCK FIELDS (TRADE SPECIFICS)
    qty,  # (N,)
    strike,  # (N,)
    log_strike,  # (N,)
    cp_mult,  # (N,)
    sqrt_tte,  # (N,)
    half_tte,  # (N,)
    # BLOCK FIELDS (LINKAGES / UIX)
    forward_price_strike_ccy_uix,  # (N,)
    option_log_forward_price_strike_ccy_uix,  # (N,)
    vol_uix,  # (N,)
    effective_discount_factor_uix,  # (N,)
) -> None:
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        # --- TRADE SPECIFICS (FOR ROW) ---
        qty_i = qty[i]
        strike_i = strike[i]
        log_strike_i = log_strike[i]
        sqrt_tte_i = sqrt_tte[i]
        half_tte_i = half_tte[i]
        cp_mult_i = cp_mult[i]

        # --- UIX (FOR ROW) AND MARKET DATA ---

        # FORWARD PRICE (STRIKE CCY)
        fwd_price_strike_ccy_id = forward_price_strike_ccy_uix[i]
        fwd_price_strike_ccy_row = forward_prices_strike_ccy[fwd_price_strike_ccy_id]

        # LOG FORWARD PRICE (STRIKE CCY)
        option_log_fwd_price_strike_ccy_id = option_log_forward_price_strike_ccy_uix[i]
        option_log_fwd_price_strike_ccy_row = option_log_forward_prices_strike_ccy[
            option_log_fwd_price_strike_ccy_id
        ]

        # VOL
        vol_id = vol_uix[i]
        vols_row = vols[vol_id]

        # EFFECTIVE DISCOUNT FACTOR (SETTLE CCY)
        # This is the discount factor that converts from the option payoff currency
        # (strike ccy) to the settlement currency.
        eff_df_id = effective_discount_factor_uix[i]
        eff_dfs_row = effective_discount_factors[eff_df_id]

        # --- PRICING ACROSS SCENARIOS (ACROSS ROW) ---
        for s in range(s_scenarios):
            log_fk = option_log_fwd_price_strike_ccy_row[s] - log_strike_i

            vol_s = vols_row[s]
            vol_sqrt_tte_s = vol_s * sqrt_tte_i
            vol2_half_tte_s = vol_s * vol_s * half_tte_i

            fv_s = _black76_fv_kernel_primitive(
                fwd_price_strike_ccy_row[s],
                log_fk,
                strike_i,
                vol_sqrt_tte_s,
                vol2_half_tte_s,
                cp_mult_i,
            )

            # PRESENT VALUE
            #
            # PV = FX(StrikeCcy->SettleCcy) * DF(StrikeCcy) * QTY * FV(StrikeCcy)
            #
            pv_s = eff_dfs_row[s] * qty_i * fv_s

            out[i, s] = pv_s
