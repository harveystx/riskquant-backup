# src/riskquant/valuation/pricer_kernels/vanilla_european_option.py
"""
Pricing kernels for vanilla European options.

    FV = cp * ( F_k * Phi(cp * d1) - K * Phi(cp * d2) )        (strike ccy, at T_exp)
    PV = effective_discount_factor * qty * FV                  (settlement ccy, at 0)

Preconditions (established by the block compiler, not here)
----------------------------------------------------------
* every `_uix` array holds valid dense row indices into its matrix;
* `vols[v, s] > 0` and `sqrt_tte[i] > 0`, so `vol_sqrt_tte > 0` and the d1
  division is well-defined;
* `forward_prices_strike_ccy[p, s] > 0` and `strike[i] > 0`;
* `option_log_forward_prices_strike_ccy[j, s] == log(forward_prices_strike_ccy[p, s])`
  for the paired (p, j) of each row -- see the block docstring;
* every market matrix has `shape[1] >= out.shape[1]`.

The kernel performs no branch and no validation.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import (
    _F64_1D_C_RO,
    _F64_2D_C,
    _F64_2D_C_RO,
    _I8_1D_C_RO,
    _I32_1D_C_RO,
    _NB_VOID,
    parallel_njit,
)
from riskquant.valuation.pricing_models.black76.model import (
    _black76_fv_kernel_primitive,
)


@parallel_njit(
    _NB_VOID(
        # OUTPUT
        _F64_2D_C,  # out
        # MARKET DATA
        _F64_2D_C_RO,  # forward_prices_strike_ccy
        _F64_2D_C_RO,  # option_log_forward_prices_strike_ccy
        _F64_2D_C_RO,  # vols
        _F64_2D_C_RO,  # effective_discount_factors
        # BLOCK FIELDS (TRADE SPECIFICS)
        _F64_1D_C_RO,  # qty
        _F64_1D_C_RO,  # strike
        _F64_1D_C_RO,  # log_strike
        _I8_1D_C_RO,  # cp_mult
        _F64_1D_C_RO,  # sqrt_tte
        _F64_1D_C_RO,  # half_tte
        # BLOCK FIELDS (LINKAGES / UIX)
        _I32_1D_C_RO,  # forward_price_strike_ccy_uix
        _I32_1D_C_RO,  # option_log_forward_price_strike_ccy_uix
        _I32_1D_C_RO,  # vol_uix
        _I32_1D_C_RO,  # effective_discount_factor_uix
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
