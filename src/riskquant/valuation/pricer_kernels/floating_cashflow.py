# src/riskquant/valuation/kernels/floating_cashflow.py
"""
Floating cashflow pricing kernel.
"""

from __future__ import annotations

import numba as nb

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    _I8_1D,
    _I32_1D,
    parallel_njit,
)


@parallel_njit(
    nb.void(
        # OUTPUT
        _F64_2D,  # out
        # MARKET DATA
        _F64_2D,  # forward_rates
        _F64_2D,  # effective_discount_factors
        # BLOCK FIELDS
        _F64_1D,  # notional
        _F64_1D,  # accrual_factor
        _F64_1D,  # spread
        _I8_1D,  # pay_receive_mult
        _I32_1D,  # forward_rate_uix
        _I32_1D,  # effective_discount_factor_uix
    ),
)
def _kernel_price_floating_cashflow(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    forward_rates,  # (P, S)
    effective_discount_factors,  # (D, S)
    # BLOCK FIELDS
    notional,  # (N,)
    accrual_factor,  # (N,)
    spread,  # (N,)
    pay_receive_mult,  # (N,)
    forward_rate_uix,  # (N,)
    effective_discount_factor_uix,  # (N,)
) -> None:
    m_cashflows, s_scenarios = out.shape

    for i in nb.prange(m_cashflows):
        # --- TRADE SPECIFIC DATA ---
        notional_i = notional[i]
        accrual_factor_i = accrual_factor[i]
        spread_i = spread[i]
        pay_receive_mult_i = pay_receive_mult[i]

        accrued_notional_i = notional_i * accrual_factor_i * pay_receive_mult_i

        # --- LINKAGES TO MARKET DATA ---

        fwd_rate_id = forward_rate_uix[i]
        fwd_rates_row = forward_rates[fwd_rate_id]

        eff_df_id = effective_discount_factor_uix[i]
        eff_dfs_row = effective_discount_factors[eff_df_id]

        out_row = out[i]

        for s in range(s_scenarios):
            floating_rate_s = fwd_rates_row[s] + spread_i

            out_row[s] = accrued_notional_i * floating_rate_s * eff_dfs_row[s]
