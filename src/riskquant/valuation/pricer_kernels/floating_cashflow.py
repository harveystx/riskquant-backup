# src/riskquant/valuation/pricer_kernels/floating_cashflow.py
"""
Floating cashflow pricing kernel.
"""

from __future__ import annotations

import numba as nb

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _I8_1D_C,
    _I32_1D_C,
    _NB_VOID,
    parallel_njit,
)


@parallel_njit(
    _NB_VOID(
        # OUTPUT
        _F64_2D_C,  # out
        # MARKET DATA
        _F64_2D_C,  # forward_rates
        _F64_2D_C,  # effective_discount_factors
        # BLOCK FIELDS
        _F64_1D_C,  # notional
        _F64_1D_C,  # accrual_factor
        _F64_1D_C,  # spread
        _I8_1D_C,  # pay_receive_mult
        _I32_1D_C,  # forward_rate_uix
        _I32_1D_C,  # effective_discount_factor_uix
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
