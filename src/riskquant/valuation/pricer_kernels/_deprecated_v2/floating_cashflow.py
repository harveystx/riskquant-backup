# src/riskquant/valuation/kernels/floating_cashflow.py
"""
Floating cashflow pricing kernel.
"""

from __future__ import annotations

import numba as nb

from riskquant.jit import (
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
        _F64_2D,  # discount_factors
        _F64_2D,  # fx_spots_effective
        # BLOCK FIELDS
        _F64_1D,  # notional
        _F64_1D,  # accrual_factor
        _F64_1D,  # spread
        _I8_1D,  # pay_receive_mult
        _I32_1D,  # forward_rate_uix
        _I32_1D,  # discount_factor_uix
        _I32_1D,  # fx_spot_effective_uix
    ),
)
def _kernel_price_floating_cashflow(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    forward_rates,  # (P, S)
    discount_factors,  # (D, S)
    fx_spots_effective,  # (F, S)
    # BLOCK FIELDS
    notional,  # (N,)
    accrual_factor,  # (N,)
    spread,  # (N,)
    pay_receive_mult,  # (N,)
    forward_rate_uix,  # (N,)
    discount_factor_uix,  # (N,)
    fx_spot_effective_uix,  # (N,)
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
        df_id = discount_factor_uix[i]
        fx_id = fx_spot_effective_uix[i]

        fwd_rates_row = forward_rates[fwd_rate_id]
        df_row = discount_factors[df_id]
        fx_row = fx_spots_effective[fx_id]

        out_row = out[i]

        for s in range(s_scenarios):
            fwd_rate_s = fwd_rates_row[s]

            floating_rate_s = fwd_rate_s + spread_i

            out_row[s] = accrued_notional_i * floating_rate_s * df_row[s] * fx_row[s]
