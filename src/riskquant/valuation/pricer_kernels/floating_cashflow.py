# src/riskquant/valuation/pricer_kernels/floating_cashflow.py
"""
Floating cashflow pricing kernel.

    PV = notional * accrual_factor * pay_receive_mult
         * (forward_rate + spread)
         * effective_discount_factor

The accrued-notional product is scenario-independent and is hoisted out of the
scenario loop; the spread add stays inside it because the forward rate is
scenario-dependent.

Preconditions (established by the block compiler, not here)
----------------------------------------------------------
* both `_uix` arrays hold valid dense row indices into their matrices;
* both market matrices have `shape[1] >= out.shape[1]`;
* `forward_rates` are annualised SIMPLE rates on the same basis as
  `accrual_factor`.
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


@parallel_njit(
    _NB_VOID(
        # OUTPUT
        _F64_2D_C,  # out
        # MARKET DATA
        _F64_2D_C_RO,  # forward_rates
        _F64_2D_C_RO,  # effective_discount_factors
        # BLOCK FIELDS
        _F64_1D_C_RO,  # notional
        _F64_1D_C_RO,  # accrual_factor
        _F64_1D_C_RO,  # spread
        _I8_1D_C_RO,  # pay_receive_mult
        _I32_1D_C_RO,  # forward_rate_uix
        _I32_1D_C_RO,  # effective_discount_factor_uix
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
