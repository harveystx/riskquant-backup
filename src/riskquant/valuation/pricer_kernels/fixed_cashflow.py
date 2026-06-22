# src/riskquant/valuation/kernels/fixed_cashflow.py
"""
Fixed cashflow pricing kernel.
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
        _F64_2D,  # effective_discount_factors
        # BLOCK FIELDS
        _F64_1D,  # notional
        _F64_1D,  # accrual_factor
        _F64_1D,  # fixed_rate
        _I8_1D,  # pay_receive_mult
        _I32_1D,  # effective_discount_factor_uix
    ),
)
def _kernel_price_fixed_cashflow(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    effective_discount_factors,  # (D, S)
    # BLOCK FIELDS
    notional,  # (N,)
    accrual_factor,  # (N,)
    fixed_rate,  # (N,)
    pay_receive_mult,  # (N,)
    effective_discount_factor_uix,  # (N,)
) -> None:
    """
    Pricing kernel for fixed cashflows.
    """

    m_cashflows, s_scenarios = out.shape

    for i in nb.prange(m_cashflows):
        cf_i = notional[i] * accrual_factor[i] * fixed_rate[i] * pay_receive_mult[i]

        eff_df_id = effective_discount_factor_uix[i]
        eff_dfs_row = effective_discount_factors[eff_df_id]

        out_row = out[i]

        for s in range(s_scenarios):
            out_row[s] = cf_i * eff_dfs_row[s]
