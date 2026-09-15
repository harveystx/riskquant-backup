# src/riskquant/valuation/pricer_kernels/fixed_cashflow.py
"""
Fixed cashflow pricing kernel.
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
        _F64_2D_C,  # effective_discount_factors
        # BLOCK FIELDS
        _F64_1D_C,  # notional
        _F64_1D_C,  # accrual_factor
        _F64_1D_C,  # fixed_rate
        _I8_1D_C,  # pay_receive_mult
        _I32_1D_C,  # effective_discount_factor_uix
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
