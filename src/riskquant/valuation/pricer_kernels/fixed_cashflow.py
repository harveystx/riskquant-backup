# src/riskquant/valuation/pricer_kernels/fixed_cashflow.py
"""
Fixed cashflow pricing kernel.

    PV = notional * accrual_factor * fixed_rate * pay_receive_mult
         * effective_discount_factor

The trade-specific product is scenario-independent and is hoisted out of the
scenario loop.

Preconditions (established by the block compiler, not here)
----------------------------------------------------------
* `effective_discount_factor_uix[i]` is a valid dense row index;
* `effective_discount_factors.shape[1] >= out.shape[1]`.
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
        _F64_2D_C_RO,  # effective_discount_factors
        # BLOCK FIELDS
        _F64_1D_C_RO,  # notional
        _F64_1D_C_RO,  # accrual_factor
        _F64_1D_C_RO,  # fixed_rate
        _I8_1D_C_RO,  # pay_receive_mult
        _I32_1D_C_RO,  # effective_discount_factor_uix
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
