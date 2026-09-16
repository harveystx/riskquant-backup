# src/riskquant/valuation/pricer_kernels/principal_cashflow.py
"""
Principal cashflow pricing kernel.

    PV = amount * effective_discount_factor

`amount` is signed: receive > 0, pay < 0. There is no pay/receive multiplier on
this block -- see `blocks/discounted_cashflow_block.py`.

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
        _F64_1D_C_RO,  # amount
        _I32_1D_C_RO,  # effective_discount_factor_uix
    ),
)
def _kernel_price_principal_cashflow(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    effective_discount_factors,  # (D, S)
    # BLOCK FIELDS
    amount,  # (N,)
    effective_discount_factor_uix,  # (N,)
) -> None:
    """
    Pricing kernel for principal cashflows.
    """
    m_cashflows, s_scenarios = out.shape

    for i in nb.prange(m_cashflows):
        cf_i = amount[i]

        eff_df_id = effective_discount_factor_uix[i]
        eff_dfs_row = effective_discount_factors[eff_df_id]

        out_row = out[i]

        for s in range(s_scenarios):
            out_row[s] = cf_i * eff_dfs_row[s]
