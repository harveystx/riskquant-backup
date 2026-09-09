# src/riskquant/valuation/kernels/principal_cashflow.py
"""
Principal cashflow pricing kernel.
"""

from __future__ import annotations

import numba as nb

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _I32_1D_C,
    parallel_njit,
)


@parallel_njit(
    nb.void(
        # OUTPUT
        _F64_2D_C,  # out
        # MARKET DATA
        _F64_2D_C,  # discount_factors
        _F64_2D_C,  # fx_spots_effective
        # BLOCK FIELDS
        _F64_1D_C,  # amount
        _I32_1D_C,  # discount_factor_uix
        _I32_1D_C,  # fx_spot_effective_uix
    ),
)
def _kernel_price_principal_cashflow(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    discount_factors,  # (D, S)
    fx_spots_effective,  # (F, S)
    # BLOCK FIELDS
    amount,  # (N,)
    discount_factor_uix,  # (N,)
    fx_spot_effective_uix,  # (N,)
) -> None:
    """
    Pricing kernel for principal cashflows.
    """
    m_cashflows, s_scenarios = out.shape

    for i in nb.prange(m_cashflows):
        cf_i = amount[i]

        df_id = discount_factor_uix[i]
        fx_id = fx_spot_effective_uix[i]

        df_row = discount_factors[df_id]
        fx_row = fx_spots_effective[fx_id]

        out_row = out[i]

        for s in range(s_scenarios):
            out_row[s] = cf_i * df_row[s] * fx_row[s]
