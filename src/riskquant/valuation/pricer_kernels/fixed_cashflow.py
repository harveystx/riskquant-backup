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
        _F64_2D,  # discount_factors
        _F64_2D,  # fx_spots_effective
        # BLOCK FIELDS
        _F64_1D,  # notional
        _F64_1D,  # accrual_factor
        _F64_1D,  # fixed_rate
        _I8_1D,  # pay_receive_mult
        _I32_1D,  # discount_factor_uix
        _I32_1D,  # fx_spot_effective_uix
    ),
)
def _kernel_price_fixed_cashflow(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    discount_factors,  # (D, S)
    fx_spots_effective,  # (F, S)
    # BLOCK FIELDS
    notional,  # (N,)
    accrual_factor,  # (N,)
    fixed_rate,  # (N,)
    pay_receive_mult,  # (N,)
    discount_factor_uix,  # (N,)
    fx_spot_effective_uix,  # (N,)
) -> None:
    """
    Pricing kernel for fixed cashflows.
    """

    m_cashflows, s_scenarios = out.shape

    for i in nb.prange(m_cashflows):
        cf_i = notional[i] * accrual_factor[i] * fixed_rate[i] * pay_receive_mult[i]

        df_id = discount_factor_uix[i]
        fx_id = fx_spot_effective_uix[i]

        df_row = discount_factors[df_id]
        fx_row = fx_spots_effective[fx_id]

        out_row = out[i]

        for s in range(s_scenarios):
            out_row[s] = cf_i * df_row[s] * fx_row[s]
