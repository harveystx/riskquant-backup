# src/riskquant/valuation/pricers/fixed_cashflow.py
"""
Fixed cashflow pricer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.pricer_kernels.fixed_cashflow import (
    _kernel_price_fixed_cashflow,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import ValuationState
    from riskquant.valuation.blocks.discounted_cashflow_block import FixedCashflowBlock


def price_fixed_cashflow_block(
    *,
    block: FixedCashflowBlock,
    valuation_state: ValuationState,
    out: npt.NDArray[np.float64],
) -> None:
    """Fixed cashflow pricer entrypoint. All arrays must be C-contiguous."""
    _kernel_price_fixed_cashflow(
        # OUTPUT
        out,
        # MARKET DATA
        valuation_state.discount_factors,
        valuation_state.fx_spots_effective,
        # BLOCK FIELDS
        block.notional,
        block.accrual_factor,
        block.fixed_rate,
        block.pay_receive_mult,
        block.discount_factor_uix,
        block.fx_spot_effective_uix,
    )
