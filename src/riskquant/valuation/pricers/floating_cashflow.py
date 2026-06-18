# src/riskquant/valuation/pricers/floating_cashflow.py
"""
Floating cashflow pricer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.pricer_kernels.floating_cashflow import (
    _kernel_price_floating_cashflow,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import ValuationState
    from riskquant.valuation.blocks.discounted_cashflow_block import (
        FloatingCashflowBlock,
    )


def price_floating_cashflow_block(
    *,
    block: FloatingCashflowBlock,
    valuation_state: ValuationState,
    out: npt.NDArray[np.float64],
) -> None:
    """Floating cashflow pricer entrypoint. All arrays must be C-contiguous."""
    _kernel_price_floating_cashflow(
        # OUTPUT
        out,
        # MARKET DATA
        valuation_state.forward_rates,
        valuation_state.discount_factors,
        valuation_state.fx_spots_effective,
        # BLOCK FIELDS
        block.notional,
        block.accrual_factor,
        block.spread,
        block.pay_receive_mult,
        block.forward_rate_uix,
        block.discount_factor_uix,
        block.fx_spot_effective_uix,
    )
