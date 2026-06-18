# src/riskquant/valuation/pricers/vanilla_forward.py
"""
Vanilla forward pricer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.pricer_kernels.vanilla_forward import (
    _kernel_price_vanilla_forward,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import ValuationState
    from riskquant.valuation.blocks.vanilla_forward_block import (
        VanillaForwardBlock,
    )


def price_vanilla_forward_block(
    *,
    block: VanillaForwardBlock,
    valuation_state: ValuationState,
    out: npt.NDArray[np.float64],
) -> None:
    """
    Price a compiled vanilla-forward block.

    All market-data interpretation has already been resolved into
    valuation factors by the market-data compiler.
    """
    _kernel_price_vanilla_forward(
        # OUTPUT
        out,
        # MARKET DATA
        valuation_state.forward_prices,
        valuation_state.discount_factors,
        valuation_state.fx_spots_effective,
        # BLOCK FIELDS
        block.qty,
        block.strike,
        block.forward_price_uix,
        block.discount_factor_u_uix,
        block.discount_factor_k_uix,
        block.fx_spot_effective_u2s_uix,
        block.fx_spot_effective_k2s_uix,
    )
