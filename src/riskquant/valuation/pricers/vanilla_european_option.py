# src/riskquant/valuation/pricers/vanilla_european_option.py
"""
Vanilla European option pricer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.pricer_kernels.vanilla_european_option import (
    _kernel_price_vanilla_european_option,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import ValuationState
    from riskquant.valuation.blocks.vanilla_european_option_block import (
        VanillaEuropeanOptionBlock,
    )


def price_vanilla_european_option_block(
    *,
    block: VanillaEuropeanOptionBlock,
    valuation_state: ValuationState,
    out: npt.NDArray[np.float64],
) -> None:
    """
    Price a compiled vanilla European option block.

    All market-data interpretation has already been resolved into
    valuation factors by the market-data compiler.
    """
    _kernel_price_vanilla_european_option(
        # OUTPUT
        out,
        # MARKET DATA
        valuation_state.forward_prices_strike_ccy,
        valuation_state.option_log_forward_prices_strike_ccy,
        valuation_state.vols,
        valuation_state.effective_discount_factors,
        # BLOCK FIELDS (TRADE SPECIFICS)
        block.qty,
        block.strike,
        block.log_strike,
        block.cp_mult,
        block.sqrt_tte,
        block.half_tte,
        # BLOCK FIELDS (LINKAGES / UIX)
        block.forward_price_strike_ccy_uix,
        block.option_log_forward_price_strike_ccy_uix,
        block.vol_uix,
        block.effective_discount_factor_uix,
    )
