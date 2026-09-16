# src/riskquant/valuation/pricers/spot.py
"""
Spot pricer.

Unpacks `SpotBlock` into flat arrays and calls the kernel. No arithmetic, no
market-data interpretation, no validation -- the dataclass does not cross the
JIT boundary.

    PV = qty * effective_spot_price

FX is already folded into `effective_spot_prices` by the market-data compiler;
there is no FX gather and no inversion at this tier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.pricer_kernels.spot import _kernel_price_spot

if TYPE_CHECKING:
    from riskquant.market_data.state import ValuationState
    from riskquant.valuation.blocks.spot_block import SpotBlock


def price_spot_block(
    *,
    block: SpotBlock,
    valuation_state: ValuationState,
    out: npt.NDArray[np.float64],
) -> None:
    """Spot pricer entrypoint. All arrays must be C-contiguous and dtype-stable."""
    _kernel_price_spot(
        # OUTPUT
        out,
        # MARKET DATA
        valuation_state.effective_spot_prices,
        # BLOCK FIELDS
        block.qty,
        block.effective_spot_price_uix,
    )
