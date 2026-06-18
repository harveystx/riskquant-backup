# src/riskquant/valuation/pricers/spot.py
"""
Spot pricer.

PV = qty * spot_price * fx_effective

`fx_spots_effective` rows are already the resolved transform (direct or
pre-inverted) — pricer kernels do no inversion math. Per-trade FX is one
gather + one multiply per scenario. The `fx_effective_uix < 0` sentinel
selects the "no FX" branch which is hoisted outside the scenario loop.
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
        valuation_state.spot_prices,
        valuation_state.fx_spots_effective,
        # BLOCK FIELDS
        block.qty,
        block.spot_price_uix,
        block.fx_spot_effective_uix,
    )
