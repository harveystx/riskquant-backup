# src/riskquant/valuation/pricers/vanilla_forward_curve.py
"""
Vanilla forward pricer, curve variant.

Identical math to the node variant; the only structural change is that the
underlying scalar comes from `price_curve_grid[curve_uix, delivery_uix, s]`
instead of `forward_nodes[uix, s]`.

FX rows are already EFFECTIVE in `MarketDataMatrix.fx_spots_effective`;
no inversion math survives into this kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.kernels.vanilla_forward_curve import (
    _kernel_price_vanilla_forward_curve,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import MarketDataMatrix
    from riskquant.portfolio.blocks.vanilla_forward_block import (
        VanillaForwardCurveBlock,
    )


def price_vanilla_forward_curve_block(
    *,
    block: VanillaForwardCurveBlock,
    market_data_mtx: MarketDataMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    """Forward-curve pricer entrypoint."""
    _kernel_price_vanilla_forward_curve(
        out,
        market_data_mtx.price_curve_grid,
        market_data_mtx.fx_spots_effective,
        market_data_mtx.discount_factor_grid,
        block.qty,
        block.strike,
        block.price_curve_uix,
        block.delivery_date_uix,
        block.fx_under_to_settle_effective_uix,
        block.under_discount_curve_uix,
        block.fx_strike_to_settle_effective_uix,
        block.strike_discount_curve_uix,
        block.settlement_date_uix,
    )
