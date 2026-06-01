# src/riskquant/valuation/pricers/vanilla_forward_node.py
"""
Vanilla forward pricer, node variant.

    leg_u = forward_node * fx_u_effective * df_u(stl)
    leg_k = strike       * fx_k_effective * df_k(stl)
    PV    = qty * (leg_u - leg_k)

FX rows are already EFFECTIVE (direct or pre-inverted) inside
`MarketDataMatrix.fx_spots_effective` — pricer kernels do NO inversion math.
Each FX gather is one indexed read + one multiply per scenario.

Optional legs (FX / DF) are signalled by `INVALID_UIX (< 0)` in the
corresponding `_uix` field. These predicates are trade-level invariants;
Numba's LICM hoists them outside the inner scenario loop, so the inner loop
streams straight-line arithmetic over C-contiguous memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.valuation.kernels.vanilla_forward_node import (
    _kernel_price_vanilla_forward_node,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import MarketDataMatrix
    from riskquant.portfolio.blocks.vanilla_forward_block import (
        VanillaForwardNodeBlock,
    )


def price_vanilla_forward_node_block(
    *,
    block: VanillaForwardNodeBlock,
    market_data_mtx: MarketDataMatrix,
    out: npt.NDArray[np.float64],
) -> None:
    """Forward-node pricer entrypoint."""
    _kernel_price_vanilla_forward_node(
        out,
        market_data_mtx.forward_nodes,
        market_data_mtx.fx_spots_effective,
        market_data_mtx.discount_factor_grid,
        block.qty,
        block.strike,
        block.price_uix,
        block.fx_under_to_settle_effective_uix,
        block.under_discount_curve_uix,
        block.fx_strike_to_settle_effective_uix,
        block.strike_discount_curve_uix,
        block.settlement_date_uix,
    )
