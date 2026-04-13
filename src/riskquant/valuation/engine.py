# src/riskquant/valuation/engine.py
from datetime import date

import numpy as np
from typing import Dict, Callable, Type

# INTERNAL IMPORTS
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.market_data.state import MarketDataMatrix

# BLOCKS
from riskquant.portfolio.blocks.spot_block import SpotBlock
from riskquant.portfolio.blocks.forward_block import (
    ForwardCurveBlock,
    ForwardNodeBlock,
)
from riskquant.portfolio.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionCurveBlock,
    VanillaEuropeanOptionNodeBlock,
)

# PRICERS
from riskquant.valuation.pricers.spot import price_spot_block
from riskquant.valuation.pricers.forward_node import price_forward_node_block
from riskquant.valuation.pricers.forward_curve import price_forward_curve_block
from riskquant.valuation.pricers.vanilla_european_option_node import (
    price_vanilla_european_option_node_block,
)
from riskquant.valuation.pricers.vanilla_european_option_curve import (
    price_vanilla_european_option_curve_block,
)


# PRICER DISPATCH MAP
BLOCK_PRICER_MAP: Dict[Type, Callable] = {
    SpotBlock: price_spot_block,
    ForwardNodeBlock: price_forward_node_block,
    ForwardCurveBlock: price_forward_curve_block,
    VanillaEuropeanOptionNodeBlock: price_vanilla_european_option_node_block,
    VanillaEuropeanOptionCurveBlock: price_vanilla_european_option_curve_block,
}


def run_valuation(
    *,
    portfolio: CompiledPortfolio,
    market_data_mtx: MarketDataMatrix,
) -> np.ndarray:
    n_trades = portfolio.total_trades
    n_scenarios = market_data_mtx.num_scenarios

    # np.empty avoids the O(N) zero-fill overhead of np.zeros
    total_pvs = np.empty((n_trades, n_scenarios), dtype=np.float64)

    # Local binding for maximum speed inside the loop
    # Make local copy of the pricer map to allow mutation (caching) without affecting global state outside this function
    pricer_map = dict(BLOCK_PRICER_MAP)

    cursor = 0  # Cursor to track the position in the total_pvs array

    for block in portfolio.iter_blocks():
        block_type = type(block)
        pricer = pricer_map.get(block_type)

        # Lazy-Caching Dispatch:
        # If not an exact match, find the subclass match and cache it for O(1) next time.
        if pricer is None:
            for cls, candidate in pricer_map.items():
                if isinstance(block, cls):
                    pricer = candidate
                    pricer_map[block_type] = (
                        pricer  # Mutate global map to cache the subclass
                    )
                    break
            else:
                raise ValueError(
                    f"No pricer mapped for block type: {block_type.__name__}"
                )

        # Execute the vectorized pricer
        pvs = pricer(block=block, market_data_mtx=market_data_mtx)

        # Slice-write PVs back contiguously using the cursor
        num_block_trades = block.num_trades
        total_pvs[cursor : cursor + num_block_trades, :] = pvs

        cursor += num_block_trades

    return total_pvs
