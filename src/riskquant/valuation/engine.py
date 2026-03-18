# src/riskquant/valuation/engine.py
import numpy as np
from typing import Dict, Callable

from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.market_data.state import MarketDataMatrix
from riskquant.valuation.pricers.spot import price_spot_block
from riskquant.valuation.pricers.forward_node import price_forward_node_block
from riskquant.valuation.pricers.forward_curve import price_forward_curve_block
from riskquant.valuation.pricers.vanilla_european_option_node import (
    price_vanilla_european_option_node_block,
)
from riskquant.valuation.pricers.vanilla_european_option_curve import (
    price_vanilla_european_option_curve_block,
)

BLOCK_PRICER_MAP: Dict[str, Callable] = {
    "SpotBlock": price_spot_block,
    "ForwardNodeBlock": price_forward_node_block,
    "ForwardCurveBlock": price_forward_curve_block,
    "VanillaEuropeanOptionNodeBlock": price_vanilla_european_option_node_block,
    "VanillaEuropeanOptionCurveBlock": price_vanilla_european_option_curve_block,
}


def run_valuation(
    *,
    portfolio: CompiledPortfolio,
    market_data_mtx: MarketDataMatrix,
) -> np.ndarray:
    n_trades = portfolio.total_trades
    n_scenarios = market_data_mtx.num_scenarios

    total_pvs = np.zeros((n_trades, n_scenarios), dtype=np.float64)
    cursor = 0

    for block in portfolio.iter_blocks():
        block_name = block.__class__.__name__
        num_block_trades = len(block.trade_uids)

        if block_name in BLOCK_PRICER_MAP:
            pvs = BLOCK_PRICER_MAP[block_name](
                block=block,
                market_data_mtx=market_data_mtx,
            )
            total_pvs[cursor : cursor + num_block_trades, :] = pvs

        cursor += num_block_trades

    return total_pvs
