# src/riskquant/valuation/engine.py

"""
Portfolio valuation engine.

Architectural role
------------------
The valuation engine is responsible for dispatching compiled portfolio
blocks to their registered block pricers.

The engine performs no pricing logic itself.

Responsibilities:

    * iterate compiled blocks
    * resolve block pricers from BLOCK_TYPE_ID
    * route block output into the correct slice of the portfolio output matrix

Non-responsibilities:

    * market-data interpretation
    * pricing-model mathematics
    * output allocation
    * result aggregation
    * reporting

All pricing logic resides in block pricers and pricing kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from riskquant.market_data.state import ValuationState
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.valuation.pricer_registry import BLOCK_PRICERS, Pricer


def run_valuation(
    *,
    portfolio: CompiledPortfolio,
    valuation_state: ValuationState,
    out: npt.NDArray[np.float64],
) -> None:
    """
    Execute full portfolio valuation across all scenarios in valuation batch.

    Parameters
    ----------
    portfolio
        Compiled portfolio.

    valuation_state
        Fully-resolved valuation factors for the current scenario batch.

    out
        Preallocated output matrix.

        Shape:

            (portfolio.total_trades, valuation_state.batch_size)

        Row ordering matches portfolio.trade_uids.
    """

    blocks = portfolio.blocks
    offsets = portfolio.block_offsets
    block_pricers: tuple[Pricer[Any], ...] = BLOCK_PRICERS

    for block_idx in range(portfolio.total_blocks):
        block = blocks[block_idx]

        start = offsets[block_idx]
        end = start + block.size

        block_pricers[block.BLOCK_TYPE_ID](
            block=block,
            valuation_state=valuation_state,
            out=out[start:end],
        )
