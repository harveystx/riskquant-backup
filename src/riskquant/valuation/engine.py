# src/riskquant/valuation/engine.py

"""
Portfolio valuation engine.

Architectural role
------------------
The valuation engine is responsible for dispatching compiled portfolio
blocks to their registered block pricers.

The engine performs no pricing logic itself.

Responsibilities:

    * validate the caller-supplied output matrix against the portfolio
      and the scenario batch
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

Why the output check is here
----------------------------
`out[start:end]` is a Python slice, and Python slices CLAMP rather than
raise: an undersized `out` silently yields a short view, every kernel sizes
its work from `out.shape`, and the block is partially priced with no signal.
The surplus rows keep whatever `np.empty` left in them. This is the earliest
layer that sees both the portfolio and the scenario batch, so it is the
correct owner of the shape invariant (D045/VAL-03). The kernels stay dumb.
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

        Must be C-contiguous float64. Row slices of a C-contiguous matrix
        are C-contiguous; column slices are not, and would fail the pinned
        kernel signatures.

    Raises
    ------
    ValueError
        If `out` is not a C-contiguous 2-D float64 matrix of exactly
        `(portfolio.total_trades, valuation_state.batch_size)`.
    """

    expected_rows = portfolio.total_trades
    expected_cols = valuation_state.batch_size

    if out.dtype != np.float64:
        raise ValueError(f"out must be float64; got {out.dtype}")

    if out.ndim != 2:
        raise ValueError(f"out must be 2-D; got ndim={out.ndim}")

    if out.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"out has shape {out.shape}; portfolio has {expected_rows} rows "
            f"and the scenario batch has {expected_cols} scenarios. A slice of "
            "a mis-sized output matrix clamps silently and leaves the block "
            "partially priced."
        )

    if not out.flags.c_contiguous:
        raise ValueError(
            "out must be C-contiguous. Row slices of a C-contiguous grid are "
            "C-contiguous; column slices are not."
        )

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
