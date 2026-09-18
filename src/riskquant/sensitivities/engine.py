# src/riskquant/sensitivities/engine.py
"""
Portfolio sensitivity engine.

Sibling of `valuation.engine`, not a consumer of it. Both read the same
compiled portfolio and the same market state; neither calls the other:

    CompiledPortfolio
          |
          +---------------------------+
          |                           |
          v                           v
      valuation                 sensitivities
    (ValuationState)           (MarketDataMatrix)
          |                           |
          v                           v
    ValuationResult            BlockSensitivities

Sensitivities are computed in closed form from the payoff, never by
bump-and-revalue, so `run_valuation` is not invoked here at all.

The dispatch loop is the same shape as `run_valuation`: iterate compiled
blocks, index the registry by `block.BLOCK_TYPE_ID`, call the block's own
implementation. Unlike valuation, output is not written into one preallocated
matrix -- sensitivity entry counts are block- and trade-dependent (an
FX-legged spot trade emits more rows than a plain one), so each block returns
its own SoA result and the engine concatenates them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivities.sensitizer_registry import resolve_sensitizer

if TYPE_CHECKING:
    from riskquant.market_data.state import MarketDataMatrix
    from riskquant.portfolio.compiled import CompiledPortfolio
    from riskquant.universe import UixMapper


def _concat_block_sensitivities(
    parts: list[BlockSensitivities],
) -> BlockSensitivities:
    """Concatenate per-block SoA results into one portfolio-level result."""
    if len(parts) == 1:
        return parts[0]

    return BlockSensitivities(
        position_uids=np.ascontiguousarray(
            np.concatenate([p.position_uids for p in parts]), dtype=np.int64
        ),
        factor_rf_ids=np.ascontiguousarray(
            np.concatenate([p.factor_rf_ids for p in parts]), dtype=np.int32
        ),
        first_order=np.ascontiguousarray(
            np.concatenate([p.first_order for p in parts]), dtype=np.float64
        ),
        second_order=np.ascontiguousarray(
            np.concatenate([p.second_order for p in parts]), dtype=np.float64
        ),
        cross_position_uids=np.ascontiguousarray(
            np.concatenate([p.cross_position_uids for p in parts]), dtype=np.int64
        ),
        cross_factor_rf_ids_i=np.ascontiguousarray(
            np.concatenate([p.cross_factor_rf_ids_i for p in parts]), dtype=np.int32
        ),
        cross_factor_rf_ids_j=np.ascontiguousarray(
            np.concatenate([p.cross_factor_rf_ids_j for p in parts]), dtype=np.int32
        ),
        cross_second_order=np.ascontiguousarray(
            np.concatenate([p.cross_second_order for p in parts]), dtype=np.float64
        ),
    )


def run_sensitivities(
    *,
    portfolio: CompiledPortfolio,
    uix_mapper: UixMapper,
    market_data: MarketDataMatrix,
    scenario: int = 0,
) -> BlockSensitivities:
    """
    Compute closed-form sensitivities for every block in a compiled portfolio.

    Parameters
    ----------
    portfolio
        Compiled portfolio.
    uix_mapper
        The portfolio's own compiled identity graph (the same instance used to
        compile it). Supplies every factor relationship the sensitizers need.
    market_data
        Raw, universe-aligned market grids. Sensitizers need the UNFOLDED
        price and FX levels; see `sensitivities.sensitizers.spot`.
    scenario
        Scenario column to evaluate against (base snapshot is 0).

    Returns
    -------
    BlockSensitivities
        Portfolio-level SoA sensitivities keyed by global `rf_id`.

    Raises
    ------
    NotImplementedError
        If any block in the portfolio has no authored sensitizer. Blocks are
        never silently skipped.
    """
    if portfolio.total_blocks == 0:
        raise ValueError("Portfolio contains no blocks; nothing to differentiate.")
    # AUD-02 / D064: sensitizers gather `market_data` rows through the portfolio's
    # block UIXs and `uix_mapper` tables; all three must share one universe.
    if portfolio.uix_fingerprint != uix_mapper.fingerprint:
        raise ValueError("portfolio was compiled against a different UixMapper.")
    if market_data.uix_fingerprint != uix_mapper.fingerprint:
        raise ValueError("MarketDataMatrix was built against a different UixMapper.")

    parts: list[BlockSensitivities] = []
    for block in portfolio.iter_blocks():
        sensitizer = resolve_sensitizer(block)
        parts.append(
            sensitizer(
                block=block,
                uix_mapper=uix_mapper,
                market_data=market_data,
                scenario=scenario,
            )
        )

    return _concat_block_sensitivities(parts)
