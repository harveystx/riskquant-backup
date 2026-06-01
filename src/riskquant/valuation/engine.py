# src/riskquant/valuation/engine.py

"""
Portfolio valuation engine.

Owns the instrument-id -> evaluator dispatch table and the scenario valuation
loop. Evaluator resolution lives here, in the valuation layer: `run_valuation`
reads each block's `INSTRUMENT_ID` class constant and indexes `BLOCK_EVALUATORS`
directly. `CompiledPortfolio` carries no evaluator references, so the dependency
runs one way only — `valuation -> portfolio` — with no import cycle.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.blocks.base_block import Block
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.instruments import (
    NUM_INSTRUMENTS,
    SPOT_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_CURVE_ID,
    VANILLA_FORWARD_NODE_ID,
    Instrument,
)

# VALUATORS
from riskquant.valuation.pricers.spot import price_spot_block
from riskquant.valuation.pricers.vanilla_european_option_curve import (
    price_vanilla_european_option_curve_block,
)
from riskquant.valuation.pricers.vanilla_european_option_node import (
    price_vanilla_european_option_node_block,
)
from riskquant.valuation.pricers.vanilla_forward_curve import (
    price_vanilla_forward_curve_block,
)
from riskquant.valuation.pricers.vanilla_forward_node import (
    price_vanilla_forward_node_block,
)
from riskquant.valuation.result import ValuationResult

# ==============================================================================
# EVALUATOR PROTOCOL
# ==============================================================================

# 1. Define a contravariant TypeVar bound to Block
B_contra = TypeVar("B_contra", bound=Block, contravariant=True)


@runtime_checkable
class Evaluator(Protocol[B_contra]):
    """
    Structural contract for all block evaluators (pricers + risk evaluators).
    """

    def __call__(
        self,
        *,
        block: B_contra,  # 2. Use the TypeVar instead of base Block
        market_data_mtx: MarketDataMatrix,
        out: npt.NDArray[np.float64],
    ) -> None: ...


# ==============================================================================
# DISPATCH TABLE
# ==============================================================================

# 3. Explicitly type the list elements as Evaluator[Any] | None
_BLOCK_EVALUATORS_MUT: list[Evaluator[Any] | None] = [None] * NUM_INSTRUMENTS


def _register(instrument_id: int, evaluator: Evaluator[Any]) -> None:
    # runtime_checkable ignores TypeVars, so this check still works natively
    if not isinstance(evaluator, Evaluator):
        raise TypeError(
            f"Evaluator for instrument_id={instrument_id} does not satisfy the "
            f"Evaluator protocol: {evaluator!r}"
        )
    _BLOCK_EVALUATORS_MUT[instrument_id] = evaluator


_register(
    int(SPOT_ID),
    price_spot_block,
)
_register(
    int(VANILLA_FORWARD_NODE_ID),
    price_vanilla_forward_node_block,
)
_register(
    int(VANILLA_FORWARD_CURVE_ID),
    price_vanilla_forward_curve_block,
)
_register(
    int(VANILLA_EUROPEAN_OPTION_NODE_ID),
    price_vanilla_european_option_node_block,
)
_register(
    int(VANILLA_EUROPEAN_OPTION_CURVE_ID),
    price_vanilla_european_option_curve_block,
)


# Freeze. No mutations after module load.
# This tells the type checker: "Trust me, the runtime guard above handled the Nones."
BLOCK_EVALUATORS: Final[tuple[Evaluator[Any], ...]] = cast(
    tuple[Evaluator[Any], ...], tuple(_BLOCK_EVALUATORS_MUT)
)

# Hard fail at import if any instrument has no evaluator: catches the case where
# a new Instrument variant is added but not wired up here.
_unregistered = [
    Instrument(i).name for i, ev in enumerate(BLOCK_EVALUATORS) if ev is None
]
if _unregistered:
    raise RuntimeError(
        f"No evaluator registered for instruments: {_unregistered}. "
        f"Register them in engine.py."
    )


def resolve_evaluator(block: Block) -> Evaluator:
    """
    Resolve the evaluator for a compiled block via its `INSTRUMENT_ID`.

    Single-array index — no `type()`, no isinstance ladder, no string keys.
    """
    evaluator = BLOCK_EVALUATORS[block.INSTRUMENT_ID]
    if evaluator is None:  # unreachable given the import-time guard above
        raise RuntimeError(
            f"No evaluator for INSTRUMENT_ID={block.INSTRUMENT_ID} "
            f"({type(block).__name__})."
        )
    return evaluator


# ==============================================================================
# VALUATION ENGINE
# ==============================================================================


def run_valuation(
    *,
    portfolio: CompiledPortfolio,
    market_data_mtx: MarketDataMatrix,
) -> ValuationResult:
    """
    Execute full portfolio valuation across all scenarios.

    Iterates pre-compiled blocks in portfolio order, resolves each block's
    evaluator from its `INSTRUMENT_ID`, and dispatches it. All writes target
    contiguous slices of a single pre-allocated (n_trades x n_scenarios) C-order
    output array. Zero intermediate allocations; the GIL is bypassed inside the
    Numba kernels.

    Returns
    -------
    npt.NDArray[np.float64]
        Shape (n_trades, n_scenarios), C-contiguous. Row i corresponds to the
        trade at portfolio offset i.
    """
    n_trades: int = portfolio.total_trades
    n_scenarios: int = market_data_mtx.num_scenarios

    out: npt.NDArray[np.float64] = np.empty(
        (n_trades, n_scenarios),
        dtype=np.float64,
        order="C",
    )

    # Hoist repeated attribute lookups out of the loop.
    blocks = portfolio.blocks
    offsets = portfolio.block_offsets
    evaluators_table = BLOCK_EVALUATORS
    md = market_data_mtx

    for i in range(portfolio.total_blocks):
        block = blocks[i]
        start = int(offsets[i])
        end = start + block.num_trades

        # id -> evaluator: one O(1) table index, resolved here in the engine.
        evaluator = evaluators_table[block.INSTRUMENT_ID]

        evaluator(
            block=block,
            market_data_mtx=md,
            out=out[start:end, :],
        )

    return ValuationResult(
        trade_uids=portfolio.trade_uids,
        values=out,
    )
