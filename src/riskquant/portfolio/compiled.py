# src/riskquant/portfolio/compiled.py

"""
Compiled portfolio container.

`CompiledPortfolio` is a pure portfolio-layer artifact: the frozen, valuation-
ready output of block compilation. It holds compiled SoA blocks plus the
metadata the engine needs to walk them — and nothing from the valuation layer.

Layering
--------
This module imports only `portfolio.blocks`. It deliberately does NOT import
pricers, risk evaluators, or the `Evaluator` protocol. Evaluator resolution is
the valuation engine's responsibility: it dispatches on each block's
`INSTRUMENT_ID` class constant via its own `BLOCK_EVALUATORS` table.

The dependency therefore points one way only:

    valuation  ──▶  portfolio        (allowed)
    portfolio  ──▶  valuation        (never)

The previous design bound pricer callables into `_evaluators` inside
`__post_init__`, which made a data container reach up into the valuation layer
and created the `compiled <-> valuation.engine` import cycle. Removing those
references eliminates the cycle by construction — there is no edge left to
break, deferred-import trick or otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

from riskquant.valuation.blocks.spot_block import SpotBlock
from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionCurveBlock,
    VanillaEuropeanOptionNodeBlock,
)
from riskquant.valuation.blocks.vanilla_forward_block import (
    VanillaForwardCurveBlock,
    VanillaForwardNodeBlock,
)

BlockT: TypeAlias = (
    SpotBlock
    | VanillaForwardNodeBlock
    | VanillaForwardCurveBlock
    | VanillaEuropeanOptionNodeBlock
    | VanillaEuropeanOptionCurveBlock
)


class CompiledPortfolioFields(TypedDict, total=False):
    unique_dates_ord32: npt.NDArray[np.int32]
    spot: SpotBlock | None
    vanilla_forward_node: VanillaForwardNodeBlock | None
    vanilla_forward_curve: VanillaForwardCurveBlock | None
    vanilla_european_option_node: VanillaEuropeanOptionNodeBlock | None
    vanilla_european_option_curve: VanillaEuropeanOptionCurveBlock | None


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledPortfolio:
    """
    Valuation-ready portfolio: an ordered set of compiled SoA blocks.

    Each populated block contributes one contiguous row-band to the eventual
    (n_trades x n_scenarios) valuation matrix; `block_offsets[i]` is the first
    row of block `i`. `trade_uids` aggregates every block's uids into one flat
    int64 array in the same order, so a portfolio-row index maps straight to a
    trade uid without per-block bookkeeping.
    """

    # Unique dates across the portfolio, as ordinals truncated to int32 for
    # engine-friendly storage and fast comparisons. Sorted ascending, unique.
    unique_dates_ord32: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    # Instrument blocks (SoA arrays); None blocks are compacted in __post_init__.
    spot: SpotBlock | None = None
    vanilla_forward_node: VanillaForwardNodeBlock | None = None
    vanilla_forward_curve: VanillaForwardCurveBlock | None = None
    vanilla_european_option_node: VanillaEuropeanOptionNodeBlock | None = None
    vanilla_european_option_curve: VanillaEuropeanOptionCurveBlock | None = None

    # Derived, set in __post_init__.
    _blocks: tuple[BlockT, ...] = field(init=False, repr=False)
    _block_offsets: npt.NDArray[np.int64] = field(init=False, repr=False)
    _total_blocks: int = field(init=False, repr=False)
    _total_trades: int = field(init=False, repr=False)
    _trade_uids: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Construction-ordered slots; None slots are compacted out below. This
        # tuple order defines block ordering and therefore output-matrix bands.
        ordered: tuple[BlockT | None, ...] = (
            self.spot,
            self.vanilla_forward_node,
            self.vanilla_forward_curve,
            self.vanilla_european_option_node,
            self.vanilla_european_option_curve,
        )
        blocks: tuple[BlockT, ...] = tuple(b for b in ordered if b is not None)
        total_blocks: int = len(blocks)

        # Offsets via exclusive prefix-sum of per-block trade counts.
        block_offsets = np.zeros(total_blocks, dtype=np.int64)
        trade_counts = np.fromiter(
            (b.num_trades for b in blocks), dtype=np.int64, count=total_blocks
        )
        total_trades = int(trade_counts.sum())
        if total_blocks > 1:
            np.cumsum(trade_counts[:-1], out=block_offsets[1:])

        # Aggregate every block's uids into one flat array in portfolio order.
        trade_uids = np.empty(total_trades, dtype=np.int64)
        offset = 0
        for block in blocks:
            n = block.num_trades
            trade_uids[offset : offset + n] = block.trade_uids
            offset += n

        object.__setattr__(self, "_blocks", blocks)
        object.__setattr__(self, "_block_offsets", block_offsets)
        object.__setattr__(self, "_total_blocks", total_blocks)
        object.__setattr__(self, "_total_trades", total_trades)
        object.__setattr__(self, "_trade_uids", trade_uids)

    # ------------------------------------------------------------------ #
    # Read-only accessors
    # ------------------------------------------------------------------ #

    @property
    def blocks(self) -> tuple[BlockT, ...]:
        return self._blocks

    @property
    def block_offsets(self) -> npt.NDArray[np.int64]:
        return self._block_offsets

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def total_trades(self) -> int:
        return self._total_trades

    @property
    def trade_uids(self) -> npt.NDArray[np.int64]:
        return self._trade_uids

    def iter_blocks(self) -> Iterator[BlockT]:
        yield from self._blocks

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        """Print a console-friendly overview of block layout and trade counts."""
        print("=" * 90)
        print(f"{'COMPILED PORTFOLIO SUMMARY':^90}")
        print("=" * 90)
        print(f"Total Trades: {self._total_trades:,}")
        print(f"Total Blocks: {self._total_blocks}")
        print(f"Unique Dates: {len(self.unique_dates_ord32)}")
        print("-" * 90)
        print(f"{'Block Type':<32} | {'Trades':<10} | {'Offset'}")
        print("-" * 90)

        if self._total_blocks == 0:
            print(f"{'Empty Portfolio':^90}")
        else:
            for i, block in enumerate(self._blocks):
                print(
                    f"{type(block).__name__:<32} | "
                    f"{block.num_trades:<10,} | {self._block_offsets[i]}"
                )
        print("=" * 90 + "\n")

    def print_block_details(
        self,
        *,
        max_items: int | None = 10,
        float_precision: int = 0,
        sep: str = ", ",
        pad: int = 12,
        thousand_sep: str = "_",
    ) -> None:
        """Cascade into each block and print its raw SoA arrays."""
        print("=" * 90)
        print(f"{f'BLOCK DETAILS (MAX {max_items} ITEMS PER BLOCK)':^90}")
        print("=" * 90 + "\n")
        for block in self._blocks:
            title = f"Block: {type(block).__name__}"
            print("-" * len(title))
            print(title)
            print("-" * len(title))
            block.print_block(
                max_items=max_items,
                float_precision=float_precision,
                sep=sep,
                pad=pad,
                thousand_sep=thousand_sep,
            )
        print("=" * 90 + "\n")
