# src/riskquant/credit/jtd/compiled.py
"""
Compiled JTD portfolio container.

`CompiledJtdPortfolio` is the credit-layer analogue of `CompiledExposure`: an
ordered set of per-model obligor blocks. Each populated block contributes one
contiguous row-band to the eventual (n_obligors x n_scenarios) JTD-loss matrix;
`block_offsets[i]` is its first row. `obligor_uids` aggregates every block's
uids into one flat int64 array in band order — the row order the engine's
`ead_by_obligor` argument must match.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.copula import (
    GaussianCopulaJtdBlock,
    TCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.deterministic import DeterministicJtdBlock

JtdBlockT: TypeAlias = GaussianCopulaJtdBlock | TCopulaJtdBlock | DeterministicJtdBlock


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledJtdPortfolio:
    """JTD-ready credit portfolio: an ordered set of compiled obligor blocks."""

    # Compiled blocks
    gaussian_copula_jtd: GaussianCopulaJtdBlock | None = None
    t_copula_jtd: TCopulaJtdBlock | None = None
    deterministic_jtd: DeterministicJtdBlock | None = None

    # Internal fields derived from the above and used by the engine.
    _blocks: tuple[JtdBlockT, ...] = field(init=False, repr=False)
    _block_offsets: npt.NDArray[np.int64] = field(init=False, repr=False)
    _total_blocks: int = field(init=False, repr=False)
    _total_obligors: int = field(init=False, repr=False)
    _obligor_uids: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ordered: tuple[JtdBlockT | None, ...] = (
            self.gaussian_copula_jtd,
            self.t_copula_jtd,
            self.deterministic_jtd,
        )
        blocks: tuple[JtdBlockT, ...] = tuple(b for b in ordered if b is not None)
        total_blocks = len(blocks)

        block_offsets = np.zeros(total_blocks, dtype=np.int64)
        counts = np.fromiter(
            (b.num_obligors for b in blocks), dtype=np.int64, count=total_blocks
        )
        total_obligors = int(counts.sum())
        if total_blocks > 1:
            np.cumsum(counts[:-1], out=block_offsets[1:])

        obligor_uids = np.empty(total_obligors, dtype=np.int64)
        offset = 0
        for b in blocks:
            n = b.num_obligors
            obligor_uids[offset : offset + n] = b.obligor_uids
            offset += n

        object.__setattr__(self, "_blocks", blocks)
        object.__setattr__(self, "_block_offsets", block_offsets)
        object.__setattr__(self, "_total_blocks", total_blocks)
        object.__setattr__(self, "_total_obligors", total_obligors)
        object.__setattr__(self, "_obligor_uids", obligor_uids)

    @property
    def blocks(self) -> tuple[JtdBlockT, ...]:
        return self._blocks

    @property
    def block_offsets(self) -> npt.NDArray[np.int64]:
        return self._block_offsets

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def total_obligors(self) -> int:
        return self._total_obligors

    @property
    def obligor_uids(self) -> npt.NDArray[np.int64]:
        return self._obligor_uids

    def iter_blocks(self) -> Iterator[JtdBlockT]:
        yield from self._blocks

    def summary(self) -> None:
        """Print a console-friendly overview of block layout and obligor counts."""
        print("=" * 90)
        print(f"{'COMPILED PORTFOLIO SUMMARY':^90}")
        print("=" * 90)
        print(f"Total Obligors: {self._total_obligors:,}")
        print(f"Total Blocks: {self._total_blocks}")
        print("-" * 90)
        print(f"{'Block Type':<32} | {'Obligors':<10} | {'Offset'}")
        print("-" * 90)

        if self._total_blocks == 0:
            print(f"{'Empty Portfolio':^90}")
        else:
            for i, block in enumerate(self._blocks):
                print(
                    f"{type(block).__name__:<32} | "
                    f"{block.num_obligors:<10,} | {self._block_offsets[i]}"
                )
        print("=" * 90 + "\n")

    def print_block_details(self, max_items: int | None = 10) -> None:
        """Cascade into each block and print its raw SoA arrays."""
        print("=" * 90)
        print(f"{f'BLOCK DETAILS (MAX {max_items} ITEMS PER BLOCK)':^90}")
        print("=" * 90 + "\n")
        for block in self._blocks:
            title = f"Block: {type(block).__name__}"
            print("-" * len(title))
            print(title)
            print("-" * len(title))
            block.print_block(max_items=max_items)
        print("=" * 90 + "\n")
