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

__all__ = [
    "CompiledJtdPortfolio",
]

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.beta_binomial import (
    BetaBinomialJtdBlock,
)
from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock
from riskquant.credit.jtd.blocks.independent_bernoulli import (
    IndependentBernoulliJtdBlock,
)
from riskquant.credit.jtd.blocks.marshall_olkin import MarshallOlkinJtdBlock
from riskquant.credit.jtd.blocks.mf_copula_gaussian import MfGaussianCopulaJtdBlock
from riskquant.credit.jtd.blocks.mf_copula_students_t import (
    MfStudentsTCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.sf_copula_gaussian import SfGaussianCopulaJtdBlock
from riskquant.credit.jtd.blocks.sf_copula_students_t import (
    SfStudentsTCopulaJtdBlock,
)
from riskquant.credit.jtd.blocks.stress_regime_bernoulli import (
    StressRegimeBernoulliJtdBlock,
)
from riskquant.credit.jtd.blocks.type_alias import JtdBlockTypeUnion


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CompiledJtdPortfolio:
    """JTD-ready credit portfolio: an ordered set of compiled obligor blocks."""

    # --- COMPILED BLOCKS ---

    # Non-copula blocks
    forced_default_jtd_block: ForcedDefaultJtdBlock | None = None
    independent_bernoulli_jtd_block: IndependentBernoulliJtdBlock | None = None
    beta_binomial_jtd_block: BetaBinomialJtdBlock | None = None
    stress_regime_bernoulli_jtd_block: StressRegimeBernoulliJtdBlock | None = None

    # SF Copula blocks
    sf_gaussian_copula_jtd_block: SfGaussianCopulaJtdBlock | None = None
    sf_students_t_copula_jtd_block: SfStudentsTCopulaJtdBlock | None = None

    # MF Copula blocks
    mf_gaussian_copula_jtd_block: MfGaussianCopulaJtdBlock | None = None
    mf_students_t_copula_jtd_block: MfStudentsTCopulaJtdBlock | None = None

    # Common-shock blocks
    marshall_olkin_jtd_block: MarshallOlkinJtdBlock | None = None

    # Internal fields derived from the above and used by the engine.
    _blocks: tuple[JtdBlockTypeUnion, ...] = field(init=False, repr=False)
    _block_offsets: npt.NDArray[np.int64] = field(init=False, repr=False)
    _total_blocks: int = field(init=False, repr=False)
    _total_obligors: int = field(init=False, repr=False)
    _obligor_uids: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Band order + slot names are owned by the model registry; getattr by
        # compiled_slot fails fast here if a spec names a slot this dataclass
        # does not declare.
        from riskquant.credit.jtd.registry import JTD_MODEL_SPECS

        ordered: tuple[JtdBlockTypeUnion | None, ...] = tuple(
            getattr(self, spec.compiled_slot) for spec in JTD_MODEL_SPECS
        )
        blocks: tuple[JtdBlockTypeUnion, ...] = tuple(
            b for b in ordered if b is not None
        )
        total_blocks = len(blocks)

        # 1. Calculate the total size needed
        total_obligors = sum(b.num_obligors for b in blocks)

        # 2. Allocate uninitialized arrays (faster than zeros)
        block_offsets = np.empty(total_blocks, dtype=np.int64)
        obligor_uids = np.empty(total_obligors, dtype=np.int64)

        # 3. Populate both arrays in a single, clean pass
        offset = 0
        for i, b in enumerate(blocks):
            block_offsets[i] = offset
            n = b.num_obligors
            obligor_uids[offset : offset + n] = b.obligor_uids
            offset += n

        object.__setattr__(self, "_blocks", blocks)
        object.__setattr__(self, "_block_offsets", block_offsets)
        object.__setattr__(self, "_total_blocks", total_blocks)
        object.__setattr__(self, "_total_obligors", total_obligors)
        object.__setattr__(self, "_obligor_uids", obligor_uids)

    @property
    def blocks(self) -> tuple[JtdBlockTypeUnion, ...]:
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

    def iter_blocks(self) -> Iterator[JtdBlockTypeUnion]:
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
