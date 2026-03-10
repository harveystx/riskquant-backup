# src/riskquant/portfolio/compiled.py
"""
Compiles IngestBatch SoA containers into runtime-oriented Struct-of-Arrays blocks.

Design goals
------------
- Pure C-level NumPy slicing from ingestion to runtime.
- Zero Python loops over individual trade rows.
- Immutable compiled runtime container for pricing/risk kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Optional, TypeAlias

from riskquant.portfolio.compilers.registry import COMPILER_REGISTRY
from riskquant.portfolio.ingest.batch import IngestBatch

from riskquant.portfolio.blocks.spot_block import SpotBlock
from riskquant.portfolio.blocks.forward_block import (
    ForwardNodeBlock,
    ForwardCurveBlock,
)
from riskquant.portfolio.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionNodeBlock,
    VanillaEuropeanOptionCurveBlock,
)

# ============================================================
# Type aliases
# ============================================================

BlockType: TypeAlias = (
    SpotBlock
    | ForwardNodeBlock
    | ForwardCurveBlock
    | VanillaEuropeanOptionNodeBlock
    | VanillaEuropeanOptionCurveBlock
)

# Compilers now take the full batch and return a Block if any trades match.
CompilerFn: TypeAlias = Callable[[IngestBatch], Optional[BlockType]]

# ============================================================
# Runtime container
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledPortfolio:
    """
    Immutable runtime container of instrument-homogeneous SoA blocks.
    """

    spot: Optional[SpotBlock] = None
    forward_node: Optional[ForwardNodeBlock] = None
    forward_curve: Optional[ForwardCurveBlock] = None
    vanilla_european_option_node: Optional[VanillaEuropeanOptionNodeBlock] = (
        None
    )
    vanilla_european_option_curve: Optional[VanillaEuropeanOptionCurveBlock] = (
        None
    )

    def iter_blocks(self) -> tuple[BlockType, ...]:
        """
        Return all non-None blocks in a stable field order.
        """
        blocks: list[BlockType] = []
        for f in fields(self):
            block = getattr(self, f.name)
            if block is not None:
                blocks.append(block)
        return tuple(blocks)

    @property
    def total_trades(self) -> int:
        """
        Total number of trades across all compiled blocks.
        """
        total = 0
        for block in self.iter_blocks():
            total += len(block.trade_uids)
        return total

    @classmethod
    def from_ingest_batch(
        cls,
        batch: IngestBatch,
    ) -> "CompiledPortfolio":
        """
        Consume a validated SoA IngestBatch and compile it directly into SoA blocks
        using vectorized NumPy masking. Zero scalar instantiation.
        """
        compiled_fields: dict[str, Optional[BlockType]] = {}

        # Dispatch the full batch to every compiler in the registry.
        # The compiler will apply its own boolean mask and slice the arrays.
        for instrument_id, spec in COMPILER_REGISTRY.items():
            compiled_fields[spec.field_name] = spec.compiler(batch)

        return cls(**compiled_fields)
