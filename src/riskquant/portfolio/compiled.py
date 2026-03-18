# src/riskquant/portfolio/compiled.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypeAlias
import numpy as np

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

# ============================================================
# Field order (CRITICAL for performance)
# ============================================================

_FIELD_NAMES = (
    "spot",
    "forward_node",
    "forward_curve",
    "vanilla_european_option_node",
    "vanilla_european_option_curve",
)

# ============================================================
# Runtime container
# ============================================================


@dataclass(frozen=True, slots=True)
class CompiledPortfolio:
    """
    Immutable runtime container of instrument-homogeneous SoA blocks.

    Design:
    - Zero reflection (no dataclasses.fields usage)
    - Predictable field order
    - Fast iteration
    - Minimal Python overhead
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

    # Temporal coordinates required by the MarketDataMatrix builder
    unique_dates_ordinal32: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    # ------------------------------------------------------------
    # Fast iteration (no reflection)
    # ------------------------------------------------------------
    def iter_blocks(self) -> tuple[BlockType, ...]:
        """
        Return all non-None blocks in stable order.
        """
        return tuple(
            block
            for name in _FIELD_NAMES
            if (block := getattr(self, name)) is not None
        )

    # ------------------------------------------------------------
    # Fast aggregate metrics
    # ------------------------------------------------------------
    @property
    def total_trades(self) -> int:
        """
        Total number of trades across all compiled blocks.
        """
        total = 0
        for name in _FIELD_NAMES:
            block = getattr(self, name)
            if block is not None:
                total += len(block.trade_uids)
        return total

    # ------------------------------------------------------------
    # Constructor helper (avoids kwargs overhead)
    # ------------------------------------------------------------
    @classmethod
    def from_compiled_fields(
        cls, compiled_fields: dict[str, BlockType], unique_dates: np.ndarray
    ) -> "CompiledPortfolio":
        """
        Fast construction from dict without dynamic kwargs overhead.
        """
        return cls(
            spot=compiled_fields.get("spot"),
            forward_node=compiled_fields.get("forward_node"),
            forward_curve=compiled_fields.get("forward_curve"),
            vanilla_european_option_node=compiled_fields.get(
                "vanilla_european_option_node"
            ),
            vanilla_european_option_curve=compiled_fields.get(
                "vanilla_european_option_curve"
            ),
            unique_dates_ordinal32=unique_dates,
        )
