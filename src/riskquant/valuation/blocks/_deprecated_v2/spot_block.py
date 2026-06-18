# src/riskquant/portfolio/blocks/spot_block.py
"""
Compiled, memory-efficient container for spot positions.

`SpotBlock` is the struct-of-arrays representation of a set of spot trades.
The block is intentionally compact (NumPy arrays, primitive numeric dtypes)
so it can be held efficiently in memory and passed between processes in
high-throughput valuation workflows.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import BlockType


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotBlock(Block):
    """Struct-of-arrays for compiled spot trades."""

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.SPOT)

    # --- TRADE SPECIFICS ---

    # Signed quantities / notionals for the spot positions.
    qty: npt.NDArray[np.float64]

    # --- LINKAGES (UIX) ---

    # Index into `ValuationState.spot_prices` for this row.
    spot_price_uix: npt.NDArray[np.int32]

    # Index into `ValuationState.fx_spots_effective`.
    fx_spot_effective_uix: npt.NDArray[np.int32]
