# src/riskquant/valuation/blocks/spot_block.py
"""
Compiled, memory-efficient container for spot positions.

`SpotBlock` is the struct-of-arrays representation of a set of spot trades.
The block is intentionally compact (NumPy arrays, primitive numeric dtypes)
so it can be held efficiently in memory and passed between processes in
high-throughput valuation workflows.

Economic contract
-----------------
    PV(0) = Qty * EffectiveSpotPrice(0)

A spot position has no time value, so no discounting is applied. FX is
already folded into the effective spot price by the market-data compiler;
the kernel performs one gather and one multiply and does no inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import BlockType


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SpotBlock(Block):
    """Struct-of-arrays for compiled spot trades."""

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.SPOT)

    # --- TRADE SPECIFICS ---

    # Signed quantities / notionals for the spot positions.
    # Units: contracts / asset units. Long > 0, short < 0.
    qty: npt.NDArray[np.float64]

    # --- LINKAGES (UIX) ---

    # Index into `ValuationState.effective_spot_prices` for this row.
    #
    #   Economic quantity : spot price already converted to settlement ccy
    #   Value             : SPOT_PRICE_CCY(0) * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #   Currency          : SETTLEMENT
    #   Date              : t = 0
    #   Discounted        : NO  (spot has no time value)
    #   FX-converted      : YES (spot FX, folded by the market-data compiler)
    #   Scenario-dependent: YES
    #
    # Must be a valid dense row index into that matrix. The kernel runs under
    # `boundscheck=False` and performs no sentinel branch: a negative value
    # wraps to the last row and an out-of-range value reads arbitrary memory.
    # UIX validity is established by the block compiler.
    effective_spot_price_uix: npt.NDArray[np.int32]
