# src/riskquant/portfolio/blocks/vanilla_forward_block.py
"""
Compiled blocks for vanilla forward trades.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import (
    BlockType,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaForwardBlock(Block):
    """Common SoA fields shared by all vanilla forward blocks."""

    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_FORWARD)

    # --- TRADE SPECIFICS ---
    qty: npt.NDArray[np.float64]
    strike: npt.NDArray[np.float64]

    # ---  LINKAGES (UIX) ---

    # Index into `ValuationState.effective_forward_prices` for the underlying forward price in
    # the FORWARD PRICE CURRENCY.
    effective_forward_price_uix: npt.NDArray[np.int32]

    # Link to effective discount factors for discounting the STRIKE.
    # Effective discount factor is in SETTLEMENT CCY:
    #
    # EFF_DY(0,T) = DF_CCY(0, T) * FX_SPOT_EFFECTIVE(CCY -> SettleCCY)
    #
    # THat is, when an amount is multiplied by the effective discount factor,
    # it is BOTH DISCOUNTED and CONVERTED to settlement currency.
    effective_discount_factor_uix: npt.NDArray[np.int32]
