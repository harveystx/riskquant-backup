# src/riskquant/portfolio/blocks/spot_block.py
"""
Compiled, memory-efficient container for spot positions.

`SpotBlock` is the struct-of-arrays representation of a set of spot trades.
The block is intentionally compact (NumPy arrays, primitive numeric dtypes)
so it can be held efficiently in memory and passed between processes in
high-throughput valuation workflows.

Architectural contract
----------------------
* Row `i` of every array corresponds to the trade with UID `trade_uids[i]`.
* Every `_uix` field is a zero-based, contiguous index produced by the
  portfolio engine's `UixMapper` — NOT a global static-config ID. The block
  carries execution coordinates only.
* `fx_spots_effective_uix` indexes into `MarketDataMatrix.fx_spots_effective`,
  the deduplicated effective FX-transform matrix. No invert flag is carried;
  the inversion (if any) has been folded into the effective matrix at the
  builder edge. Pricer kernels do one unconditional gather + multiply.
* `INVALID_UIX (< 0)` in any `_uix` field signals "no factor required for
  this trade" and is the only branch the pricer kernel performs (hoisted
  outside the scenario loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.portfolio.blocks.base_block import Block
from riskquant.portfolio.instruments import SPOT_ID


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotBlock(Block):
    """Struct-of-arrays for compiled spot trades."""

    INSTRUMENT_ID: ClassVar[int] = int(SPOT_ID)

    # ========== TRADE IDENTITY ==========
    # Unique trade identifiers, one per row.
    trade_uids: npt.NDArray[np.int64]

    # ========== ECONOMIC TERMS ==========
    # Signed quantities / notionals for the spot positions.
    qty: npt.NDArray[np.float64]

    # ========== MARKET LINKAGES (EXECUTION UIX) ==========
    # Index into `MarketDataMatrix.prices` for this row.
    price_uix: npt.NDArray[np.int32]

    # Index into `MarketDataMatrix.fx_spots_effective`, or INVALID_UIX
    # (< 0) when no FX conversion is required for this trade.
    fx_spots_effective_uix: npt.NDArray[np.int32]
