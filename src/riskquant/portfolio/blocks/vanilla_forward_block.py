# src/riskquant/portfolio/blocks/vanilla_forward_block.py
"""
Compiled blocks for vanilla forward trades.

Architectural contract
----------------------
* Row `i` of every array corresponds to the trade with UID `trade_uids[i]`.
* Every `_uix` field is a zero-based, contiguous index produced by the
  portfolio engine's `UixMapper` — NOT a global static-config ID.
* FX legs are carried as `fx_*_to_settle_effective_uix` indexing the
  deduplicated `MarketDataMatrix.fx_spots_effective` matrix. No invert
  flags survive into the block layer — inversion is folded into the
  effective matrix at the builder edge.
* `INVALID_UIX (< 0)` in any optional `_uix` field signals "no factor
  required for this trade" and is the only branch the pricer kernel
  performs (hoisted outside the scenario loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.portfolio.blocks.base_block import Block
from riskquant.portfolio.instruments import (
    VANILLA_FORWARD_CURVE_ID,
    VANILLA_FORWARD_NODE_ID,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _VanillaForwardBaseBlock(Block):
    """Common SoA fields shared by all vanilla forward blocks."""

    # ========== TRADE IDENTITY ==========
    trade_uids: npt.NDArray[np.int64]

    # ========== ECONOMIC TERMS ==========
    qty: npt.NDArray[np.float64]
    strike: npt.NDArray[np.float64]

    # ========== SETTLEMENT DATE (DATE UIX) ==========
    # Index into `unique_dates_ord32` for cashflow settlement.
    settlement_date_uix: npt.NDArray[np.int32]

    # ========== FX LINKAGES (EFFECTIVE UIX) ==========
    # Index into `MarketDataMatrix.fx_spots_effective` for the under->settle
    # transform. INVALID_UIX (< 0) when no FX needed for this leg.
    fx_under_to_settle_effective_uix: npt.NDArray[np.int32]

    # Index into `MarketDataMatrix.fx_spots_effective` for the strike->settle
    # transform. INVALID_UIX (< 0) when no FX needed for this leg.
    fx_strike_to_settle_effective_uix: npt.NDArray[np.int32]

    # ========== DISCOUNT LINKAGES (CURVE UIX) ==========
    # Indices into `MarketDataMatrix.discount_factor_grid` axis 0.
    # INVALID_UIX (< 0) when no discount applies on that leg.
    under_discount_curve_uix: npt.NDArray[np.int32]
    strike_discount_curve_uix: npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaForwardNodeBlock(_VanillaForwardBaseBlock):
    """Forward priced directly off a scalar node."""

    INSTRUMENT_ID: ClassVar[int] = int(VANILLA_FORWARD_NODE_ID)

    # ========== UNDERLYING MARKET LINKAGE ==========
    # Index into `MarketDataMatrix.forward_nodes` axis 0.
    price_uix: npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaForwardCurveBlock(_VanillaForwardBaseBlock):
    """Forward priced off a curve requiring delivery-date interpolation."""

    INSTRUMENT_ID: ClassVar[int] = int(VANILLA_FORWARD_CURVE_ID)

    # ========== DELIVERY DATE (DATE UIX) ==========
    delivery_date_uix: npt.NDArray[np.int32]

    # ========== UNDERLYING MARKET LINKAGE ==========
    # Index into `MarketDataMatrix.price_curve_grid` axis 0.
    price_curve_uix: npt.NDArray[np.int32]
