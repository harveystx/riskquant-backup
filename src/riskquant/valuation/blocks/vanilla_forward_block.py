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

    # Index into `ValuationState.forward_prices` for the underlying forward price in
    # the FORWARD PRICE CURRENCY.
    forward_price_uix: npt.NDArray[np.int32]

    # Link to discount factors for discounting the FORWARD PRICE.
    # Discount factor is in FORWARD PRICE CCY. Cashflow is
    # converted to SETTLEMENT CCY via `fx_spot_effective_u2s_uix`
    # if required.
    discount_factor_u_uix: npt.NDArray[np.int32]

    # Link to discount factors for discounting the STRIKE.
    # Discount factor is in STRIKE CCY. Cashflow is
    # converted to SETTLEMENT CCY via `fx_spot_effective_k2s_uix`
    # if required.
    discount_factor_k_uix: npt.NDArray[np.int32]

    # Index into `ValuationState.fx_spots_effective` for the under->settle FX
    # spot rate at t = 0.
    fx_spot_effective_u2s_uix: npt.NDArray[np.int32]

    # Index into `ValuationState.fx_spots_effective` for the strike->settle FX
    # spot rate at t = 0.
    fx_spot_effective_k2s_uix: npt.NDArray[np.int32]


# @dataclass(frozen=True, slots=True, kw_only=True)
# class _VanillaForwardBaseBlock(Block):
#     """Common SoA fields shared by all vanilla forward blocks."""

#     # ========== BLOCK ID FOR DISPATCH ==========
#                     BLOCK_TYPE_ID: ClassVar[int] = UNSET_BLOCK_TYPE_ID_SENTINEL

#     # ========== ECONOMIC TERMS ==========
#     qty: npt.NDArray[np.float64]
#     strike: npt.NDArray[np.float64]

#     # ========== SETTLEMENT DATE (DATE UIX) ==========
#     # Index into `unique_dates_ord32` for cashflow settlement.
#     settlement_date_uix: npt.NDArray[np.int32]

#     # ========== FX LINKAGES (EFFECTIVE UIX) ==========
#     # Index into `MarketDataMatrix.fx_spots_effective` for the under->settle
#     # transform. INVALID_UIX (< 0) when no FX needed for this leg.
#     fx_under_to_settle_effective_uix: npt.NDArray[np.int32]

#     # Index into `MarketDataMatrix.fx_spots_effective` for the strike->settle
#     # transform. INVALID_UIX (< 0) when no FX needed for this leg.
#     fx_strike_to_settle_effective_uix: npt.NDArray[np.int32]

#     # ========== DISCOUNT LINKAGES (CURVE UIX) ==========
#     # Indices into `MarketDataMatrix.discount_factor_grid` axis 0.
#     # INVALID_UIX (< 0) when no discount applies on that leg.
#     under_discount_curve_uix: npt.NDArray[np.int32]
#     strike_discount_curve_uix: npt.NDArray[np.int32]


# @dataclass(frozen=True, slots=True, kw_only=True)
# class VanillaForwardNodeBlock(_VanillaForwardBaseBlock):
#     """Forward priced directly off a scalar node."""

#     # ========== BLOCK ID FOR DISPATCH ==========
#     BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_FORWARD_NODE)

#     # ========== UNDERLYING MARKET LINKAGE ==========
#     # Index into `MarketDataMatrix.forward_nodes` axis 0.
#     price_uix: npt.NDArray[np.int32]


# @dataclass(frozen=True, slots=True, kw_only=True)
# class VanillaForwardCurveBlock(_VanillaForwardBaseBlock):
#     """Forward priced off a curve requiring delivery-date interpolation."""

#     # ========== BLOCK ID FOR DISPATCH ==========
#     BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_FORWARD_CURVE)

#     # ========== DELIVERY DATE (DATE UIX) ==========
#     delivery_date_uix: npt.NDArray[np.int32]

#     # ========== UNDERLYING MARKET LINKAGE ==========
#     # Index into `MarketDataMatrix.price_curve_grid` axis 0.
#     price_curve_uix: npt.NDArray[np.int32]
