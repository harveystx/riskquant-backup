# src/riskquant/portfolio/blocks/vanilla_european_option_block.py
"""
Compiled blocks for vanilla European option trades.

Architectural contract
----------------------
* Row `i` of every array corresponds to the trade with UID `trade_uids[i]`.
* Every `_uix` field is a zero-based, contiguous execution index produced by
  the portfolio engine's `UixMapper` — NOT a global static-config ID.
* FX legs are carried as `fx_*_to_settle_effective_uix` indexing
  `MarketDataMatrix.fx_spots_effective`. No invert flags survive into the
  block layer.
* `cp_mult` packs `is_call` into `+1` / `-1` for branchless Black-76 payoff.
* `INVALID_UIX (< 0)` in any optional `_uix` field signals "no factor
  required for this trade" and is the only branch the pricer kernel
  performs (hoisted outside the scenario loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from riskquant.valuation.blocks._base_block import Block
from riskquant.valuation.blocks.block_type import BlockType


@dataclass(frozen=True, slots=True)
class VanillaEuropeanOptionBlock(Block):
    # --- BLOCK ID FOR DISPATCH ---
    BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_EUROPEAN_OPTION)

    # --- TRADE SPECIFICS ---
    qty: NDArray[np.float64]
    strike: NDArray[np.float64]
    cp_mult: NDArray[np.int8]
    sqrt_tte: NDArray[np.float64]
    half_tte: NDArray[np.float64]

    # --- LINKAGES (UIX) ---

    # Index into ValuationState.forward_prices for the underlying forward price in
    # the STRIKE CURRENCY. This is the forward price at the expiry date of the option.
    forward_price_uix: NDArray[np.int32]

    # Index into ValuationState.vols for the volatility of the underlying forward price.
    # This is the volatility at the expiry date of the option.
    vol_uix: NDArray[np.int32]

    # Index into ValuationState.discount_factors for the discount factor in the STRIKE
    # CURRENCY. Discounts the cashflow from the SETTLEMENT DATE of the option to t = 0.
    discount_factor_k_uix: NDArray[np.int32]

    # Index into ValuationState.fx_spots_effective for the FX rate from the STRIKE
    # CURRENCY to the SETTLEMENT CURRENCY at t = 0.
    fx_spot_effective_k2s_uix: NDArray[np.int32]


# @dataclass(frozen=True, slots=True, kw_only=True)
# class _VanillaEuropeanOptionBaseBlock(Block):
#     """Common SoA fields shared by all vanilla European option blocks."""

#     # ========== BLOCK ID FOR DISPATCH ==========
#     BLOCK_TYPE_ID: ClassVar[int] = UNSET_BLOCK_TYPE_ID_SENTINEL

#     # ========== ECONOMIC TERMS ==========
#     qty: npt.NDArray[np.float64]
#     strike: npt.NDArray[np.float64]
#     # +1 for call, -1 for put, precomputed for branchless payoff sign.
#     cp_mult: npt.NDArray[np.int8]

#     # ========== DATES (DATE UIX) ==========
#     # Indices into `unique_dates_ord32`.
#     expiry_date_uix: npt.NDArray[np.int32]
#     settlement_date_uix: npt.NDArray[np.int32]

#     # ========== FX LINKAGES (EFFECTIVE UIX) ==========
#     # Indices into `MarketDataMatrix.fx_spots_effective`. INVALID_UIX (< 0)
#     # signals "no FX conversion required" for that leg.
#     fx_under_to_settle_effective_uix: npt.NDArray[np.int32]
#     fx_strike_to_settle_effective_uix: npt.NDArray[np.int32]

#     # ========== DISCOUNT LINKAGES (CURVE UIX) ==========
#     # Indices into `MarketDataMatrix.discount_factor_grid` axis 0.
#     # INVALID_UIX (< 0) when no discount applies on that leg.
#     under_discount_curve_uix: npt.NDArray[np.int32]
#     strike_discount_curve_uix: npt.NDArray[np.int32]


# @dataclass(frozen=True, slots=True, kw_only=True)
# class VanillaEuropeanOptionNodeBlock(_VanillaEuropeanOptionBaseBlock):
#     """European option whose underlying is sourced from a scalar node."""

#     # ========== BLOCK ID FOR DISPATCH ==========
#     BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_EUROPEAN_OPTION_NODE)

#     # ========== MARKET LINKAGES ==========
#     # Index into `MarketDataMatrix.forward_nodes`.
#     underlying_price_uix: npt.NDArray[np.int32]

#     # Index into `MarketDataMatrix.vol_nodes`.
#     vol_uix: npt.NDArray[np.int32]


# @dataclass(frozen=True, slots=True, kw_only=True)
# class VanillaEuropeanOptionCurveBlock(_VanillaEuropeanOptionBaseBlock):
#     """European option whose underlying and vol come from curves/surfaces."""

#     # ========== BLOCK ID FOR DISPATCH ==========
#     BLOCK_TYPE_ID: ClassVar[int] = int(BlockType.VANILLA_EUROPEAN_OPTION_CURVE)

#     # ========== DELIVERY DATE (DATE UIX) ==========
#     delivery_date_uix: npt.NDArray[np.int32]

#     # ========== MARKET LINKAGES ==========
#     # Index into `MarketDataMatrix.price_curve_grid` axis 0.
#     underlying_price_curve_uix: npt.NDArray[np.int32]

#     # Index into `MarketDataMatrix.vol_surfaces` axis 0.
#     vol_surface_uix: npt.NDArray[np.int32]

#     # Y-axis kind for the vol surface (e.g., strike, log-moneyness).
#     vol_surface_y_axis_type_id: npt.NDArray[np.int8]
