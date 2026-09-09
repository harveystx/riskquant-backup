# src/riskquant/valuation/blocks/vanilla_european_option_block.py
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
    log_strike: NDArray[np.float64]
    cp_mult: NDArray[np.int8]
    sqrt_tte: NDArray[np.float64]
    half_tte: NDArray[np.float64]

    # --- LINKAGES (UIX) ---

    # Index into ValuationState.forward_prices for the underlying forward price in
    # the STRIKE CURRENCY. This is the forward price at the expiry date of the option.
    forward_price_strike_ccy_uix: NDArray[np.int32]

    # Index into ValuationState.log_forward_prices for the log of the underlying
    # forward price in the STRIKE CURRENCY. This is the log of the forward price
    # in STRIKE CURRENCY at the expiry date of the option.
    option_log_forward_price_strike_ccy_uix: NDArray[np.int32]

    # Index into ValuationState.vols for the volatility of the underlying forward price.
    # This is the volatility at the expiry date of the option.
    vol_uix: NDArray[np.int32]

    # Index into ValuationState.effective_discount_factors for the discount factor in
    # the STRIKE CURRENCY. Discounts the cashflow from the SETTLEMENT DATE of the
    # option to t = 0.
    effective_discount_factor_uix: NDArray[np.int32]
