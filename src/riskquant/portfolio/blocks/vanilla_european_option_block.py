"""
Compiled block for vanilla European option trades.

This module defines `VanillaEuropeanOptionBlock`, a struct-of-arrays
representation of option trades optimized for vectorized valuation and
simulation. The block stores only numeric identifiers and primitive
arrays; pricers and resolvers map indices to full market objects.

Key points:
  - Row i corresponds to trade UID `trade_uids[i]`.
  - `price_uix` and `vol_uix` index compiled factor universes (not risk-factor IDs).
  - `expiry_date_ordinal32` stores expiry as int32 days-since-epoch; use
    `NO_EXPIRY_DATE` to indicate a missing expiry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.blocks.base_block import Block


@dataclass(frozen=True, slots=True)
class VanillaEuropeanOptionBlock(Block):
    """
    Struct-of-arrays for compiled plain-vanilla European options.

    Attributes are NumPy arrays of length `m_opt` (number of option trades).
    The block uses frozen dataclass semantics and slots for memory efficiency.
    """

    # --- TRADE IDENTITY ---
    # Unique trade identifiers for each option row. dtype: int64 (np.int64)
    trade_uids: np.ndarray

    # --- ECONOMIC TERMS ---

    # Option quantities (signed). dtype: float64 (np.float64)
    qty: np.ndarray

    # Option strikes. dtype: float64 (np.float64)
    strike: np.ndarray

    # Option type flag: 1 = call, 0 = put. dtype: int8 (np.int8)
    is_call: np.ndarray

    # --- MARKET LINKAGES (UNIVERSE INDICES) ---

    # Index into compiled price factor universe for underlying exposure. dtype: int32
    price_uix: np.ndarray

    # Index into compiled volatility factor universe for implied vols. dtype: int32
    vol_uix: np.ndarray

    # --- EXPIRY DATE ---

    # Expiry date encoded as int32 ordinal (days since DATE_EPOCH)
    expiry_date_ordinal32: np.ndarray

    # --- OPTIONAL FX LINKAGES (UNIVERSE INDICES) ---

    # FX index for underlying_ccy -> settlement_ccy or sentinel (-1) if none. dtype: int32
    fx_under_to_settle_uix: np.ndarray

    # FX index for strike_ccy -> settlement_ccy or sentinel (-1) if none. dtype: int32
    fx_strike_to_settle_uix: np.ndarray
