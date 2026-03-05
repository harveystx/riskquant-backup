"""
Compiled block for vanilla European option trades.

This module defines `VanillaEuropeanOptionNodeBlock` and `VanillaEuropeanOptionCurveBlock`,
which are struct-of-arrays representations of option trades optimized for vectorized
valuation and simulation. The blocks store only numeric identifiers and primitive
arrays; pricers and resolvers map these compiled universe indices (uix) to full
market data matrices during runtime.

Key design:
  - Row i across all arrays corresponds to trade UID `trade_uids[i]`.
  - `_uix` fields (e.g., `vol_uix`, `underlying_price_uix`) index into the
    compiled `RiskFactorMatrix` universe, replacing the slower `_rf_id` lookups.
  - Expiry and settlement dates are encoded as int32 days-since-epoch for
    fast array-based interpolation and discounting.
  - `is_call` is packed as an `np.int8` array (1 for call, 0 for put) to minimize
    memory bandwidth during the inner pricing loops.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class _VanillaEuropeanOptionBaseBlock:
    """
    Common struct-of-arrays fields shared by all vanilla European option blocks.
    """

    # ========== TRADE IDENTITY ==========

    # Unique trade identifiers (one per row). dtype: int64 (np.int64)
    trade_uids: np.ndarray

    # ========== ECONOMIC TERMS ==========

    # Signed position sizes (contracts / notional). dtype: float64 (np.float64)
    qty: np.ndarray

    # Strike prices K expressed in strike currency units. dtype: float64 (np.float64)
    strike: np.ndarray

    # Option type indicator: 1 => call, 0 => put. dtype: int8 (np.int8)
    is_call: np.ndarray

    # ========== DATES ==========

    # Expiry date encoded as int32 ordinal (days since DATE_EPOCH).
    # Determines the time-to-expiry for pricing and interpolating the vol surface.
    expiry_date_ordinal32: np.ndarray

    # Settlement/payment date encoded as int32 ordinal (days since DATE_EPOCH).
    # Used for discounting the option payoff.
    payoff_settlement_date_ordinal32: np.ndarray

    # ========== FX LINKAGES ==========

    # Index into compiled FX universe mapping underlying_ccy -> settlement_ccy,
    # or sentinel when no FX conversion is required. dtype: int32
    fx_under_to_settle_uix: np.ndarray

    # Index into compiled FX universe mapping strike_ccy -> settlement_ccy,
    # or sentinel when no FX conversion is required. dtype: int32
    fx_strike_to_settle_uix: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaEuropeanOptionNodeBlock(_VanillaEuropeanOptionBaseBlock):
    """
    Struct-of-arrays representation of compiled vanilla European options
    where the underlying is sourced from a specific scalar node.
    """

    # ========== MARKET LINKAGES (UNIVERSE INDICES) ==========

    # Index into compiled price factor universe for the underlying asset. dtype: int32
    underlying_price_uix: np.ndarray

    # Index into compiled volatility factor universe (the vol node/series). dtype: int32
    vol_uix: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaEuropeanOptionCurveBlock(_VanillaEuropeanOptionBaseBlock):
    """
    Struct-of-arrays representation of compiled vanilla European options
    where the underlying and volatility are sourced from curves/surfaces
    that require date-based interpolation.
    """

    # ========== DATES ==========

    # Physical delivery date encoded as int32 ordinal (days since DATE_EPOCH).
    # Used to interpolate the underlying price curve to find the forward price.
    delivery_date_ordinal32: np.ndarray

    # ========== MARKET LINKAGES (UNIVERSE INDICES) ==========

    # Index into compiled price curve universe for the underlying asset. dtype: int32
    underlying_price_curve_uix: np.ndarray

    # Index into compiled volatility surface universe. dtype: int32
    vol_surface_uix: np.ndarray
