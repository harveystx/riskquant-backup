"""
Compiled block for forward trades (struct-of-arrays).

This module defines the compiled runtime representation of forward positions.
Produced by the portfolio compiler from `ForwardPayload` objects, it stores
NumPy arrays of numeric indices into a `CompiledPortfolio` factor universe,
enabling fast vectorized valuation and scenario simulation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class _ForwardBaseBlock:
    """
    Common struct-of-arrays fields shared by all forward blocks.
    """

    # ========== TRADE IDENTITY ==========
    trade_uids: np.ndarray

    # ========== ECONOMIC TERMS ==========
    qty: np.ndarray
    strike: np.ndarray

    # ========== SETTLEMENT DATE ==========
    settlement_date_ordinal32: np.ndarray

    # ========== FX MARKET LINKAGES ==========
    
    # Index into the compiled FX factor universe, or sentinel when no FX needed. dtype: int32 (np.int32)
    fx_under_to_settle_uix: np.ndarray
    fx_strike_to_settle_uix: np.ndarray

    # Flag to indicate if the math needs to invert the FX rate (1/X)
    fx_under_to_settle_invert: np.ndarray  # bool or int8
    fx_strike_to_settle_invert: np.ndarray # bool or int8

    # ========== DISCOUNT LINKAGES ==========
    under_discount_curve_uix: np.ndarray
    strike_discount_curve_uix: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardNodeBlock(_ForwardBaseBlock):
    """
    Struct-of-arrays representation of compiled forward trades
    priced directly off a node/scalar.
    """

    # ========== UNDERLYING MARKET LINKAGE ==========
    # Index into compiled price factor universe for exposure timing.
    price_uix: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardCurveBlock(_ForwardBaseBlock):
    """
    Struct-of-arrays representation of compiled forward trades
    priced off a curve requiring delivery date interpolation.
    """

    # ========== DELIVERY DATE ==========
    # Physical delivery date encoded as int32 ordinal.
    delivery_date_ordinal32: np.ndarray

    # ========== UNDERLYING MARKET LINKAGE ==========
    # Index into compiled price curve universe.
    price_curve_uix: np.ndarray
