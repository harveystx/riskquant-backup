"""
Compiled block for forward trades (struct-of-arrays).

This module defines `ForwardBlock`, the compiled runtime representation of
forward positions. Produced by the portfolio compiler from `ForwardPayload`
objects, it stores NumPy arrays of numeric indices into a `CompiledPortfolio`
factor universe, enabling fast vectorized valuation and scenario simulation.

Key design:
  - Row i across all arrays corresponds to trade UID `trade_uids[i]`.
  - `price_uix` identifies the exposure timing (forward curve node index)
    and is distinct from `settlement_date_ordinal32`, which is used only
    for discounting the cash payoff.
  - FX indices use sentinel values (e.g., -1) to indicate when no FX
    conversion is required; this keeps arrays numeric and vectorizable.
  - No `price_source_type` is stored; that interpretation is resolved at
    compile time by the portfolio compiler.
"""

from dataclasses import dataclass

import numpy as np

# Import base Block interface for type checking and polymorphism
from riskquant.portfolio.blocks.base_block import Block


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardNodeBlock(Block):
    """
    Struct-of-arrays representation of compiled forward trades.
    """

    # ========== TRADE IDENTITY ==========

    # Unique trade identifiers (one per row). dtype: int64 (np.int64)
    trade_uids: np.ndarray

    # ========== ECONOMIC TERMS ==========

    # Signed contract quantities/notionals. dtype: float64 (np.float64)
    qty: np.ndarray

    # Strike prices for each forward contract. dtype: float64 (np.float64)
    strike: np.ndarray

    # ========== SETTLEMENT DATE ==========

    # Settlement/payment date encoded as int32 ordinal (days since DATE_EPOCH).
    # Index into `CompiledPortfolio` discount factor curves. Use sentinel
    # NO_SETTLEMENT_DATE_ORDINAL32 to indicate no discounting.
    settlement_date_ordinal32: np.ndarray

    # ========== MARKET LINKAGES (UNIVERSE INDICES) ==========

    # Index into compiled price factor universe for exposure timing.
    # Resolves to a forward curve node or surface used in scenario shocks.
    price_uix: np.ndarray

    # Index into compiled FX factor universe mapping underlying_ccy -> settlement_ccy,
    # or sentinel (-1) when no FX conversion is required. dtype: int32
    fx_under_to_settle_uix: np.ndarray

    # Index into compiled FX factor universe mapping strike_ccy -> settlement_ccy,
    # or sentinel (-1) when no FX conversion is required. dtype: int32
    fx_strike_to_settle_uix: np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardCurveBlock(Block):
    """
    Struct-of-arrays representation of compiled forward trades.
    """

    # ========== TRADE IDENTITY ==========

    # Unique trade identifiers (one per row). dtype: int64 (np.int64)
    trade_uids: np.ndarray

    # ========== ECONOMIC TERMS ==========

    # Signed contract quantities/notionals. dtype: float64 (np.float64)
    qty: np.ndarray

    # Strike prices for each forward contract. dtype: float64 (np.float64)
    strike: np.ndarray

    # ========== SETTLEMENT DATE ==========

    # Physical delivery date encoded as int32 ordinal (days since DATE_EPOCH).
    # Default is sentinel NO_DELIVERY_DATE_ORDINAL32 if not applicable.
    delivery_date_ordinal32: np.ndarray

    # Settlement/payment date encoded as int32 ordinal (days since DATE_EPOCH).
    # Index into `CompiledPortfolio` discount factor curves. Use sentinel
    # NO_SETTLEMENT_DATE_ORDINAL32 to indicate no discounting.
    settlement_date_ordinal32: np.ndarray

    # ========== MARKET LINKAGES (UNIVERSE INDICES) ==========

    # Index into compiled price curve universe.
    price_curve_uix: np.ndarray

    # Index into compiled FX factor universe mapping underlying_ccy -> settlement_ccy,
    # or sentinel (-1) when no FX conversion is required. dtype: int32
    fx_under_to_settle_uix: np.ndarray

    # Index into compiled FX factor universe mapping strike_ccy -> settlement_ccy,
    # or sentinel (-1) when no FX conversion is required. dtype: int32
    fx_strike_to_settle_uix: np.ndarray
