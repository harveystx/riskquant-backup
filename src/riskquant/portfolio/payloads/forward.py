"""
Payload definition for forward contract trades.

This module defines a lightweight, immutable container for forward positions.
The payload stores only numeric identifiers, primitive fields, and date ordinals
(int32 days-since-epoch). External pricers and valuation engines resolve the
identifiers to market data and perform pricing, discounting, and FX conversions.

Key design principle:
  Exposure timing (via `price_rf_id` and `price_source_type`) is intentionally
  decoupled from settlement timing (`settlement_date_ordinal32`). This enables
  support for forwards on long-dated production with arbitrary payment schedules,
  power and commodity contracts, and custom discounting logic.
"""

from dataclasses import dataclass

import numpy as np

# Import price source type indicators and date/FX sentinels for defaults
from riskquant.constants import (
    NO_FX_RF_REQUIRED,
    NO_SETTLEMENT_DATE_ORDINAL32,
)
from riskquant.portfolio.payloads.base_payload import Payload


@dataclass(frozen=True, slots=True, kw_only=True)
class _ForwardBasePayload(Payload):
    """Common fields shared by forward payload variants."""

    # ========== ECONOMIC TERMS ==========

    # Signed quantity/notional (long > 0, short < 0). dtype: np.float64
    quantity: np.float64

    # Strike/forward level K in trade currency units. dtype: np.float64
    strike: np.float64

    # ========== SETTLEMENT DATE ==========

    # Settlement/payment date encoded as int32 ordinal (days since DATE_EPOCH).
    # Used exclusively for discounting. Default sentinel indicates no discounting.
    settlement_date_ordinal32: np.int32 = NO_SETTLEMENT_DATE_ORDINAL32

    # ========== FX LINKAGES ==========

    # FX risk factor ID mapping underlying_ccy -> settlement_ccy.
    # Sentinel NO_FX_RF_REQUIRED (-3) indicates no FX conversion needed.
    fx_under_to_settle_rf_id: np.int32 = NO_FX_RF_REQUIRED

    # FX risk factor ID mapping strike_ccy -> settlement_ccy.
    # Sentinel NO_FX_RF_REQUIRED (-3) indicates no FX conversion needed.
    fx_strike_to_settle_rf_id: np.int32 = NO_FX_RF_REQUIRED

    # ========== VALIDATION IN __post_init__ ==========

    def __post_init__(self) -> None:
        # Force strict NumPy dtypes to prevent Python object boxing
        object.__setattr__(
            self,
            "quantity",
            np.float64(self.quantity),
        )
        object.__setattr__(
            self,
            "strike",
            np.float64(self.strike),
        )
        object.__setattr__(
            self,
            "settlement_date_ordinal32",
            np.int32(self.settlement_date_ordinal32),
        )
        object.__setattr__(
            self,
            "fx_under_to_settle_rf_id",
            np.int32(self.fx_under_to_settle_rf_id),
        )
        object.__setattr__(
            self,
            "fx_strike_to_settle_rf_id",
            np.int32(self.fx_strike_to_settle_rf_id),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardNodePayload(_ForwardBasePayload):
    """
    Immutable payload for a forward contract position.

    Encodes the minimal economic and market-linkage information required
    by pricers and valuation engines. All fields use compact NumPy dtypes
    to support efficient array-based processing and serialization.

    """

    # ========== UNDERLYING LINKAGES ==========

    # Unique identifier for the forward price/curve node used by the pricer.
    # Resolves to a time series or market surface in scenario shocks.
    price_rf_id: np.int32

    # ========== VALIDATION IN __post_init__ ==========

    def __post_init__(self) -> None:
        _ForwardBasePayload.__post_init__(self)  # Cast base fields
        object.__setattr__(
            self,
            "price_rf_id",
            np.int32(self.price_rf_id),
        )

    def __str__(self) -> str:
        """Return a compact, single-line string representation for logging and debugging."""
        return (
            f"ForwardNodePayload("
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"settlement_date_ordinal32={self.settlement_date_ordinal32}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}, "
            f"price_rf_id={self.price_rf_id}"
            f")"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ForwardCurvePayload(_ForwardBasePayload):
    """
    Immutable payload for a forward contract position when the price is sourced from a curve and the delivery_date_ordinal32 is used to determine the curve point for pricing and valuation.

    """

    # ========== UNDERLYING LINKAGES ==========

    # Unique identifier for the forward price/curve node used by the pricer.
    # Resolves to a time series or market surface in scenario shocks.
    price_curve_id: np.int32

    # ========== DELIVERY DATE (ORDINAL) ==========

    # Physical delivery date encoded as int32 ordinal (days since DATE_EPOCH).
    # Default is sentinel NO_DELIVERY_DATE_ORDINAL32 if not applicable.
    delivery_date_ordinal32: np.int32

    # ========== VALIDATION IN __post_init__ ==========

    def __post_init__(self) -> None:
        _ForwardBasePayload.__post_init__(self)  # Cast base fields
        object.__setattr__(
            self,
            "price_curve_id",
            np.int32(self.price_curve_id),
        )
        object.__setattr__(
            self,
            "delivery_date_ordinal32",
            np.int32(self.delivery_date_ordinal32),
        )

    def __str__(self) -> str:
        """Return a compact, single-line string representation for logging and debugging."""
        return (
            f"ForwardCurvePayload("
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"settlement_date_ordinal32={self.settlement_date_ordinal32}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}, "
            f"price_curve_id={self.price_curve_id}, "
            f"delivery_date_ordinal32={self.delivery_date_ordinal32}"
            f")"
        )
