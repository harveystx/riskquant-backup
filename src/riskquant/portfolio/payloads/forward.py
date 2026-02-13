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
from riskquant.portfolio.constants import (
    NO_ID,
    NO_DELIVERY_DATE_ORDINAL32,
    NO_SETTLEMENT_DATE_ORDINAL32,
    NO_FX_RF_REQUIRED,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ForwardBasePayload:
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
    # Sentinel NO_FX_RF_REQUIRED (-1) indicates no FX conversion needed.
    fx_under_to_settle_rf_id: np.int32 = NO_FX_RF_REQUIRED

    # FX risk factor ID mapping strike_ccy -> settlement_ccy.
    # Sentinel NO_FX_RF_REQUIRED (-1) indicates no FX conversion needed.
    fx_strike_to_settle_rf_id: np.int32 = NO_FX_RF_REQUIRED

    def __repr__(self) -> str:
        return self.__str__()


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

    def __post_init__(self) -> None:
        if self.price_rf_id == NO_ID:
            raise ValueError(
                "ForwardNodePayload requires price_rf_id (cannot be sentinel)."
            )
        if self.price_rf_id < 0:
            raise ValueError("price_rf_id must be non-negative.")

    def __str__(self) -> str:
        """Return a compact, single-line string representation for logging and debugging."""
        return (
            f"ForwardNodePayload("
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"settlement_date_ordinal32={self.settlement_date_ordinal32}, "
            f"price_rf_id={self.price_rf_id}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}"
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

    def __post_init__(self) -> None:
        if self.delivery_date_ordinal32 == NO_DELIVERY_DATE_ORDINAL32:
            raise ValueError(
                "ForwardCurvePayload requires delivery_date_ordinal32 (cannot be sentinel)."
            )
        if self.delivery_date_ordinal32 < 0:
            raise ValueError("delivery_date_ordinal32 must be non-negative.")
        if self.price_curve_id == NO_ID:
            raise ValueError(
                "ForwardCurvePayload requires price_curve_id (cannot be sentinel)."
            )
        if self.price_curve_id < 0:
            raise ValueError("price_curve_id must be non-negative.")

    def __str__(self) -> str:
        """Return a compact, single-line string representation for logging and debugging."""
        return (
            f"ForwardCurvePayload("
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"delivery_date_ordinal32={self.delivery_date_ordinal32}, "
            f"settlement_date_ordinal32={self.settlement_date_ordinal32}, "
            f"price_curve_id={self.price_curve_id}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}"
            f")"
        )


if __name__ == "__main__":
    # Example instantiation of ForwardNodePayload for testing and demonstration
    fwd_node_payload = ForwardNodePayload(
        quantity=np.float64(1_000.0),
        strike=np.float64(1.2345),
        settlement_date_ordinal32=np.int32(19_100),
        price_rf_id=np.int32(101),
        fx_under_to_settle_rf_id=NO_FX_RF_REQUIRED,
        fx_strike_to_settle_rf_id=NO_FX_RF_REQUIRED,
    )
    print("Example ForwardNodePayload:")
    print(fwd_node_payload)

    # Example instantiation of ForwardCurvePayload for testing and demonstration
    fwd_curve_payload = ForwardCurvePayload(
        quantity=np.float64(1_000.0),
        strike=np.float64(1.2345),
        delivery_date_ordinal32=np.int32(19_150),
        settlement_date_ordinal32=np.int32(19_100),
        price_curve_id=np.int32(202),
        fx_under_to_settle_rf_id=NO_FX_RF_REQUIRED,
        fx_strike_to_settle_rf_id=NO_FX_RF_REQUIRED,
    )
    print("\nExample ForwardCurvePayload:")
    print(fwd_curve_payload)
