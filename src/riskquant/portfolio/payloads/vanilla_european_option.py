"""
Payload definition for vanilla European option trades.

This module defines a compact, immutable container that encodes the
minimal economic and market-linkage information required by option pricers
and valuation engines. The payload deliberately carries identifiers
for risk factors (price, vol, FX) and a date encoded as an ordinal32
(days since DATE_EPOCH), rather than storing strings or full market
objects. This keeps memory footprint small and makes the payload
well-suited for large portfolios and high-throughput simulation.

Notes on expiry encoding:
  - The payload uses `expiry_date_ordinal32` (int32 days-since-epoch)
    to represent the option expiry date, aligning with how settlement
    dates are encoded in ForwardPayload (`settlement_date_ordinal32`).
  - The sentinel value `NO_EXPIRY_DATE` indicates a missing expiry.
"""

from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.constants import (
    NO_DELIVERY_DATE_ORDINAL32,
    NO_EXPIRY_DATE_ORDINAL32,
    NO_FX_RF_REQUIRED,
    NO_ID,
    NO_SETTLEMENT_DATE_ORDINAL32,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _VanillaEuropeanOptionBasePayload:
    """
    Common fields shared by vanilla European option payload variants.
    """

    # ========== ECONOMIC TERMS ==========

    # Signed position size (contracts / notional)
    quantity: np.float64

    # Strike price K expressed in strike currency units
    strike: np.float64

    # 1 => call, 0 => put; stored as int8 for compactness
    is_call: np.int8

    # ========== EXPIRY DATE ==========

    # Expiry encoded as int32 ordinal (days since DATE_EPOCH). Matches ForwardPayload.
    expiry_date_ordinal32: np.int32

    # ========== SETTLEMENT DATE ==========

    # Optional settlement date (days-since-epoch). Used for discounting only. dtype: np.int32
    payoff_settlement_date_ordinal32: np.int32 = NO_SETTLEMENT_DATE_ORDINAL32

    # ========== FX LINKAGES ==========

    # FX id for underlying_ccy -> settlement_ccy, or FX_NONE if no conversion
    fx_under_to_settle_rf_id: np.int32 = NO_FX_RF_REQUIRED

    # FX id for strike_ccy -> settlement_ccy, or FX_NONE if no conversion
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
            "is_call",
            np.int8(self.is_call),
        )
        object.__setattr__(
            self,
            "expiry_date_ordinal32",
            np.int32(self.expiry_date_ordinal32),
        )
        object.__setattr__(
            self,
            "payoff_settlement_date_ordinal32",
            np.int32(self.payoff_settlement_date_ordinal32),
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

        # basic call/put validation (optional but cheap)
        if int(self.is_call) not in (0, 1):
            raise ValueError("is_call must be 0 (put) or 1 (call).")

        # Expiry date ordinal must be non-negative and cannot be the NO_EXPIRY_DATE_ORDINAL32 sentinel
        if self.expiry_date_ordinal32 == NO_EXPIRY_DATE_ORDINAL32:
            raise ValueError(
                "European option payload requires expiry_date_ordinal32 (cannot be sentinel)."
            )
        if self.expiry_date_ordinal32 < 0:
            raise ValueError("expiry_date_ordinal32 must be non-negative.")

        # Settlement date ordinal must be non-negative or the NO_SETTLEMENT_DATE_ORDINAL32 sentinel
        if (
            self.payoff_settlement_date_ordinal32 < 0
            and self.payoff_settlement_date_ordinal32
            != NO_SETTLEMENT_DATE_ORDINAL32
        ):
            raise ValueError(
                "payoff_settlement_date_ordinal32 must be non-negative or NO_SETTLEMENT_DATE_ORDINAL32 sentinel."
            )

        # FX risk factor IDs must be either valid non-negative integers or the NO_FX_RF_REQUIRED sentinel
        for field_name in [
            "fx_under_to_settle_rf_id",
            "fx_strike_to_settle_rf_id",
        ]:
            field_value = getattr(self, field_name)
            if field_value == NO_ID:
                raise ValueError(
                    f"{field_name} cannot be NO_ID sentinel. Use NO_FX_RF_REQUIRED if no FX conversion is needed."
                )
            if field_value != NO_FX_RF_REQUIRED and field_value < 0:
                raise ValueError(
                    f"{field_name} must be a non-negative integer or NO_FX_RF_REQUIRED sentinel."
                )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaEuropeanOptionNodePayload(_VanillaEuropeanOptionBasePayload):
    """
    Immutable payload for a single vanilla European option position when
    the underlying price is sourced from a node.
    """

    # ========== UNDERLYING LINKAGE ==========

    # ID referencing the underlying price/forward risk factor used by the pricer
    underlying_price_rf_id: np.int32

    # ========== VOL LINKAGE ==========

    # Implied volatility risk factor id (volatility series used by pricer). This is a node-level risk factor id that resolves to a time series in scenario shocks. It is a separate risk factor from the price_rf_id because some pricers require a separate vol input, and the vol may be sourced from a different node than the price (e.g., a volatility surface vs. a forward curve).
    vol_rf_id: np.int32

    # ========== VALIDATION IN __post_init__ ==========

    def __post_init__(self) -> None:
        _VanillaEuropeanOptionBasePayload.__post_init__(self)

        # Underlying price_rf_id must be a valid non-negative integer (no sentinel allowed)
        if self.underlying_price_rf_id == NO_ID:
            raise ValueError(
                "underlying_price_rf_id is required (cannot be sentinel)."
            )
        if self.underlying_price_rf_id < 0:
            raise ValueError("underlying_price_rf_id must be non-negative.")

        # vol_rf_id must be a valid non-negative integer (no sentinel allowed)
        if self.vol_rf_id == NO_ID:
            raise ValueError("vol_rf_id is required (cannot be sentinel).")
        if self.vol_rf_id < 0:
            raise ValueError("vol_rf_id must be non-negative.")

    def __str__(self) -> str:
        """Return a compact single-line representation for logging/debugging."""
        return (
            "VanillaEuropeanOptionNodePayload("
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"is_call={int(self.is_call)}, "
            f"expiry_date_ordinal32={self.expiry_date_ordinal32}, "
            f"delivery_date_ordinal32={self.payoff_settlement_date_ordinal32},"
            f"payoff_settlement_date_ordinal32={self.payoff_settlement_date_ordinal32}, "
            f"underlying_price_rf_id={self.underlying_price_rf_id}, "
            f"vol_rf_id={self.vol_rf_id}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}"
            ")"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VanillaEuropeanOptionCurvePayload(_VanillaEuropeanOptionBasePayload):
    """
    Immutable payload for a single vanilla European option position when
    the underlying price is sourced from a curve.
    """

    # ========== UNDERLYING CURVE LINKAGE ==========

    # ID referencing the underlying price/forward risk factor used by the pricer
    underlying_price_curve_rf_id: np.int32

    # ========== DELIVERY LINKAGE ==========

    # Physical delivery date encoded as int32 ordinal (days since DATE_EPOCH). Used to determine curve point for pricing and valuation. Default is sentinel NO_DELIVERY_DATE_ORDINAL32 if not applicable.
    delivery_date_ordinal32: np.int32

    # ========== VOL LINKAGE ==========

    # Implied volatility risk factor id (volatility series used by pricer). This is a surface-level risk factor id that resolves to a surface in scenario shocks. It is a separate risk factor from the price_rf_id because some pricers require a separate vol input, and the vol may be sourced from a different node than the price (e.g., a volatility surface vs. a forward curve). The final vol used for pricing is interpolated from the vol surface based on the expiry date and strike (money-ness) of the option.
    vol_surface_rf_id: np.int32

    # ========== VALIDATION IN __post_init__ ==========

    def __post_init__(self) -> None:
        _VanillaEuropeanOptionBasePayload.__post_init__(self)

        # Underlying price curve_rf_id must be a valid non-negative integer (no sentinel allowed)
        if self.underlying_price_curve_rf_id == NO_ID:
            raise ValueError(
                "underlying_price_curve_rf_id is required (cannot be sentinel)."
            )
        if self.underlying_price_curve_rf_id < 0:
            raise ValueError(
                "underlying_price_curve_rf_id must be non-negative."
            )

        # delivery_date_ordinal32 must be non-negative or the NO_DELIVERY_DATE_ORDINAL32 sentinel
        if self.delivery_date_ordinal32 == NO_DELIVERY_DATE_ORDINAL32:
            raise ValueError(
                "VanillaEuropeanOptionCurvePayload requires delivery_date_ordinal32 (cannot be sentinel)."
            )
        if self.delivery_date_ordinal32 < 0:
            raise ValueError("delivery_date_ordinal32 must be non-negative.")

        # vol_surface_rf_id must be a valid non-negative integer (no sentinel allowed)
        if self.vol_surface_rf_id == NO_ID:
            raise ValueError(
                "vol_surface_rf_id is required (cannot be sentinel)."
            )
        if self.vol_surface_rf_id < 0:
            raise ValueError("vol_surface_rf_id must be non-negative.")

    def __str__(self) -> str:
        """Return a compact single-line representation for logging/debugging."""
        return (
            "VanillaEuropeanOptionCurvePayload("
            f"quantity={self.quantity}, "
            f"strike={self.strike}, "
            f"is_call={int(self.is_call)}, "
            f"expiry_date_ordinal32={self.expiry_date_ordinal32}, "
            f"payoff_settlement_date_ordinal32={self.payoff_settlement_date_ordinal32}, "
            f"underlying_price_curve_rf_id={self.underlying_price_curve_rf_id}, "
            f"vol_surface_rf_id={self.vol_surface_rf_id}, "
            f"fx_under_to_settle_rf_id={self.fx_under_to_settle_rf_id}, "
            f"fx_strike_to_settle_rf_id={self.fx_strike_to_settle_rf_id}"
            ")"
        )
