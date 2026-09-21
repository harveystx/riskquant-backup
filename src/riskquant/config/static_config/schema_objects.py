# src/riskquant/config/static_config/schema_objects.py
"""
Immutable value objects for the god-tier static config pipeline.

One dataclass per input table. Every field is non-null in its own table
unless the domain makes it method-specific: no nullability cascades across
tables. The three first-class object kinds (Node, Curve, Surface) each get
their own metadata table and child point table.

  NodeMeta              - scalar observable. A node IS its own risk factor.
                          FX rate nodes additionally carry fx_base_ccy.
  CurveMeta             - 1D function metadata. Owns its X-axis tenor-
                          resolution convention directly (roll_method plus
                          method-specific parameters). Optional discount-
                          basis tag marks IR discount curves.
  CurvePointMeta        - one row per point on a curve. The point's
                          tenor_token is parsed using the parent's
                          x_axis_kind.
  SurfaceMeta           - 2D function metadata (expiry x strike-axis). Owns
                          its X-axis tenor-resolution convention directly,
                          exactly as CurveMeta does.
  SurfacePointMeta      - one row per point on a surface; x_tenor_token
                          is parsed using the surface's x_axis_kind,
                          y_label using its y_axis_kind.
  HolidayCalendar       - one calendar with sorted int32 holiday ordinals.
  Ccy                   - one currency with its dense int8 id.

There is no standalone roll-rule table. A curve / surface interprets its
own X-axis; the resolution convention is part of the container's metadata,
not a shared entity behind a name reference. See
``riskquant.config.enums.roll_methods`` for the authoritative RollMethod,
allowed-pair, and parameter contract.

All system-defined taxonomies (asset class, unit, shift, interp/extrap,
x/y axis kind, roll method, discount basis) live in riskquant.config.enums
and riskquant.market_data — never inside these dataclasses as derived state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# ============================================================
# CURRENCY
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class Ccy:
    """A currency in the organisation's active universe."""

    ccy_name: str
    ccy_id: int


# ============================================================
# HOLIDAY CALENDAR
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class HolidayCalendar:
    """
    Holiday dates for one named calendar, stored as sorted int32
    ordinals (days since 1970-01-01) for O(log N) np.searchsorted
    bisection in the business-day engine.
    """

    holiday_calendar_name: str
    holiday_dates_ord: npt.NDArray[np.int32]


# ============================================================
# NODE  (scalar; IS its own risk factor)
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeMeta:
    """
    A scalar observable. Every node is both a container (for ingest
    classification) and a risk factor (for market-data lookup). The
    node_id IS the rf_id; there is no child point table.

    fx_base_ccy contract:
      asset_class == "FX"  <=> fx_base_ccy is not None
      For FX nodes, node_name == fx_base_ccy + quote_ccy is enforced.
    """

    node_id: int
    node_name: str

    quote_ccy: str
    asset_class: str
    asset_subclass: str

    quote_type: str
    unit_scale: str
    shift_type: str

    fx_base_ccy: str | None


# ============================================================
# CURVE
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class CurveMeta:
    """
    A 1D function metadata row.

    X-axis tenor-resolution contract (self-describing — no rule table):
      x_axis_kind == LABEL   -> roll_method and every roll parameter
                                (anchor_offset, roll_month, roll_day,
                                fh_bh_mmdd) are None. A LABEL axis does not
                                depend on the as-of date.
      x_axis_kind == OFFSET_* -> roll_method is non-None and the
                                (x_axis_kind, roll_method) pair is in
                                ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS. Only
                                the method's required parameters are
                                populated; the rest are None. See
                                ROLL_METHOD_REQUIRED_PARAMS.

      Parameter semantics:
        anchor_offset  CALENDAR_PERIOD only. Offset (months for OFFSET_M,
                       years for OFFSET_Y) added to the as-of period to form
                       the spot anchor before the token offset applies.
        roll_month/roll_day  ROLL_ON_MMDD only. The annual roll boundary.
        fh_bh_mmdd     HALF_YEAR_BY_DATE / HALF_YEAR_BY_LABEL only. The
                       'MMDD|MMDD' half-year boundary convention.

    D080 construction metadata (curve_compounding, maturity_calendar,
    spot_lag_days, business_day_convention, end_of_month):
      Everything a constructed discount curve needs beyond its levels. All are
      None unless the curve is declared constructible; a missing field is a
      configuration error at construction time, never a default. They are read
      only by the market-data curve layer and never reach valuation.

    curve_input_type contract (D079 clause 3):
      The ECONOMIC representation of this curve's levels (PAR_RATE, ZERO_RATE,
      DISCOUNT_FACTOR, LOG_DISCOUNT_FACTOR), orthogonal to quote_type. Declared
      once here; curve points inherit it through curve_id. None means "not yet
      declared": the curve may still be used as a market-data input, and any
      future curve construction for it stays fail-closed under D061.

    discount_basis contract:
      "NOT_A_DISCOUNT_CURVE"   -> non-discount curve (e.g. commodity forward).
                                  is_default_discount_for_ccy must be False.
      any other value          -> discount curve. asset_class must be "IR".
                                  Exactly one discount curve per (ccy) is
                                  flagged is_default_discount_for_ccy=True
                                  and its basis must be the firm default.
    """

    curve_id: int
    curve_name: str

    quote_ccy: str
    asset_class: str
    asset_subclass: str

    quote_type: str
    unit_scale: str
    shift_type: str

    x_axis_kind: str
    x_interp_method: str
    x_extrap_method: str

    roll_method: str | None
    anchor_offset: int | None
    roll_month: int | None
    roll_day: int | None
    fh_bh_mmdd: str | None

    curve_input_type: str | None
    curve_compounding: str | None
    maturity_calendar: str | None
    spot_lag_days: int | None
    business_day_convention: str | None
    end_of_month: bool | None

    discount_basis: str
    is_default_discount_for_ccy: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class CurvePointMeta:
    """One addressable point on a curve."""

    rf_id: int
    curve_id: int
    curve_name: str
    tenor_token: str


# ============================================================
# SURFACE
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceMeta:
    """
    A 2D function metadata row. Same X-axis tenor-resolution contract as
    CurveMeta (roll_method plus method-specific parameters, owned directly).
    The X-axis resolution convention is independent of the Y axis: y_axis_kind
    has no roll coupling because the y coordinate is value-typed (a strike,
    log-strike, moneyness, or delta).
    """

    surface_id: int
    surface_name: str

    quote_ccy: str
    asset_class: str
    asset_subclass: str

    quote_type: str
    unit_scale: str
    shift_type: str

    x_axis_kind: str
    x_interp_method: str
    x_extrap_method: str

    roll_method: str | None
    anchor_offset: int | None
    roll_month: int | None
    roll_day: int | None
    fh_bh_mmdd: str | None

    y_axis_kind: str
    y_interp_method: str
    y_extrap_method: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfacePointMeta:
    """One addressable point on a surface."""

    rf_id: int
    surface_id: int
    surface_name: str
    x_tenor_token: str
    y_label: str
