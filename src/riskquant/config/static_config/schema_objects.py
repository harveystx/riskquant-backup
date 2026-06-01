# src/riskquant/config/static_config/schema_objects.py
"""
Immutable value objects for the god-tier static config pipeline.

One dataclass per input table. Every field is non-null in its own table:
no nullability cascades, no "must be null when this is null" rules. The
three first-class object kinds (Node, Curve, Surface) each get their own
metadata table and child point table.

  NodeMeta              - scalar observable. A node IS its own risk factor.
                          FX rate nodes additionally carry fx_base_ccy.
  CurveMeta             - 1D function metadata. Optional discount-basis tag
                          marks IR discount curves.
  CurvePointMeta        - one row per point on a curve. The point's
                          tenor_token is parsed using the parent's
                          x_axis_kind.
  SurfaceMeta           - 2D function metadata (expiry x strike-axis).
  SurfacePointMeta      - one row per point on a surface; x_tenor_token
                          is parsed using the surface's x_axis_kind,
                          y_label using its y_axis_kind.
  SpotRollRule          - one named rule. PK is the rule name.
  HolidayCalendar       - one calendar with sorted int32 holiday ordinals.
  Ccy                   - one currency with its dense int8 id.

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
# SPOT ROLL RULE
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotRollRule:
    """
    Named spot-roll rule.

    PK is `rule_name`. The rule's `roll_method` determines which
    parameters are meaningful — the validator checks each parameter
    against the method:

      CALENDAR_PERIOD   -> uses offset_in_tenor_units
      ROLL_ON_MMDD      -> uses roll_month, roll_day, post_roll_shift
      HALF_YEAR_BY_DATE -> uses fh_bh_mmdd
      HALF_YEAR_BY_LABEL-> uses fh_bh_mmdd

    Unused parameters carry sentinel zeros (offset, post_roll_shift)
    or None (roll_month, roll_day, fh_bh_mmdd) — these are never read by
    the resolution code for a non-applicable method.
    """

    rule_name: str
    roll_method: str
    offset_in_tenor_units: int
    roll_month: int | None
    roll_day: int | None
    fh_bh_mmdd: str | None
    post_roll_shift: int


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

    unit_type: str
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

    x_axis_kind contract:
      LABEL                    -> spot_roll_rule_name is None
      OFFSET_M / OFFSET_Y /
      OFFSET_HY                -> spot_roll_rule_name is non-None

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

    unit_type: str
    unit_scale: str
    shift_type: str

    x_axis_kind: str
    x_interp_method: str
    x_extrap_method: str
    spot_roll_rule_name: str | None

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
    A 2D function metadata row. Same x_axis_kind / spot_roll_rule
    contract as CurveMeta. y_axis_kind has no roll-method coupling: the
    y coordinate is value-typed (a strike, log-strike, moneyness, delta).
    """

    surface_id: int
    surface_name: str

    quote_ccy: str
    asset_class: str
    asset_subclass: str

    unit_type: str
    unit_scale: str
    shift_type: str

    x_axis_kind: str
    x_interp_method: str
    x_extrap_method: str
    spot_roll_rule_name: str | None

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
