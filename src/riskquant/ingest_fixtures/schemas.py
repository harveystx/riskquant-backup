# src/riskquant/ingest_fixtures/schemas.py
"""
Canonical TypedDict schemas for every human-facing ingest frame.

Schemas mirror the SoA contracts they feed:
- Market-risk frames carry trade-keyed columns + optional credit linkage.
- JTD frames carry obligor-keyed *model-specific* columns ONLY. PD and
  LGD live on the obligor master; they are joined onto JTD rows at
  compile time and are never duplicated on per-model frames.
- Obligor- and netting-set-master frames are first-class schemas, one
  per master.
"""

from collections.abc import Sequence
from typing import TypedDict

######################
# RISK STATIC CONFIG #
######################


# ============================================================
# CURRENCY
# ============================================================


class RiskStaticConfigCcy(TypedDict):
    ccy_name: Sequence[str]
    ccy_id: Sequence[int]


# ============================================================
# HOLIDAY CALENDAR
# ============================================================


class RiskStaticConfigHolidayCalendar(TypedDict):
    holiday_calendar_name: Sequence[str]
    holiday_date: Sequence[str]  # ISO yyyy-mm-dd; compiler converts to int32 ordinals


# ============================================================
# SPOT ROLL RULE
# ============================================================


class RiskStaticConfigSpotRollRule(TypedDict):
    rule_name: Sequence[str]
    roll_method: Sequence[str]
    offset_in_tenor_units: Sequence[int]
    roll_month: Sequence[int | None]
    roll_day: Sequence[int | None]
    fh_bh_mmdd: Sequence[str | None]
    post_roll_shift: Sequence[int]


# ============================================================
# NODE  (scalar; IS its own risk factor)
# ============================================================


class RiskStaticConfigNodeMeta(TypedDict):
    node_id: Sequence[int]
    node_name: Sequence[str]
    quote_ccy: Sequence[str]
    asset_class: Sequence[str]
    asset_subclass: Sequence[str]
    unit_type: Sequence[str]
    unit_scale: Sequence[str]
    shift_type: Sequence[str]
    fx_base_ccy: Sequence[str | None]


# ============================================================
# CURVE
# ============================================================


class RiskStaticConfigCurveMeta(TypedDict):
    curve_id: Sequence[int]
    curve_name: Sequence[str]
    quote_ccy: Sequence[str]
    asset_class: Sequence[str]
    asset_subclass: Sequence[str]
    unit_type: Sequence[str]
    unit_scale: Sequence[str]
    shift_type: Sequence[str]
    x_axis_kind: Sequence[str]
    x_interp_method: Sequence[str]
    x_extrap_method: Sequence[str]
    spot_roll_rule_name: Sequence[str | None]
    discount_basis: Sequence[str]
    is_default_discount_for_ccy: Sequence[bool]


class RiskStaticConfigCurvePointMeta(TypedDict):
    rf_id: Sequence[int]
    curve_id: Sequence[int]
    tenor_token: Sequence[str]


# ============================================================
# SURFACE
# ============================================================


class RiskStaticConfigSurfaceMeta(TypedDict):
    surface_id: Sequence[int]
    surface_name: Sequence[str]
    quote_ccy: Sequence[str]
    asset_class: Sequence[str]
    asset_subclass: Sequence[str]
    unit_type: Sequence[str]
    unit_scale: Sequence[str]
    shift_type: Sequence[str]
    x_axis_kind: Sequence[str]
    x_interp_method: Sequence[str]
    x_extrap_method: Sequence[str]
    spot_roll_rule_name: Sequence[str | None]
    y_axis_kind: Sequence[str]
    y_interp_method: Sequence[str]
    y_extrap_method: Sequence[str]


class RiskStaticConfigSurfacePointMeta(TypedDict):
    rf_id: Sequence[int]
    surface_id: Sequence[int]
    x_tenor_token: Sequence[str]
    y_label: Sequence[str]


# ============================================================
# TOP-LEVEL
# ============================================================


class RiskStaticConfigData(TypedDict):
    ccy: RiskStaticConfigCcy
    holiday_calendars: RiskStaticConfigHolidayCalendar
    spot_roll_rule: RiskStaticConfigSpotRollRule
    node_meta: RiskStaticConfigNodeMeta
    curve_meta: RiskStaticConfigCurveMeta
    curve_point_meta: RiskStaticConfigCurvePointMeta
    surface_meta: RiskStaticConfigSurfaceMeta
    surface_point_meta: RiskStaticConfigSurfacePointMeta


#####################################
# END OF RISK STATIC CONFIG SCHEMAS #
#####################################

#####################
# RISK PROXY CONFIG #
#####################


class RiskProxyConfigData(TypedDict):
    proxying_level: Sequence[str]  # "CONTAINER" | "RISK_FACTOR"
    required_name: Sequence[str]
    proxy_name: Sequence[str]
    vol_scaling_factor: Sequence[float]


# ============================================================
# BASE
# ============================================================


class _BaseIngestData(TypedDict):
    uid: Sequence[int]
    instrument: Sequence[str]


# ============================================================
# MARKET RISK
# ============================================================


class _MarketCoreIngestData(_BaseIngestData):
    quantity: Sequence[float]
    underlying: Sequence[str]
    settlement_ccy: Sequence[str]


class _OptionalNettingSet(TypedDict, total=False):
    # Effective-netting-set linkage. Optional: absent/None -> NO_NETTING_SET.
    netting_set_id: Sequence[int | None]


class MarketRiskIngestData(_MarketCoreIngestData, _OptionalNettingSet):
    strike_price: Sequence[float | None]
    strike_ccy: Sequence[str | None]

    is_call: Sequence[float | None]

    expiry_date: Sequence[str | None]
    delivery_date: Sequence[str | None]
    settlement_date: Sequence[str | None]

    under_discount_curve: Sequence[str | None]
    strike_discount_curve: Sequence[str | None]

    # Name of the volatility container (vol surface for option-curve;
    # vol surface or vol-node container for option-node) the option's
    # implied vol is sourced from. Required for OPTION-family rows,
    # None for SPOT and FORWARD rows. Resolved to an int32 container
    # id at compile time against `static_config.container_name_to_id`.
    volatility_container_name: Sequence[str | None]


# ============================================================
# CREDIT MASTERS
# ============================================================


class _OptionalObligorName(TypedDict, total=False):
    # Effective-netting-set linkage. Optional: absent/None -> NO_NETTING_SET.
    name: Sequence[str | None]


class ObligorCreditIngestData(_OptionalObligorName):
    """
    Obligor master ingest frame.

    Single source of truth for obligor-level credit attributes. PD and
    LGD live here, not on per-model JTD frames — that's the duplication
    the master is designed to prevent.

    `parent_uid` is nullable: NA/None marks an ultimate obligor and is
    mapped to `NO_PARENT_OBLIGOR` at compile time. The rating/sector/
    country int codes are placeholders for IntEnum taxonomies wired in a
    follow-up; `NO_RATING_ID`, `NO_SECTOR_ID`, `NO_COUNTRY_ID` mark
    unknowns.
    """

    uid: Sequence[int]
    pd: Sequence[float]
    lgd: Sequence[float]

    rating_id: Sequence[int]  # int8 enum, source: config/enums/credit_ratings (TBD)
    sector_id: Sequence[int]  # int32 enum (TBD)
    country_id: Sequence[int]  # int32 enum (TBD)

    parent_uid: Sequence[int | None]  # None -> NO_PARENT_OBLIGOR


class NettingSetMasterIngestData(TypedDict):
    """
    Netting-set master ingest frame.

    Carries the netting-set identifier (joined to the portfolio's
    `netting_set_id` column), the owning obligor, and net collateral
    (haircut-adjusted, >= 0).
    """

    uid: Sequence[int]
    obligor_uid: Sequence[int]
    net_collateral: Sequence[float]


# ============================================================
# CREDIT RISK | DETERMINISTIC JTD
# ============================================================


class DeterministicJTDIngestData(_BaseIngestData):
    """
    Deterministic JTD model frame.

    Carries only the dispatch keys (`uid`, `instrument`). LGD comes from
    the obligor master; PD is not read by the deterministic kernel; EAD
    flows in per scenario from the exposure engine. The frame is
    deliberately minimal — anything else here would duplicate state that
    already has a home elsewhere.
    """


# ============================================================
# CREDIT RISK | COPULA JTD (single-factor copulas)
# ============================================================


class _SingleFactorCopulaJTDCoreIngestData(_BaseIngestData):
    """
    Single-factor copula JTD core: model-specific fields only.

    PD and LGD are joined off the obligor master at compile time and
    intentionally absent from this schema.
    """

    beta: Sequence[float]
    systemic_factor: Sequence[str]
    idiosyncratic_factor: Sequence[str]


class GaussianCopulaJTDIngestData(_SingleFactorCopulaJTDCoreIngestData):
    pass


class TCopulaJTDIngestData(_SingleFactorCopulaJTDCoreIngestData):
    chi_squared_factor: Sequence[str]
    nu: Sequence[float]
