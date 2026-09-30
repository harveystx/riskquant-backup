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
# NODE  (scalar; IS its own risk factor)
# ============================================================


class RiskStaticConfigNodeMeta(TypedDict):
    node_id: Sequence[int]
    node_name: Sequence[str]
    quote_ccy: Sequence[str]
    asset_class: Sequence[str]
    asset_subclass: Sequence[str]
    quote_type: Sequence[str]
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
    quote_type: Sequence[str]
    unit_scale: Sequence[str]
    shift_type: Sequence[str]
    x_axis_kind: Sequence[str]
    x_interp_method: Sequence[str]
    x_extrap_method: Sequence[str]
    roll_method: Sequence[str | None]
    anchor_offset: Sequence[int | None]
    roll_month: Sequence[int | None]
    roll_day: Sequence[int | None]
    fh_bh_mmdd: Sequence[str | None]
    curve_input_type: Sequence[str | None]
    curve_compounding: Sequence[str | None]
    maturity_calendar: Sequence[str | None]
    spot_lag_days: Sequence[int | None]
    business_day_convention: Sequence[str | None]
    end_of_month: Sequence[bool | None]
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
    quote_type: Sequence[str]
    unit_scale: Sequence[str]
    shift_type: Sequence[str]
    x_axis_kind: Sequence[str]
    x_interp_method: Sequence[str]
    x_extrap_method: Sequence[str]
    roll_method: Sequence[str | None]
    anchor_offset: Sequence[int | None]
    roll_month: Sequence[int | None]
    roll_day: Sequence[int | None]
    fh_bh_mmdd: Sequence[str | None]
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
    node_meta: RiskStaticConfigNodeMeta
    curve_meta: RiskStaticConfigCurveMeta
    curve_point_meta: RiskStaticConfigCurvePointMeta
    surface_meta: RiskStaticConfigSurfaceMeta
    surface_point_meta: RiskStaticConfigSurfacePointMeta


#####################################
# END OF RISK STATIC CONFIG SCHEMAS #
#####################################

####################
# RISK PROXY MODEL #
####################


class RiskProxyModelTermData(TypedDict):
    """One row per authored Proxy V1 term (D088 §12.3, D093).

    `term_kind` "NODE" -> `NodeProxyTerm` (target/driver are factor-node
    names); "CONTAINER" -> `ContainerProxyTerm` (target/driver are curve or
    surface names). No target chart and no horizon are authored.
    """

    term_kind: Sequence[str]  # "NODE" | "CONTAINER"
    target_name: Sequence[str]
    driver_name: Sequence[str]
    driver_chart: Sequence[str]  # "LOG_RETURN" | "SIMPLE_RETURN" | "DIFFERENCE"
    loading: Sequence[float]


# ============================================================
# BASE
# ============================================================


class _BaseIngestData(TypedDict):
    uid: Sequence[int]
    instrument: Sequence[str]


# ============================================================
# MARKET RISK
# ============================================================


class _OptionalNettingSet(TypedDict, total=False):
    # Effective-netting-set linkage. Optional: absent/None -> NO_NETTING_SET.
    netting_set_id: Sequence[int | None]


class _SpotForwardOptionFields(TypedDict, total=False):
    # --- SPOT / FORWARD / OPTION ---

    quantity: Sequence[float | None]
    underlying: Sequence[str | None]
    settlement_ccy: Sequence[str | None]

    strike_price: Sequence[float | None]
    strike_ccy: Sequence[str | None]

    is_call: Sequence[bool | None]

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


class _VanillaIrsFields(TypedDict, total=False):
    notional: Sequence[float | None]

    pay_receive_fixed: Sequence[str | None]

    fixed_rate: Sequence[float | None]
    float_spread: Sequence[float | None]

    settlement_ccy: Sequence[str | None]

    effective_date: Sequence[str | None]
    maturity_date: Sequence[str | None]

    fixed_freq: Sequence[str | None]
    float_freq: Sequence[str | None]

    fixed_daycount: Sequence[str | None]
    float_daycount: Sequence[str | None]

    forecast_curve: Sequence[str | None]
    discount_curve: Sequence[str | None]

    fixed_holiday_calendar: Sequence[str | None]
    float_holiday_calendar: Sequence[str | None]

    fixed_business_day_convention: Sequence[str | None]
    float_business_day_convention: Sequence[str | None]

    date_generation_rule: Sequence[str | None]  # "FORWARD" or "BACKWARD"

    payment_lag: Sequence[int | None]
    fixing_lag: Sequence[int | None]


class MarketRiskIngestData(
    _BaseIngestData,
    _OptionalNettingSet,
    _SpotForwardOptionFields,
    _VanillaIrsFields,
):
    """
    Canonical market-risk fixture schema.

    Represents the union of all supported market-risk instrument
    columns. Each instrument family populates only the subset of
    columns it requires; all other columns are None.

    This schema exists for fixture authoring and validation only.
    It is NOT a runtime contract.
    """

    pass


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


class _OptionalFactorNu(TypedDict, total=False):
    # Chi degrees of freedom. Present (and > 0) only for CHI rows; absent /
    # None for every other kind, mapped to NO_CHI_NU at compile time.
    nu: Sequence[float | None]


class _OptionalFactorBetaParams(TypedDict, total=False):
    # Beta distribution parameters for Beta PD factors. Present and > 0 only for
    # BETA_PD rows; absent / None for every other kind, mapped to NO_BETA_PARAMS
    # at compile time.
    alpha: Sequence[float | None]
    beta: Sequence[float | None]


class CreditFactorUniverseIngestData(_OptionalFactorNu, _OptionalFactorBetaParams):
    """
    Credit factor-universe master ingest frame.

    Single source of truth for the credit factor space. One row per factor:
    a stable `factor_id` identity (sorted ascending -> dense UIX = row position
    in the (F, S) CreditFactorMatrix), the human `factor_name` obligor JTD
    frames reference, and the `factor_kind` (a `CreditFactorKind` name). `nu`
    is the only kind-specific column: required and > 0 for CHI factors, omitted
    elsewhere — so each chi factor's degrees of freedom live here, once, rather
    than being duplicated onto obligor rows.
    """

    factor_id: Sequence[int]
    factor_name: Sequence[str]
    factor_kind: Sequence[str]  # a CreditFactorKind name (SYSTEMIC, CHI, ...)


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

    factor_loading: Sequence[float]
    systemic_factor: Sequence[str]
    idiosyncratic_factor: Sequence[str]


class GaussianCopulaJTDIngestData(_SingleFactorCopulaJTDCoreIngestData):
    pass


class TCopulaJTDIngestData(_SingleFactorCopulaJTDCoreIngestData):
    chi_squared_factor: Sequence[str]
    nu: Sequence[float]


# ============================================================
# CREDIT RISK | BERNOULLI-FAMILY JTD (conditional-independence)
# ============================================================


class IndependentBernoulliJTDIngestData(_BaseIngestData):
    """
    Independent-Bernoulli JTD frame.

    Each obligor defaults if its OWN uniform draw falls below its (fixed,
    master-sourced) PD: ``u_i < pd_i``. The uniforms MUST be per-obligor
    distinct UNIFORM factors — sharing a row makes defaults comonotonic rather
    than independent. PD/LGD are joined off the obligor master.
    """

    uniform_factor: Sequence[str]


class StressRegimeBernoulliJTDIngestData(_BaseIngestData):
    """
    Stress-regime Bernoulli JTD frame.

    Independent-Bernoulli with a single shared STRESS_INDICATOR factor that, when
    on, scales each obligor's PD by ``stress_pd_multiplier``. Uniforms are
    per-obligor distinct; the stress regime factor is global (shared).
    """

    uniform_factor: Sequence[str]
    stress_regime_factor: Sequence[str]
    stress_pd_multiplier: Sequence[float]


class BetaBinomialJTDIngestData(_BaseIngestData):
    """
    Beta-binomial JTD frame.

    A shared per-scenario default probability ``P_s ~ Beta(alpha, beta)`` (drawn
    into the BETA_PD factor row) drives conditionally-independent obligor
    defaults ``u_i < P_s``. Uniforms are per-obligor distinct; the beta factor is
    shared (the common shock). PD is NOT carried — it is stochastic; only LGD is
    joined off the obligor master.
    """

    uniform_factor: Sequence[str]
    beta_factor: Sequence[str]


# ============================================================
# CREDIT RISK | MULTI-FACTOR COPULA JTD
# ============================================================


class MfGaussianCopulaJTDIngestData(_BaseIngestData):
    """
    Multi-factor Gaussian copula per-obligor frame: one row per obligor.

    The systemic loadings are NOT here — they are ragged (each obligor loads on
    a different subset of systemic factors), so they live in a separate
    long-format loadings frame (`MfCopulaLoadingsData`), one row per
    obligor x systemic factor, packed into CSR at compile time. Only the
    per-obligor idiosyncratic factor lives here. PD/LGD are joined off the
    obligor master.
    """

    idiosyncratic_factor: Sequence[str]


class MfCopulaLoadingsData(TypedDict):
    """
    Long-format multi-factor copula loadings: one row per obligor x systemic
    factor (`obligor_uid`, `factor_name`, `factor_loading`). Packed into CSR in
    block-band order at compile time; `sum factor_loading^2 <= 1` per obligor.
    """

    obligor_uid: Sequence[int]
    factor_name: Sequence[str]
    factor_loading: Sequence[float]
