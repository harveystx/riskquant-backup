# scripts/demos/e2e/_fixtures.py
"""
Shared production-input fixture for the sensitivity-based VaR E2E demos.

Builds every typed object `riskquant.solutions.risk_run.api.run_scenario_var_es`
requires, once:

    static source tables
        -> validate -> compile RiskStaticConfig
        -> proxy source table -> validate -> compile ProxyTopology
        -> positions (human-readable edge) -> ingest -> CompiledPortfolio
        -> risk-factor level snapshot (name -> rf_id resolved once, at the edge)
        -> market-data assembly -> pricing (MTM) -> trade sensitivities
        -> canonical `solutions.sensitivity_pnl` book (SensitivityPnlCalculator)
        -> the canonical run axis (`derive_run_factor_axis`: valuation
           universe, then any reporting-only FX factor) and its shock universe
           (required_factor_ids, required_shift_type_id, source_factor_ids,
           shock_scale, base_levels) via the ONE proxy resolution call every
           scenario model downstream shares
        -> construction data for each scenario-model family
           (RiskFactorMatrix for Historical/SBB/FilteredHistorical/
           FilteredBootstrap; ParametricRiskData for Parametric)
        -> ReportingSpec (the reporting currency only; the run derives every
           settlement currency and reporting FX route itself, D085)

This module owns NO scenario-model choice, NO shock application, NO
reporting-currency conversion, and NO VaR/ES computation: those are
`run_scenario_var_es`'s job. `demo_e2e_sensi_based_var.py` calls
`build_fixture()`; the certification fixtures of `tests/solutions/risk_run`
load this module as well.

Golden fixture (D019, LOCKED): the static universe, positions, proxy scale,
base levels, and 10-day history below back the independent oracle checks in
`demo_e2e_sensi_based_var.py` and `tests/e2e/test_demo_e2e_sensi_based_var.py`.
Do not change a value here without updating both together.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.proxy_config.audit import audit_proxy_topology
from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.static_config.audit import audit_risk_static_config
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.builder_matrix import (
    build_risk_factor_matrix_from_df,
)
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.proxy_resolver.proxy import (
    resolve_required_factor_ids_to_market_data_factor_ids,
)
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.risk_run.api import derive_run_factor_axis
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    RunFactorAxis,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.sensitivity_pnl.api import (
    SensitivityPnlCalculator,
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorOrder,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
)
from riskquant.universe import UixMapper
from riskquant.utils.dates import date_to_ord32
from riskquant.utils.printer import print_df, print_header
from riskquant.valuation.engine import run_valuation
from riskquant.valuation.result import ValuationResult

# =============================================================================
# MINIMAL RISK STATIC CONFIG INPUTS  (D019 LOCKED golden fixture)
# =============================================================================

_CCY_DF = pl.DataFrame({
    "ccy_name": ["EUR", "USD"],
    "ccy_id": [1, 2],
})

_HOLIDAY_CALENDARS_DF = pl.DataFrame({
    "holiday_calendar_name": [
        "NL",
        "NL",
        "US",
        "US",
    ],
    "holiday_date": [
        "2027-01-01",
        "2027-01-25",
        "2027-01-01",
        "2027-01-25",
    ],
})

_NODE_META_DF = pl.DataFrame({
    "node_id": [0, 1, 2, 3],
    "node_name": [
        "USDEUR",
        "AAPL_EQ",
        "BRENT_CRUDE",
        "WTI_CRUDE",
    ],
    "quote_ccy": [
        "EUR",
        "USD",
        "USD",
        "USD",
    ],
    "asset_class": [
        "FX",
        "EQUITY",
        "COMMODITY",
        "COMMODITY",
    ],
    "asset_subclass": [
        "FX_G10",
        "EQ_SINGLE_NAME",
        "CM_OIL",
        "CM_OIL",
    ],
    "quote_type": [
        "FX_RATE",
        "PRICE",
        "PRICE",
        "PRICE",
    ],
    "unit_scale": [
        "DECIMAL",
        "DECIMAL",
        "DECIMAL",
        "DECIMAL",
    ],
    "shift_type": [
        "LOG_RETURN",
        "LOG_RETURN",
        "LOG_RETURN",
        "LOG_RETURN",
    ],
    "fx_base_ccy": [
        "USD",
        None,
        None,
        None,
    ],
})

# The validator requires every active quote currency to have at least one
# discount curve. These are the smallest curves needed to satisfy that
# invariant while keeping the demo independent of the canonical fixture.
_CURVE_META_DF = pl.DataFrame({
    "curve_id": [250000000, 250001000],
    "curve_name": [
        "EUR_ESTR_OIS",
        "USD_SOFR_OIS",
    ],
    "quote_ccy": [
        "EUR",
        "USD",
    ],
    "asset_class": [
        "IR",
        "IR",
    ],
    "asset_subclass": [
        "IR_OIS",
        "IR_OIS",
    ],
    "quote_type": [
        "RATE",
        "RATE",
    ],
    "unit_scale": [
        "PERCENT",
        "PERCENT",
    ],
    "shift_type": [
        "DIFFERENCE",
        "DIFFERENCE",
    ],
    "x_axis_kind": [
        "LABEL",
        "LABEL",
    ],
    "x_interp_method": [
        "LINEAR",
        "LINEAR",
    ],
    "x_extrap_method": [
        "FLAT",
        "FLAT",
    ],
    "roll_method": [
        None,
        None,
    ],
    "anchor_offset": [
        None,
        None,
    ],
    "roll_month": [
        None,
        None,
    ],
    "roll_day": [
        None,
        None,
    ],
    "fh_bh_mmdd": [
        None,
        None,
    ],
    "discount_basis": [
        "OIS",
        "OIS",
    ],
    "is_default_discount_for_ccy": [
        True,
        True,
    ],
})

_CURVE_POINT_META_DF = pl.DataFrame({
    "rf_id": [1000, 2000],
    "curve_id": [250000000, 250001000],
    "tenor_token": [
        "[3M]",
        "[3M]",
    ],
})

# Surface risk factors are not needed for this demo, but the production
# static-config contract still includes both surface tables.
_SURFACE_META_DF = pl.DataFrame({
    "surface_id": pl.Series([], dtype=pl.Int64),
    "surface_name": pl.Series([], dtype=pl.String),
    "quote_ccy": pl.Series([], dtype=pl.String),
    "asset_class": pl.Series([], dtype=pl.String),
    "asset_subclass": pl.Series([], dtype=pl.String),
    "quote_type": pl.Series([], dtype=pl.String),
    "unit_scale": pl.Series([], dtype=pl.String),
    "shift_type": pl.Series([], dtype=pl.String),
    "x_axis_kind": pl.Series([], dtype=pl.String),
    "x_interp_method": pl.Series([], dtype=pl.String),
    "x_extrap_method": pl.Series([], dtype=pl.String),
    "roll_method": pl.Series([], dtype=pl.String),
    "anchor_offset": pl.Series([], dtype=pl.Int32),
    "roll_month": pl.Series([], dtype=pl.Int32),
    "roll_day": pl.Series([], dtype=pl.Int32),
    "fh_bh_mmdd": pl.Series([], dtype=pl.String),
    "y_axis_kind": pl.Series([], dtype=pl.String),
    "y_interp_method": pl.Series([], dtype=pl.String),
    "y_extrap_method": pl.Series([], dtype=pl.String),
})

_SURFACE_POINT_META_DF = pl.DataFrame({
    "rf_id": pl.Series([], dtype=pl.Int64),
    "surface_id": pl.Series([], dtype=pl.Int64),
    "x_tenor_token": pl.Series([], dtype=pl.String),
    "y_label": pl.Series([], dtype=pl.String),
})

_RISK_STATIC_CONFIG_DICT_OF_DFS: dict[str, pl.DataFrame] = {
    "ccy": _CCY_DF,
    "holiday_calendars": _HOLIDAY_CALENDARS_DF,
    "node_meta": _NODE_META_DF,
    "curve_meta": _CURVE_META_DF,
    "curve_point_meta": _CURVE_POINT_META_DF,
    "surface_meta": _SURFACE_META_DF,
    "surface_point_meta": _SURFACE_POINT_META_DF,
}


def _run_risk_static_config(*, verbose: bool = True) -> RiskStaticConfig:
    """Build and audit the minimal static risk-factor universe."""
    if verbose:
        print_header("Importing, Validating and Compiling Risk Static Config")
        for sheet_name, df in _RISK_STATIC_CONFIG_DICT_OF_DFS.items():
            print(f"Sheet: {sheet_name}  (rows={df.height})")
            print(df)
            print("---")

    rsc = compile_risk_static_config_from_dict_of_dfs(_RISK_STATIC_CONFIG_DICT_OF_DFS)

    if verbose:
        rsc.summary()
    audit_risk_static_config(rsc=rsc, n=200)

    return rsc


# =============================================================================
# MINIMAL PROXY TOPOLOGY  (D019 LOCKED: WTI_CRUDE -> BRENT_CRUDE, scale 0.95)
# =============================================================================

_PROXY_CONFIG_DF = pl.DataFrame({
    "proxying_level": [
        "RISK_FACTOR",
    ],
    "required_name": [
        "WTI_CRUDE",
    ],
    "proxy_name": [
        "BRENT_CRUDE",
    ],
    "vol_scaling_factor": [
        0.95,
    ],
})


def _run_proxy_topology(
    rsc: RiskStaticConfig, *, verbose: bool = True
) -> ProxyTopology:
    """Build and audit the minimal risk-proxy topology."""
    if verbose:
        print_header("Importing, Validating and Compiling Proxy Topology")
        print("Proxy configuration:")
        print(_PROXY_CONFIG_DF)
        print("---")

    rpt = compile_proxy_topology_from_df(
        df=_PROXY_CONFIG_DF,
        static_config=rsc,
    )

    if verbose:
        rpt.summary()
    audit_proxy_topology(topology=rpt, static_config=rsc, n=200)

    return rpt


# =============================================================================
# MINIMAL POSITIONS -> PORTFOLIO  (D019 LOCKED: 3 trades, multi-currency)
# =============================================================================

# Human-readable edge representation: three SPOT positions. Names resolve to
# rf_ids inside the ingest pipeline (`compile_raw_df_into_ingest_batch`);
# nothing downstream of that call touches a string again. AAPL and WTI settle
# in USD; Brent settles in EUR -- the multi-currency property the reporting-
# currency conversion step below exists to demonstrate.
_POSITIONS_DF = pd.DataFrame({
    "uid": [1_010, 1_020, 1_030],
    "instrument": ["SPOT", "SPOT", "SPOT"],
    "quantity": [100.0, -50.0, 75.0],
    "underlying": ["AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE"],
    "settlement_ccy": ["USD", "EUR", "USD"],
})


def _settlement_ccy_by_uid() -> dict[int, str]:
    return dict(zip(_POSITIONS_DF["uid"], _POSITIONS_DF["settlement_ccy"], strict=True))


def _underlying_by_uid() -> dict[int, str]:
    return dict(zip(_POSITIONS_DF["uid"], _POSITIONS_DF["underlying"], strict=True))


def _run_portfolio(
    rsc: RiskStaticConfig, *, verbose: bool = True
) -> tuple[CompiledPortfolio, UixMapper, IngestBatch]:
    """Ingest the minimal positions frame and compile it into SoA blocks.

    positions (pandas, human-readable)
        -> compile_raw_df_into_ingest_batch  : validated, rf_id-resolved SoA
        -> build_uix_mapper                  : global rf_id -> block-local uix
        -> compile_portfolio_from_ingest_batch : per-instrument SoA blocks

    Only SPOT is registered in `portfolio.compilers.registry.COMPILERS`
    today; forward/option compilers are not yet authored, so the portfolio
    here is intentionally spot-only.

    Returns the `UixMapper` alongside the portfolio: it is the same global
    ID <-> uix translation table `market_data.assemble.build_universe_mapping`
    (pricing) and the sensitivity derivation below both key off, so it is
    built once here and threaded through rather than rebuilt.
    """
    if verbose:
        print_header("Ingesting and Compiling Positions into a Portfolio")
        print_df(_POSITIONS_DF, title="Positions (edge representation)")

    ingest_batch = compile_raw_df_into_ingest_batch(
        raw_df=_POSITIONS_DF,
        static_config=rsc,
    )
    uix_mapper = build_uix_mapper(
        [ingest_batch], as_of_ord32=int(date_to_ord32(_SNAPSHOT_AS_OF_DATE))
    )
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=ingest_batch,
        uix_mapper=uix_mapper,
    )

    if verbose:
        portfolio.summary()
        portfolio.print_block_details()

    return portfolio, uix_mapper, ingest_batch


# =============================================================================
# MINIMAL RISK-FACTOR LEVEL SNAPSHOT  (D019 LOCKED base levels)
# =============================================================================

# Deterministic as-of levels for every risk factor referenced above,
# including WTI_CRUDE (the proxy target's own level, independent of the
# proxy topology -- proxying only rewires SHOCKS, never spot levels).
_RISK_FACTOR_LEVELS: dict[str, float] = {
    "USDEUR": 1.0800,
    "AAPL_EQ": 200.00,
    "BRENT_CRUDE": 82.30,
    "WTI_CRUDE": 70.00,
}

_SNAPSHOT_AS_OF_DATE = date(2027, 1, 4)


def _run_risk_factor_snapshot(
    rsc: RiskStaticConfig, *, verbose: bool = True
) -> RiskFactorLevelSnapshot:
    """Build a deterministic risk-factor level snapshot, resolved to rf_id."""
    if verbose:
        print_header("Building Risk-Factor Level Snapshot")
        print("Risk factor levels (edge representation):")
        for name, level in _RISK_FACTOR_LEVELS.items():
            print(f"  {name:<12} {level}")
        print("---")

    snapshot = build_risk_factor_level_snapshot_from_dict(
        as_of_date=_SNAPSHOT_AS_OF_DATE,
        levels_dict=_RISK_FACTOR_LEVELS,
        static_config=rsc,
    )

    if verbose:
        snapshot.summary()

    return snapshot


# =============================================================================
# PRICING
# =============================================================================


def _run_market_data(
    *,
    rsc: RiskStaticConfig,
    uix_mapper: UixMapper,
    snapshot: RiskFactorLevelSnapshot,
) -> MarketDataMatrix:
    """Assemble the raw, universe-aligned market grids from the snapshot.

    One scenario (the deterministic base snapshot itself) -- this is a t0
    state, not a simulation batch. Simulation batches are produced later by
    whichever `ScenarioGenerator` `run_scenario_var_es` builds internally.
    """
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=uix_mapper
    )
    return build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=_SNAPSHOT_AS_OF_DATE,
        uix_mapper=uix_mapper,
        universe_map=universe_map,
        snapshot=snapshot,
        num_scenarios=1,
    )


def _run_pricing(
    *,
    portfolio: CompiledPortfolio,
    uix_mapper: UixMapper,
    market_data: MarketDataMatrix,
    verbose: bool = True,
) -> ValuationResult:
    """Price the compiled portfolio against the assembled market data.

    No discounting is required here: `SpotBlock`'s own kernel is
    `PV = qty * spot_price * fx_effective`, which reads no discount factor.
    """
    if verbose:
        print_header("Pricing the Compiled Portfolio")

    valuation_state = assemble_valuation_state(
        matrix=market_data, uix_mapper=uix_mapper
    )

    out = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=valuation_state, out=out)

    result = ValuationResult(trade_uids=portfolio.trade_uids, values=out)

    if verbose:
        settle_ccy_by_uid = _settlement_ccy_by_uid()
        print("Trade-level MTM (per-trade settlement currency; not summed):")
        for uid, pv in zip(
            result.trade_uids.tolist(), result.values[:, 0].tolist(), strict=True
        ):
            print(f"  uid={uid:<6} PV={pv:>14,.2f} {settle_ccy_by_uid[uid]}")
        print("---")

    return result


# =============================================================================
# TRADE SENSITIVITIES -> COMPILED BOOK
# =============================================================================


def _run_sensitivities(
    *,
    rsc: RiskStaticConfig,
    portfolio: CompiledPortfolio,
    uix_mapper: UixMapper,
    market_data: MarketDataMatrix,
    verbose: bool = True,
) -> SensitivityPnlCalculator:
    """Compute trade sensitivities and compile them into the canonical book.

    Pure orchestration -- all sensitivity mathematics lives in the production
    layer: `run_sensitivities` dispatches per block, `build_sensitivity_book_spec`
    is the edge adapter, `SensitivityPnlCalculator.compile` is the real
    production book compiler. This module contributes no mathematics of its own.
    """
    if verbose:
        print_header("Deriving Trade Sensitivities")

    sensitivities = run_sensitivities(
        portfolio=portfolio, uix_mapper=uix_mapper, market_data=market_data
    )

    if verbose:
        rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
        display = pl.DataFrame({
            "position_id": sensitivities.position_uids,
            "factor_id": sensitivities.factor_rf_ids,
            "first_order": sensitivities.first_order,
            "second_order": sensitivities.second_order,
        }).with_columns(
            pl.col("factor_id").replace_strict(rf_id_to_name).alias("factor_name")
        )
        print_df(
            display.select(
                "position_id", "factor_id", "factor_name", "first_order", "second_order"
            ).to_pandas(),
            title="Trade sensitivities (first/second order)",
        )

    # standalone sensitivity report: the book's own factors, declared explicitly
    book_spec = build_sensitivity_book_spec(
        sensitivities=sensitivities,
        required_factor_ids=sensitivities.active_factor_rf_ids,
    )
    calculator = SensitivityPnlCalculator.compile(book_spec)

    if verbose:
        rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
        active_names = ", ".join(
            rf_id_to_name[int(rf)] for rf in calculator.book.active_factor_ids
        )
        print(
            f"Book compiled: {calculator.book.position_uids.shape[0]} position(s), "
            f"{calculator.book.active_factor_ids.shape[0]} active factor(s) "
            f"({active_names})."
        )
        print("---")

    return calculator


# =============================================================================
# SHOCK-UNIVERSE AXIS  (the exact inputs run_scenario_var_es requires)
# =============================================================================

# Source factors that actually carry market data. WTI_CRUDE is deliberately
# absent: it has no history of its own and is proxied from BRENT_CRUDE by
# the compiled ProxyTopology (resolved once, below, for every scenario model).
_SCENARIO_SOURCE_NAMES: tuple[str, ...] = ("USDEUR", "AAPL_EQ", "BRENT_CRUDE")


@dataclass(frozen=True, slots=True, kw_only=True)
class ShockUniverseAxis:
    """The canonical run axis, on the same order for every array below.

    The first `num_valuation_factors` rows are the valuation universe (also
    the Taylor axis); any further rows are reporting-only FX factors (D085).
    """

    num_valuation_factors: int
    required_factor_ids: npt.NDArray[np.int32]
    required_shift_type_id: npt.NDArray[np.int8]
    source_factor_ids: npt.NDArray[np.int32]
    shock_scale: npt.NDArray[np.float32]
    base_levels: npt.NDArray[np.float64]
    resolution_code: npt.NDArray[np.int8]


def _run_shock_universe(
    *,
    rsc: RiskStaticConfig,
    proxy_topology: ProxyTopology,
    run_axis: RunFactorAxis,
    snapshot: RiskFactorLevelSnapshot,
    verbose: bool = True,
) -> ShockUniverseAxis:
    """Bind the run's scenario data to the canonical run axis.

    Composes the existing public production path -- this module adds no
    scenario machinery of its own:

        derive_run_factor_axis        : the run-owned axis (valuation
                                         universe, then reporting-only FX)
                                         + each factor's declared shift_type;
                                         computed by the caller and passed in
        extract_levels_from_snapshot  : t0 BASE levels for those factors
        resolve_required_factor_ids_.. : required -> (source, shock_scale),
                                         i.e. the compiled ProxyTopology
                                         applied to a concrete market-data
                                         universe

    This is the ONE place proxy resolution happens: every scenario model
    downstream (the validation-only deterministic case in the canonical
    demo, or any production `ScenarioModelConfig` in either demo) reads this
    SAME axis, so WTI_CRUDE is proxied to BRENT_CRUDE exactly once here, not
    re-derived per model. Proxying changes SHOCK PROVENANCE, never a base
    level: WTI_CRUDE keeps its own base level (70.00) and its own identity
    in the sensitivity book; only the shock driving it is sourced from
    BRENT_CRUDE and scaled by 0.95.
    """
    # The run axis is a read-only descriptor; the scenario data bound to it is
    # this fixture's own execution arrays (the scenario building blocks below
    # and in the demos take writable arrays).
    required_factor_ids = np.array(run_axis.factor_ids, copy=True)

    base_levels = extract_levels_from_snapshot(
        required_ids=required_factor_ids, snapshot=snapshot
    )

    available_factor_ids = np.ascontiguousarray(
        [rsc.rel_name_to_rf_id[name] for name in _SCENARIO_SOURCE_NAMES],
        dtype=np.int32,
    )

    proxy_resolution = resolve_required_factor_ids_to_market_data_factor_ids(
        static_config=rsc,
        proxy_topology=proxy_topology,
        required_factor_ids=required_factor_ids,
        market_data_factor_ids=available_factor_ids,
    )

    if verbose:
        rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
        print_header("Shock-Universe Axis (proxy resolution)")
        print_df(
            pd.DataFrame({
                "required_factor": [
                    rf_id_to_name[int(rf)] for rf in required_factor_ids
                ],
                "base_level": base_levels,
                "source_factor": [
                    rf_id_to_name[int(rf)] for rf in proxy_resolution.source_factor_ids
                ],
                "shock_scale": proxy_resolution.shock_scale,
            }),
            title="Proxy resolution (exposure identity vs shock provenance)",
            float_format="_.4f",
        )
        print(
            "WTI_CRUDE keeps its OWN base level and its OWN sensitivity identity.\n"
            "Only its shock provenance changes: it reads BRENT_CRUDE's shock,\n"
            "scaled by 0.95, inside the exponent. This axis is resolved ONCE and\n"
            "shared by every scenario model demonstrated downstream."
        )
        print("---")

    return ShockUniverseAxis(
        num_valuation_factors=run_axis.num_valuation_factors,
        required_factor_ids=required_factor_ids,
        required_shift_type_id=np.array(run_axis.shift_type_ids, copy=True),
        source_factor_ids=proxy_resolution.source_factor_ids,
        shock_scale=proxy_resolution.shock_scale,
        base_levels=base_levels,
        resolution_code=proxy_resolution.resolution_code,
    )


# =============================================================================
# HISTORICAL OBSERVATIONS
# (construction data for Historical / SBB / FilteredHistorical / FilteredBootstrap)
# =============================================================================

# Tiny hand-authored history over the SOURCE factors only. WTI_CRUDE has NO
# history column by design: it is not observed here, and the ProxyTopology
# sources its shock from BRENT_CRUDE. It is never given synthetic history and
# then overwritten.
_HISTORY_DATES: tuple[str, ...] = (
    "2026-12-01",
    "2026-12-02",
    "2026-12-03",
    "2026-12-04",
    "2026-12-07",
    "2026-12-08",
    "2026-12-09",
    "2026-12-10",
    "2026-12-11",
    "2026-12-14",
)

# Column order matches `_SCENARIO_SOURCE_NAMES`.
_HISTORY_LEVELS: tuple[tuple[float, ...], ...] = (
    #  USDEUR   AAPL_EQ  BRENT_CRUDE
    (1.0700, 196.00, 80.10),
    (1.0750, 198.50, 81.40),
    (1.0720, 197.20, 80.90),
    (1.0810, 201.30, 83.20),
    (1.0780, 200.10, 82.60),
    (1.0835, 203.40, 84.10),
    (1.0795, 199.80, 82.90),
    (1.0760, 198.20, 81.70),
    (1.0825, 202.60, 83.60),
    (1.0800, 200.00, 82.30),
)


def _build_historical_matrix(rsc: RiskStaticConfig) -> RiskFactorMatrix:
    """Turn the hand-authored history into the canonical observations container.

    `build_risk_factor_matrix_from_df` is the canonical EDGE builder: a wide
    pandas frame (dates on the index, one column per factor NAME) plus a
    name -> rf_id map. Names die here; everything downstream is int32 factor
    ids and an F-contiguous float64 `(T, N)` values block -- the exact `data`
    object `run_scenario_var_es` accepts for every historical-family model.
    """
    frame = pd.DataFrame(
        np.asarray(_HISTORY_LEVELS, dtype=np.float64),
        index=pd.to_datetime(list(_HISTORY_DATES)),
        columns=list(_SCENARIO_SOURCE_NAMES),
    )
    return build_risk_factor_matrix_from_df(df=frame, static_config=rsc)


# =============================================================================
# PARAMETRIC CONSTRUCTION DATA  (for the Parametric model)
# =============================================================================


def _build_parametric_data(rsc: RiskStaticConfig) -> ParametricRiskData:
    """Illustrative Cholesky/vol construction data for the Parametric model.

    Vols and correlation are the SAMPLE statistics of the same tiny
    hand-authored history's daily log returns, so the parametric model's
    covariance structure is not an arbitrary invention -- it is the same
    9-observation sample the historical-family models above read directly.
    `cholesky_matrix` is the Cholesky factor of the CORRELATION matrix
    (`ParametricGenerator` scales its correlated unit-normal draws by
    `volatilities` afterward, it does not take a covariance Cholesky).

    This is construction DATA, not a scenario: `ParametricGenerator` draws
    its own seeded Monte-Carlo paths from it, exactly as `HistoricalGenerator`
    draws its rolling windows from `_build_historical_matrix`.
    """
    levels = np.asarray(_HISTORY_LEVELS, dtype=np.float64)
    log_returns = np.diff(np.log(levels), axis=0)  # (9, 3)

    vols = np.std(log_returns, axis=0, ddof=1)
    corr = np.corrcoef(log_returns, rowvar=False)
    cholesky = np.linalg.cholesky(corr)

    source_ids = np.ascontiguousarray(
        sorted(int(rsc.rel_name_to_rf_id[name]) for name in _SCENARIO_SOURCE_NAMES),
        dtype=np.int32,
    )
    # Re-order vols/cholesky rows+cols to match the SORTED id axis
    # (ParametricRiskData requires factor_ids sorted ascending).
    name_by_id = {int(rsc.rel_name_to_rf_id[n]): n for n in _SCENARIO_SOURCE_NAMES}
    order = [_SCENARIO_SOURCE_NAMES.index(name_by_id[i]) for i in source_ids.tolist()]
    vols = np.ascontiguousarray(vols[order], dtype=np.float64)
    cholesky = np.ascontiguousarray(cholesky[np.ix_(order, order)], dtype=np.float64)

    return ParametricRiskData(
        factor_ids=source_ids,
        cholesky_matrix=cholesky,
        volatilities=vols,
    )


# =============================================================================
# REPORTING SPEC
# =============================================================================

_REPORTING_CCY: str = "USD"


def _certify_sensitivities(
    *,
    rsc: RiskStaticConfig,
    portfolio: CompiledPortfolio,
    uix_mapper: UixMapper,
    market_data: MarketDataMatrix,
    pricing: ValuationResult,
    required_factor_ids: npt.NDArray[np.int32],
    base_levels: npt.NDArray[np.float64],
) -> SensitivityPnlCalculator:
    """
    P1-A: the production book is a CERTIFIED TAYLOR/DERIVED representation.

    Same sensitizer output as `_run_sensitivities`, declared SECOND_ORDER_FULL
    over the sensitizer's structural F(p), and bound to the exact base it was
    produced at: as-of, the base levels of the run axis's valuation prefix
    (the Taylor axis), discounting mode, the static-config content digest,
    the valuation-contract fingerprint and the V_0 of the same pricing pass.
    """
    sensitivities = run_sensitivities(
        portfolio=portfolio, market_data=market_data, uix_mapper=uix_mapper
    )
    pv_by_uid = dict(
        zip(pricing.trade_uids.tolist(), pricing.values[:, 0].tolist(), strict=True)
    )
    book_uids = np.unique(np.asarray(sensitivities.position_uids, dtype=np.int64))
    v0 = np.array([pv_by_uid[int(u)] for u in book_uids.tolist()], dtype=np.float64)
    binding = TaylorBaseBinding(
        as_of_ord32=int(date_to_ord32(_SNAPSHOT_AS_OF_DATE)),
        base_levels_digest=base_levels_digest(
            factor_ids=required_factor_ids, base_levels=base_levels
        ),
        discounting_mode=DiscountingMode.NONE.value,
        static_config_digest=static_config_content_digest(rsc),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=book_uids, values=v0
        ),
    )
    # the production DERIVED book: built on the run axis's valuation prefix
    # (F-1, D082 as amended); reporting FX is never a Taylor factor
    spec = build_sensitivity_book_spec(
        sensitivities=sensitivities, required_factor_ids=required_factor_ids
    )
    declaration = derive_taylor_declaration(
        sensitivities=sensitivities,
        order=TaylorOrder.SECOND_ORDER_FULL,
        base_binding=binding,
    )
    return SensitivityPnlCalculator.compile(
        dataclasses.replace(spec, taylor_declaration=declaration)
    )


def _build_derived_block(*, calculator: SensitivityPnlCalculator) -> TaylorBlock:
    """TAYLOR/DERIVED block: the certified book only. The run values V_0 from
    its own `ValuationContext` and verifies the book's binding against it
    (P1-B)."""
    return TaylorBlock(calculator=calculator)


def _build_valuation_context(
    *, rsc: RiskStaticConfig, ingest_batch: IngestBatch, uix_mapper: UixMapper
) -> ValuationContext:
    """The run-owned base valuation context (P1-B, N1): the static
    configuration, the canonical ingest batch (the run compiles it), the
    mapper (as-of) and the discounting mode."""
    return ValuationContext(
        static_config=rsc,
        ingest_batch=ingest_batch,
        uix_mapper=uix_mapper,
        discounting_mode=DiscountingMode.NONE,
    )


def _build_methodology(calculator: SensitivityPnlCalculator) -> MethodologyDeclaration:
    """D019 book: every position TAYLOR/DERIVED; U stated explicitly (D081 §11)."""
    uids = np.asarray(calculator.book.position_uids, dtype=np.int64)
    return MethodologyDeclaration(
        production_position_uids=uids,
        full_reval_uids=np.empty(0, dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=uids,
    )


# =============================================================================
# FIXTURE BUNDLE
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class DemoFixture:
    """Every typed input `run_scenario_var_es` needs, built once and shared
    by `demo_e2e_sensi_based_var.py` and the `tests/solutions/risk_run` fixtures."""

    rsc: RiskStaticConfig
    proxy_topology: ProxyTopology
    portfolio: CompiledPortfolio
    uix_mapper: UixMapper
    snapshot: RiskFactorLevelSnapshot
    market_data: MarketDataMatrix
    pricing: ValuationResult
    calculator: SensitivityPnlCalculator
    shock_universe: ShockUniverseAxis
    historical_matrix: RiskFactorMatrix
    parametric_data: ParametricRiskData
    reporting: ReportingSpec
    methodology: MethodologyDeclaration
    taylor_derived: TaylorBlock
    valuation_context: ValuationContext


def build_fixture(*, verbose: bool = True) -> DemoFixture:
    """Build every typed input the production scenario-VaR entry point needs.

    Mirrors the required end-state pipeline exactly:

        human-readable inputs -> validated static configuration
            -> compiled portfolio -> market state -> valuation / sensitivities
            -> canonical compiled sensitivity book -> shock-universe axis
            -> per-model construction data -> ReportingSpec

    Everything after this point (scenario model choice, shock application,
    reporting-currency conversion, VaR/ES, RunManifest/RiskRun) is
    `run_scenario_var_es`'s job, not this module's.
    """
    rsc = _run_risk_static_config(verbose=verbose)
    proxy_topology = _run_proxy_topology(rsc, verbose=verbose)
    portfolio, uix_mapper, ingest_batch = _run_portfolio(rsc, verbose=verbose)
    snapshot = _run_risk_factor_snapshot(rsc, verbose=verbose)
    market_data = _run_market_data(rsc=rsc, uix_mapper=uix_mapper, snapshot=snapshot)
    pricing = _run_pricing(
        portfolio=portfolio,
        uix_mapper=uix_mapper,
        market_data=market_data,
        verbose=verbose,
    )
    calculator = _run_sensitivities(
        rsc=rsc,
        portfolio=portfolio,
        uix_mapper=uix_mapper,
        market_data=market_data,
        verbose=verbose,
    )
    valuation_context = _build_valuation_context(
        rsc=rsc, ingest_batch=ingest_batch, uix_mapper=uix_mapper
    )
    shock_universe = _run_shock_universe(
        rsc=rsc,
        proxy_topology=proxy_topology,
        run_axis=derive_run_factor_axis(
            valuation_context=valuation_context, reporting_ccy=_REPORTING_CCY
        ),
        snapshot=snapshot,
        verbose=verbose,
    )
    k_val = shock_universe.num_valuation_factors
    # P1-A: the risk run consumes a CERTIFIED TAYLOR/DERIVED book bound to the
    # shock universe's base levels and the pricing pass's V_0.
    calculator = _certify_sensitivities(
        rsc=rsc,
        portfolio=portfolio,
        uix_mapper=uix_mapper,
        market_data=market_data,
        pricing=pricing,
        required_factor_ids=shock_universe.required_factor_ids[:k_val],
        base_levels=shock_universe.base_levels[:k_val],
    )
    historical_matrix = _build_historical_matrix(rsc)
    parametric_data = _build_parametric_data(rsc)
    # D085: the caller chooses only the reporting currency.
    reporting = ReportingSpec(reporting_ccy=_REPORTING_CCY)

    return DemoFixture(
        rsc=rsc,
        proxy_topology=proxy_topology,
        portfolio=portfolio,
        uix_mapper=uix_mapper,
        snapshot=snapshot,
        market_data=market_data,
        pricing=pricing,
        calculator=calculator,
        shock_universe=shock_universe,
        historical_matrix=historical_matrix,
        parametric_data=parametric_data,
        reporting=reporting,
        methodology=_build_methodology(calculator),
        taylor_derived=_build_derived_block(calculator=calculator),
        valuation_context=valuation_context,
    )


# =============================================================================
# FIXTURE-LEVEL GOLDEN CHECKS  (pricing + sensitivity identities)
# =============================================================================


def run_fixture_golden_checks(
    *,
    rsc: RiskStaticConfig,
    pricing: ValuationResult,
    calculator: SensitivityPnlCalculator,
    verbose: bool = True,
) -> bool:
    """Deterministic, hand-verifiable checks for the minimal spot universe.

    Independent of any scenario model: validates the PRICING and SENSITIVITY
    COMPILATION steps directly against hand-derived economic identities, by
    reading `calculator.book` fields directly rather than calling
    `calculator.run(...)`. Given quantity=-50, BRENT_CRUDE=82.30 USD,
    USDEUR=1.0800:

        Brent MTM  = -4444.20 EUR
        dV/dBrent  =   -54.00 EUR/USD
        dV/dUSDEUR = -4115.00 EUR per unit FX
    """
    name_to_rf = rsc.rel_name_to_rf_id
    aapl_rf, brent_rf, usdeur_rf, wti_rf = (
        int(name_to_rf["AAPL_EQ"]),
        int(name_to_rf["BRENT_CRUDE"]),
        int(name_to_rf["USDEUR"]),
        int(name_to_rf["WTI_CRUDE"]),
    )

    pv_by_uid = dict(
        zip(pricing.trade_uids.tolist(), pricing.values[:, 0].tolist(), strict=True)
    )
    book = calculator.book

    def first_order(position_id: int, factor_id: int) -> float:
        m = int(np.searchsorted(book.position_uids, position_id))
        start, end = book.position_offsets[m], book.position_offsets[m + 1]
        row_factor_uix = book.factor_uix[start:end]
        row_first_order = book.first_order[start:end]
        for local, active_uix in enumerate(row_factor_uix):
            if int(book.active_factor_ids[active_uix]) == factor_id:
                return float(row_first_order[local])
        raise KeyError(
            f"No sensitivity row for position={position_id}, factor={factor_id}"
        )

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, actual: float, expected: float, *, tol: float = 1e-2) -> None:
        # tol is the D019 golden-fixture display tolerance (cents). Snapshot
        # levels are canonical float64 (D069), so the computed values agree
        # with the hand-derived identities far inside it.
        passed = abs(actual - expected) <= tol
        checks.append((name, passed, f"actual={actual:.6f} expected={expected:.6f}"))

    check("AAPL MTM = qty * price", pv_by_uid[1_010], 100.0 * 200.00)
    check("WTI MTM = qty * price", pv_by_uid[1_030], 75.0 * 70.00)
    check("Brent MTM (quanto)", pv_by_uid[1_020], -50.0 * 82.30 * 1.0800)

    check("dV/dAAPL_EQ = qty (no FX leg)", first_order(1_010, aapl_rf), 100.0)
    check("dV/dWTI_CRUDE = qty (no FX leg)", first_order(1_030, wti_rf), 75.0)
    check("dV/dBRENT_CRUDE = qty * USDEUR", first_order(1_020, brent_rf), -54.00)
    check("dV/dUSDEUR = qty * Brent_USD", first_order(1_020, usdeur_rf), -4115.00)

    active_names = {int(rf) for rf in book.active_factor_ids}
    check(
        "no unexpected risk factors in the book",
        float(len(active_names - {aapl_rf, brent_rf, usdeur_rf, wti_rf})),
        0.0,
    )

    if verbose:
        print_header("Golden Checks (pricing + sensitivities)")
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        if verbose:
            print(f"  {status:<4} | {name:<70} | {detail}")
    if verbose:
        print("---")
        print(f"{len(checks)} pricing/sensitivity checks run.")
        print()

    return all_passed
