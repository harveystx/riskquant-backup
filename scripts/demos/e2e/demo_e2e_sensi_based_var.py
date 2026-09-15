# scripts/demos/e2e/demo_e2e_sensi_based_var.py
"""
Minimal end-to-end demo for sensitivity-based Value at Risk (VaR).

This intentionally uses a tiny static universe rather than the canonical
static-config fixture:

    - 2 currencies
    - 4 scalar risk factors
        * 1 FX rate
        * 1 equity price
        * 2 commodity prices
    - 2 minimal discount curves, one for each active currency

The fourth risk factor, WTI_CRUDE, deliberately has insufficient historical
data and is therefore configured to proxy to BRENT_CRUDE.

The purpose of this script is to demonstrate the E2E wiring:

    static source tables
        -> validate
        -> compile RiskStaticConfig
        -> proxy source table
        -> validate against RiskStaticConfig
        -> compile ProxyTopology
        -> inspect / audit
        -> positions (human-readable edge)
        -> ingest -> validated SoA IngestBatch
        -> compile -> CompiledPortfolio (SoA blocks, e.g. SpotBlock)
        -> risk-factor level snapshot (name -> rf_id resolved once, at the edge)
        -> market-data assembly (snapshot -> MarketDataMatrix -> ValuationState)
        -> pricing (run_valuation -> per-trade MTM)
        -> trade sensitivities (closed-form, in the canonical
           `solutions.sensitivity_pnl` book contract)
        -> feed downstream historical/simulated shocks + sensitivity P&L + VaR/ES

This pass wires positions through to a compiled portfolio, prices it, and
derives trade-level sensitivities in the canonical sensitivity-P&L book
shape. It stops there: historical/simulated shocks, proxy-adjusted shocks,
sensitivity P&L, and VaR/ES are out of scope for this increment and are not
touched.
"""

import math
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from riskquant.config.proxy_config.audit import audit_proxy_topology
from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.proxy_config.proxy_topology import ProxyTopology
from riskquant.config.static_config.audit import audit_risk_static_config
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import INVALID_UIX
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
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.proxy_resolver.proxy import (
    RES_CONTAINER_PROXY,
    RES_DIRECT,
    RES_ID_PROXY,
    RES_NOT_SIMULATED,
    RES_TENOR_FALLBACK,
    resolve_required_factor_ids_to_market_data_factor_ids,
)
from riskquant.reporting.fx_conversion import (
    ReportingCurrencyPnl,
    build_fx_conversion_plan,
    convert_scenario_pnl_to_reporting_currency,
)
from riskquant.scenario_metrics._result import VarEsResult
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.sensitivities.book_adapter import build_sensitivity_book_spec
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.shocks.apply import (
    apply_shocks_to_base_levels,
    validate_apply_shocks_buffers,
)
from riskquant.solutions.sensitivity_pnl.api import (
    FactorAxis,
    SensitivityPnlCalculator,
    SensitivityPnlResult,
)
from riskquant.universe import UixMapper
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_header,
    print_title,
)
from riskquant.valuation.engine import run_valuation
from riskquant.valuation.result import ValuationResult

# =============================================================================
# MINIMAL RISK STATIC CONFIG INPUTS
# =============================================================================

_CCY_DF = pl.DataFrame(
    {
        "ccy_name": ["EUR", "USD"],
        "ccy_id": [1, 2],
    }
)

_HOLIDAY_CALENDARS_DF = pl.DataFrame(
    {
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
    }
)

_NODE_META_DF = pl.DataFrame(
    {
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
    }
)

# The validator requires every active quote currency to have at least one
# discount curve. These are the smallest curves needed to satisfy that
# invariant while keeping the demo independent of the canonical fixture.
_CURVE_META_DF = pl.DataFrame(
    {
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
    }
)

_CURVE_POINT_META_DF = pl.DataFrame(
    {
        "rf_id": [1000, 2000],
        "curve_id": [250000000, 250001000],
        "tenor_token": [
            "[3M]",
            "[3M]",
        ],
    }
)

# Surface risk factors are not needed for this demo, but the production
# static-config contract still includes both surface tables.
_SURFACE_META_DF = pl.DataFrame(
    {
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
    }
)

_SURFACE_POINT_META_DF = pl.DataFrame(
    {
        "rf_id": pl.Series([], dtype=pl.Int64),
        "surface_id": pl.Series([], dtype=pl.Int64),
        "x_tenor_token": pl.Series([], dtype=pl.String),
        "y_label": pl.Series([], dtype=pl.String),
    }
)

_RISK_STATIC_CONFIG_DICT_OF_DFS: dict[str, pl.DataFrame] = {
    "ccy": _CCY_DF,
    "holiday_calendars": _HOLIDAY_CALENDARS_DF,
    "node_meta": _NODE_META_DF,
    "curve_meta": _CURVE_META_DF,
    "curve_point_meta": _CURVE_POINT_META_DF,
    "surface_meta": _SURFACE_META_DF,
    "surface_point_meta": _SURFACE_POINT_META_DF,
}


def _run_risk_static_config() -> RiskStaticConfig:
    """Build and audit the minimal static risk-factor universe."""
    print_header("Importing, Validating and Compiling Risk Static Config")

    for sheet_name, df in _RISK_STATIC_CONFIG_DICT_OF_DFS.items():
        print(f"Sheet: {sheet_name}  (rows={df.height})")
        print(df)
        print("---")

    rsc = compile_risk_static_config_from_dict_of_dfs(_RISK_STATIC_CONFIG_DICT_OF_DFS)

    rsc.summary()

    audit_risk_static_config(rsc=rsc, n=200)

    return rsc


# =============================================================================
# MINIMAL PROXY TOPOLOGY
# =============================================================================

_PROXY_CONFIG_DF = pl.DataFrame(
    {
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
    }
)


def _run_proxy_topology(rsc: RiskStaticConfig) -> ProxyTopology:
    """Build and audit the minimal risk-proxy topology."""
    print_header("Importing, Validating and Compiling Proxy Topology")

    print("Proxy configuration:")
    print(_PROXY_CONFIG_DF)
    print("---")

    rpt = compile_proxy_topology_from_df(
        df=_PROXY_CONFIG_DF,
        static_config=rsc,
    )

    rpt.summary()

    audit_proxy_topology(
        topology=rpt,
        static_config=rsc,
        n=200,
    )

    return rpt


# =============================================================================
# MINIMAL POSITIONS -> PORTFOLIO
# =============================================================================

# Human-readable edge representation: two SPOT positions. Names resolve to
# rf_ids inside the ingest pipeline (`compile_raw_df_into_ingest_batch`);
# nothing downstream of that call touches a string again.
_POSITIONS_DF = pd.DataFrame(
    {
        "uid": [1_010, 1_020, 1_030],
        "instrument": ["SPOT", "SPOT", "SPOT"],
        "quantity": [100.0, -50.0, 75.0],
        "underlying": ["AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE"],
        "settlement_ccy": ["USD", "EUR", "USD"],
    }
)


def _run_portfolio(rsc: RiskStaticConfig) -> tuple[CompiledPortfolio, UixMapper]:
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
    print_header("Ingesting and Compiling Positions into a Portfolio")

    print_df(_POSITIONS_DF, title="Positions (edge representation)")

    ingest_batch = compile_raw_df_into_ingest_batch(
        raw_df=_POSITIONS_DF,
        static_config=rsc,
    )
    uix_mapper = build_uix_mapper([ingest_batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=ingest_batch,
        uix_mapper=uix_mapper,
    )

    portfolio.summary()
    portfolio.print_block_details()

    return portfolio, uix_mapper


# =============================================================================
# MINIMAL RISK-FACTOR LEVEL SNAPSHOT
# =============================================================================

# Deterministic as-of levels for every risk factor referenced above,
# including WTI_CRUDE (the proxy target's own level, independent of the
# proxy topology -- proxying only rewires SHOCKS, never spot levels).
#
# BRENT_CRUDE and USDEUR are pinned to the values used in this increment's
# hand-verifiable golden checks (see `_run_golden_checks` below); do not
# change them without updating those checks.
_RISK_FACTOR_LEVELS: dict[str, float] = {
    "USDEUR": 1.0800,
    "AAPL_EQ": 200.00,
    "BRENT_CRUDE": 82.30,
    "WTI_CRUDE": 70.00,
}

_SNAPSHOT_AS_OF_DATE = date(2027, 1, 4)


def _run_risk_factor_snapshot(rsc: RiskStaticConfig) -> RiskFactorLevelSnapshot:
    """Build a deterministic risk-factor level snapshot, resolved to rf_id.

    Names resolve once, at the edge, via `rsc.rel_name_to_rf_id` (the same
    name -> rf_id map RiskStaticConfig already exposes); the resulting
    snapshot is a compact (factor_ids: int32, factor_levels: float32) pair,
    sorted and deduplicated by construction.
    """
    print_header("Building Risk-Factor Level Snapshot")

    print("Risk factor levels (edge representation):")
    for name, level in _RISK_FACTOR_LEVELS.items():
        print(f"  {name:<12} {level}")
    print("---")

    snapshot = build_risk_factor_level_snapshot_from_dict(
        as_of_date=_SNAPSHOT_AS_OF_DATE,
        levels_dict=_RISK_FACTOR_LEVELS,
        name_to_id_map=rsc.rel_name_to_rf_id,
    )

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

    RiskFactorLevelSnapshot + RiskStaticConfig + UixMapper
        -> build_universe_mapping             : run-scoped market-data layout
                                                 plan (RiskStaticConfig SSOT +
                                                 the portfolio's own uix
                                                 universe; nothing inferred)
        -> build_market_data_matrix_from_snapshot : snapshot levels -> raw,
                                                 universe-aligned market grids

    This is the shared input to BOTH downstream branches, which is why it is
    its own step rather than an internal detail of pricing:

        MarketDataMatrix --+--> assemble_valuation_state -> run_valuation
                           |        (folded effective factors)
                           +--> run_sensitivities
                                    (raw, UNFOLDED price and FX levels)

    Pricing folds `spot_price * fx_effective` into one effective factor;
    sensitivities must keep them separate. Both legitimately read this same
    raw tier.

    One scenario (the deterministic base snapshot itself) -- this is a t0
    state, not a simulation batch.
    """
    universe_map = build_universe_mapping(static_config=rsc, uix_mapper=uix_mapper)

    return build_market_data_matrix_from_snapshot(
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
) -> ValuationResult:
    """Price the compiled portfolio against the assembled market data.

    MarketDataMatrix
        -> assemble_valuation_state : the fold -- raw grids -> the
                                       pre-multiplied EFFECTIVE factors
                                       pricing kernels gather
        -> run_valuation            : dispatches each block to its
                                       registered pricer

    No discounting is required here: `SpotBlock`'s own kernel is
    `PV = qty * spot_price * fx_effective` (see
    `valuation.pricer_kernels.spot`), which reads no discount factor. The
    two OIS curves stay in `RiskStaticConfig` only to satisfy its discount-
    curve invariant; a spot-only portfolio never references them, which is
    exactly what `uix_mapper.discount_curve_id_universe` being empty proves.
    """
    print_header("Pricing the Compiled Portfolio")

    valuation_state = assemble_valuation_state(
        matrix=market_data, uix_mapper=uix_mapper
    )

    out = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=valuation_state, out=out)

    result = ValuationResult(trade_uids=portfolio.trade_uids, values=out)

    # Settlement currency is an edge/presentation concern -- joined back from
    # the positions frame purely for display. PVs are NOT summed into one
    # portfolio total here: AAPL/WTI settle in USD and Brent settles in EUR,
    # and blindly summing mixed-currency PVs would silently misrepresent the
    # portfolio's value. A true multi-currency total needs an explicit
    # reporting-currency conversion, which is a policy choice out of scope
    # for this increment.
    settle_ccy_by_uid = dict(
        zip(_POSITIONS_DF["uid"], _POSITIONS_DF["settlement_ccy"], strict=True)
    )
    print("Trade-level MTM (per-trade settlement currency; not summed):")
    for uid, pv in zip(
        result.trade_uids.tolist(), result.values[:, 0].tolist(), strict=True
    ):
        print(f"  uid={uid:<6} PV={pv:>14,.2f} {settle_ccy_by_uid[uid]}")
    print("---")

    return result


# =============================================================================
# TRADE SENSITIVITIES
# =============================================================================


def _run_sensitivities(
    *,
    rsc: RiskStaticConfig,
    portfolio: CompiledPortfolio,
    uix_mapper: UixMapper,
    market_data: MarketDataMatrix,
) -> SensitivityPnlCalculator:
    """Compute trade sensitivities and compile them into the canonical book.

    Pure orchestration. All sensitivity mathematics lives in the production
    layer -- `sensitivities.sensitizer_kernels.spot` holds the closed-form
    partials, `sensitivities.sensitizers.spot` unpacks the compiled identity
    graph, and `sensitivities.engine` dispatches per block by
    `BLOCK_TYPE_ID`, exactly as `valuation.engine` dispatches pricers. This
    demo contributes no mathematics of its own:

        run_sensitivities          : portfolio -> BlockSensitivities (SoA,
                                      keyed by global rf_id, no Polars)
        build_sensitivity_book_spec: SoA -> validated SensitivityBookSpec
                                      (the edge adapter, where Polars lives)
        SensitivityPnlCalculator.compile : the real production book compiler

    `compile` depends only on the factor axis, not on any shock set, so
    calling it here is a genuine checkpoint -- it proves the derived book is
    well-formed in the canonical shape -- without pricing any P&L, which
    needs a shock matrix this increment does not build.
    """
    print_header("Deriving Trade Sensitivities")

    sensitivities = run_sensitivities(
        portfolio=portfolio,
        uix_mapper=uix_mapper,
        market_data=market_data,
    )

    # Names are resolved ONLY here, for display. The numerical core above
    # never sees a risk-factor name.
    rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}

    display = pl.DataFrame(
        {
            "position_id": sensitivities.position_uids,
            "factor_id": sensitivities.factor_rf_ids,
            "first_order": sensitivities.first_order,
            "second_order": sensitivities.second_order,
        }
    ).with_columns(
        pl.col("factor_id").replace_strict(rf_id_to_name).alias("factor_name")
    )
    print_df(
        display.select(
            "position_id", "factor_id", "factor_name", "first_order", "second_order"
        ).to_pandas(),
        title="Trade sensitivities (first/second order)",
    )

    if sensitivities.num_cross_entries:
        cross_display = pl.DataFrame(
            {
                "position_id": sensitivities.cross_position_uids,
                "factor_id_i": sensitivities.cross_factor_rf_ids_i,
                "factor_id_j": sensitivities.cross_factor_rf_ids_j,
                "cross_second_order": sensitivities.cross_second_order,
            }
        ).with_columns(
            pl.col("factor_id_i").replace_strict(rf_id_to_name).alias("factor_name_i"),
            pl.col("factor_id_j").replace_strict(rf_id_to_name).alias("factor_name_j"),
        )
        print_df(
            cross_display.select(
                "position_id",
                "factor_id_i",
                "factor_name_i",
                "factor_id_j",
                "factor_name_j",
                "cross_second_order",
            ).to_pandas(),
            title="Cross sensitivities",
        )

    book_spec = build_sensitivity_book_spec(sensitivities=sensitivities)
    calculator = SensitivityPnlCalculator.compile(book_spec)

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
# SCENARIO SHOCKS
# =============================================================================

# Human-readable names for `ProxyResolution.resolution_code`. Presentation
# only -- the numerical path never sees these.
_RESOLUTION_CODE_NAMES: dict[int, str] = {
    int(RES_NOT_SIMULATED): "NOT_SIMULATED",
    int(RES_DIRECT): "DIRECT",
    int(RES_TENOR_FALLBACK): "TENOR_FALLBACK",
    int(RES_ID_PROXY): "ID_PROXY",
    int(RES_CONTAINER_PROXY): "CONTAINER_PROXY",
}

# Deterministic, hand-verifiable shock set. These are LOG RETURNS, because
# every node in this universe declares `shift_type = LOG_RETURN` -- the
# reconstruction below is `level = base * exp(shock * scale)`, NOT an additive
# bump. Values are chosen asymmetric and non-zero so a wrong (additive)
# interpretation would be numerically visible.
#
# Columns are the SOURCE factors that actually carry market data. WTI_CRUDE is
# deliberately absent: it has no history of its own and is proxied from
# BRENT_CRUDE by the compiled ProxyTopology.
_SCENARIO_SOURCE_NAMES: tuple[str, ...] = ("USDEUR", "AAPL_EQ", "BRENT_CRUDE")

# (S, K_source) -- one ROW per scenario, one COLUMN per source factor.
_SCENARIO_SHOCKS: tuple[tuple[float, ...], ...] = (
    #  USDEUR   AAPL_EQ  BRENT_CRUDE
    (0.000, 0.010, -0.020),  # scenario 0
    (-0.010, 0.000, 0.015),  # scenario 1
    (0.020, -0.015, -0.010),  # scenario 2
)


def _run_scenario_shocks(rsc: RiskStaticConfig) -> npt.NDArray[np.float32]:
    """Materialise the deterministic raw shock matrix in canonical layout.

    Returns the `(S, K_source)` float32 F-CONTIGUOUS matrix that
    `ScenarioSetBatch.shocks` requires: the apply-shocks kernel reads down
    the scenario axis with unit stride, so F-order is a hard contract
    (`validate_apply_shocks_buffers` rejects anything else).

    These are RAW shocks against the SOURCE factors. Proxy resolution has not
    been applied yet -- that happens in `_run_proxy_resolved_levels`.
    """
    print_header("Deterministic Scenario Shocks (raw, source factors)")

    shocks = np.asfortranarray(np.array(_SCENARIO_SHOCKS, dtype=np.float32))

    print_df(
        pd.DataFrame(
            np.asarray(_SCENARIO_SHOCKS),
            columns=list(_SCENARIO_SOURCE_NAMES),
            index=[f"scenario {s}" for s in range(len(_SCENARIO_SHOCKS))],
        ),
        title="Raw scenario shocks (LOG RETURNS, source factors only)",
        index=True,
        float_format="_.4f",
    )
    print(
        "These are log returns, not additive bumps. WTI_CRUDE has no column: "
        "it carries\nno market data of its own and is sourced from "
        "BRENT_CRUDE via the ProxyTopology."
    )
    print("---")

    return shocks


# =============================================================================
# PROXY-RESOLVED SCENARIO LEVELS
# =============================================================================


def _run_proxy_resolved_levels(
    *,
    rsc: RiskStaticConfig,
    proxy_topology: ProxyTopology,
    uix_mapper: UixMapper,
    snapshot: RiskFactorLevelSnapshot,
    raw_shocks: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resolve proxies and reconstruct scenario levels via production code.

    Composes the existing public production path -- this demo adds no
    scenario machinery of its own:

        build_universe_mapping        : the required factor axis + each
                                         factor's declared shift_type
        extract_levels_from_snapshot  : t0 BASE levels for those factors
        resolve_required_factor_ids_.. : required -> (source, shock_scale),
                                         i.e. the compiled ProxyTopology
                                         applied to a concrete market-data
                                         universe
        build_shock_universe          : required -> source shock COLUMN
        apply_shocks_to_base_levels   : level = base * exp(shock * scale)
                                         for LOG_RETURN factors

    Proxying changes SHOCK PROVENANCE, never a base level. WTI_CRUDE keeps
    its own base level (70.00) and its own identity in the sensitivity book;
    only the shock driving it is sourced from BRENT_CRUDE and scaled by 0.95.

    Returns `(required_factor_ids, base_levels, shocked_levels)`.
    """
    print_header("Proxy Resolution and Scenario Level Reconstruction")

    universe_map = build_universe_mapping(static_config=rsc, uix_mapper=uix_mapper)
    required_factor_ids = universe_map.required_factor_ids

    base_levels = extract_levels_from_snapshot(
        required_ids=required_factor_ids, snapshot=snapshot
    )

    # The market-data universe actually available to drive scenarios. WTI is
    # absent on purpose: that absence is what forces the proxy path.
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

    shock_universe = build_shock_universe(
        required_shift_type_id=universe_map.required_shift_type_ids,
        source_factor_ids=proxy_resolution.source_factor_ids,
        available_factor_ids=available_factor_ids,
    )

    rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    print_df(
        pd.DataFrame(
            {
                "required_factor": [
                    rf_id_to_name[int(rf)] for rf in required_factor_ids
                ],
                "base_level": base_levels,
                "source_factor": [
                    rf_id_to_name[int(rf)] for rf in proxy_resolution.source_factor_ids
                ],
                "shock_scale": proxy_resolution.shock_scale,
                "resolution": [
                    _RESOLUTION_CODE_NAMES[int(code)]
                    for code in proxy_resolution.resolution_code
                ],
            }
        ),
        title="Proxy resolution (exposure identity vs shock provenance)",
        float_format="_.4f",
    )
    print(
        "WTI_CRUDE keeps its OWN base level and its OWN sensitivity identity.\n"
        "Only its shock provenance changes:\n"
        "    WTI_level(s) = 70.00 * exp(0.95 * BRENT_CRUDE_shock(s))\n"
        "This is NOT a substitution of Brent's level for WTI's."
    )
    print("---")

    batch = ScenarioSetBatch(
        shocks=raw_shocks,
        required_to_shock_uix=shock_universe.required_to_shock_uix,
        # float32 by ProxyResolution contract; promoted to the kernel's
        # float64 at the execution boundary inside `apply_shocks_to_base_levels`.
        shock_scale=proxy_resolution.shock_scale,
        required_shift_type_id=universe_map.required_shift_type_ids,
        scenario_offset=0,
    )

    num_scenarios = int(raw_shocks.shape[0])
    shocked_levels = np.empty(
        (required_factor_ids.shape[0], num_scenarios), dtype=np.float64, order="C"
    )
    validate_apply_shocks_buffers(
        base_levels=base_levels, shocks=raw_shocks, out=shocked_levels
    )
    apply_shocks_to_base_levels(
        base_levels=base_levels, batch=batch, out=shocked_levels
    )

    print_df(
        pd.DataFrame(
            shocked_levels,
            index=[rf_id_to_name[int(rf)] for rf in required_factor_ids],
            columns=[f"scenario {s}" for s in range(num_scenarios)],
        ),
        title="Proxy-resolved scenario LEVELS (base * exp(shock * scale))",
        index=True,
        float_format="_.6f",
    )
    print("---")

    return required_factor_ids, base_levels, shocked_levels


# =============================================================================
# SENSITIVITY P&L
# =============================================================================


def _run_scenario_pnl(
    *,
    rsc: RiskStaticConfig,
    calculator: SensitivityPnlCalculator,
    required_factor_ids: npt.NDArray[np.int32],
    base_levels: npt.NDArray[np.float64],
    shocked_levels: npt.NDArray[np.float64],
) -> SensitivityPnlResult:
    """Run the compiled sensitivity book against the scenario levels.

    Uses the calculator's LEVELS path (`FactorAxis.levels`), so the
    base-to-shocked conversion into a displacement is done by the production
    kernel under each factor's declared `DisplacementConvention` -- ABSOLUTE
    here, i.e. `x = shocked_level - base_level`. The demo performs no
    displacement arithmetic of its own.

    The Taylor form evaluated by the production kernel is

        PnL = SUM[d1 * x + 0.5 * d2 * x^2] + SUM[c_ij * x_i * x_j]

    Note the cross term carries NO 0.5 factor (the halving is absorbed by
    summing only i<j rather than both orderings).
    """
    print_header("Sensitivity P&L (deterministic scenarios)")

    axis = FactorAxis.from_keys(
        pl.Series("factor_id", required_factor_ids.tolist(), dtype=pl.Int32)
    )
    market = axis.levels(base_levels=base_levels, shocked_levels=shocked_levels)
    result = calculator.run(market)

    settle_ccy_by_uid = dict(
        zip(_POSITIONS_DF["uid"], _POSITIONS_DF["settlement_ccy"], strict=True)
    )
    underlying_by_uid = dict(
        zip(_POSITIONS_DF["uid"], _POSITIONS_DF["underlying"], strict=True)
    )

    print_df(
        pd.DataFrame(
            result.pnl,
            index=[
                f"{uid} ({underlying_by_uid[uid]}, {settle_ccy_by_uid[uid]})"
                for uid in result.position_uids.tolist()
            ],
            columns=[f"scenario {s}" for s in range(result.pnl.shape[1])],
        ),
        title="Trade P&L by scenario (each in its own settlement currency)",
        index=True,
        float_format="_.4f",
    )
    print(
        "No portfolio total is shown. AAPL and WTI settle in USD, Brent in "
        "EUR;\nsumming them would be economically meaningless. A "
        "reporting-currency\naggregation is a policy choice and is outside "
        "this increment."
    )
    print("---")

    return result


# =============================================================================
# HISTORICAL SCENARIOS
# =============================================================================

# Tiny hand-authored history over the SOURCE factors only. WTI_CRUDE has NO
# history column by design: it is not observed here, and the ProxyTopology
# sources its shock from BRENT_CRUDE. It is never given synthetic history and
# then overwritten.
#
# Observed LEVELS, one row per business date, oldest first. Standard
# historical simulation turns each consecutive pair into one scenario, so
# T rows yield T - horizon scenarios. Values are chosen so the resulting
# 1-day log returns are non-trivial in both directions and hand-checkable.
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

_HISTORY_HORIZON: int = 1
_HISTORY_BATCH_SIZE: int = 4


def _build_historical_matrix(rsc: RiskStaticConfig) -> RiskFactorMatrix:
    """Turn the hand-authored history into the canonical observations container.

    `build_risk_factor_matrix_from_df` is the canonical EDGE builder: a wide
    pandas frame (dates on the index, one column per factor NAME) plus a
    name -> rf_id map. Names die here; everything downstream is int32 factor
    ids and an F-contiguous float64 `(T, N)` values block.
    """
    frame = pd.DataFrame(
        np.asarray(_HISTORY_LEVELS, dtype=np.float64),
        index=pd.to_datetime(list(_HISTORY_DATES)),
        columns=list(_SCENARIO_SOURCE_NAMES),
    )
    return build_risk_factor_matrix_from_df(
        df=frame,
        name_to_id_map={
            name: int(rsc.rel_name_to_rf_id[name]) for name in _SCENARIO_SOURCE_NAMES
        },
    )


def _run_simulated_scenarios(
    *,
    rsc: RiskStaticConfig,
    proxy_topology: ProxyTopology,
    uix_mapper: UixMapper,
    snapshot: RiskFactorLevelSnapshot,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate risk-factor scenarios by standard HISTORICAL simulation.

    Uses `HistoricalGenerator` -- the repository's canonical historical
    `ScenarioGenerator`. Like every generator it consumes the resolved proxy
    outputs (`source_factor_ids`, `shock_scale`) and calls
    `build_shock_universe` internally, so the demo does no proxy work of its
    own and no second scenario abstraction is created.

    The generator produces RISK-FACTOR scenarios, never P&L:

        historical levels -> h-day shocks -> proxy -> shocked levels
            -> sensitivity P&L

    Historical data and proxying stay conceptually distinct. Each observed
    factor supplies its OWN shocks; the proxy only re-routes which column a
    required factor reads and scales it. WTI_CRUDE has no history at all and
    keeps its own base level -- only its shock provenance is Brent's.

    Shocks are exact h-day changes in each factor's declared `shift_type`
    (LOG_RETURN here), computed by the generator's own kernel. The demo
    performs no return arithmetic.
    """
    print_header("Historical Scenarios (standard historical simulation)")

    universe_map = build_universe_mapping(static_config=rsc, uix_mapper=uix_mapper)
    required_factor_ids = universe_map.required_factor_ids
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

    matrix = _build_historical_matrix(rsc)
    generator = HistoricalGenerator(matrix=matrix, horizon=_HISTORY_HORIZON)

    shocked_chunks: list[npt.NDArray[np.float64]] = []
    for batch in generator.iter_batches(
        required_shift_type_id=universe_map.required_shift_type_ids,
        source_factor_ids=proxy_resolution.source_factor_ids,
        shock_scale=proxy_resolution.shock_scale,
        batch_size=_HISTORY_BATCH_SIZE,
    ):
        batch_scenarios = int(batch.shocks.shape[0])
        chunk = np.empty(
            (required_factor_ids.shape[0], batch_scenarios),
            dtype=np.float64,
            order="C",
        )
        validate_apply_shocks_buffers(
            base_levels=base_levels, shocks=batch.shocks, out=chunk
        )
        apply_shocks_to_base_levels(base_levels=base_levels, batch=batch, out=chunk)
        shocked_chunks.append(chunk)

    shocked_levels = np.ascontiguousarray(
        np.concatenate(shocked_chunks, axis=1), dtype=np.float64
    )

    rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    print(
        f"Observation dates : {len(_HISTORY_DATES)} "
        f"({_HISTORY_DATES[0]} .. {_HISTORY_DATES[-1]})\n"
        f"Horizon           : {_HISTORY_HORIZON}-day\n"
        f"Scenarios         : {shocked_levels.shape[1]} "
        f"(= T - horizon)\n"
        f"Batch size        : {_HISTORY_BATCH_SIZE} "
        f"({len(shocked_chunks)} batches)\n"
        f"Observed factors  : {len(_SCENARIO_SOURCE_NAMES)} source "
        f"({', '.join(_SCENARIO_SOURCE_NAMES)})\n"
        f"Required factors  : {required_factor_ids.shape[0]} "
        f"(WTI_CRUDE has NO history; its shock is proxied from BRENT_CRUDE)"
    )
    print("---")

    print_df(
        pd.DataFrame(
            {
                "base_level": base_levels,
                "mean": shocked_levels.mean(axis=1),
                "std": shocked_levels.std(axis=1),
                "min": shocked_levels.min(axis=1),
                "max": shocked_levels.max(axis=1),
            },
            index=[rf_id_to_name[int(rf)] for rf in required_factor_ids],
        ),
        title="Simulated shocked-level statistics",
        index=True,
        float_format="_.4f",
    )
    print("---")

    return required_factor_ids, base_levels, shocked_levels


# =============================================================================
# REPORTING CURRENCY
# =============================================================================

_REPORTING_CCY: str = "USD"


def _run_reporting_currency(
    *,
    rsc: RiskStaticConfig,
    calculator: SensitivityPnlCalculator,
    required_factor_ids: npt.NDArray[np.int32],
    base_levels: npt.NDArray[np.float64],
    shocked_levels: npt.NDArray[np.float64],
    label: str,
) -> tuple[SensitivityPnlResult, ReportingCurrencyPnl]:
    """Convert settlement-currency scenario P&L into the reporting currency.

    CONVENTION (stated, not left to be inferred from the code):

        Reporting-currency scenario P&L is obtained by converting the
        SETTLEMENT-currency scenario P&L using the SCENARIO FX rate:

            dV_report[i, s] = dV_settlement[i, s] * F_settlement->report[i, s]

    It is NOT `V_shocked*F_shocked - V_base*F_base`. For the EUR-settled,
    USD-priced Brent position that alternative cancels the FX factor
    algebraically and deletes the USDEUR exposure the sensitivity book
    carries (dV/dUSDEUR = -4115). The two disagree in sign once FX moves.

    The FX quotation convention comes from the FX authority
    (`portfolio.fx`), never hard-coded here: `USDEUR` multiplies USD -> EUR,
    so converting EUR -> USD uses its reciprocal (invert=1).

    Aggregation happens ONLY after every row is in the reporting currency.
    """
    print_header(f"Reporting Currency ({label})")

    axis = FactorAxis.from_keys(
        pl.Series("factor_id", required_factor_ids.tolist(), dtype=pl.Int32)
    )
    settlement_result = calculator.run(
        axis.levels(base_levels=base_levels, shocked_levels=shocked_levels)
    )

    settle_by_uid = dict(
        zip(_POSITIONS_DF["uid"], _POSITIONS_DF["settlement_ccy"], strict=True)
    )
    uids = settlement_result.position_uids.tolist()
    settlement_ccys = [settle_by_uid[uid] for uid in uids]

    # FX factor rows, by name, taken from the SCENARIO levels -- so the
    # conversion is scenario-specific, not a single spot rate.
    rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    fx_names: list[str] = []
    fx_rows: list[npt.NDArray[np.float64]] = []
    for row, rf in enumerate(required_factor_ids):
        name = rf_id_to_name[int(rf)]
        if name == "USDEUR":
            fx_names.append(name)
            fx_rows.append(shocked_levels[row])
    fx_levels = np.ascontiguousarray(np.vstack(fx_rows), dtype=np.float64)

    plan = build_fx_conversion_plan(
        settlement_ccys=settlement_ccys,
        reporting_ccy=_REPORTING_CCY,
        fx_factor_names=fx_names,
    )

    # Documented dtype seam: SensitivityPnlResult.pnl is float32; the
    # conversion contract requires float64 C-contiguous. Promote explicitly.
    settlement_pnl = np.ascontiguousarray(settlement_result.pnl, dtype=np.float64)

    reporting = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=settlement_pnl, fx_levels=fx_levels, plan=plan
    )

    print_df(
        pd.DataFrame(
            {
                "settlement_ccy": plan.settlement_ccys,
                "fx_route": [
                    "identity"
                    if plan.fx_row_index[i] == INVALID_UIX
                    else f"{fx_names[int(plan.fx_row_index[i])]}"
                    + (" (inverted)" if plan.invert[i] else "")
                    for i in range(plan.num_positions)
                ],
            },
            index=[f"{uid} ({rf_id_to_name and settle_by_uid[uid]})" for uid in uids],
        ),
        title=f"Settlement -> reporting ({_REPORTING_CCY}) FX routing",
        index=True,
    )
    print(
        f"Reporting currency: {_REPORTING_CCY}\n"
        f"Reporting P&L = settlement P&L * scenario FX rate.\n"
        f"USD-settled trades convert by identity; the EUR-settled Brent "
        f"trade\nconverts with 1 / USDEUR at each scenario's own FX level."
    )
    print("---")

    return settlement_result, reporting


# =============================================================================
# RISK MEASURES (VaR / ES)
# =============================================================================

# Illustrative confidence level for a 3-scenario demonstration ONLY.
#
# With S = 3 and alpha = 1 - C = 1/3, the tail mass is
#     m = alpha * S = 1.0
# exactly integral, so the canonical geometry resolves to
#     var_index = 0, full_tail_count = 1, fractional_weight = 0.
# VaR is therefore the single worst scenario and ES integrates exactly that
# one observation -- an unambiguous, hand-verifiable result.
#
# A regulatory level (97.5% / 99%) on 3 scenarios would place the entire tail
# inside a fraction of the worst observation, making the demonstration
# meaningless. The CONVENTION is not altered: `compute_var_es` is the same
# production entry point a real run uses; only the confidence input differs.
_VAR_CONFIDENCE_LEVEL: float = 2.0 / 3.0


def _run_reporting_currency_golden_checks(
    *,
    rsc: RiskStaticConfig,
    settlement_result: SensitivityPnlResult,
    reporting: ReportingCurrencyPnl,
    required_factor_ids: npt.NDArray[np.int32],
    shocked_levels: npt.NDArray[np.float64],
) -> bool:
    """Validate reporting-currency conversion against an INDEPENDENT oracle.

    The oracle is a deliberately plain nested loop over positions and
    scenarios. It never calls `convert_scenario_pnl_to_reporting_currency`:

        for each scenario s:
            portfolio = 0
            for each position i:
                fx = 1                    if settlement == reporting
                   = 1 / USDEUR[s]        if settlement is EUR, reporting USD
                portfolio += settlement_pnl[i, s] * fx
    """
    print_header("Reporting-Currency Golden Checks (production vs oracle)")

    rf_id_to_name = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    usdeur_row = next(
        row
        for row, rf in enumerate(required_factor_ids)
        if rf_id_to_name[int(rf)] == "USDEUR"
    )
    usdeur = shocked_levels[usdeur_row]

    settle_by_uid = dict(
        zip(_POSITIONS_DF["uid"], _POSITIONS_DF["settlement_ccy"], strict=True)
    )
    uids = settlement_result.position_uids.tolist()
    settlement_pnl = np.asarray(settlement_result.pnl, dtype=np.float64)

    num_scenarios = settlement_pnl.shape[1]
    oracle_converted = np.empty_like(settlement_pnl)
    oracle_portfolio = np.zeros(num_scenarios, dtype=np.float64)

    for scenario in range(num_scenarios):
        running_total = 0.0
        for position, uid in enumerate(uids):
            settlement_ccy = settle_by_uid[uid]
            if settlement_ccy == _REPORTING_CCY:
                fx_rate = 1.0
            elif settlement_ccy == "EUR" and _REPORTING_CCY == "USD":
                # USDEUR quotes USD -> EUR, so EUR -> USD is its reciprocal.
                fx_rate = 1.0 / float(usdeur[scenario])
            else:  # pragma: no cover - the demo universe has no other pair
                raise AssertionError(
                    f"Oracle has no rule for {settlement_ccy} -> {_REPORTING_CCY}."
                )
            converted = float(settlement_pnl[position, scenario]) * fx_rate
            oracle_converted[position, scenario] = converted
            running_total += converted
        oracle_portfolio[scenario] = running_total

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, actual: float, expected: float, *, tol: float) -> None:
        passed = abs(actual - expected) <= tol
        checks.append((name, passed, f"max_abs_diff={abs(actual - expected):.3e}"))

    converted_diff = float(np.max(np.abs(reporting.converted_pnl - oracle_converted)))
    portfolio_diff = float(np.max(np.abs(reporting.portfolio_pnl - oracle_portfolio)))
    check("converted per-position P&L matches oracle", converted_diff, 0.0, tol=1e-9)
    check("portfolio P&L matches oracle", portfolio_diff, 0.0, tol=1e-9)

    # Aggregation must happen AFTER conversion, never before.
    naive_total = settlement_pnl.sum(axis=0)
    mixed_currency_diff = float(np.max(np.abs(reporting.portfolio_pnl - naive_total)))
    checks.append(
        (
            "portfolio total is NOT a raw mixed-currency sum",
            mixed_currency_diff > 1e-6,
            f"max_abs_diff_vs_naive={mixed_currency_diff:.6f}",
        )
    )

    # USD-settled rows must be untouched by conversion (identity route).
    for position, uid in enumerate(uids):
        if settle_by_uid[uid] != _REPORTING_CCY:
            continue
        identity_diff = float(
            np.max(np.abs(reporting.converted_pnl[position] - settlement_pnl[position]))
        )
        check(
            f"uid={uid} ({_REPORTING_CCY}) converts by identity",
            identity_diff,
            0.0,
            tol=0.0,
        )

    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name:<70} | {detail}")
    print("---")
    print(f"{len(checks)} reporting-currency checks run.")
    print()

    return all_passed


def _run_risk_measures(
    *,
    pnl: SensitivityPnlResult,
    reporting: ReportingCurrencyPnl,
) -> tuple[VarEsResult, list[str], npt.NDArray[np.float64]]:
    """Compute canonical unweighted empirical VaR and ES.

    Uses the production entry point `scenario_metrics.api.var_es.compute_var_es`
    directly. The demo implements no quantile mathematics.

    Convention (established by that package, not by this demo)
    ---------------------------------------------------------
        alpha = 1 - confidence_level
        m     = alpha * n_scenarios
        VaR   = the ceil(m)-th smallest P&L (a LOWER-TAIL ORDER STATISTIC)
        ES    = integrates exactly m observations' worth of probability mass,
                with a fractional weight on the boundary observation when m is
                non-integral.

    VaR is the P&L quantile ITSELF, so a loss is NEGATIVE. It is never
    sign-flipped into a positive loss number.

    Currency handling
    -----------------
    `solutions.sensitivity_pnl` is currency-agnostic: it carries no reporting
    currency, so there is no canonical cross-currency aggregation to reuse and
    inventing one is out of scope here. Risk measures are therefore reported
    per trade, plus for the USD sub-portfolio (AAPL + WTI), which is the only
    economically coherent aggregate available. The EUR-settled Brent trade is
    never summed into it.
    """
    print_header("Risk Measures (unweighted empirical VaR / ES)")

    uids = pnl.position_uids.tolist()
    underlying_by_uid = dict(
        zip(_POSITIONS_DF["uid"], _POSITIONS_DF["underlying"], strict=True)
    )
    settle_by_uid = dict(
        zip(_POSITIONS_DF["uid"], _POSITIONS_DF["settlement_ccy"], strict=True)
    )

    # Per-trade rows stay in each trade's SETTLEMENT currency; the portfolio
    # row is the reporting-currency aggregate produced after conversion. The
    # two are never mixed in a single sum.
    trade_rows = np.asarray(pnl.pnl, dtype=np.float64)

    group_value_matrix = np.ascontiguousarray(
        np.vstack([trade_rows, reporting.portfolio_pnl[None, :]]), dtype=np.float64
    )
    labels = [
        f"{uid} ({underlying_by_uid[uid]}, {settle_by_uid[uid]})" for uid in uids
    ] + [f"PORTFOLIO (all trades, {reporting.reporting_ccy})"]

    result = compute_var_es(group_value_matrix, _VAR_CONFIDENCE_LEVEL)

    alpha = 1.0 - _VAR_CONFIDENCE_LEVEL
    num_scenarios = group_value_matrix.shape[1]
    print(
        f"Confidence level      : {_VAR_CONFIDENCE_LEVEL:.6f}\n"
        f"Lower-tail alpha      : {alpha:.6f}\n"
        f"Scenarios             : {num_scenarios}\n"
        f"Tail mass (alpha * S) : {alpha * num_scenarios:.6f}"
    )
    print("---")

    print_df(
        pd.DataFrame(
            {"VaR": result.var, "ES": result.es},
            index=labels,
        ),
        title=f"VaR / ES at {_VAR_CONFIDENCE_LEVEL:.4f} confidence",
        index=True,
        float_format="_.4f",
    )
    print(
        "VaR is the lower P&L quantile itself: a loss is NEGATIVE.\n"
        "More negative = greater downside risk. It is not reported as a\n"
        "positive loss number.\n"
        f"Per-trade rows are in each trade's own settlement currency. The\n"
        f"PORTFOLIO row is in {reporting.reporting_ccy}, aggregated only AFTER\n"
        f"every trade was converted to the reporting currency."
    )
    print("---")

    return result, labels, group_value_matrix


def _run_risk_measure_golden_checks(
    *,
    result: VarEsResult,
    labels: list[str],
    group_value_matrix: npt.NDArray[np.float64],
) -> bool:
    """Validate VaR / ES against an INDEPENDENT oracle.

    The oracle re-derives the canonical definition from first principles
    rather than calling `compute_var_es`:

        alpha = 1 - C ; m = alpha * S
        VaR = sorted(pnl)[ceil(m) - 1]        (lower-tail order statistic)
        ES  = (1/m) * [ sum of the first floor(m) sorted observations
                        + frac(m) * the next one ]
    """
    print_header("Risk-Measure Golden Checks (production vs independent oracle)")

    num_scenarios = group_value_matrix.shape[1]
    alpha = 1.0 - _VAR_CONFIDENCE_LEVEL
    tail_mass = alpha * num_scenarios
    # Match the production float-noise snap: `(1 - 2/3) * 9` evaluates to
    # 3.0000000000000004, and a naive ceil() would pick rank 4 instead of 3.
    # `scenario_metrics._computer` documents and applies the same snap, so the
    # oracle must model the definition, not a floating-point artefact of it.
    nearest = round(tail_mass)
    if abs(tail_mass - nearest) <= 1e-9 * max(1.0, abs(tail_mass)):
        tail_mass = float(nearest)
    full_count = math.floor(tail_mass + 1e-12)
    frac = tail_mass - full_count

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, actual: float, expected: float, *, tol: float = 1e-9) -> None:
        passed = abs(actual - expected) <= tol
        checks.append((name, passed, f"actual={actual:.6f} expected={expected:.6f}"))

    for row_index, label in enumerate(labels):
        ordered = np.sort(group_value_matrix[row_index])

        var_rank = max(1, math.ceil(tail_mass))
        expected_var = float(ordered[var_rank - 1])

        tail_sum = float(ordered[:full_count].sum())
        if frac > 0.0:
            tail_sum += frac * float(ordered[full_count])
        expected_es = tail_sum / tail_mass

        check(f"VaR {label}", float(result.var[row_index]), expected_var)
        check(f"ES  {label}", float(result.es[row_index]), expected_es)

        # VaR is a LOSS here and must stay negative -- never sign-flipped.
        check(
            f"VaR sign is negative (loss) {label}",
            float(result.var[row_index] < 0.0),
            1.0,
        )

    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name:<70} | {detail}")
    print("---")
    print(f"{len(checks)} risk-measure checks run.")
    print()

    return all_passed


# =============================================================================
# GOLDEN CHECKS
# =============================================================================


def _run_scenario_golden_checks(
    *,
    rsc: RiskStaticConfig,
    proxy_topology: ProxyTopology,
    required_factor_ids: npt.NDArray[np.int32],
    base_levels: npt.NDArray[np.float64],
    shocked_levels: npt.NDArray[np.float64],
    raw_shocks: npt.NDArray[np.float32],
    pnl: SensitivityPnlResult,
) -> bool:
    """Validate scenario P&L against an INDEPENDENT economic oracle.

    The oracle is derived from the instrument identities, NOT from the
    sensitivity coefficients and NOT by re-running the production algorithm:

        AAPL   V = Q * S            ->  dV = Q * dS
        WTI    V = Q * S            ->  dV = Q * dS
                 with S(s) = S0 * exp(0.95 * BRENT_shock(s))
        Brent  V = Q * B * F        ->  dV = Q * (B0*dF + F0*dB + dB*dF)

    The Brent form is the EXACT increment of the bilinear payoff, so it is a
    genuine independent check of the production Taylor expansion (whose
    first-order + cross terms must reproduce it exactly for a bilinear
    payoff).
    """
    print_header("Scenario Golden Checks (production vs independent oracle)")

    name_to_rf = rsc.rel_name_to_rf_id
    row_of = {int(rf): i for i, rf in enumerate(required_factor_ids)}
    usdeur_i = row_of[int(name_to_rf["USDEUR"])]
    aapl_i = row_of[int(name_to_rf["AAPL_EQ"])]
    brent_i = row_of[int(name_to_rf["BRENT_CRUDE"])]
    wti_i = row_of[int(name_to_rf["WTI_CRUDE"])]

    f0 = float(base_levels[usdeur_i])
    b0 = float(base_levels[brent_i])

    d_aapl = shocked_levels[aapl_i] - base_levels[aapl_i]
    d_wti = shocked_levels[wti_i] - base_levels[wti_i]
    d_brent = shocked_levels[brent_i] - base_levels[brent_i]
    d_fx = shocked_levels[usdeur_i] - base_levels[usdeur_i]

    oracle = {
        1_010: 100.0 * d_aapl,
        1_030: 75.0 * d_wti,
        1_020: -50.0 * (b0 * d_fx + f0 * d_brent + d_brent * d_fx),
    }

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, actual: float, expected: float, *, tol: float) -> None:
        passed = abs(actual - expected) <= tol
        checks.append((name, passed, f"actual={actual:.6f} expected={expected:.6f}"))

    # --- P&L, elementwise per trade per scenario -------------------------
    # float32 P&L on values of order 1e2 carries ~1e-2 absolute resolution;
    # tolerance is set from the dtype, not loosened to hide an axis error.
    for uid, expected_row in oracle.items():
        produced = pnl.row_for(uid)
        for s, expected in enumerate(expected_row.tolist()):
            check(
                f"P&L uid={uid} scenario {s}",
                float(produced[s]),
                expected,
                tol=5e-2,
            )

    # --- Base levels are NOT altered by proxying -------------------------
    check("WTI base level unchanged", float(base_levels[wti_i]), 70.00, tol=1e-4)
    check("BRENT base level unchanged", float(base_levels[brent_i]), 82.30, tol=1e-4)
    check("USDEUR base level unchanged", f0, 1.0800, tol=1e-6)

    # --- WTI shock provenance: Brent x 0.95, inside the exponent ---------
    brent_shock_col = _SCENARIO_SOURCE_NAMES.index("BRENT_CRUDE")
    for s in range(raw_shocks.shape[0]):
        expected_wti_level = 70.00 * math.exp(
            0.95 * float(raw_shocks[s, brent_shock_col])
        )
        check(
            f"WTI level = 70*exp(0.95*dBrent) scenario {s}",
            float(shocked_levels[wti_i, s]),
            expected_wti_level,
            tol=1e-4,
        )

    # --- WTI P&L must NOT use the raw (absent) WTI shock ------------------
    # A zero WTI shock would leave WTI P&L identically zero; it must not be.
    check(
        "WTI P&L is non-zero (proxy shock applied, not a raw zero shock)",
        float(np.abs(np.asarray(pnl.row_for(1_030))).sum() > 1.0),
        1.0,
        tol=0.0,
    )

    # --- Unproxied factors pass through unchanged -------------------------
    for label, idx, col in (
        ("USDEUR", usdeur_i, _SCENARIO_SOURCE_NAMES.index("USDEUR")),
        ("AAPL_EQ", aapl_i, _SCENARIO_SOURCE_NAMES.index("AAPL_EQ")),
        ("BRENT_CRUDE", brent_i, brent_shock_col),
    ):
        for s in range(raw_shocks.shape[0]):
            expected_level = float(base_levels[idx]) * math.exp(
                float(raw_shocks[s, col])
            )
            check(
                f"{label} passes through unscaled scenario {s}",
                float(shocked_levels[idx, s]),
                expected_level,
                tol=1e-4,
            )

    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name:<70} | {detail}")
    print("---")
    print(f"{len(checks)} scenario checks run.")
    print()

    return all_passed


def _run_golden_checks(
    *,
    rsc: RiskStaticConfig,
    pricing: ValuationResult,
    calculator: SensitivityPnlCalculator,
) -> bool:
    """Deterministic, hand-verifiable checks for the minimal spot universe.

    Given quantity=-50, BRENT_CRUDE=82.30 USD, USDEUR=1.0800, the straight
    `EUR MTM = quantity * Brent_USD * USDEUR` convention (confirmed above to
    be exactly what `SpotBlock`'s kernel computes) gives:

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
        # tol accounts for the risk-factor snapshot's float32 level storage
        # (`RiskFactorLevelSnapshot.factor_levels` is float32/float64 -- see
        # `market_data.risk_factors.snapshot`), not any computation error:
        # 82.30 and 1.0800 are not exactly representable in float32, so the
        # PV/sensitivity values inherit ~1e-4 relative rounding from the
        # snapshot's own storage precision before this function ever runs.
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

    print_header("Golden Checks (pricing + sensitivities)")
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name:<70} | {detail}")
    print("---")
    print(f"{len(checks)} pricing/sensitivity checks run.")
    print()

    return all_passed


# =============================================================================
# MAIN DEMO
# =============================================================================


def main() -> None:
    """Run the minimal end-to-end sensitivity-based VaR demo."""
    print_calling_file()
    print_title("END-TO-END DEMO: SENSITIVITY-BASED VAR")

    # -------------------------------------------------------------------------
    # RISK STATIC CONFIG
    # -------------------------------------------------------------------------
    rsc = _run_risk_static_config()

    # -------------------------------------------------------------------------
    # RISK PROXY CONFIG
    # -------------------------------------------------------------------------
    rpt = _run_proxy_topology(rsc=rsc)

    # -------------------------------------------------------------------------
    # POSITIONS -> PORTFOLIO
    # -------------------------------------------------------------------------
    portfolio, uix_mapper = _run_portfolio(rsc=rsc)

    # -------------------------------------------------------------------------
    # RISK-FACTOR LEVEL SNAPSHOT
    # -------------------------------------------------------------------------
    snapshot = _run_risk_factor_snapshot(rsc=rsc)

    # -------------------------------------------------------------------------
    # PRICING
    # -------------------------------------------------------------------------
    market_data = _run_market_data(rsc=rsc, uix_mapper=uix_mapper, snapshot=snapshot)

    pricing = _run_pricing(
        portfolio=portfolio, uix_mapper=uix_mapper, market_data=market_data
    )

    # -------------------------------------------------------------------------
    # TRADE SENSITIVITIES
    # -------------------------------------------------------------------------
    calculator = _run_sensitivities(
        rsc=rsc,
        portfolio=portfolio,
        uix_mapper=uix_mapper,
        market_data=market_data,
    )

    # -------------------------------------------------------------------------
    # SCENARIO SHOCKS
    # -------------------------------------------------------------------------
    raw_shocks = _run_scenario_shocks(rsc=rsc)

    # -------------------------------------------------------------------------
    # PROXY-RESOLVED SCENARIO LEVELS
    # -------------------------------------------------------------------------
    required_factor_ids, base_levels, shocked_levels = _run_proxy_resolved_levels(
        rsc=rsc,
        proxy_topology=rpt,
        uix_mapper=uix_mapper,
        snapshot=snapshot,
        raw_shocks=raw_shocks,
    )

    # -------------------------------------------------------------------------
    # SENSITIVITY P&L
    # -------------------------------------------------------------------------
    scenario_pnl = _run_scenario_pnl(
        rsc=rsc,
        calculator=calculator,
        required_factor_ids=required_factor_ids,
        base_levels=base_levels,
        shocked_levels=shocked_levels,
    )

    # -------------------------------------------------------------------------
    # GOLDEN CHECKS
    # -------------------------------------------------------------------------
    base_checks_passed = _run_golden_checks(
        rsc=rsc, pricing=pricing, calculator=calculator
    )

    # Reporting-currency conversion of the DETERMINISTIC scenario P&L, then
    # its own independent-oracle validation.
    deterministic_settlement, deterministic_reporting = _run_reporting_currency(
        rsc=rsc,
        calculator=calculator,
        required_factor_ids=required_factor_ids,
        base_levels=base_levels,
        shocked_levels=shocked_levels,
        label="deterministic scenarios",
    )

    # -------------------------------------------------------------------------
    # SIMULATED SCENARIOS
    # -------------------------------------------------------------------------
    sim_factor_ids, sim_base_levels, sim_shocked_levels = _run_simulated_scenarios(
        rsc=rsc,
        proxy_topology=rpt,
        uix_mapper=uix_mapper,
        snapshot=snapshot,
    )

    sim_settlement, sim_reporting = _run_reporting_currency(
        rsc=rsc,
        calculator=calculator,
        required_factor_ids=sim_factor_ids,
        base_levels=sim_base_levels,
        shocked_levels=sim_shocked_levels,
        label="simulated scenarios",
    )

    risk_measures, risk_labels, risk_matrix = _run_risk_measures(
        pnl=sim_settlement, reporting=sim_reporting
    )

    scenario_checks_passed = _run_scenario_golden_checks(
        rsc=rsc,
        proxy_topology=rpt,
        required_factor_ids=required_factor_ids,
        base_levels=base_levels,
        shocked_levels=shocked_levels,
        raw_shocks=raw_shocks,
        pnl=scenario_pnl,
    )

    reporting_checks_passed = _run_reporting_currency_golden_checks(
        rsc=rsc,
        settlement_result=deterministic_settlement,
        reporting=deterministic_reporting,
        required_factor_ids=required_factor_ids,
        shocked_levels=shocked_levels,
    )

    simulated_reporting_checks_passed = _run_reporting_currency_golden_checks(
        rsc=rsc,
        settlement_result=sim_settlement,
        reporting=sim_reporting,
        required_factor_ids=sim_factor_ids,
        shocked_levels=sim_shocked_levels,
    )

    risk_checks_passed = _run_risk_measure_golden_checks(
        result=risk_measures,
        labels=risk_labels,
        group_value_matrix=risk_matrix,
    )

    if (
        base_checks_passed
        and scenario_checks_passed
        and reporting_checks_passed
        and simulated_reporting_checks_passed
        and risk_checks_passed
    ):
        print("ALL CHECKS PASSED")
    else:
        print("VALIDATION FAILED")
    print()

    # -------------------------------------------------------------------------
    # E2E SUMMARY
    # -------------------------------------------------------------------------
    print_header("Priced Portfolio + Sensitivity Representation Ready")

    print(
        "Risk factors available to the downstream VaR demo:\n"
        "  rf_id=0  USDEUR\n"
        "  rf_id=1  AAPL_EQ\n"
        "  rf_id=2  BRENT_CRUDE\n"
        "  rf_id=3  WTI_CRUDE  -> BRENT_CRUDE (scale=0.95)\n"
    )

    print(
        "Discount curves available to satisfy the static-config valuation "
        "invariant:\n"
        "  EUR -> EUR_ESTR_OIS\n"
        "  USD -> USD_SOFR_OIS\n"
        "(unreferenced by this spot-only portfolio; SpotBlock's own kernel "
        "does not discount.)\n"
    )

    print(
        "Proxy topology is compiled separately from RiskStaticConfig and \n"
        "will substitute BRENT_CRUDE historical data for WTI_CRUDE at the \n"
        "risk-factor level with a 0.95 volatility scaling factor.\n"
    )

    print(
        f"Portfolio compiled: {portfolio.total_trades} trade(s) across \n"
        f"{portfolio.total_blocks} block(s) (SPOT only -- forward/option \n"
        f"block compilers are not yet registered).\n"
    )

    print(
        f"Risk-factor level snapshot built for {snapshot.num_factors} \n"
        f"factor(s) as of {snapshot.as_of_date}.\n"
    )

    print(
        f"Portfolio priced against the snapshot: {pricing.num_trades} \n"
        f"trade(s) valued (per-trade settlement currency; see above).\n"
    )

    print(
        f"Trade sensitivities compiled into the canonical \n"
        f"`solutions.sensitivity_pnl` book: "
        f"{calculator.book.position_uids.shape[0]} position(s), \n"
        f"{calculator.book.active_factor_ids.shape[0]} active risk factor(s). \n"
        f"The EUR-settled BRENT_CRUDE position correctly carries BOTH a \n"
        f"commodity and an FX sensitivity -- it is not treated as a single-\n"
        f"factor exposure.\n"
    )

    print(
        "RiskStaticConfig, ProxyTopology, the compiled portfolio, the \n"
        "risk-factor level snapshot, the priced portfolio, and the compiled \n"
        "sensitivity book are ready to supply the inputs required by the \n"
        "downstream historical/simulated shocks, proxy-adjusted shocks, \n"
        "sensitivity P&L, and VaR/ES. Those stages are not part of this \n"
        "demo yet.\n"
    )


if __name__ == "__main__":
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    main()
