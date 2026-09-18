# scripts/demos/e2e/demo_e2e_canonical_base_valuation.py
"""
End-to-end demo: canonical ingest -> compile -> market-data build -> valuation.

Uses the canonical exhaustive market-risk trade fixture AND the canonical
as-of-date market-data snapshot. One deterministic "scenario" (the t0
levels) so we exercise the full hydration path without standing up a
scenario generator.

Coverage scope (current registry state, checked against
`portfolio.compilers.registry.COMPILERS` at demo-write time)
-------------------------------------------------------------
We price the SPOT family only:

    SPOT

`VANILLA_FORWARD_NODE`, `VANILLA_FORWARD_CURVE`, both OPTION families, and
`VANILLA_IRS` all ship in the exhaustive fixture for INGEST coverage, but
none has a registered block compiler yet
(`portfolio.compilers.registry.UNREGISTERED_INSTRUMENT_IDS`) --
`compile_portfolio_from_ingest_batch` fails loud on any of them, by design
(Constitution: fail fast, never silently drop a trade). This demo filters
them out before compiling, rather than let the engine raise.

Within SPOT, trades whose `underlying` is not an admissible scalar spot
source (D063: a PRICE, FX_RATE or INDEX risk factor) are ALSO excluded -- e.g.
a SPOT on an interest-rate curve point such as `USD_LIBOR_3M[10Y]`. The
ingest identity gate rejects such rows by design. SPOT trades on FX-rate
underlyings are priced: the producer now owns their `prices` rows (D072).

The exhaustive fixture also carries VANILLA_IRS-only ingest columns
(`fixed_rate`, `notional`, `maturity_date`, ...) not in
`portfolio.ingest.schema.ALL_ALLOWED_COLS_SET`, and a `netting_set_id`
credit-exposure linkage column whose NA handling in
`portfolio.ingest.materialize._materialize_nullable_int_col` currently
raises on real NaN despite the column being documented optional. Both are
dropped from the ingest frame below -- neither is used by SPOT valuation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.rf_resolution import (
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_proxy_config.canonical_risk_proxy_config import (
    CANONICAL_RISK_PROXY_CONFIG_DF,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.ingest_fixtures.market_data.canonical_market_data_snapshot import (
    CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT,
)
from riskquant.ingest_fixtures.portfolio.exhaustive import (
    CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.portfolio.ingest.schema import ALL_ALLOWED_COLS_SET
from riskquant.portfolio.instruments import InstrumentType
from riskquant.portfolio.validation import validate_ingest_batches_against_uix_universe
from riskquant.utils.dates import date_to_ord32
from riskquant.utils.printer import (
    print_array,
    print_calling_file,
    print_df,
    print_dict,
    print_dict_of_dfs,
    print_header,
    print_subheader,
    print_success_message,
    print_text,
    print_title,
)
from riskquant.valuation.engine import run_valuation
from riskquant.valuation.result import ValuationResult

# ============================================================
# CANONICAL INPUTS
# ============================================================

_AS_OF_DATE = CANONICAL_AS_OF_DATE
_RISK_STATIC_CONFIG_DICT_OF_DFS = CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
_PROXY_CONFIG_DF = CANONICAL_RISK_PROXY_CONFIG_DF
_PORTFOLIO_DATA = CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA
_PORTFOLIO_DATA_DF = pd.DataFrame(_PORTFOLIO_DATA)
_RF_LEVELS_SNAPSHOT = CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT

# Single deterministic scenario — the canonical t0 snapshot.
_NUM_SCENARIOS: int = 1

# Only SPOT has a registered block compiler today
# (`portfolio.compilers.registry.COMPILERS`); every other family in the
# exhaustive fixture ships for ingest coverage only.
_PRICEABLE_INSTRUMENTS: frozenset[str] = frozenset({InstrumentType.SPOT.name})

# Names that are NOT admissible SPOT sources under D063, read directly from the
# canonical static-config tables: nodes and curve points whose quote type is
# not PRICE / FX_RATE / INDEX. SPOT trades on these are excluded.
_SPOT_QUOTE_TYPES: frozenset[str] = frozenset({"PRICE", "FX_RATE", "INDEX"})
_NODE_META = _RISK_STATIC_CONFIG_DICT_OF_DFS["node_meta"]
_CURVE_META = _RISK_STATIC_CONFIG_DICT_OF_DFS["curve_meta"]
_CURVE_POINTS = _RISK_STATIC_CONFIG_DICT_OF_DFS["curve_point_meta"]
_CURVE_POINT_KINDS = _CURVE_POINTS.join(
    _CURVE_META.select(["curve_id", "curve_name", "quote_type"]), on="curve_id"
)
_NON_SPOT_SOURCE_NAMES: frozenset[str] = frozenset(
    _NODE_META.filter(~_NODE_META["quote_type"].is_in(list(_SPOT_QUOTE_TYPES)))[
        "node_name"
    ].to_list()
    + [
        f"{name}{token}"
        for name, token in _CURVE_POINT_KINDS
        .filter(~_CURVE_POINT_KINDS["quote_type"].is_in(list(_SPOT_QUOTE_TYPES)))
        .select(["curve_name", "tenor_token"])
        .iter_rows()
    ]
)

# VANILLA_IRS-only ingest columns present in the exhaustive fixture but not
# in `ALL_ALLOWED_COLS_SET`, plus the optional credit-exposure linkage
# column whose NA handling currently raises — see the module docstring.
_UNPRICED_COLUMNS_TO_DROP: frozenset[str] = frozenset({"netting_set_id"})


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    print_calling_file()
    print_title("START OF DEMO - BASE VALUATION")
    print_text(
        "This demo showcases the end-to-end valuation process for SPOT "
        "instruments using canonical base data."
    )
    print_header(title="DEMO: SPOT END-TO-END VALUATION (PVs)")

    # ---- INGEST (LOCAL BINDS OF CANONICAL DATA) --------------------------

    print_subheader(title="SECTION: INGEST CANONICAL DATA FROM FIXTURES")

    as_of_date = _AS_OF_DATE
    print_text(f"AS OF DATE: {as_of_date}")

    risk_static_config_dict_of_dfs = _RISK_STATIC_CONFIG_DICT_OF_DFS
    # `print_dict_of_dfs`/`print_df` are pandas-only (`d: dict[str,
    # pd.DataFrame]`); the canonical static-config tables are Polars
    # (Constitution D006: config ingest is Polars, not pandas). Convert for
    # display only -- nothing downstream of this print reads the pandas copy.
    print_dict_of_dfs(
        {k: v.to_pandas() for k, v in risk_static_config_dict_of_dfs.items()},
        title="CANONICAL RISK STATIC CONFIGURATION",
    )

    proxy_config_df = _PROXY_CONFIG_DF
    print_df(
        proxy_config_df.to_pandas(),
        title="CANONICAL RISK PROXY CONFIGURATION",
        float_format=">12_.0f",
    )

    # Filter exhaustive trade-data down to the pricer-runnable subset: SPOT
    # only, excluding SPOT trades on non-spot source kinds (D063).
    full_trade_df = _PORTFOLIO_DATA_DF
    portfolio_trade_data_df = full_trade_df[
        full_trade_df["instrument"].isin(_PRICEABLE_INSTRUMENTS)
        & ~full_trade_df["underlying"].isin(_NON_SPOT_SOURCE_NAMES)
    ].reset_index(drop=True)
    # Drop VANILLA_IRS-only / credit-exposure-only columns not needed for
    # SPOT valuation (see module docstring); every remaining column stays.
    portfolio_trade_data_df = portfolio_trade_data_df.loc[
        :, portfolio_trade_data_df.columns.isin(ALL_ALLOWED_COLS_SET)
    ].drop(columns=list(_UNPRICED_COLUMNS_TO_DROP), errors="ignore")
    print_df(
        full_trade_df,
        title=(
            "CANONICAL EXHAUSTIVE MARKET RISK TRADE DATA "
            f"({len(full_trade_df)} rows, all instruments)"
        ),
    )
    n_dropped = len(full_trade_df) - len(portfolio_trade_data_df)
    print_text(
        f"Filtered to pricer-runnable subset: {len(portfolio_trade_data_df)} rows "
        f"(dropped {n_dropped} rows — forward-node/forward-curve/option/IRS "
        f"families have no registered block compiler yet, and SPOT trades on "
        f"non-spot source kinds are rejected by D063; see module docstring)."
    )
    print_df(
        portfolio_trade_data_df,
        title="PRICER-RUNNABLE TRADE DATA",
    )

    # ---- COMPILE MASTERS -------------------------------------------------

    print_header(title="SECTION: COMPILE MASTERS")

    risk_static_config = compile_risk_static_config_from_dict_of_dfs(
        risk_static_config_dict_of_dfs
    )
    print_text("COMPILED RISK STATIC CONFIG OBJECT:")
    risk_static_config.summary()

    print_dict(
        risk_static_config.container_name_to_id,
        title="CONTAINER NAME TO CONTAINER ID MAPPING",
    )

    proxy_config = compile_proxy_topology_from_df(
        df=proxy_config_df,
        static_config=risk_static_config,
    )
    print_subheader(title="COMPILED PROXY CONFIG OBJECT")
    proxy_config.summary()

    # ---- COMPILE RISK FACTORS SNAPSHOT ----

    # RF Levels Snapshot Canonical Input
    print_subheader(title="CANONICAL RISK FACTOR LEVELS SNAPSHOT INPUT")
    print_dict(
        _RF_LEVELS_SNAPSHOT, title="RISK FACTOR LEVELS SNAPSHOT - CANONICAL INPUT"
    )

    # Compile RF Levels Snapshot
    rf_lvl_snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=_AS_OF_DATE,
        levels_dict=_RF_LEVELS_SNAPSHOT,
        static_config=risk_static_config,
    )

    # Display Compiled Snapshot
    print_subheader(title="COMPILED RISK FACTOR LEVELS SNAPSHOT")
    rf_lvl_snapshot.summary()
    print_df(
        rf_lvl_snapshot.df.T,
        index=True,
        title="RISK FACTOR LEVELS SNAPSHOT (COMPILED OUTPUT, TRANSPOSED FOR DISPLAY)",
        float_format=">12_.6f",
    )

    # ---- RF NAME MAPPING -------------------------------------------------

    print_header(
        title="SECTION: RF ABSOLUTE NAME TO RELATIVE NAME AND ID MAPPING FOR AS-OF DATE"
    )

    rf_abs_to_rel_name_and_id = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=risk_static_config,
        as_of_date=as_of_date,
    )

    print_dict(
        rf_abs_to_rel_name_and_id.rel_name_to_abs_name,
        title="RISK FACTOR RELATIVE NAME TO ABSOLUTE NAME MAPPING",
    )
    print_dict(
        rf_abs_to_rel_name_and_id.rel_name_to_rf_id,
        title="RISK FACTOR RELATIVE NAME TO ID MAPPING",
    )

    # ---- PHASE 1: INGEST BATCHING ----------------------------------------

    print_header(title="SECTION: BUILD INGEST BATCHES")

    ingest_batch = compile_raw_df_into_ingest_batch(
        raw_df=portfolio_trade_data_df,
        static_config=risk_static_config,
        copy=False,
    )
    print_text(
        f"COMPILED INGEST BATCH (DENSE SOA) (TOTAL TRADES = {ingest_batch.size})"
    )
    print_df(
        ingest_batch.df,
        title="INGEST BATCH - PRINTED AS DATAFRAME (FOR DEBUGGING ONLY)",
        float_format=">12_.0f",
    )

    ingest_batches: list[IngestBatch] = [ingest_batch]

    # ---- PHASE 2: BUILD UIX MAPPER UNIVERSE ------------------------------

    print_header(title="SECTION: BUILD UIX MAPPER UNIVERSE FROM INGEST BATCHES")

    uix_mapper = build_uix_mapper(
        ingest_batches, as_of_ord32=int(date_to_ord32(as_of_date))
    )

    print_array(
        uix_mapper.market_id_universe,
        title="UIX MAPPER: MARKET ID UNIVERSE",
        float_format=">12_.0f",
    )
    print_array(
        uix_mapper.discount_curve_id_universe,
        title="UIX MAPPER: DISCOUNT CURVE ID UNIVERSE",
        float_format=">12_.0f",
    )
    print_array(
        uix_mapper.date_ord32_universe,
        title="UIX MAPPER: DATE ORD32 UNIVERSE",
        float_format=">12_.0f",
    )
    print_array(
        uix_mapper.fx_effective_pairs,
        title="UIX MAPPER: FX EFFECTIVE PAIR TABLE (uix, invert)",
        float_format=">12_.0f",
    )

    validate_ingest_batches_against_uix_universe(
        ingest_batches=ingest_batches,
        uix_mapper=uix_mapper,
    )

    # ---- PHASE 3: COMPILE EXECUTION BLOCKS -------------------------------

    print_header(title="SECTION: COMPILE PORTFOLIO INTO EXECUTION BLOCKS")

    compiled_portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=ingest_batch,
        uix_mapper=uix_mapper,
    )

    print_text(
        f"COMPILED PORTFOLIO EXTRACTED "
        f"{len(compiled_portfolio.__dataclass_fields__)} POTENTIAL BLOCKS."
    )

    print_header(title="SUMMARY OF COMPILED PORTFOLIO")
    compiled_portfolio.summary()

    print_subheader(title="Details of Blocks in Compiled Portfolio")
    compiled_portfolio.print_block_details()

    # ---- PHASE 5: BUILD MARKET DATA MATRIX -------------------------------

    print_header(title="SECTION: BUILD MARKET DATA MATRIX")

    universe_map = build_universe_mapping(
        static_config=risk_static_config,
        uix_mapper=uix_mapper,
    )
    market_data_mtx = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=as_of_date,
        uix_mapper=uix_mapper,
        universe_map=universe_map,
        snapshot=rf_lvl_snapshot,
        num_scenarios=_NUM_SCENARIOS,
    )

    print_text(
        f"MARKET DATA MATRIX BUILT  "
        f"(scenarios={market_data_mtx.num_scenarios}, "
        f"fx_eff_pairs={market_data_mtx.fx_spots_effective.shape[0]}, "
        f"prices={market_data_mtx.prices.shape[0]}, "
        f"discount_curves={market_data_mtx.discount_factor_grid.shape[0]}, "
        f"price_curves={market_data_mtx.price_curve_grid.shape[0]})"
    )
    print_array(
        market_data_mtx.fx_spots_effective,
        title="MARKET DATA: FX SPOTS EFFECTIVE (per fx_eff_uix, scenario)",
        float_format=">12_.6f",
    )
    print_array(
        market_data_mtx.tte_grid,
        title="MARKET DATA: TTE GRID (years from as_of, per date_uix)",
        float_format=">12_.6f",
    )

    # ---- PHASE 6: VALUATION ----------------------------------------------

    print_header(title="SECTION: RUN VALUATION")

    # `run_valuation` takes the FOLDED `ValuationState` (`effective_spot_prices
    # = spot * fx`), not the raw `MarketDataMatrix` -- `assemble_valuation_state`
    # is the fold step. It also writes in place into a caller-preallocated
    # `out` (zero-allocation hot path) and returns nothing; this demo wraps
    # the result into a `ValuationResult` for display, matching
    # `demo_e2e_sensi_based_var.py`'s `_fixtures._run_pricing`.
    valuation_state = assemble_valuation_state(
        matrix=market_data_mtx, uix_mapper=uix_mapper
    )
    out = np.empty(
        (compiled_portfolio.total_trades, market_data_mtx.num_scenarios),
        dtype=np.float64,
    )
    run_valuation(
        portfolio=compiled_portfolio, valuation_state=valuation_state, out=out
    )
    base_valuation_result = ValuationResult(
        trade_uids=compiled_portfolio.trade_uids, values=out
    )

    print_array(
        base_valuation_result.trade_uids,
        title="VALUATION: TRADE UIDS",
        float_format=">12_.0f",
    )
    print_array(
        base_valuation_result.values,
        title="VALUATION: PVs (n_trades, n_scenarios)",
        float_format=">14_.4f",
    )

    # ---- END OF DEMO -----------------------------------------------------
    print_title("END OF DEMO")
    print_success_message("Success! End-to-end demo complete.")
