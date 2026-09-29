# scripts/demos/e2e/00_spot_getting_started.py
"""
RiskQuant SPOT -- executable user guide, chapter 00: getting started.

The complete certified SPOT market-risk pipeline, one step at a time, on a
three-position book small enough to check by hand:

    STEP 1   define the book                (human-readable positions frame)
    STEP 2   build market / valuation inputs (static config, levels, history)
    STEP 3   value the base world            (ingest -> compile -> V0)
    STEP 4   choose a scenario methodology   (model config + methodology)
    STEP 5   generate shocks                 (scenario seam, float32 F-order)
    STEP 6   construct shocked levels        (base + shock, float64)
    STEP 7   calculate scenario P&L          (full revaluation, float32 grid)
    STEP 8   convert to reporting currency   (D081 ordinary MTM)
    STEP 9   calculate VaR / ES              (empirical lower tail)
    STEP 10  inspect the resulting RiskRun   (the ONE production call)

Steps 3-9 call the SAME core building blocks `run_scenario_var_es` composes
internally, so each intermediate is visible. They teach; they are not an
integration surface. Step 10 then makes the single production call a real
integration makes and proves its VaR/ES bitwise equal to steps 3-9 (chapter
07 shows how an integration assembles that call's inputs). Every number is
also checked against an INDEPENDENT hand oracle (q * P * X arithmetic), never
against RiskQuant itself.

Why the numbers are exact: the factors use DIFFERENCE shifts and every level
and move is a dyadic rational (1.25, 0.0625, 200, 4, ...). Binary floating
point represents them exactly, so the oracle comparisons below are EXACT
equality, not a tolerance. Later chapters add a realistic LOG_RETURN world.

Suite map
---------
    00 getting started: the whole pipeline, FULL_REVAL, by hand and in one call
    01 TAYLOR/DERIVED: certification, orders, identity, what the run refuses
    02 mixed book: FULL_REVAL + TAYLOR/DERIVED composed in one run
    03 FX legs and the reporting currency: D081, routes, inverted legs (D084)
    04 scenario models: five models and a horizon through one call
    05 reproducibility and identity: manifest, reproduce=, result integrity
    06 batching: an operational partition of the run's scenario world
    07 production example: the integration template

    demo_e2e_sensi_based_var.py: the D019 golden reference (proxy WTI <-
    BRENT, reciprocal USDEUR quote, certified TAYLOR/DERIVED book)

Run:  python scripts/demos/e2e/00_spot_getting_started.py
"""

from __future__ import annotations

import time

import _spot_world as sw
import numpy as np

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.reporting.fx_conversion import (
    build_fx_conversion_plan,
    convert_scenario_pnl_to_reporting_currency,
)
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.simlab.shocks.axis_operator import lower_axis_operator
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
from riskquant.solutions.risk_run.api import derive_run_factor_axis, run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    RiskRun,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import build_scenario_generator
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig
from riskquant.solutions.scenario_models.world_columns import bind_operator_to_world
from riskquant.utils.business_days import ord_to_date
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

#: S = 4 historical scenarios and C = 0.75 give tail mass m = (1 - C) * S = 1
#: exactly: VaR is the single worst scenario and ES averages exactly that one.
CONFIDENCE: float = 0.75
BATCH_SIZE: int = 2


def main() -> None:
    t0 = time.perf_counter()
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 00 getting started")
    sw.para(
        """
        This chapter runs the whole certified SPOT market-risk path on three
        positions. Read it top to bottom: each step names the production
        function it calls, prints what that function produced, and asserts
        the claim it makes.
        """
    )

    # ------------------------------------------------------------------ 1
    sw.step("STEP 1", "Define the book")
    positions = sw.TINY_POSITIONS
    sw.para(
        """
        A SPOT position is a quantity q of one scalar underlying (a price or
        an FX rate), settled in one currency. Its value in settlement currency
        is V = q * P, times the FX rate when the underlying is quoted in a
        different currency (a quanto read). The book arrives at the EDGE as a
        human-readable frame: this is the only place names and currency codes
        appear as strings.
        """
    )
    sw.table(
        ["uid", "instrument", "quantity", "underlying", "settlement_ccy"],
        positions.itertuples(index=False),
        align="rlrll",
    )
    sw.kv([
        ("uid", "forged economic position identity (int64)"),
        ("underlying", "risk-factor NAME, resolved to an rf_id at ingest"),
        ("settlement_ccy", "currency the position's value is booked in"),
    ])

    # ------------------------------------------------------------------ 2
    sw.step("STEP 2", "Build market and valuation inputs")
    sw.para(
        """
        Three inputs describe the market. The STATIC CONFIG declares each risk
        factor once: its quote currency, its quote type and its shift type
        (how a scenario move is expressed). The LEVEL SNAPSHOT gives the
        as-of value of each factor. The HISTORY is the construction data the
        scenario model reads. Names are resolved to integer rf_ids here and
        never travel further.
        """
    )
    rsc = sw.compile_static_config(sw.TINY_FACTORS, sw.TINY_CCYS)
    proxy_terms = sw.author_proxy_terms()  # no authored terms: identity rows only
    sw.table(
        ["factor", "rf_id", "quote ccy", "quote type", "shift type"],
        [
            (
                f.name,
                rsc.rel_name_to_rf_id[f.name],
                f.quote_ccy,
                f.quote_type,
                f.shift_type,
            )
            for f in sw.TINY_FACTORS
        ],
        align="lrlll",
    )
    snapshot = build_risk_factor_level_snapshot_from_dict(
        as_of_date=sw.TINY_AS_OF, levels_dict=sw.TINY_LEVELS, static_config=rsc
    )
    history = sw.history_matrix(
        rsc,
        dates=sw.TINY_HISTORY_DATES,
        names=sw.TINY_HISTORY_NAMES,
        values=sw.TINY_HISTORY_VALUES,
    )
    sw.table(
        ["date", *sw.TINY_HISTORY_NAMES],
        [
            (d, *(sw.num(v) for v in row))
            for d, row in zip(
                sw.TINY_HISTORY_DATES, sw.TINY_HISTORY_VALUES, strict=True
            )
        ],
    )
    sw.kv([
        ("as-of date", sw.TINY_AS_OF.isoformat()),
        ("base levels", ", ".join(f"{k}={v}" for k, v in sw.TINY_LEVELS.items())),
        (
            "history shape",
            f"T={history.values.shape[0]} dates x N={history.values.shape[1]} factors",
        ),
        (
            "history layout",
            f"{history.values.dtype}, F-contiguous={history.values.flags.f_contiguous}",
        ),
    ])
    ok.check(
        "history is float64 and F-contiguous (time axis unit stride)",
        history.values.dtype == np.float64 and history.values.flags.f_contiguous,
    )

    # ------------------------------------------------------------------ 3
    sw.step("STEP 3", "Value the base world")
    sw.para(
        """
        Ingest validates the frame and resolves names to rf_ids. The
        UixMapper assigns compact int32 indices and fixes the as-of. The
        compiler builds a struct-of-arrays SpotBlock. The universe mapping is
        the VALUATION axis. Here it is also the run axis (STEP 10 shows why),
        so every later array (base levels, shocks, shocked levels) uses
        exactly this row order.
        """
    )
    ingest_batch = compile_raw_df_into_ingest_batch(
        raw_df=positions.copy(), static_config=rsc
    )
    uix_mapper = build_uix_mapper(
        [ingest_batch], as_of_ord32=int(date_to_ord32(sw.TINY_AS_OF))
    )
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=ingest_batch, uix_mapper=uix_mapper
    )
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=uix_mapper
    )
    axis = universe_map.required_factor_ids
    names = {int(v): k for k, v in rsc.rel_name_to_rf_id.items()}
    base_levels = extract_levels_from_snapshot(required_ids=axis, snapshot=snapshot)
    market_data = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=sw.TINY_AS_OF,
        uix_mapper=uix_mapper,
        universe_map=universe_map,
        snapshot=snapshot,
        num_scenarios=1,
    )
    v0_rows = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=portfolio,
        valuation_state=assemble_valuation_state(
            matrix=market_data, uix_mapper=uix_mapper
        ),
        out=v0_rows,
    )
    v0 = dict(zip(portfolio.trade_uids.tolist(), v0_rows[:, 0].tolist(), strict=True))
    sw.kv([
        ("required-factor axis", [names[int(r)] for r in axis]),
        (
            "axis dtype",
            f"{axis.dtype} (rf_ids), shift ids {universe_map.required_shift_type_ids.dtype}",
        ),
        ("base levels x0", base_levels.tolist()),
    ])
    # independent oracle: V = q * P (* EURUSD for the EUR-quoted, USD-settled quanto)
    oracle_v0 = {101: 10.0 * 200.0, 102: -4.0 * 120.0, 103: 8.0 * 120.0 * 1.25}
    sw.table(
        ["uid", "hand formula", "settle", "V0 RiskQuant", "V0 hand"],
        [
            (101, "10 x 200", "USD", sw.num(v0[101], 2), sw.num(oracle_v0[101], 2)),
            (102, "-4 x 120", "EUR", sw.num(v0[102], 2), sw.num(oracle_v0[102], 2)),
            (
                103,
                "8 x 120 x 1.25",
                "USD",
                sw.num(v0[103], 2),
                sw.num(oracle_v0[103], 2),
            ),
        ],
        align="rllrr",
    )
    ok.check(
        "axis order is the universe map's (FX first, then scalar nodes)",
        [names[int(r)] for r in axis] == ["EURUSD", "AAPL_EQ", "SAP_EQ"],
    )
    ok.check("V0 equals the hand oracle exactly for every position", v0 == oracle_v0)

    # ------------------------------------------------------------------ 4
    sw.step("STEP 4", "Choose a scenario methodology")
    scenario_config = HistoricalGeneratorConfig(model="historical", horizon=1)
    full_reval = MethodologyDeclaration(
        production_position_uids=np.array([101, 102, 103], dtype=np.int64),
        full_reval_uids=np.array([101, 102, 103], dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=np.empty(0, dtype=np.int64),
    )
    sw.para(
        """
        Two independent choices. The SCENARIO MODEL decides how market moves
        are generated: here, deterministic historical simulation with 1-day
        overlapping windows. The METHODOLOGY decides how each position turns
        a move into P&L: here every position is FULL_REVAL (revalued at the
        shocked levels). Chapters 01-02 introduce TAYLOR/DERIVED.
        """
    )
    sw.kv([
        ("model", scenario_config.model),
        ("horizon", f"{scenario_config.horizon} day"),
        (
            "scenarios",
            f"T - horizon = {history.values.shape[0]} - 1 = {history.values.shape[0] - 1}",
        ),
        ("seed", "none: historical simulation is deterministic"),
        ("confidence", CONFIDENCE),
        ("methodology", "U = {101, 102, 103}, all FULL_REVAL"),
    ])

    # ------------------------------------------------------------------ 5
    sw.step("STEP 5", "Generate scenario shocks")
    # Proxy V1 (D088): the run compiles the authored terms over its axis. With
    # no authored term every factor has its identity row (factor, own chart,
    # 1.0); lowering adds each row's target chart kappa; binding maps every
    # (driver, driver chart) the rows read to a world column at the world's h.
    proxy_model = compile_proxy_model(
        authored_terms=proxy_terms,
        static_config=rsc,
        as_of_date=sw.TINY_AS_OF,
        run_axis_rf_ids=axis,
    )
    operator = lower_axis_operator(
        model=proxy_model,
        axis_rf_ids=axis,
        axis_shift_type_ids=universe_map.required_shift_type_ids,
    )
    binding = bind_operator_to_world(operator, scenario_config, history)
    generator = build_scenario_generator(scenario_config, data=history)
    # The generator REUSES one batch buffer across yields (zero-allocation
    # streaming). Consume each batch before advancing: here the shocks are
    # copied for display and the shocked levels are built immediately, which
    # is exactly what the production run does batch by batch.
    shock_chunks: list[np.ndarray] = []
    level_chunks: list[np.ndarray] = []
    batch_facts: list[bool] = []
    base_before = base_levels.copy()
    for batch in generator.iter_batches(driver_rf_ids=binding.request_driver_rf_ids,
        chart_ids=binding.request_chart_ids,
        batch_size=BATCH_SIZE,
    ):
        batch_facts.append(
            batch.shocks.dtype == np.float32
            and batch.shocks.flags.f_contiguous
            and batch.shocks.shape[0] <= BATCH_SIZE
        )
        shock_chunks.append(batch.shocks.copy())
        out = np.empty(
            (axis.shape[0], batch.shocks.shape[0]), dtype=np.float64, order="C"
        )
        apply_linear_terms_to_base_levels(
            base_levels=base_levels,
            row_term_offsets=operator.row_term_offsets,
            term_column_uix=binding.term_column_uix,
            term_loadings=operator.term_loadings,
            target_shift_types=operator.target_shift_type_ids,
            shocks=batch.shocks,
            out=out,
        )
        level_chunks.append(out)
    shocks = np.vstack(shock_chunks)
    sw.para(
        f"""
        The generator streams scenarios in batches of {BATCH_SIZE}. Each batch
        holds (B, K) float32 shocks in F-order, so the scenario axis has unit
        stride for the apply kernel. The shocked levels (STEP 6) are built
        inside the same loop, batch by batch. The compiled Proxy V1 model states
        which world column drives each factor (here every factor has its
        identity row: its own column in its own chart, loading 1). The base
        valuation world is not touched by this step.
        """
    )
    hist = np.asarray(sw.TINY_HISTORY_VALUES, dtype=np.float64)
    oracle_shocks = np.diff(hist, axis=0).astype(np.float32)  # DIFFERENCE, h = 1
    sw.table(
        ["scenario", *[names[int(r)] for r in axis]],
        [(s, *(sw.num(x) for x in shocks[s])) for s in range(shocks.shape[0])],
    )
    ok.check(
        f"generated {generator.total_scenarios} scenarios (= T - horizon)",
        generator.total_scenarios == 4 and shocks.shape == (4, 3),
    )
    ok.check(
        f"streamed as {len(batch_facts)} batches of <= {BATCH_SIZE}, float32, F-contiguous",
        len(batch_facts) == 2 and all(batch_facts),
    )
    ok.check(
        "every factor reads its own column: identity rows, world request = axis",
        np.array_equal(binding.request_driver_rf_ids, axis)
        and np.array_equal(binding.request_chart_ids, universe_map.required_shift_type_ids),
    )
    ok.check(
        "shocks equal the hand-computed 1-day differences exactly",
        np.array_equal(shocks, oracle_shocks),
    )

    # ------------------------------------------------------------------ 6
    sw.step("STEP 6", "Construct shocked levels")
    shocked = np.ascontiguousarray(np.hstack(level_chunks))
    sw.para(
        """
        One canonical (K, S) float64 buffer of shocked levels. For a
        DIFFERENCE factor, x_s = x0 + shock; for LOG_RETURN it would be
        x0 * exp(sum of loading * shock). Every methodology reads this ONE buffer.
        """
    )
    sw.table(
        ["factor", "x0", *[f"s={s}" for s in range(4)]],
        [
            (names[int(r)], sw.num(base_levels[i]), *(sw.num(x) for x in shocked[i]))
            for i, r in enumerate(axis)
        ],
    )
    ok.check(
        "x_s = x0 + shock, exactly",
        np.array_equal(
            shocked, base_levels[:, None] + oracle_shocks.T.astype(np.float64)
        ),
    )
    ok.check("base levels were not mutated", np.array_equal(base_levels, base_before))

    # ------------------------------------------------------------------ 7
    sw.step("STEP 7", "Calculate scenario P&L (settlement currency)")
    vs_rows = np.empty((portfolio.total_trades, 4), dtype=np.float64)
    run_valuation(
        portfolio=portfolio,
        valuation_state=assemble_valuation_state(
            matrix=build_market_data_matrix_from_flat_levels(
                as_of_date=ord_to_date(int(uix_mapper.as_of_ord32)),
                uix_mapper=uix_mapper,
                universe_map=universe_map,
                flat_levels=shocked,
                num_scenarios=4,
                discounting_mode=DiscountingMode.NONE,
            ),
            uix_mapper=uix_mapper,
        ),
        out=vs_rows,
    )
    uids = [int(u) for u in portfolio.trade_uids.tolist()]
    dv = (
        (vs_rows - v0_rows).astype(np.float32).astype(np.float64)
    )  # the float32 PnlGrid

    fx_row, aapl_row, sap_row = 0, 1, 2

    oracle_vs = {
        101: 10.0 * shocked[aapl_row],
        102: -4.0 * shocked[sap_row],
        103: 8.0 * shocked[sap_row] * shocked[fx_row],
    }

    sw.para(
        """
        FULL_REVAL takes each shocked market-level column from STEP 6,
        revalues the position at those levels, and subtracts its base-world
        value:

            dV_s = V_s - V_0.

        For a single-currency SPOT position:

            dV_s = q * P_s - q * P_0.

        For position 103, the underlying SAP is EUR-quoted while the position
        settles in USD, so the full value is:

            dV_s = q * P_s * X_s - q * P_0 * X_0,

        where X is EURUSD. The tables below deliberately expand these
        equations using the signed quantity and the actual shocked levels from
        STEP 6, so the path from market shock to P&L is visible by hand.
        """
    )

    # ------------------------------ 7a
    sw.para(
        """
        Position 101: +10 AAPL, settled in USD.

        The shocked AAPL levels are read directly from STEP 6. Because the
        position is long 10 units, the P&L moves in the same direction as
        the shocked price.
        """
    )
    sw.table(
        ["scenario", "AAPL P_s from STEP 6", "full MTM calculation", "dV (USD)"],
        [
            (
                s,
                sw.num(shocked[aapl_row, s], 4),
                (
                    f"10 x {sw.num(shocked[aapl_row, s], 4)}"
                    f" - 10 x 200"
                    f" = {sw.num(oracle_vs[101][s] - oracle_v0[101], 2)}"
                ),
                sw.num(dv[0, s], 2),
            )
            for s in range(4)
        ],
        align="rlrl",
    )

    # ------------------------------ 7b
    sw.para(
        """
        Position 102: -4 SAP, settled in EUR.

        The quantity is negative. Therefore a higher shocked SAP price creates
        a negative P&L, while a lower shocked SAP price creates a positive P&L.
        """
    )
    sw.table(
        ["scenario", "SAP P_s from STEP 6", "full MTM calculation", "dV (EUR)"],
        [
            (
                s,
                sw.num(shocked[sap_row, s], 4),
                (
                    f"(-4) x {sw.num(shocked[sap_row, s], 4)}"
                    f" - [(-4) x 120]"
                    f" = {sw.num(oracle_vs[102][s] - oracle_v0[102], 2)}"
                ),
                sw.num(dv[1, s], 2),
            )
            for s in range(4)
        ],
        align="rlrl",
    )

    # ------------------------------ 7c
    sw.para(
        """
        Position 103: +8 SAP, where SAP is quoted in EUR but the position is
        settled in USD.

        Both the SAP level and EURUSD level come directly from STEP 6. The
        shocked USD value therefore contains both market moves:

            V_s = 8 * P_s * X_s.

        This is the full-revaluation MTM, not a first-order sensitivity
        approximation: the SAP and FX moves multiply each other.
        """
    )
    sw.table(
        [
            "scenario",
            "SAP P_s",
            "EURUSD X_s",
            "full MTM calculation",
            "dV (USD)",
        ],
        [
            (
                s,
                sw.num(shocked[sap_row, s], 4),
                sw.num(shocked[fx_row, s], 4),
                (
                    f"8 x {sw.num(shocked[sap_row, s], 4)}"
                    f" x {sw.num(shocked[fx_row, s], 4)}"
                    f" - 8 x 120 x 1.25"
                    f" = {sw.num(oracle_vs[103][s] - oracle_v0[103], 2)}"
                ),
                sw.num(dv[2, s], 2),
            )
            for s in range(4)
        ],
        align="rrlrr",
    )

    # ------------------------------ 7d
    sw.para(
        """
        Reconciliation: the expanded hand calculations above must reproduce
        the same float32 P&L grid used by the production run. The settlement
        currency is still the position's own currency at this point: 102 is
        EUR, while 101 and 103 are USD. STEP 8 performs the reporting-currency
        conversion.
        """
    )
    sw.table(
        ["uid", "ccy", *[f"dV s={s}" for s in range(4)]],
        [
            (
                u,
                positions.set_index("uid").loc[u, "settlement_ccy"],
                *(sw.num(x, 2) for x in dv[i]),
            )
            for i, u in enumerate(uids)
        ],
        align="rlrrrr",
    )

    ok.check(
        "V_s equals q * P_s (* EURUSD_s) exactly for every position and scenario",
        all(np.array_equal(vs_rows[i], oracle_vs[u]) for i, u in enumerate(uids)),
    )
    ok.check(
        "full MTM hand calculations produce the displayed dV exactly",
        np.array_equal(
            dv,
            np.asarray(
                [
                    [oracle_vs[101][s] - oracle_v0[101] for s in range(4)],
                    [oracle_vs[102][s] - oracle_v0[102] for s in range(4)],
                    [oracle_vs[103][s] - oracle_v0[103] for s in range(4)],
                ],
                dtype=np.float64,
            ),
        ),
    )

    # ------------------------------------------------------------------ 8
    sw.step("STEP 8", "Convert to the reporting currency (D081)")
    # The caller chooses ONLY the reporting currency (D085). The run derives
    # each position's settlement currency and FX route itself; this step
    # builds the same plan by hand, from the trade record, to show it.
    reporting = ReportingSpec(reporting_ccy="USD")
    settlement_ccys = [
        str(positions.set_index("uid").loc[u, "settlement_ccy"]) for u in uids
    ]
    plan = build_fx_conversion_plan(
        settlement_ccys=settlement_ccys,
        reporting_ccy=reporting.reporting_ccy,
        fx_factor_names=["EURUSD"],
    )
    converted = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=np.ascontiguousarray(dv),
        fx_levels=np.ascontiguousarray(shocked[[fx_row]]),
        plan=plan,
        base_settlement_values=np.array([v0[u] for u in uids], dtype=np.float64),
        base_fx_levels=base_levels[[fx_row]].copy(),
    )
    sw.para(
        """
        Reporting value is W = V * q, with q the settlement -> reporting rate.
        Production reporting P&L is ordinary mark-to-market (D081):
        """
    )
    sw.formula(
        "Let dV = V_s - V_0. Then V_s = V_0 + dV",
    )
    sw.formula(
        "dW = V_s * q_s - V_0 * q_0",
        "   = (V_0 + dV) * q_s - V_0 * q_0",
        "   = dV * q_s + V_0 * q_s - V_0 * q_0",
        "   = dV * q_s + V_0 * (q_s - q_0)",
    )
    sw.para(
        """
        For the EUR-settled 102, q = EURUSD (direct pair, no inversion). For
        USD-settled positions q = 1 exactly. The second form shows why V_0 is
        a required input: the base value itself is re-translated when FX
        moves. Portfolio P&L is summed ONLY after every row is in USD.
        """
    )
    q0 = {101: 1.0, 102: 1.25, 103: 1.0}
    qs = {101: np.ones(4), 102: shocked[fx_row], 103: np.ones(4)}
    oracle_dw = np.vstack([oracle_vs[u] * qs[u] - oracle_v0[u] * q0[u] for u in uids])
    sw.table(
        ["uid", *[f"dW s={s}" for s in range(4)]],
        [
            (u, *(sw.num(x, 2) for x in converted.converted_pnl[i]))
            for i, u in enumerate(uids)
        ]
        + [("TOTAL", *(sw.num(x, 2) for x in converted.portfolio_pnl))],
    )
    ok.check(
        "dW equals V_s*q_s - V_0*q_0 (hand oracle) exactly",
        np.array_equal(converted.converted_pnl, oracle_dw),
    )
    ok.check(
        "portfolio P&L is the column sum of converted rows",
        np.array_equal(converted.portfolio_pnl, oracle_dw.sum(axis=0)),
    )

    # ------------------------------------------------------------------ 9
    sw.step("STEP 9", "Calculate VaR and ES")
    groups = np.ascontiguousarray(
        np.vstack([converted.converted_pnl, converted.portfolio_pnl[None, :]])
    )
    metrics = compute_var_es(groups, CONFIDENCE)
    sw.para(
        f"""
        VaR is the ceil(m)-th worst scenario and ES averages the worst m
        scenarios, with m = (1 - C) * S = (1 - {CONFIDENCE}) * 4 = 1. Both are
        reported as P&L: a loss is NEGATIVE. VaR and ES are computed per
        position and for the portfolio, all in USD.
        """
    )
    labels = [str(u) for u in uids] + ["PORTFOLIO"]
    sw.table(
        ["group", "VaR (USD)", "ES (USD)", "worst scenario (hand)"],
        [
            (
                labels[g],
                sw.num(metrics.var[g], 2),
                sw.num(metrics.es[g], 2),
                sw.num(groups[g].min(), 2),
            )
            for g in range(4)
        ],
    )
    ok.check(
        "VaR = ES = worst scenario for every group (m = 1)",
        np.array_equal(metrics.var, groups.min(axis=1))
        and np.array_equal(metrics.es, groups.min(axis=1)),
    )
    ok.check(
        "portfolio VaR is less severe than the sum of position VaRs "
        "(worst cases fall in different scenarios)",
        metrics.var[-1] == -71.0 and metrics.var[-1] > metrics.var[:-1].sum(),
        f"portfolio {sw.num(metrics.var[-1], 2)} vs sum {sw.num(metrics.var[:-1].sum(), 2)}",
    )

    # ------------------------------------------------------------------ 10
    sw.step("STEP 10", "The production call and the RiskRun")
    sw.para(
        """
        Steps 3-9 exist to teach. A real integration makes ONE call. It
        receives the typed inputs, checks the composition contract, builds
        the run identity BEFORE any numerics, values V0 itself from the
        ValuationContext, and returns an immutable RiskRun.
        """
    )
    run_axis = derive_run_factor_axis(
        valuation_context=ValuationContext(
            static_config=rsc,
            ingest_batch=ingest_batch,
            uix_mapper=uix_mapper,
            discounting_mode=DiscountingMode.NONE,
        ),
        reporting_ccy=reporting.reporting_ccy,
    )
    sw.para(
        """
        The run owns ONE factor axis (D082, D085): the valuation universe,
        then any FX factor that only the reporting conversion needs. Here the
        quanto 103 already reads EURUSD, so USD reporting adds no row and the
        run axis is the STEP 3 axis. `derive_run_factor_axis` is how a caller
        binds its scenario data to that axis; the run refuses any other.
        """
    )
    ok.check(
        "the run axis is the valuation universe here (no reporting-only FX row)",
        np.array_equal(run_axis.factor_ids, axis)
        and run_axis.reporting_factor_ids.size == 0,
    )
    run: RiskRun = run_scenario_var_es(
        scenario_config=scenario_config,
        data=history,
        required_factor_ids=axis,
        required_shift_type_id=universe_map.required_shift_type_ids,
        proxy_terms=proxy_terms,
        base_levels=base_levels,
        methodology=full_reval,
        valuation_context=ValuationContext(
            static_config=rsc,
            ingest_batch=ingest_batch,
            uix_mapper=uix_mapper,
            discounting_mode=DiscountingMode.NONE,
        ),
        reporting=reporting,
        confidence_level=CONFIDENCE,
        batch_size=BATCH_SIZE,
    )
    m = run.manifest
    sw.kv([
        ("position_uids", run.position_uids.tolist()),
        ("var", run.var.tolist()),
        ("es", run.es.tolist()),
        (
            "portfolio VaR / ES",
            f"{sw.num(run.portfolio_var, 2)} / {sw.num(run.portfolio_es, 2)} {run.reporting_ccy}",
        ),
        ("reproducible", run.reproducible),
        ("identity", run.identity[:32] + "..."),
        ("manifest.scenario_config", m.scenario_config),
        (
            "manifest.risk_measure",
            f"confidence_level={m.risk_measure.confidence_level}",
        ),
        ("manifest.reporting", f"currency={m.reporting.reporting_currency}"),
        (
            "manifest.engine",
            f"riskquant {m.engine.riskquant_version}, numba {m.engine.numba_version}",
        ),
    ])
    ok.check("RiskRun VaR equals STEP 9 bitwise", np.array_equal(run.var, metrics.var))
    ok.check("RiskRun ES equals STEP 9 bitwise", np.array_equal(run.es, metrics.es))
    ok.check("identity is the manifest's content hash", run.identity == m.identity())
    restored = RiskRun.from_json(run.to_json())
    ok.check(
        "to_json / from_json round trip is bitwise (IEEE-754 hex)",
        np.array_equal(restored.var, run.var)
        and np.array_equal(restored.es, run.es)
        and restored.identity == run.identity,
    )
    ok.check(
        "no TAYLOR block in this run, so no Taylor provenance stamp",
        run.pnl_provenance is None,
    )

    sw.para(
        f"""
        Next: 01 replaces full revaluation with a certified Taylor book; 03
        covers FX legs and the reporting currency. Wall time of this
        chapter: {time.perf_counter() - t0:.2f} s.
        """
    )
    sw.write_evidence(
        "00_spot_getting_started",
        {"var": run.var, "es": run.es, "identity": run.identity},
    )
    ok.done()


if __name__ == "__main__":
    main()
