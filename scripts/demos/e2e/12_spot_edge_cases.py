# scripts/demos/e2e/12_spot_edge_cases.py
"""
RiskQuant SPOT -- chapter 12: boundary cases and FAIL FAST.

Part A runs VALID boundary cases and checks each against a hand value:
single position / single factor, a single scenario, batch_size = 1 and
batch_size > S, and an inverted-FX position under full revaluation.

Part B passes deliberately INVALID inputs to the real APIs. Each case states
what was changed, which contract rejects it, the exact exception type, and at
which stage the rejection happens. Only the exact expected exception type
counts as PASS; any other exception propagates and stops the demo.

The stage is reported as the current source places it. Composition contracts,
the confidence level and the run axis (including the length of every
axis-aligned input) reject before identity and numerics (PR-04). Checks owned
by the scenario generator (horizon, non-finite history) reject when the
generator is built, after the S = 1 base valuation and before any scenario.

Run:  python scripts/demos/e2e/12_spot_edge_cases.py
"""

from __future__ import annotations

from typing import Any

import _spot_world as sw
import numpy as np
import pandas as pd

from riskquant.exceptions import PrettyKeyError
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    ReportingSpec,
    RiskRun,
)
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig

HIST = HistoricalGeneratorConfig(model="historical", horizon=1)


def _positions(rows: list[tuple[int, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame({
        "uid": [r[0] for r in rows],
        "instrument": ["SPOT"] * len(rows),
        "quantity": [r[1] for r in rows],
        "underlying": [r[2] for r in rows],
        "settlement_ccy": [r[3] for r in rows],
    })


def _inputs(fx: sw.TinyFixture, **override: Any) -> dict[str, Any]:
    """Every argument of run_scenario_var_es for a FULL_REVAL tiny run."""
    w = fx.world
    kwargs: dict[str, Any] = dict(
        scenario_config=HIST,
        data=fx.history,
        required_factor_ids=w.required_factor_ids,
        required_shift_type_id=w.required_shift_type_id,
        source_factor_ids=w.source_factor_ids,
        shock_scale=w.shock_scale,
        base_levels=w.base_levels,
        methodology=sw.methodology(full=w.uids.tolist()),
        valuation_context=sw.valuation_context(w),
        reporting=sw.reporting_spec(w, reporting_ccy="USD"),
        confidence_level=0.75,
        batch_size=2,
    )
    kwargs.update(override)
    return kwargs


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 12 edge cases and FAIL FAST")

    # ================================================================ A
    sw.step("A1", "Single position, single factor")
    one = sw.tiny_fixture(_positions([(1, 10.0, "AAPL_EQ", "USD")]))
    r1: RiskRun = run_scenario_var_es(**_inputs(one))
    moves = np.diff(np.asarray(sw.TINY_HISTORY_VALUES)[:, 1])
    sw.kv([("run axis", one.world.axis_names()), ("AAPL 1-day moves", moves.tolist()),
           ("VaR / ES (m = 1)", f"{r1.var.tolist()} / {r1.es.tolist()}")])
    ok.check("the axis holds exactly the one factor the book reads", one.world.axis_names() == ["AAPL_EQ"])
    ok.check("no FX factor is needed when everything is USD (empty FX plan)",
             one.world.required_factor_ids.shape == (1,))
    ok.check("VaR = ES = 10 x worst move, for the position and the portfolio",
             r1.var.tolist() == [10.0 * moves.min()] * 2 and r1.es.tolist() == [10.0 * moves.min()] * 2)

    sw.step("A2", "A single scenario")
    base_world = sw.tiny_fixture().world
    short = sw.TinyFixture(
        world=base_world,
        history=sw.history_matrix(base_world.rsc, dates=sw.TINY_HISTORY_DATES[-2:],
                                  names=sw.TINY_HISTORY_NAMES, values=sw.TINY_HISTORY_VALUES[-2:]),
    )
    r_s = run_scenario_var_es(**_inputs(short, confidence_level=0.5))
    an_s = sw.scenario_pnl_anatomy(short.world, scenario_config=HIST, data=short.history,
                                   method=sw.methodology(full=short.world.uids.tolist()),
                                   taylor_derived=None,
                                   reporting=sw.reporting_spec(short.world, reporting_ccy="USD"),
                                   confidence_level=0.5, batch_size=2)
    sw.para(
        """
        T = 2 observations, horizon 1: S = 1. With C = 0.5 the tail mass is
        m = 0.5; ES integrates half an observation with fractional weight,
        so ES equals the single scenario, as does VaR.
        """
    )
    ok.check("anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(an_s, r_s))
    ok.check("S = 1 and VaR = ES = that scenario's P&L for every group",
             an_s.num_scenarios == 1 and np.array_equal(r_s.var, r_s.es)
             and r_s.portfolio_var == float(an_s.reporting.portfolio_pnl[0]))

    sw.step("A3", "batch_size = 1 and batch_size > S")
    tiny = sw.tiny_fixture()
    ref = run_scenario_var_es(**_inputs(tiny, batch_size=4))
    for b in (1, 3, 5, 1_000):
        r = run_scenario_var_es(**_inputs(tiny, batch_size=b))
        ok.check(f"batch_size={b} (S=4): same identity, bitwise same VaR/ES",
                 r.identity == ref.identity and np.array_equal(r.var, ref.var) and np.array_equal(r.es, ref.es))

    sw.step("A4", "Inverted FX leg: FULL_REVAL only (D084)")
    inv = sw.tiny_fixture(_positions([(1, 8.0, "AAPL_EQ", "EUR")]))
    sw.para(
        """
        8 AAPL (USD) settled in EUR: V = 8 x AAPL / EURUSD. Only EURUSD is
        quoted, so the FX leg is INVERTED. Full revaluation prices it. The
        closed-form SPOT sensitizer refuses inverted legs, so no TAYLOR/DERIVED
        book exists for it; the run also refuses such a leg in U_D (D084).
        """
    )
    r_inv = run_scenario_var_es(**_inputs(inv))
    ok.check("V0 = 8 x 200 / 1.25 = 1280 EUR", inv.world.v0_by_uid[1] == 1280.0)
    ok.check("the FULL_REVAL run completes with finite VaR/ES",
             bool(np.isfinite(r_inv.var).all() and np.isfinite(r_inv.es).all()))
    ok.expect_rejection(
        "TAYLOR/DERIVED derivation for an inverted FX leg",
        NotImplementedError,
        lambda: sw.certified_taylor_block(inv.world, [1]),
        reason="PV = q S / X has different partials; the branch is not authored, "
        "so it is refused rather than approximated",
        stage="run_sensitivities (book construction), before any run",
    )

    # ================================================================ B
    sw.step("B1", "Ingest boundary")
    ok.expect_rejection(
        "a position with zero quantity",
        ValueError,
        lambda: sw.tiny_fixture(_positions([(1, 0.0, "AAPL_EQ", "USD")])),
        reason="zero / dust quantities are refused at ingest; a zero row is "
        "never carried silently into risk",
        stage="compile_raw_df_into_ingest_batch",
    )
    ok.expect_rejection(
        "a duplicated position uid",
        ValueError,
        lambda: sw.tiny_fixture(_positions([(1, 1.0, "AAPL_EQ", "USD"), (1, 2.0, "AAPL_EQ", "USD")])),
        reason="a uid is the economic identity of ONE position",
        stage="compile_raw_df_into_ingest_batch",
    )
    ok.expect_rejection(
        "an underlying name unknown to the static config",
        PrettyKeyError,
        lambda: sw.tiny_fixture(_positions([(1, 1.0, "NOPE_EQ", "USD")])),
        reason="names die at the edge; an unresolved name has no rf_id",
        stage="compile_raw_df_into_ingest_batch",
    )
    ok.expect_rejection(
        "a settlement currency absent from the static config",
        PrettyKeyError,
        lambda: sw.tiny_fixture(_positions([(1, 1.0, "AAPL_EQ", "CHF")])),
        reason="every currency must be declared once, centrally",
        stage="compile_raw_df_into_ingest_batch",
    )
    ok.expect_rejection(
        "a level snapshot missing a factor the book needs",
        ValueError,
        lambda: sw.tiny_fixture(levels={"EURUSD": 1.25, "AAPL_EQ": 200.0}),
        reason="a missing base level is never filled or defaulted",
        stage="market-data assembly (build_market_data_matrix_from_snapshot)",
    )

    sw.step("B2", "Run composition and factor-axis contracts")
    w = tiny.world
    bad_shift = w.required_shift_type_id.copy()
    bad_shift[w.row("AAPL_EQ")] = np.int8(1)  # LOG_RETURN instead of DIFFERENCE
    ok.expect_rejection(
        "invalid shift type on one axis row (LOG_RETURN for a DIFFERENCE factor)",
        CompositionContractError,
        lambda: run_scenario_var_es(**_inputs(tiny, required_shift_type_id=bad_shift)),
        reason="the shock axis must equal the run axis row for row; the "
        "wrong shift type would reconstruct wrong levels",
        stage="composition contract, before identity and numerics",
    )
    ok.expect_rejection(
        "a permuted factor axis",
        CompositionContractError,
        lambda: run_scenario_var_es(**_inputs(tiny, required_factor_ids=w.required_factor_ids[::-1].copy())),
        reason="a permuted float row is indistinguishable from the right one; "
        "order equality is required, not set equality",
        stage="composition contract, before identity and numerics",
    )
    ok.expect_rejection(
        "U omits a position the portfolio values",
        CompositionContractError,
        lambda: run_scenario_var_es(**_inputs(
            tiny, methodology=sw.methodology(full=[101, 102]),
            reporting=sw.reporting_spec(w))),
        reason="valuation rows are never dropped and absent positions never zero-filled",
        stage="composition contract, before identity and numerics",
    )
    ok.expect_rejection(
        "caller-supplied settlement currencies",
        TypeError,
        lambda: ReportingSpec(reporting_ccy="USD", settlement_ccys=("USD", "EUR", "USD")),
        reason="the run derives each position's settlement currency from the trade "
        "record; a second, caller-owned copy could contradict it (D085)",
        stage="ReportingSpec construction",
    )
    ok.expect_rejection(
        "an empty production universe",
        CompositionContractError,
        lambda: MethodologyDeclaration(
            production_position_uids=np.empty(0, dtype=np.int64), full_reval_uids=np.empty(0, dtype=np.int64),
            taylor_supplied_uids=np.empty(0, dtype=np.int64), taylor_derived_uids=np.empty(0, dtype=np.int64)),
        reason="a production run needs at least one economic position",
        stage="MethodologyDeclaration construction",
    )
    ok.expect_rejection(
        "float position uids",
        CompositionContractError,
        lambda: MethodologyDeclaration(
            production_position_uids=np.array([101.0]), full_reval_uids=np.array([101.0]),
            taylor_supplied_uids=np.empty(0, dtype=np.int64), taylor_derived_uids=np.empty(0, dtype=np.int64)),
        reason="a cast would truncate silently; identities are integers",
        stage="MethodologyDeclaration construction",
    )
    ok.expect_rejection(
        "a malformed reporting currency code",
        ValueError,
        lambda: ReportingSpec(reporting_ccy="US"),
        reason="currencies are 3 uppercase letters; nothing is coerced",
        stage="ReportingSpec construction",
    )

    sw.step("B3", "Operational and scenario parameters")
    ok.expect_rejection(
        "batch_size = 0",
        ValueError,
        lambda: run_scenario_var_es(**_inputs(tiny, batch_size=0)),
        reason="a non-positive batch cannot partition the scenario stream",
        stage="first check of run_scenario_var_es",
    )
    ok.expect_rejection(
        "horizon >= number of observations",
        ValueError,
        lambda: run_scenario_var_es(**_inputs(tiny,
                                              scenario_config=HistoricalGeneratorConfig(model="historical", horizon=5))),
        reason="T - horizon must leave at least one scenario",
        stage="generator construction, after the base valuation",
    )
    ok.expect_rejection(
        "a non-finite value in the history",
        ValueError,
        lambda: run_scenario_var_es(**_inputs(tiny, data=sw.history_matrix(
            w.rsc, dates=sw.TINY_HISTORY_DATES, names=sw.TINY_HISTORY_NAMES,
            values=[sw.TINY_HISTORY_VALUES[0], (1.3125, float("nan"), 120.0), *sw.TINY_HISTORY_VALUES[2:]]))),
        reason="every shift type requires finite level observations",
        stage="scenario generation (shift-domain validation)",
    )
    ok.expect_rejection(
        "confidence_level = 1.0",
        ValueError,
        lambda: run_scenario_var_es(**_inputs(tiny, confidence_level=1.0)),
        reason="VaR needs 0 < C < 1",
        stage="first checks of run_scenario_var_es, before identity (PR-04)",
    )
    ok.expect_rejection(
        "a base-level vector shorter than the axis",
        CompositionContractError,
        lambda: run_scenario_var_es(**_inputs(tiny, base_levels=w.base_levels[:2].copy())),
        reason="every axis-aligned input has one entry per run-axis row",
        stage="composition (run-axis gate), before identity and numerics",
    )
    ok.done()


if __name__ == "__main__":
    main()
