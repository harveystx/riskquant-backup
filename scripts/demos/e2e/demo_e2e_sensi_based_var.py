# scripts/demos/e2e/demo_e2e_sensi_based_var.py
"""
Canonical end-to-end demo for sensitivity-based Value at Risk (VaR).

Demonstrates the CURRENT production usage model for scenario-based VaR/ES:

    human-readable inputs (static config, proxy config, positions, snapshot)
        -> validated RiskStaticConfig / ProxyTopology
        -> compiled portfolio (SoA blocks)
        -> market state -> pricing (MTM) -> trade sensitivities
        -> canonical compiled sensitivity book (SensitivityPnlCalculator)
        -> shock-universe axis (proxy-resolved once, shared by every model)
        -> scenario-model configuration (HistoricalGeneratorConfig here)
        -> riskquant.solutions.risk_run.api.run_scenario_var_es
        -> RiskRun (VaR/ES + an immutable, verifiable run identity)

`run_scenario_var_es` is the canonical production scenario-risk entry point
(D042-D044): scenario generation, proxy-resolved shock application,
sensitivity P&L, reporting-currency conversion, and VaR/ES all happen INSIDE
that one call. This demo builds its typed inputs and calls it -- it does not
re-derive scenarios, re-apply shocks, re-convert currency, or re-run VaR/ES
by hand. See `_fixtures.py` for the shared input-construction code and
`04_spot_scenario_models.py` for every other scenario model
(FilteredHistorical, SBB, FilteredBootstrap, Parametric) and a multi-day
horizon through the same call.

The multi-currency property is preserved end to end: AAPL and WTI are
USD-priced/settled; Brent is USD-priced but EUR-settled. Scenario P&L is
computed per position in its OWN settlement currency, converted to the USD
reporting currency at each scenario's OWN FX level (never a single t0 rate),
and aggregated to a portfolio total ONLY after conversion -- all inside
`run_scenario_var_es`; this demo never sums mixed-currency P&L itself.

Before the canonical call, a small VALIDATION section hand-verifies the
wiring: a 3-scenario deterministic shock set (never fed through
`run_scenario_var_es` -- there is no production model for "these exact
numbers") drives the same low-level building blocks `run_scenario_var_es`
composes internally (Proxy V1 compile, lowering, world-column binding,
`apply_linear_terms_to_base_levels`, `calculator.run`, `compute_var_es`),
and every result is checked against an
economic identity computed from FIRST PRINCIPLES, never by calling the
function under test. This proves the shock/proxy/FX/VaR wiring is correct
by hand; it is deliberately NOT presented as "how to call RiskQuant" -- that
is the canonical call afterward.
"""

from __future__ import annotations

import math

import _fixtures as fx
import numpy as np
import pandas as pd
import polars as pl

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.reporting.fx_conversion import (
    build_fx_conversion_plan,
    convert_scenario_pnl_to_reporting_currency,
)
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.shocks.axis_operator import lower_axis_operator
from riskquant.simlab.shocks.column_binding import bind_term_columns
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import RiskRun
from riskquant.solutions.run_identity.api import InputMismatchError
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig
from riskquant.solutions.sensitivity_pnl.api import FactorAxis
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_header,
    print_success_message,
    print_title,
)

# =============================================================================
# VALIDATION-ONLY: hand-authored 3-scenario shock set
# =============================================================================
# NOT a production scenario model -- no ScenarioModelConfig represents "these
# exact numbers". Exists solely to drive the low-level building blocks with a
# hand-verifiable input so the checks below can compare against an
# independent, from-first-principles oracle. LOG RETURNS (every node in this
# universe declares shift_type=LOG_RETURN): level = base * exp(shock * scale).
_VALIDATION_SHOCKS: tuple[tuple[float, ...], ...] = (
    #  USDEUR   AAPL_EQ  BRENT_CRUDE
    (0.000, 0.010, -0.020),  # scenario 0
    (-0.010, 0.000, 0.015),  # scenario 1
    (0.020, -0.015, -0.010),  # scenario 2
)

# With S=3 and alpha=1-C=1/3, tail mass m=alpha*S=1.0 exactly integral, so
# VaR is the single worst scenario and ES integrates exactly that one
# observation -- an unambiguous, hand-verifiable result. Same confidence
# level the canonical run below uses (D019): C=2/3 is calibrated so BOTH
# S=3 (this validation) and the real 9-scenario historical run give an
# exactly integral tail mass.
_VALIDATION_CONFIDENCE_LEVEL: float = 2.0 / 3.0

def _run_validation(fixture: fx.DemoFixture) -> bool:
    """Hand-verify proxy/shock/P&L/reporting-currency/VaR-ES wiring.

    Uses `fixture.shock_universe` (the run axis and the authored Proxy V1
    terms shared with the canonical call below) to reconstruct scenario
    levels for the deterministic 3-scenario shock set through the D088
    building blocks, then checks the result against economic identities
    derived independently of the Taylor kernel:

        AAPL   V = Q * S            ->  dV = Q * dS
        WTI    V = Q * S            ->  dV = Q * dS,  S(s) = S0*exp(0.95*BRENT_shock(s))
        Brent  V = Q * B * F        ->  dV = Q * (B0*dF + F0*dB + dB*dF)

    The Brent form is the EXACT increment of the bilinear payoff, so it is a
    genuine independent check of the production Taylor expansion.
    """
    print_header("VALIDATION: Hand-Verifiable Scenario/Proxy/FX/VaR Wiring")

    su = fixture.shock_universe
    rsc = fixture.rsc

    raw_shocks = np.asfortranarray(np.array(_VALIDATION_SHOCKS, dtype=np.float32))
    print_df(
        pd.DataFrame(
            np.asarray(_VALIDATION_SHOCKS),
            columns=list(fx._SCENARIO_SOURCE_NAMES),
            index=[f"scenario {s}" for s in range(len(_VALIDATION_SHOCKS))],
        ),
        title="Hand-authored validation shocks (LOG RETURNS, source factors only)",
        index=True,
        float_format="_.4f",
    )
    print("---")

    # The hand-authored shocks are a world of one-day LOG_RETURN columns, one
    # per source factor, in `_SCENARIO_SOURCE_NAMES` order: keys (d, LOG, 1).
    world_ids = np.ascontiguousarray(
        [rsc.rel_name_to_rf_id[name] for name in fx._SCENARIO_SOURCE_NAMES],
        dtype=np.int32,
    )
    world_charts = np.full(world_ids.shape[0], SHIFT_LOG_RETURN_ID, dtype=np.int8)
    operator = lower_axis_operator(
        model=compile_proxy_model(
            authored_terms=su.proxy_terms,
            static_config=rsc,
            as_of_date=fx._SNAPSHOT_AS_OF_DATE,
            run_axis_rf_ids=su.required_factor_ids,
        ),
        axis_rf_ids=su.required_factor_ids,
        axis_shift_type_ids=su.required_shift_type_id,
    )
    term_column_uix = bind_term_columns(
        term_driver_rf_ids=operator.term_driver_rf_ids,
        term_chart_ids=operator.term_chart_ids,
        requested_horizon=1,
        world_column_driver_rf_ids=world_ids,
        world_column_chart_ids=world_charts,
        world_horizon=1,
    )
    num_scenarios = int(raw_shocks.shape[0])
    shocked_levels = np.empty(
        (su.required_factor_ids.shape[0], num_scenarios), dtype=np.float64, order="C"
    )
    apply_linear_terms_to_base_levels(
        base_levels=su.base_levels,
        row_term_offsets=operator.row_term_offsets,
        term_column_uix=term_column_uix,
        term_loadings=operator.term_loadings,
        target_shift_types=operator.target_shift_type_ids,
        shocks=raw_shocks,
        out=shocked_levels,
    )

    # The Taylor book lives on the run axis's valuation prefix (D082, D085).
    k = su.num_valuation_factors
    axis = FactorAxis.from_keys(
        pl.Series("factor_id", su.required_factor_ids[:k].tolist(), dtype=pl.Int32)
    )
    market = axis.levels(
        base_levels=su.base_levels[:k], shocked_levels=shocked_levels[:k]
    )
    pnl = fixture.calculator.run(market)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, actual: float, expected: float, *, tol: float) -> None:
        passed = abs(actual - expected) <= tol
        checks.append((name, passed, f"actual={actual:.6f} expected={expected:.6f}"))

    # ---- scenario-level reconstruction + proxy semantics (economic oracle) --
    row_of = {int(rf): i for i, rf in enumerate(su.required_factor_ids)}
    usdeur_i = row_of[int(rsc.rel_name_to_rf_id["USDEUR"])]
    aapl_i = row_of[int(rsc.rel_name_to_rf_id["AAPL_EQ"])]
    brent_i = row_of[int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])]
    wti_i = row_of[int(rsc.rel_name_to_rf_id["WTI_CRUDE"])]

    f0 = float(su.base_levels[usdeur_i])
    b0 = float(su.base_levels[brent_i])
    d_aapl = shocked_levels[aapl_i] - su.base_levels[aapl_i]
    d_wti = shocked_levels[wti_i] - su.base_levels[wti_i]
    d_brent = shocked_levels[brent_i] - su.base_levels[brent_i]
    d_fx = shocked_levels[usdeur_i] - su.base_levels[usdeur_i]

    oracle_pnl = {
        1_010: 100.0 * d_aapl,
        1_030: 75.0 * d_wti,
        1_020: -50.0 * (b0 * d_fx + f0 * d_brent + d_brent * d_fx),
    }
    for uid, expected_row in oracle_pnl.items():
        produced = pnl.row_for(uid)
        for s, expected in enumerate(expected_row.tolist()):
            check(f"P&L uid={uid} scenario {s}", float(produced[s]), expected, tol=5e-2)

    check(
        "WTI base level unchanged by proxying",
        float(su.base_levels[wti_i]),
        70.00,
        tol=1e-4,
    )
    brent_shock_col = fx._SCENARIO_SOURCE_NAMES.index("BRENT_CRUDE")
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

    # ---- reporting-currency conversion (independent nested-loop oracle) -----
    settle_by_uid = fx._settlement_ccy_by_uid()
    uids = pnl.position_uids.tolist()
    settlement_pnl = np.ascontiguousarray(pnl.pnl, dtype=np.float64)
    usdeur_row = shocked_levels[usdeur_i]

    plan = build_fx_conversion_plan(
        settlement_ccys=[settle_by_uid[u] for u in uids],
        reporting_ccy=fx._REPORTING_CCY,
        fx_factor_names=("USDEUR",),
    )
    # D081: ordinary-MTM reporting P&L, dW = dV*q_s + V0*(q_s - q0), with V0 the
    # book position's base settlement value from the SAME pricing pass.
    pv_by_uid = dict(
        zip(
            fixture.pricing.trade_uids.tolist(),
            fixture.pricing.values[:, 0].tolist(),
            strict=True,
        )
    )
    base_values = np.array([pv_by_uid[int(u)] for u in uids], dtype=np.float64)
    reporting = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=settlement_pnl,
        fx_levels=usdeur_row[None, :],
        plan=plan,
        base_settlement_values=base_values,
        base_fx_levels=np.array([f0], dtype=np.float64),
    )
    oracle_converted = np.empty_like(settlement_pnl)
    oracle_portfolio = np.zeros(num_scenarios, dtype=np.float64)
    for s in range(num_scenarios):
        total = 0.0
        for i, uid in enumerate(uids):
            if settle_by_uid[uid] == fx._REPORTING_CCY:
                rate_s, rate_0 = 1.0, 1.0
            else:
                rate_s, rate_0 = 1.0 / float(usdeur_row[s]), 1.0 / f0
            # independent L0 reference: V_s*q_s - V_0*q_0 = dV*q_s + V0*(q_s - q0)
            converted = float(settlement_pnl[i, s]) * rate_s + float(base_values[i]) * (
                rate_s - rate_0
            )
            oracle_converted[i, s] = converted
            total += converted
        oracle_portfolio[s] = total

    check(
        "converted per-position P&L matches oracle",
        float(np.max(np.abs(reporting.converted_pnl - oracle_converted))),
        0.0,
        tol=1e-9,
    )
    check(
        "portfolio P&L matches oracle",
        float(np.max(np.abs(reporting.portfolio_pnl - oracle_portfolio))),
        0.0,
        tol=1e-9,
    )
    naive_sum = settlement_pnl.sum(axis=0)
    mixed_diff = float(np.max(np.abs(reporting.portfolio_pnl - naive_sum)))
    checks.append((
        "portfolio total is NOT a raw mixed-currency sum",
        mixed_diff > 1e-6,
        f"max_abs_diff_vs_naive={mixed_diff:.6f}",
    ))

    # ---- VaR/ES (independent order-statistic oracle) ------------------------
    group_matrix = np.ascontiguousarray(
        np.vstack([settlement_pnl, reporting.portfolio_pnl[None, :]]), dtype=np.float64
    )
    labels = [str(uid) for uid in uids] + ["PORTFOLIO"]
    metrics = compute_var_es(group_matrix, _VALIDATION_CONFIDENCE_LEVEL)

    alpha = 1.0 - _VALIDATION_CONFIDENCE_LEVEL
    tail_mass = alpha * num_scenarios
    nearest = round(tail_mass)
    if abs(tail_mass - nearest) <= 1e-9 * max(1.0, abs(tail_mass)):
        tail_mass = float(nearest)
    full_count = math.floor(tail_mass + 1e-12)
    frac = tail_mass - full_count
    for row, label in enumerate(labels):
        ordered = np.sort(group_matrix[row])
        var_rank = max(1, math.ceil(tail_mass))
        expected_var = float(ordered[var_rank - 1])
        tail_sum = float(ordered[:full_count].sum())
        if frac > 0.0:
            tail_sum += frac * float(ordered[full_count])
        expected_es = tail_sum / tail_mass
        check(f"VaR {label}", float(metrics.var[row]), expected_var, tol=1e-9)
        check(f"ES  {label}", float(metrics.es[row]), expected_es, tol=1e-9)

    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name:<70} | {detail}")
    print("---")
    print(f"{len(checks)} validation checks run.")
    print()

    return all_passed


# =============================================================================
# CANONICAL PRODUCTION CALL
# =============================================================================

_BATCH_SIZE_A: int = 4
_BATCH_SIZE_B: int = 64


def _run_canonical(fixture: fx.DemoFixture) -> tuple[RiskRun, bool]:
    """Call the production scenario-VaR entry point and demonstrate its
    reproducibility contract (D040-D044): identical inputs give an identical
    identity and bitwise-identical numbers regardless of batch_size;
    `reproduce=` against a matching manifest succeeds; `reproduce=` against a
    manifest whose inputs were substituted fails BEFORE any numerical work.
    """
    print_header("Scenario Model Configuration")
    scenario_config = HistoricalGeneratorConfig(model="historical", horizon=1)
    print(
        f"model             : {scenario_config.model}\n"
        f"horizon           : {scenario_config.horizon}-day\n"
        f"observation dates : {len(fx._HISTORY_DATES)} "
        f"({fx._HISTORY_DATES[0]} .. {fx._HISTORY_DATES[-1]})\n"
        f"scenarios         : {len(fx._HISTORY_DATES) - scenario_config.horizon} "
        f"(= T - horizon)\n"
        f"reporting currency: {fixture.reporting.reporting_ccy}\n"
        f"settlement ccys   : {sorted(set(fx._settlement_ccy_by_uid().values()))} "
        f"(run-derived, D085)"
    )
    print("---")

    su = fixture.shock_universe

    print_header("riskquant.solutions.risk_run.api.run_scenario_var_es")
    run_a = run_scenario_var_es(
        scenario_config=scenario_config,
        data=fixture.historical_matrix,
        required_factor_ids=su.required_factor_ids,
        required_shift_type_id=su.required_shift_type_id,
        proxy_terms=su.proxy_terms,
        base_levels=su.base_levels,
        methodology=fixture.methodology,
        valuation_context=fixture.valuation_context,
        taylor_derived=fixture.taylor_derived,
        reporting=fixture.reporting,
        confidence_level=_VALIDATION_CONFIDENCE_LEVEL,
        batch_size=_BATCH_SIZE_A,
    )

    labels = [str(uid) for uid in run_a.position_uids.tolist()] + ["PORTFOLIO"]
    print_df(
        pd.DataFrame({"VaR": run_a.var, "ES": run_a.es}, index=labels),
        title=f"VaR / ES at {_VALIDATION_CONFIDENCE_LEVEL:.4f} confidence ({run_a.reporting_ccy})",
        index=True,
        float_format="_.4f",
    )
    print(
        f"identity     : {run_a.identity}\n"
        f"reproducible : {run_a.reproducible}\n"
        f"reporting_ccy: {run_a.reporting_ccy}"
    )
    print("---")

    # ---- reproducibility: batch_size is operational, not identity (D041) ----
    run_b = run_scenario_var_es(
        scenario_config=scenario_config,
        data=fixture.historical_matrix,
        required_factor_ids=su.required_factor_ids,
        required_shift_type_id=su.required_shift_type_id,
        proxy_terms=su.proxy_terms,
        base_levels=su.base_levels,
        methodology=fixture.methodology,
        valuation_context=fixture.valuation_context,
        taylor_derived=fixture.taylor_derived,
        reporting=fixture.reporting,
        confidence_level=_VALIDATION_CONFIDENCE_LEVEL,
        batch_size=_BATCH_SIZE_B,
    )
    same_identity = run_b.identity == run_a.identity
    same_numbers = np.array_equal(run_b.var, run_a.var) and np.array_equal(
        run_b.es, run_a.es
    )

    # ---- reproducibility: reproduce= against a matching manifest succeeds ---
    run_c = run_scenario_var_es(
        scenario_config=scenario_config,
        data=fixture.historical_matrix,
        required_factor_ids=su.required_factor_ids,
        required_shift_type_id=su.required_shift_type_id,
        proxy_terms=su.proxy_terms,
        base_levels=su.base_levels,
        methodology=fixture.methodology,
        valuation_context=fixture.valuation_context,
        taylor_derived=fixture.taylor_derived,
        reporting=fixture.reporting,
        confidence_level=_VALIDATION_CONFIDENCE_LEVEL,
        batch_size=_BATCH_SIZE_A,
        reproduce=run_a.manifest,
    )
    reproduce_ok = run_c.identity == run_a.identity

    # ---- reproducibility: a substituted input is refused BEFORE execution ---
    stale_values = np.array(fixture.historical_matrix.values, copy=True)
    stale_values[-1, 1] *= 1.01  # nudge the most recent AAPL_EQ level
    stale_matrix = RiskFactorMatrix(
        dates=fixture.historical_matrix.dates,
        factor_ids=fixture.historical_matrix.factor_ids,
        values=np.asfortranarray(stale_values),
    )
    stale_refused = False
    try:
        run_scenario_var_es(
            scenario_config=scenario_config,
            data=stale_matrix,
            required_factor_ids=su.required_factor_ids,
            required_shift_type_id=su.required_shift_type_id,
            proxy_terms=su.proxy_terms,
            base_levels=su.base_levels,
            methodology=fixture.methodology,
            valuation_context=fixture.valuation_context,
            taylor_derived=fixture.taylor_derived,
            reporting=fixture.reporting,
            confidence_level=_VALIDATION_CONFIDENCE_LEVEL,
            batch_size=_BATCH_SIZE_A,
            reproduce=run_a.manifest,
        )
    except InputMismatchError:
        stale_refused = True

    print_header("Reproducibility (RunManifest / RiskRun identity)")
    repro_checks = [
        ("batch_size does not change identity (operational only)", same_identity),
        ("batch_size does not change VaR/ES (bitwise)", same_numbers),
        ("reproduce= against own manifest succeeds", reproduce_ok),
        ("stale historical data refused before execution", stale_refused),
    ]
    all_passed = True
    for name, passed in repro_checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name}")
    print("---")
    print(f"{len(repro_checks)} reproducibility checks run.")
    print()

    return run_a, all_passed


# =============================================================================
# MAIN DEMO
# =============================================================================


def main() -> None:
    """Run the canonical end-to-end sensitivity-based VaR demo."""
    print_calling_file()
    print_title("END-TO-END DEMO: SENSITIVITY-BASED VAR")

    fixture = fx.build_fixture(verbose=True)

    fixture_checks_passed = fx.run_fixture_golden_checks(
        rsc=fixture.rsc, pricing=fixture.pricing, calculator=fixture.calculator
    )
    validation_passed = _run_validation(fixture)
    run, repro_passed = _run_canonical(fixture)

    if fixture_checks_passed and validation_passed and repro_passed:
        print("ALL CHECKS PASSED")
    else:
        # A failed check must fail the process, not only the printed banner:
        # the demo is a registered regression gate (tests/e2e, PR-07).
        raise SystemExit("VALIDATION FAILED")
    print()

    print_header("E2E Summary")
    print(
        f"Portfolio compiled: {fixture.portfolio.total_trades} trade(s), "
        f"{fixture.calculator.book.active_factor_ids.shape[0]} active risk factor(s).\n"
        f"Proxy topology: WTI_CRUDE -> BRENT_CRUDE (scale=0.95), resolved once "
        f"and shared by every scenario model.\n"
        f"Scenario model demonstrated here: {run.manifest.scenario_config.get('model')} "
        f"(see 04_spot_scenario_models.py for FilteredHistorical, SBB, "
        f"FilteredBootstrap, Parametric, and a multi-day horizon).\n"
        f"Portfolio VaR: {run.portfolio_var:,.4f} {run.reporting_ccy}\n"
        f"Portfolio ES : {run.portfolio_es:,.4f} {run.reporting_ccy}\n"
        f"Run identity : {run.identity}"
    )

    print_title("END OF DEMO")

    print_success_message("DEMO RAN SUCCESSFULLY", add_space_after=False)


if __name__ == "__main__":
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    main()
