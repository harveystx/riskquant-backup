# scripts/demos/e2e/08_spot_model_comparison.py
"""
RiskQuant SPOT -- chapter 08: one valuation world, five scenario models.

The SAME realistic book (120 SPOT positions, 3 currencies, a proxied
factor) and the SAME valuation context run through every certified model:

    historical          deterministic 1-day overlapping windows of history
    filtered_historical deterministic; AR(1)/EWMA-filtered returns rescaled
                        to TODAY's volatility
    sbb                 stationary block bootstrap of history (seeded)
    filtered_bootstrap  bootstrap of the filtered scenarios (seeded)
    parametric          seeded Monte Carlo from vols + correlation Cholesky
                        (standard normal, and Student-t with df = 4)

Only `scenario_config` and its construction `data` change between rows;
every other argument of `run_scenario_var_es` is identical. The table is a
DESCRIPTION of how each methodology shapes the P&L distribution. It is not a
ranking and declares no winner.

Distribution statistics come from the anatomy recomposition, which is
proven bitwise-equal to each RiskRun before any statistic is printed.

Run:  python scripts/demos/e2e/08_spot_model_comparison.py
"""

from __future__ import annotations

import time

import _spot_world as sw
import numpy as np

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.scenario_models.schemas import (
    FilteredBootstrapGeneratorConfig,
    FilteredHistoricalGeneratorConfig,
    HistoricalGeneratorConfig,
    ParametricGeneratorConfig,
    SbbGeneratorConfig,
    StandardNormalDistribution,
    StudentTDistribution,
)

CONFIDENCE: float = 0.99
BATCH_SIZE: int = 512
SEED: int = 7
PATHS: int = 2_000


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 08 scenario-model comparison")
    t0 = time.perf_counter()
    real = sw.realistic_fixture()
    world = real.world
    method = sw.methodology(full=world.uids.tolist())
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    context = sw.valuation_context(world)

    sw.step("1", "The fixed valuation world")
    sw.kv([
        ("positions", f"{world.uids.shape[0]} SPOT, all FULL_REVAL"),
        ("run axis", f"{world.required_factor_ids.shape[0]} factors, LOG_RETURN"),
        ("history", f"{real.num_observations} business days (matrix data)"),
        ("parametric data", f"{real.parametric.factor_ids.shape[0]} factors: sample vols + corr Cholesky"),
        ("reporting", "USD (EURUSD, GBPUSD)"),
        ("confidence", CONFIDENCE),
    ])

    models = [
        ("historical", "horizon=1", HistoricalGeneratorConfig(model="historical", horizon=1), real.history),
        ("filtered_historical", "horizon=1, schema defaults",
         FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=1), real.history),
        ("sbb", f"block=5, seed={SEED}",
         SbbGeneratorConfig(model="sbb", horizon=1, total_scenarios=PATHS, expected_block_size=5.0, seed=SEED),
         real.history),
        ("filtered_bootstrap", f"seed={SEED}",
         FilteredBootstrapGeneratorConfig(model="filtered_bootstrap", horizon=1, total_scenarios=PATHS, seed=SEED),
         real.history),
        ("parametric normal", f"seed={SEED}",
         ParametricGeneratorConfig(model="parametric", total_scenarios=PATHS, seed=SEED,
                                   distribution=StandardNormalDistribution(kind="standard_normal")),
         real.parametric),
        ("parametric t(4)", f"df=4, seed={SEED}",
         ParametricGeneratorConfig(model="parametric", total_scenarios=PATHS, seed=SEED,
                                   distribution=StudentTDistribution(kind="student_t", df=4.0)),
         real.parametric),
    ]

    sw.step("2", "Run every model through the same production call")
    results = {}
    for label, _, config, data in models:
        run = run_scenario_var_es(
            scenario_config=config,  # <- the only argument that varies (with data)
            data=data,
            required_factor_ids=world.required_factor_ids,
            required_shift_type_id=world.required_shift_type_id,
            source_factor_ids=world.source_factor_ids,
            shock_scale=world.shock_scale,
            base_levels=world.base_levels,
            methodology=method,
            valuation_context=context,
            reporting=reporting,
            confidence_level=CONFIDENCE,
            batch_size=BATCH_SIZE,
        )
        anatomy = sw.scenario_pnl_anatomy(world, scenario_config=config, data=data, method=method,
                                          taylor_derived=None, reporting=reporting,
                                          confidence_level=CONFIDENCE, batch_size=BATCH_SIZE)
        ok.check(f"{label}: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(anatomy, run))
        results[label] = (run, anatomy)
    ok.check("every model produced a distinct run identity",
             len({r.identity for r, _ in results.values()}) == len(models))

    sw.step("3", "Comparison table (USD, portfolio)")
    sw.table(["model", "key configuration", "scenarios S"],
             [(label, cfg_text, f"{results[label][1].num_scenarios:,}") for label, cfg_text, _, _ in models],
             align="llr")
    rows = []
    for label, _, _, _ in models:
        run, an = results[label]
        p = an.reporting.portfolio_pnl
        rows.append((label, sw.num(run.portfolio_var, 0), sw.num(run.portfolio_es, 0),
                     sw.num(p.std(ddof=1), 0), sw.num(p.min(), 0)))
    sw.table(["model", "VaR 99%", "ES 99%", "stdev", "worst"], rows, align="lrrrr")
    for label, _, _, _ in models:
        run, an = results[label]
        v, e, _ = sw.oracle_var_es(an.reporting.portfolio_pnl, CONFIDENCE)
        ok.check(f"{label}: VaR = independent full-sort order statistic, exactly; "
                 "ES within the float64 summation-order bound",
                 run.portfolio_var == v
                 and abs(run.portfolio_es - e) <= sw.es_summation_bound(an.reporting.portfolio_pnl, CONFIDENCE))

    sw.step("4", "Why the numbers differ")
    hist = results["historical"][1].reporting.portfolio_pnl
    fhs = results["filtered_historical"][1].reporting.portfolio_pnl
    sbb = results["sbb"][1].reporting.portfolio_pnl
    fb = results["filtered_bootstrap"][1].reporting.portfolio_pnl
    sw.para(
        """
        Historical replays each observed 1-day move on today's levels, with
        equal weight. Filtered historical first removes AR(1) drift and
        divides each return by its EWMA volatility at the time, then rescales
        by today's EWMA volatility: calm and stressed periods are expressed
        in today's volatility regime, capped by max_vol_ratio.
        """
    )
    sw.para(
        """
        The bootstraps resample. At horizon 1 a stationary block bootstrap
        draws whole historical scenarios, so every SBB scenario IS a
        historical scenario; only the weights (sampling frequency) differ.
        The filtered bootstrap does the same with the filtered scenarios.
        Longer horizons (chapter 09) chain blocks into new multi-day paths.
        """
    )
    ok.check("horizon 1: every SBB portfolio scenario is a historical one (bitwise)",
             bool(np.isin(sbb, hist).all()))
    ok.check("horizon 1: every filtered-bootstrap scenario is a filtered-historical one",
             bool(np.isin(fb, fhs).all()))
    sw.para(
        """
        Parametric draws new joint moves from the sample volatilities and
        correlations: no observed scenario is replayed, and the tail shape
        is the chosen distribution's. The Student-t variant uses the same
        vols and correlations with heavier tails.
        """
    )
    pn = results["parametric normal"][1].reporting.portfolio_pnl
    pt = results["parametric t(4)"][1].reporting.portfolio_pnl
    ok.check("parametric scenarios are not replays of history",
             not bool(np.isin(pn, hist).any()))
    kurt = lambda x: float(((x - x.mean()) ** 4).mean() / x.var() ** 2)  # noqa: E731
    sw.kv([("excess kurtosis, parametric normal", f"{kurt(pn) - 3.0:.3f}"),
           ("excess kurtosis, parametric t(4)", f"{kurt(pt) - 3.0:.3f}")])
    ok.check("the Student-t P&L distribution has heavier tails than the normal (sample kurtosis)",
             kurt(pt) > kurt(pn))
    sw.para(
        f"""
        None of these is "the right answer". Each encodes a different view
        of which market moves are plausible tomorrow. Wall time for six
        production runs plus six anatomies: {time.perf_counter() - t0:.2f} s.
        """
    )
    sw.write_evidence("08_spot_model_comparison", {
        k: {"var": r.var, "es": r.es, "identity": r.identity} for k, (r, _) in results.items()
    })
    ok.done()


if __name__ == "__main__":
    main()
