# scripts/demos/e2e/04_spot_scenario_models.py
"""
RiskQuant SPOT -- chapter 04: one valuation world, five scenario models.

The SAME realistic book (120 SPOT positions, 3 currencies, a proxied
factor) and the SAME valuation context run through every certified model:

    historical          deterministic 1-day overlapping windows of history
    filtered_historical deterministic; AR(1)/EWMA-filtered returns rescaled
                        to TODAY's volatility
    sbb                 stationary block bootstrap of history (seeded)
    filtered_bootstrap  bootstrap of the filtered scenarios (seeded)
    parametric          seeded Monte Carlo from vols + correlation Cholesky
                        (standard normal, and Student-t with df = 4)

Historical simulation also runs at a 10-day horizon: the horizon is a model
parameter, not a separate model.

Only `scenario_config` and its construction `data` change between rows;
every other argument of `run_scenario_var_es` is identical. The table is a
DESCRIPTION of how each methodology shapes the P&L distribution. It is not a
ranking and declares no winner.

Distribution statistics come from the anatomy recomposition, which is
proven bitwise-equal to each RiskRun before any statistic is printed.

The model configuration is a strict, typed contract (Pydantic v2,
`extra="forbid"`): it can be parsed from an untyped mapping (JSON, Excel
named ranges), and a parameter the model does not own fails at the solution
boundary, before any run input is built (part 5).

Run:  python scripts/demos/e2e/04_spot_scenario_models.py
"""

from __future__ import annotations

import time
from typing import Any

import _spot_world as sw
import numpy as np
from pydantic import ValidationError

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.scenario_models.schemas import (
    ForecastPeriodConfig,
    FilteredBootstrapGeneratorConfig,
    FilteredHistoricalGeneratorConfig,
    HistoricalGeneratorConfig,
    ParametricGeneratorConfig,
    SbbGeneratorConfig,
    StandardNormalDistribution,
    StudentTDistribution,
)

#: D090: the parametric construction data describes one-business-day log
#: returns (effective period (1, BUSINESS_DAY)); the request states the same
#: period, and the run refuses any other.
ONE_BUSINESS_DAY = ForecastPeriodConfig(count=1, unit="BUSINESS_DAY")

CONFIDENCE: float = 0.99
BATCH_SIZE: int = 512
SEED: int = 7
PATHS: int = 2_000
#: An SBB request as it arrives from a scheduler, a JSON file or Excel.
SBB_REQUEST: dict[str, Any] = {
    "model": "sbb",
    "horizon": 1,
    "total_scenarios": 1000,
    "expected_block_size": 5.0,
    "seed": SEED,
}


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 04 scenario models")
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
        ("historical 10d", "horizon=10", HistoricalGeneratorConfig(model="historical", horizon=10), real.history),
        ("filtered_historical", "horizon=1, schema defaults",
         FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=1), real.history),
        ("sbb", f"block=5, seed={SEED}",
         SbbGeneratorConfig(model="sbb", horizon=1, total_scenarios=PATHS, expected_block_size=5.0, seed=SEED),
         real.history),
        ("filtered_bootstrap", f"seed={SEED}",
         FilteredBootstrapGeneratorConfig(model="filtered_bootstrap", horizon=1, total_scenarios=PATHS, seed=SEED),
         real.history),
        ("parametric normal", f"seed={SEED}",
         ParametricGeneratorConfig(model="parametric", forecast_horizon=ONE_BUSINESS_DAY, total_scenarios=PATHS, seed=SEED,
                                   distribution=StandardNormalDistribution(kind="standard_normal")),
         real.parametric),
        ("parametric t(4)", f"df=4, seed={SEED}",
         ParametricGeneratorConfig(model="parametric", forecast_horizon=ONE_BUSINESS_DAY, total_scenarios=PATHS, seed=SEED,
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
            proxy_terms=world.proxy_terms,
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
        Historical simulation at horizon h replays each observed h-day move
        (overlapping windows) on today's levels: the 10-day distribution is
        wider than the 1-day one.
        """
    )
    ok.check("historical: the 10-day portfolio VaR is more severe than the 1-day VaR",
             results["historical 10d"][0].portfolio_var < results["historical"][0].portfolio_var)
    sw.para(
        """
        The bootstraps resample. At horizon 1 a stationary block bootstrap
        draws whole historical scenarios, so every SBB scenario IS a
        historical scenario; only the weights (sampling frequency) differ.
        The filtered bootstrap does the same with the filtered scenarios.
        At longer horizons the bootstraps chain blocks into new multi-day
        paths.
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
        of which market moves are plausible tomorrow. Wall time for
        {len(models)} production runs plus {len(models)} anatomies:
        {time.perf_counter() - t0:.2f} s.
        """
    )

    sw.step("5", "The configuration contract")
    config = parse_scenario_model_config(SBB_REQUEST)
    sw.para(
        """
        `model` is the discriminator. Each model's config carries ONLY its
        own parameters; the parsed object is typed, and its canonical form is
        what the run manifest stores (chapter 05).
        """
    )
    sw.kv([(k, v) for k, v in config.model_dump().items()])
    ok.check("the untyped mapping parses to the typed SbbGeneratorConfig",
             type(config).__name__ == "SbbGeneratorConfig" and config.model_dump() == SBB_REQUEST)
    ok.expect_rejection(
        "a parameter that belongs to another model (decay_factor on sbb)",
        ValidationError,
        lambda: parse_scenario_model_config({**SBB_REQUEST, "decay_factor": 0.94}),
        reason="extra='forbid': a silently ignored parameter would change "
        "nothing while the caller believes it changed the model",
        stage="config parsing (Pydantic), before any run input is built",
    )
    sw.write_evidence("04_spot_scenario_models", {
        k: {"var": r.var, "es": r.es, "identity": r.identity} for k, (r, _) in results.items()
    })
    ok.done()


if __name__ == "__main__":
    main()
