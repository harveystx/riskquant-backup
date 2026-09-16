# scripts/demos/e2e/demo_e2e_sensi_based_var_models.py
"""
Scenario-model coverage for sensitivity-based VaR.

Reuses the SAME fixture (`_fixtures.build_fixture`) as
`demo_e2e_sensi_based_var.py` -- same portfolio, same market state, same
compiled sensitivity book, same proxy-resolved shock-universe axis -- and
drives it through every other production scenario model via the SAME
canonical entry point, `riskquant.solutions.risk_run.api.run_scenario_var_es`:

    Historical        (already the canonical demo's primary narrative;
                        repeated here at a second, multi-day horizon)
    FilteredHistorical (AR(1)/EWMA-filtered historical, deterministic --
                        same RiskFactorMatrix, no seed, same proxy path)
    SBB               (Stationary Block Bootstrap, seeded)
    FilteredBootstrap (AR(1)/EWMA-filtered bootstrap, seeded)
    Parametric        (seeded Monte-Carlo, no horizon concept)

Every row below exercises the FULL production contract identically:
proxying (WTI_CRUDE -> BRENT_CRUDE), multi-currency settlement (USD + EUR),
USD reporting-currency conversion, and a verifiable RunManifest/RiskRun
identity. What varies is only the scenario-model configuration passed to
`run_scenario_var_es` -- no per-model demo file, no duplicated fixture.

Because these models are largely stochastic, no hand-computed golden P&L
values are checked here (see `demo_e2e_sensi_based_var.py` for that, on the
deterministic 3-scenario case). Instead this file checks Layer-1,
input-agnostic STRUCTURAL invariants that must hold for every model: finite
values, one VaR/ES pair per position plus the portfolio, and ES at least as
severe as VaR (ES averages the worst tail observations VaR only bounds).
"""

from __future__ import annotations

from typing import Literal

import _fixtures as fx
import numpy as np
import pandas as pd

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import NonReproducibleRunError, RiskRun
from riskquant.solutions.scenario_models.schemas import (
    FilteredBootstrapGeneratorConfig,
    FilteredHistoricalGeneratorConfig,
    HistoricalGeneratorConfig,
    ParametricGeneratorConfig,
    SbbGeneratorConfig,
    ScenarioModelConfig,
    StandardNormalDistribution,
)
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_header,
    print_success_message,
    print_title,
)

_CONFIDENCE_LEVEL: float = 2.0 / 3.0
_BATCH_SIZE: int = 32
_SEED: int = 7

#: Which shared construction-data object a scenario-model config's family
#: requires: "matrix" -> `fixture.historical_matrix` (historical-family
#: models), "parametric" -> `fixture.parametric_data` (the parametric model).
DataKind = Literal["matrix", "parametric"]


def _coverage_configs() -> list[tuple[str, ScenarioModelConfig, DataKind]]:
    """(label, scenario_config, data_kind) for every model/horizon this file
    demonstrates. `data_kind` selects the construction-data object the
    config's model family requires -- built once in the shared fixture."""
    return [
        (
            "historical (1-day)",
            HistoricalGeneratorConfig(model="historical", horizon=1),
            "matrix",
        ),
        (
            "historical (3-day)",
            HistoricalGeneratorConfig(model="historical", horizon=3),
            "matrix",
        ),
        (
            "filtered_historical (1-day)",
            # decay_factor / max_vol_ratio left at their schema defaults
            # (EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO) -- the production
            # contract, not a hand-picked value.
            FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=1),
            "matrix",
        ),
        (
            "sbb (1-day)",
            SbbGeneratorConfig(
                model="sbb",
                horizon=1,
                total_scenarios=256,
                expected_block_size=3.0,
                seed=_SEED,
            ),
            "matrix",
        ),
        (
            "filtered_bootstrap (1-day)",
            FilteredBootstrapGeneratorConfig(
                model="filtered_bootstrap",
                horizon=1,
                total_scenarios=256,
                seed=_SEED,
            ),
            "matrix",
        ),
        (
            "parametric",
            ParametricGeneratorConfig(
                model="parametric",
                total_scenarios=256,
                seed=_SEED,
                distribution=StandardNormalDistribution(kind="standard_normal"),
            ),
            "parametric",
        ),
    ]


def _run_one(
    fixture: fx.DemoFixture, scenario_config: ScenarioModelConfig, data_kind: DataKind
) -> RiskRun:
    su = fixture.shock_universe
    data = (
        fixture.historical_matrix if data_kind == "matrix" else fixture.parametric_data
    )
    return run_scenario_var_es(
        scenario_config=scenario_config,
        data=data,
        required_factor_ids=su.required_factor_ids,
        required_shift_type_id=su.required_shift_type_id,
        source_factor_ids=su.source_factor_ids,
        shock_scale=su.shock_scale,
        base_levels=su.base_levels,
        calculator=fixture.calculator,
        reporting=fixture.reporting,
        confidence_level=_CONFIDENCE_LEVEL,
        batch_size=_BATCH_SIZE,
    )


def _run_coverage_sweep(fixture: fx.DemoFixture) -> bool:
    print_header("Scenario-Model Coverage")

    rows: list[dict[str, object]] = []
    all_passed = True
    for label, scenario_config, data_kind in _coverage_configs():
        run = _run_one(fixture, scenario_config, data_kind)

        # ---- Layer-1 structural invariants (input-agnostic, every model) ----
        n_expected = fixture.calculator.book.position_uids.shape[0] + 1
        checks = [
            run.var.shape == (n_expected,),
            run.es.shape == (n_expected,),
            bool(np.all(np.isfinite(run.var))),
            bool(np.all(np.isfinite(run.es))),
            bool(np.all(run.es <= run.var + 1e-9)),
            run.reporting_ccy == "USD",
            run.identity == run.manifest.identity(),
        ]
        passed = all(checks)
        all_passed &= passed

        rows.append({
            "model": label,
            "portfolio_var": run.portfolio_var,
            "portfolio_es": run.portfolio_es,
            "reproducible": run.reproducible,
            "identity[:12]": run.identity[:12],
            "structural checks": "PASS" if passed else "FAIL",
        })

    print_df(
        pd.DataFrame(rows).set_index("model"),
        title=f"VaR / ES at {_CONFIDENCE_LEVEL:.4f} confidence, USD reporting",
        index=True,
        float_format="_.4f",
    )
    print("---")
    print(f"{len(rows)} model/horizon combinations run.")
    print()

    return all_passed


def _run_seeded_reproducibility(fixture: fx.DemoFixture) -> bool:
    """A stochastic model with the SAME seed reproduces its identity and
    numbers exactly; a different seed changes both."""
    print_header("Seeded-Model Reproducibility (SBB)")

    def _sbb(seed: int | None) -> SbbGeneratorConfig:
        return SbbGeneratorConfig(
            model="sbb",
            horizon=1,
            total_scenarios=128,
            expected_block_size=3.0,
            seed=seed,
        )

    a = _run_one(fixture, _sbb(_SEED), "matrix")
    b = _run_one(fixture, _sbb(_SEED), "matrix")
    c = _run_one(fixture, _sbb(_SEED + 1), "matrix")

    same_seed_matches = a.identity == b.identity and np.array_equal(a.var, b.var)
    different_seed_diverges = a.identity != c.identity

    checks = [
        ("same seed -> same identity and numbers", same_seed_matches),
        ("different seed -> different identity", different_seed_diverges),
    ]
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name}")
    print("---")
    print()
    return all_passed


def _run_seed_policy(fixture: fx.DemoFixture) -> bool:
    """`mode="reproducible"` (the default) refuses a stochastic model with
    `seed=None` before any numerical work; `mode="exploratory"` permits it
    and marks the result `reproducible=False`."""
    print_header("Seed Policy (reproducible vs exploratory)")

    su = fixture.shock_universe
    no_seed_config = SbbGeneratorConfig(
        model="sbb", horizon=1, total_scenarios=64, expected_block_size=3.0, seed=None
    )

    refused = False
    try:
        run_scenario_var_es(
            scenario_config=no_seed_config,
            data=fixture.historical_matrix,
            required_factor_ids=su.required_factor_ids,
            required_shift_type_id=su.required_shift_type_id,
            source_factor_ids=su.source_factor_ids,
            shock_scale=su.shock_scale,
            base_levels=su.base_levels,
            calculator=fixture.calculator,
            reporting=fixture.reporting,
            confidence_level=_CONFIDENCE_LEVEL,
            batch_size=_BATCH_SIZE,
            mode="reproducible",
        )
    except NonReproducibleRunError:
        refused = True

    exploratory = run_scenario_var_es(
        scenario_config=no_seed_config,
        data=fixture.historical_matrix,
        required_factor_ids=su.required_factor_ids,
        required_shift_type_id=su.required_shift_type_id,
        source_factor_ids=su.source_factor_ids,
        shock_scale=su.shock_scale,
        base_levels=su.base_levels,
        calculator=fixture.calculator,
        reporting=fixture.reporting,
        confidence_level=_CONFIDENCE_LEVEL,
        batch_size=_BATCH_SIZE,
        mode="exploratory",
    )

    checks = [
        ("seed=None refused under mode='reproducible'", refused),
        ("seed=None permitted under mode='exploratory'", True),
        (
            "exploratory result marked reproducible=False",
            exploratory.reproducible is False,
        ),
    ]
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        all_passed &= passed
        print(f"  {status:<4} | {name}")
    print("---")
    print()
    return all_passed


def main() -> None:
    """Run the scenario-model coverage demo."""
    print_calling_file()
    print_title("SCENARIO-MODEL COVERAGE: SENSITIVITY-BASED VAR")

    fixture = fx.build_fixture(verbose=False)
    print(
        f"Fixture: {fixture.portfolio.total_trades} trade(s), "
        f"{fixture.calculator.book.active_factor_ids.shape[0]} active risk factor(s), "
        f"proxy WTI_CRUDE -> BRENT_CRUDE (scale=0.95), reporting {fixture.reporting.reporting_ccy}."
    )
    print("---")

    coverage_passed = _run_coverage_sweep(fixture)
    seeded_passed = _run_seeded_reproducibility(fixture)
    policy_passed = _run_seed_policy(fixture)

    if coverage_passed and seeded_passed and policy_passed:
        print("ALL CHECKS PASSED")
    else:
        print("VALIDATION FAILED")

    print_title("END OF DEMO")
    print_success_message("DEMO RAN SUCCESSFULLY", add_space_after=False)


if __name__ == "__main__":
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    main()
