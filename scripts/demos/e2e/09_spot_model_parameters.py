# scripts/demos/e2e/09_spot_model_parameters.py
"""
RiskQuant SPOT -- chapter 09: configuring a scenario model.

A run has two kinds of parameters:

    ECONOMIC     change what is being measured. They are part of the run
                 identity (RunManifest): model, horizon, total_scenarios,
                 seed, expected_block_size, decay_factor, max_vol_ratio,
                 distribution, confidence_level, and every data input.
    OPERATIONAL  change only how the work is partitioned. batch_size is the
                 one operational knob of `run_scenario_var_es`: it is
                 deliberately absent from the identity (D041) and the result
                 is bitwise invariant to it.

Each experiment below changes ONE parameter against a fixed baseline and
asserts what the identity contract says must happen. The measured effect on
VaR/ES is printed. An economic parameter need not move VaR in every setting
(block length has no effect at horizon 1, shown below), so the universal
assertion is the identity change; numerical effects are asserted case by
case.

Configs are Pydantic v2 models with `extra="forbid"`. They can be built
from typed classes or parsed from an untyped mapping (JSON, Excel named
ranges) with `parse_scenario_model_config`; invalid parameters fail at the
solution boundary, before any numerics.

Run:  python scripts/demos/e2e/09_spot_model_parameters.py
"""

from __future__ import annotations

from typing import Any

import _spot_world as sw
import numpy as np
from pydantic import ValidationError

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import RiskRun
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.schemas import (
    FilteredHistoricalGeneratorConfig,
    HistoricalGeneratorConfig,
    ScenarioModelConfig,
)

BASE_SBB: dict[str, Any] = {
    "model": "sbb",
    "horizon": 1,
    "total_scenarios": 1000,
    "expected_block_size": 5.0,
    "seed": 7,
}
BASE_CONFIDENCE: float = 0.99
BASE_BATCH: int = 256


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 09 model parameters")
    real = sw.realistic_fixture()
    world = real.world
    method = sw.methodology(full=world.uids.tolist())
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    context = sw.valuation_context(world)

    def run(config: ScenarioModelConfig, *, confidence: float = BASE_CONFIDENCE,
            batch_size: int = BASE_BATCH, data=None) -> RiskRun:
        return run_scenario_var_es(
            scenario_config=config,
            data=real.history if data is None else data,
            required_factor_ids=world.required_factor_ids,
            required_shift_type_id=world.required_shift_type_id,
            source_factor_ids=world.source_factor_ids,
            shock_scale=world.shock_scale,
            base_levels=world.base_levels,
            methodology=method,
            valuation_context=context,
            reporting=reporting,
            confidence_level=confidence,
            batch_size=batch_size,
        )

    # ---------------------------------------------------------------- 1
    sw.step("1", "A config from an untyped mapping")
    config = parse_scenario_model_config(BASE_SBB)
    sw.para(
        """
        `model` is the discriminator. Each model's config carries ONLY its
        own parameters; the parsed object is typed and immutable in use.
        """
    )
    sw.kv([(k, v) for k, v in config.model_dump().items()])
    ok.check("the mapping parses to the typed SbbGeneratorConfig",
             type(config).__name__ == "SbbGeneratorConfig" and config.model_dump() == BASE_SBB)
    ok.check("the manifest stores the canonical config (seed included)",
             run(config).manifest.scenario_config == BASE_SBB)

    # ---------------------------------------------------------------- 2
    sw.step("2", "One parameter at a time")
    base = run(config)
    experiments: list[tuple[str, str, RiskRun]] = [
        ("batch_size 256 -> 7", "operational", run(config, batch_size=7)),
        ("batch_size 256 -> 1000 (= S)", "operational", run(config, batch_size=1000)),
        ("seed 7 -> 8", "economic", run(parse_scenario_model_config({**BASE_SBB, "seed": 8}))),
        ("total_scenarios 1000 -> 1500", "economic",
         run(parse_scenario_model_config({**BASE_SBB, "total_scenarios": 1500}))),
        ("expected_block_size 5 -> 20", "economic",
         run(parse_scenario_model_config({**BASE_SBB, "expected_block_size": 20.0}))),
        ("horizon 1 -> 10", "economic", run(parse_scenario_model_config({**BASE_SBB, "horizon": 10}))),
        ("confidence 0.99 -> 0.975", "economic", run(config, confidence=0.975)),
    ]
    rows = [("baseline", "-", base.identity[:10], sw.num(base.portfolio_var, 0), sw.num(base.portfolio_es, 0))]
    for label, kind, r in experiments:
        rows.append((label, kind, r.identity[:10], sw.num(r.portfolio_var, 0), sw.num(r.portfolio_es, 0)))
    sw.table(["change", "kind", "identity", "VaR", "ES"], rows, align="llrrr")
    for label, kind, r in experiments:
        if kind == "operational":
            ok.check(f"{label}: same identity, bitwise same VaR/ES",
                     r.identity == base.identity and np.array_equal(r.var, base.var) and np.array_equal(r.es, base.es))
        else:
            ok.check(f"{label}: new identity", r.identity != base.identity)
    ok.check("the seed change moved the portfolio VaR (a different sample was drawn)",
             experiments[2][2].portfolio_var != base.portfolio_var)
    ok.check("the horizon change widened the distribution (10-day vs 1-day VaR)",
             experiments[5][2].portfolio_var < base.portfolio_var)
    sw.para(
        """
        Look at expected_block_size. At horizon 1 a sampled path is a single
        day, so the block length never gets to act: the draws, and therefore
        VaR and ES, are bitwise unchanged. The identity still changes. The
        identity names the declared CONFIGURATION, not the numbers it
        happened to produce; a reproduction must use the same declaration.
        """
    )
    ok.check("block size 5 -> 20 at horizon 1: bitwise same VaR/ES, new identity",
             np.array_equal(experiments[4][2].var, base.var) and np.array_equal(experiments[4][2].es, base.es)
             and experiments[4][2].identity != base.identity)

    # ---------------------------------------------------------------- 3
    sw.step("3", "Model-specific parameters: filtered historical")
    fhs_default = FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=1)
    fhs_fast = FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=1, decay_factor=0.90)
    fhs_capped = FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=1, max_vol_ratio=1.0)
    hist = HistoricalGeneratorConfig(model="historical", horizon=1)
    table = []
    runs = {}
    for label, cfg in (("historical (no filter)", hist), ("FHS defaults", fhs_default),
                       ("FHS decay_factor 0.90", fhs_fast), ("FHS max_vol_ratio 1.0", fhs_capped)):
        runs[label] = run(cfg)
        table.append((label, f"{getattr(cfg, 'decay_factor', '-')}", f"{getattr(cfg, 'max_vol_ratio', '-')}",
                      sw.num(runs[label].portfolio_var, 0)))
    sw.table(["config", "decay_factor", "max_vol_ratio", "VaR 99%"], table, align="lrrr")
    sw.para(
        """
        decay_factor is the EWMA memory: a smaller value reacts faster to
        recent volatility. max_vol_ratio caps how far a historical return can
        be scaled up to today's volatility (1.0 allows no amplification).
        Deterministic: no seed, identical inputs give identical output.
        """
    )
    ok.check("each filtered-historical parameter change produces a new identity",
             len({r.identity for r in runs.values()}) == 4)
    ok.check("FHS is deterministic: a rerun of the defaults is bitwise identical",
             np.array_equal(run(fhs_default).var, runs["FHS defaults"].var))

    # ---------------------------------------------------------------- 4
    sw.step("4", "Invalid configurations fail at the solution boundary")
    ok.expect_rejection(
        "a parameter that belongs to another model (decay_factor on sbb)",
        ValidationError,
        lambda: parse_scenario_model_config({**BASE_SBB, "decay_factor": 0.94}),
        reason="extra='forbid': a silently ignored parameter would change "
        "nothing while the caller believes it changed the model",
        stage="config parsing (Pydantic), before any run input is built",
    )
    ok.expect_rejection(
        "horizon given as a float (1.0)",
        ValidationError,
        lambda: parse_scenario_model_config({"model": "historical", "horizon": 1.0}),
        reason="StrictInt: a discrete window length is never coerced from a float",
        stage="config parsing (Pydantic)",
    )
    ok.expect_rejection(
        "a negative seed",
        ValidationError,
        lambda: parse_scenario_model_config({**BASE_SBB, "seed": -1}),
        reason="NumPy SeedSequence rejects negative seeds; the seam fails first",
        stage="config parsing (Pydantic)",
    )
    ok.expect_rejection(
        "an unknown model name",
        ValidationError,
        lambda: parse_scenario_model_config({"model": "garch", "horizon": 1}),
        reason="the discriminated union admits only the five certified models",
        stage="config parsing (Pydantic)",
    )
    ok.expect_rejection(
        "a historical config paired with parametric construction data",
        TypeError,
        lambda: build_scenario_generator(hist, data=real.parametric),
        reason="each model family requires exactly one construction-data type",
        stage="generator construction (build_scenario_generator)",
    )
    ok.done()


if __name__ == "__main__":
    main()
