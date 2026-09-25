# scripts/demos/e2e/06_spot_batching_and_scale.py
"""
RiskQuant SPOT -- chapter 06: batching is operational, not economic.

`batch_size` controls how the scenario stream is partitioned: how many
scenarios the generator yields per chunk, how many columns FULL_REVAL values
per market-data build. Each scenario column is computed independently, so
the partition cannot change any number. The contract (D041) is:

    any batch_size > 0  ->  bitwise-identical VaR / ES  and  identical identity

This chapter proves it on a larger, production-shaped mixed book for two
stochastic models, with batch sizes 1, 3, 7, 64, S and S + 1, and prints the
wall time of each run. Timings are informational (single process, this
machine); they are not a benchmark and are not asserted.

Why the stochastic models are the interesting case: the scenario WORLD
belongs to the run, not to a batch (D087). The seed fixes the whole world
of S scenarios; the parametric model draws all its base normals up front
(D021). A batch is only an execution window onto that world. The run
refuses a stream whose batches do not tile [0, S) exactly once, in order
(PR-06), because the identity names the configured world, not the windows.

Multi-core thread-count determinism is certified separately by the SimLab
release certificate; this demo does not depend on the core count.

Run:  python scripts/demos/e2e/06_spot_batching_and_scale.py
"""

from __future__ import annotations

import time

import _spot_world as sw
import numpy as np

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.scenario_models.schemas import (
    ParametricGeneratorConfig,
    SbbGeneratorConfig,
    StandardNormalDistribution,
)

NUM_POSITIONS: int = 400
NUM_OBSERVATIONS: int = 1_001
NUM_SCENARIOS: int = 2_500
CONFIDENCE: float = 0.99


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 06 batching and scale")
    t0 = time.perf_counter()
    real = sw.realistic_fixture(num_positions=NUM_POSITIONS, num_observations=NUM_OBSERVATIONS)
    world = real.world
    uids = world.uids.tolist()
    full_uids, derived_uids = uids[0::2], uids[1::2]
    method = sw.methodology(full=full_uids, derived=derived_uids)
    block = sw.certified_taylor_block(world, derived_uids)
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    context = sw.valuation_context(world)
    build_s = time.perf_counter() - t0

    sw.step("1", "Dimensions")
    sw.kv([
        ("positions", f"{NUM_POSITIONS} ({len(full_uids)} FULL_REVAL, {len(derived_uids)} TAYLOR/DERIVED)"),
        ("run axis", f"{world.required_factor_ids.shape[0]} factors"),
        ("history", f"{NUM_OBSERVATIONS} observations"),
        ("scenarios S", f"{NUM_SCENARIOS:,} per model"),
        ("P&L grid", f"{NUM_POSITIONS} x {NUM_SCENARIOS:,} float32 = "
                     f"{NUM_POSITIONS * NUM_SCENARIOS * 4 / 2**20:.1f} MiB"),
        ("fixture build", f"{build_s:.2f} s"),
    ])

    models = {
        "sbb": (SbbGeneratorConfig(model="sbb", horizon=1, total_scenarios=NUM_SCENARIOS,
                                   expected_block_size=5.0, seed=11), real.history),
        "parametric": (ParametricGeneratorConfig(model="parametric", total_scenarios=NUM_SCENARIOS, seed=11,
                                                 distribution=StandardNormalDistribution(kind="standard_normal")),
                       real.parametric),
    }
    batch_sizes = (1, 3, 7, 64, NUM_SCENARIOS, NUM_SCENARIOS + 1)

    for step_no, (name, (config, data)) in enumerate(models.items(), start=2):
        sw.step(str(step_no), f"{name}: six partitions of the same stream")
        rows = []
        runs = {}
        for b in batch_sizes:
            t = time.perf_counter()
            runs[b] = run_scenario_var_es(
                scenario_config=config,
                data=data,
                required_factor_ids=world.required_factor_ids,
                required_shift_type_id=world.required_shift_type_id,
                source_factor_ids=world.source_factor_ids,
                shock_scale=world.shock_scale,
                base_levels=world.base_levels,
                methodology=method,
                valuation_context=context,
                taylor_derived=block,
                reporting=reporting,
                confidence_level=CONFIDENCE,
                batch_size=b,
            )
            rows.append((f"{b:,}", f"{-(-NUM_SCENARIOS // b):,}", runs[b].identity[:12],
                         runs[b].portfolio_var.hex(), f"{time.perf_counter() - t:.2f}"))
        sw.table(["batch_size", "batches", "identity", "portfolio VaR (hex)", "sec"], rows, align="rrlll")
        ref = runs[64]
        for b in batch_sizes:
            if b == 64:
                continue
            ok.check(f"{name}: batch_size={b:,} == batch_size=64 (identity, all {NUM_POSITIONS + 1} VaR "
                     "and ES values, bitwise)",
                     runs[b].identity == ref.identity and np.array_equal(runs[b].var, ref.var)
                     and np.array_equal(runs[b].es, ref.es))

    sw.para(
        f"""
        batch_size = 1 is the slowest partition because every scenario pays
        the per-batch setup (a market-data build for the FULL_REVAL block);
        S + 1 is a single partial batch. The numbers never move. Total wall
        time: {time.perf_counter() - t0:.1f} s.
        """
    )
    ok.done()


if __name__ == "__main__":
    main()
