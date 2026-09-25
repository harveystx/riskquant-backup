# scripts/demos/e2e/05_spot_reproducibility_and_identity.py
"""
RiskQuant SPOT -- chapter 05: reproducibility and run identity.

Why identity exists
-------------------
A VaR number is only meaningful together with the exact computation that
produced it: which book, which market base, which history, which model and
seed, which confidence, which reporting currency, which engine. RiskQuant
builds a `RunManifest` of content digests of those inputs BEFORE any
numerics, hashes it into `identity`, and returns numbers and identity in one
frozen `RiskRun`. Two runs with the same identity must produce bitwise the
same numbers (for reproducible runs); a changed economic input must produce a
changed identity; an operational change (batch_size) must change neither.

Identity versus result integrity (D083)
---------------------------------------
The identity digests INPUTS only; it never digests VaR/ES. A persisted
`RiskRun` also carries `result_digest`, which binds the identity to the stored
position UIDs, VaR, ES and reproducibility flag. `RiskRun.from_json` rejects
an artifact whose stored numbers were edited (`ResultIntegrityError`).

`result_digest` is an unkeyed SHA-256: integrity, NOT authentication. A writer
who rebuilds the numbers AND the digest consistently produces an artifact that
loads. To verify the numbers themselves, reproduce the run
(`reproduce=manifest`) and compare bitwise. Both behaviours are demonstrated
below.

Run:  python scripts/demos/e2e/05_spot_reproducibility_and_identity.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import _spot_world as sw
import numpy as np

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    NonReproducibleRunError,
    ReproducibilityMode,
    ResultIntegrityError,
    RiskRun,
)
from riskquant.solutions.run_identity.api import InputMismatchError
from riskquant.solutions.scenario_models.schemas import SbbGeneratorConfig

SEED: int = 7
CONFIDENCE: float = 0.99
SBB = SbbGeneratorConfig(
    model="sbb", horizon=1, total_scenarios=1000, expected_block_size=5.0, seed=SEED
)

#: Executed in a FRESH interpreter: rebuilds the fixture from scratch and
#: prints the identity and the IEEE-754 hex of the portfolio VaR/ES.
_CHILD = r"""
import sys; sys.path.insert(0, sys.argv[1])
import _spot_world as sw
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.scenario_models.schemas import SbbGeneratorConfig
real = sw.realistic_fixture(); w = real.world
run = run_scenario_var_es(
    scenario_config=SbbGeneratorConfig(model="sbb", horizon=1, total_scenarios=1000,
                                       expected_block_size=5.0, seed=7),
    data=real.history, required_factor_ids=w.required_factor_ids,
    required_shift_type_id=w.required_shift_type_id, source_factor_ids=w.source_factor_ids,
    shock_scale=w.shock_scale, base_levels=w.base_levels,
    methodology=sw.methodology(full=w.uids.tolist()), valuation_context=sw.valuation_context(w),
    reporting=sw.reporting_spec(w, reporting_ccy="USD"),
    confidence_level=0.99, batch_size=97)
print(run.identity, run.portfolio_var.hex(), run.portfolio_es.hex())
"""


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 05 reproducibility and identity")

    def run(
        fixture: sw.RealisticFixture,
        *,
        config=SBB,
        data=None,
        confidence=CONFIDENCE,
        batch_size=256,
        reporting_ccy="USD",
        mode: ReproducibilityMode = "reproducible",
        reproduce=None,
    ) -> RiskRun:
        w = fixture.world
        return run_scenario_var_es(
            scenario_config=config,
            data=fixture.history if data is None else data,
            required_factor_ids=w.required_factor_ids,
            required_shift_type_id=w.required_shift_type_id,
            source_factor_ids=w.source_factor_ids,
            shock_scale=w.shock_scale,
            base_levels=w.base_levels,
            methodology=sw.methodology(full=w.uids.tolist()),
            valuation_context=sw.valuation_context(w),
            reporting=sw.reporting_spec(
                w, reporting_ccy=reporting_ccy
            ),
            confidence_level=confidence,
            batch_size=batch_size,
            mode=mode,
            reproduce=reproduce,
        )

    real = sw.realistic_fixture()
    base = run(real)

    # ---------------------------------------------------------------- 1
    sw.step("1", "Same inputs, same seed -> the same run")
    rebuilt = sw.realistic_fixture()  # every object rebuilt from the edge tables
    again = run(rebuilt)
    env = {**os.environ, "PYTHONUTF8": "1"}
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(Path(__file__).resolve().parent)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.split()
    sw.table(
        ["run", "identity", "portfolio VaR (hex)"],
        [
            ("base (batch 256)", base.identity[:16], base.portfolio_var.hex()),
            ("rebuilt fixture", again.identity[:16], again.portfolio_var.hex()),
            ("fresh process, batch 97", child[0][:16], child[1]),
        ],
        align="lll",
    )
    ok.check(
        "rebuilt fixture: same identity, bitwise same VaR/ES",
        again.identity == base.identity
        and np.array_equal(again.var, base.var)
        and np.array_equal(again.es, base.es),
    )
    ok.check(
        "a fresh Python process reproduces identity and portfolio VaR/ES bitwise",
        child == [base.identity, base.portfolio_var.hex(), base.portfolio_es.hex()],
    )
    sw.para(
        """
        The child process also partitions the stream differently (batch 97
        against 256). batch_size is operational: it is absent from the
        identity and never moves a number (chapter 06).
        """
    )

    # ---------------------------------------------------------------- 2
    sw.step("2", "Changed economic input -> changed identity")
    levels = dict(real.world.levels)
    levels["US_EQ_00"] = levels["US_EQ_00"] * 1.01
    bumped_history = np.array(real.history.values, copy=True)
    bumped_history[-1, 3] *= 1.001
    stale = RiskFactorMatrix(
        dates=real.history.dates,
        factor_ids=real.history.factor_ids,
        values=np.asfortranarray(bumped_history),
    )
    positions = real.world.positions.copy()
    positions.loc[0, "quantity"] = positions.loc[0, "quantity"] + 1.0

    variants = {
        "seed 7 -> 8": run(real, config=SBB.model_copy(update={"seed": 8})),
        "one history value x1.001": run(real, data=stale),
        "one base level x1.01": run(_with_levels(levels)),
        "one quantity +1": run(sw.realistic_fixture(positions=positions)),
        "confidence 0.99 -> 0.95": run(real, confidence=0.95),
    }
    sw.para(
        """
        Each row lists the manifest fields whose content differs from the
        base run. A reporting-currency change also changes the identity
        (chapter 03, section A3).
        """
    )
    rows = []
    for label, r in variants.items():
        sections = _diff_sections(base, r)
        rows.append((label, r.identity[:8], ", ".join(sections)))
        ok.check(f"{label}: new identity", r.identity != base.identity)
    sw.table(
        ["change", "identity", "manifest fields that differ"],
        rows,
        align="lll",
        indent=2,
    )

    # ---------------------------------------------------------------- 3
    sw.step("3", "reproduce= verifies inputs BEFORE numerics")
    replay = run(real, reproduce=base.manifest)
    ok.check(
        "reproduce against the run's own manifest succeeds, bitwise",
        replay.identity == base.identity and np.array_equal(replay.var, base.var),
    )
    ok.expect_rejection(
        "reproduce with a substituted history snapshot",
        InputMismatchError,
        lambda: run(real, data=stale, reproduce=base.manifest),
        reason="the data digest differs from the manifest being reproduced",
        stage="reproduce verification, before any scenario is generated",
    )
    ok.expect_rejection(
        "reproduce with a book whose quantity changed",
        InputMismatchError,
        lambda: run(sw.realistic_fixture(positions=positions), reproduce=base.manifest),
        reason="the compiled-portfolio digest (run context) differs",
        stage="reproduce verification, before any scenario is generated",
    )

    # ---------------------------------------------------------------- 4
    sw.step("4", "Seed policy")
    no_seed = SBB.model_copy(update={"seed": None})
    ok.expect_rejection(
        "seed=None under mode='reproducible' (the default)",
        NonReproducibleRunError,
        lambda: run(real, config=no_seed),
        reason="a stochastic model without a seed cannot be reproduced",
        stage="first check of run_scenario_var_es, before anything else",
    )
    e1, e2 = (
        run(real, config=no_seed, mode="exploratory"),
        run(real, config=no_seed, mode="exploratory"),
    )
    sw.para(
        """
        mode='exploratory' permits seed=None and marks the result. Note what
        the identity means then: both exploratory runs share ONE identity
        (same declared configuration), yet draw different samples. That is
        exactly why the flag exists.
        """
    )
    ok.check(
        "exploratory runs are marked reproducible=False",
        e1.reproducible is False and e2.reproducible is False,
    )
    ok.check(
        "same identity, different numbers: identity alone does not pin an unseeded run",
        e1.identity == e2.identity and not np.array_equal(e1.var, e2.var),
    )

    # ---------------------------------------------------------------- 5
    sw.step("5", "Persisted RiskRun: round trip, result integrity and its limit")
    payload = base.to_json()
    restored = RiskRun.from_json(payload)
    ok.check(
        "to_json -> from_json is bitwise (IEEE-754 hex) and keeps identity and result_digest",
        np.array_equal(restored.var, base.var)
        and np.array_equal(restored.es, base.es)
        and restored.identity == base.identity
        and restored.result_digest == base.result_digest,
    )
    d = json.loads(payload)
    d["var_hex"][-1] = (base.portfolio_var * 0.5).hex()
    ok.expect_rejection(
        "an edited VaR value is refused",
        ResultIntegrityError,
        lambda: RiskRun.from_json(json.dumps(d)),
        reason="result_digest binds identity, position UIDs, VaR, ES and the reproducible flag (D083)",
        stage="RiskRun.from_json, after the identity check",
    )
    sw.para(
        """
        Limitation, stated plainly: result_digest is an unkeyed SHA-256. It
        detects an edited or corrupted result, not a forger who rebuilds the
        numbers AND the digest. Such a self-consistent artifact loads. The
        numbers themselves are verified by reproduction: re-run with
        reproduce= and compare bitwise.
        """
    )
    forged = RiskRun(
        position_uids=base.position_uids,
        var=base.var * 0.5,
        es=base.es,
        reporting_ccy=base.reporting_ccy,
        manifest=base.manifest,
        identity=base.identity,
        reproducible=base.reproducible,
        pnl_provenance=base.pnl_provenance,
    )
    loaded = RiskRun.from_json(forged.to_json())
    ok.check(
        "a self-consistent forgery passes from_json (integrity is not authentication)",
        loaded.identity == base.identity and loaded.portfolio_var != base.portfolio_var,
    )
    check = run(real, reproduce=loaded.manifest)
    ok.check(
        "re-running with reproduce= and comparing bitwise exposes the forged number",
        check.identity == loaded.identity
        and check.portfolio_var != loaded.portfolio_var
        and np.array_equal(check.var, base.var)
        and np.array_equal(check.es, base.es),
    )
    ok.done()


def _with_levels(levels: dict[str, float]) -> sw.RealisticFixture:
    """The realistic fixture with different base levels, same book and history."""
    real = sw.realistic_fixture()
    world = sw.build_world(
        rsc=real.world.rsc,
        proxy_topology=real.world.proxy_topology,
        positions=real.world.positions,
        levels=levels,
        as_of=real.world.as_of,
        history_factor_names=[
            real.world.name_of(int(r)) for r in real.history.factor_ids
        ],
    )
    return sw.RealisticFixture(
        world=world,
        history=real.history,
        parametric=real.parametric,
        num_observations=real.num_observations,
    )


def _diff_sections(a: RiskRun, b: RiskRun) -> list[str]:
    """Dotted paths of the manifest leaves whose content differs."""
    return _leaf_diffs(a.manifest.model_dump(), b.manifest.model_dump(), "")


def _leaf_diffs(x: object, y: object, prefix: str) -> list[str]:
    if isinstance(x, dict) and isinstance(y, dict):
        out: list[str] = []
        for k in sorted(set(x) | set(y)):
            out.extend(
                _leaf_diffs(x.get(k), y.get(k), f"{prefix}.{k}" if prefix else k)
            )
        return out
    return [] if x == y else [prefix]


if __name__ == "__main__":
    main()
