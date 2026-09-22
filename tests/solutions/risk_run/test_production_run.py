# tests/solutions/risk_run/test_production_run.py
"""
Release-gate contract of the production entry point `run_scenario_var_es`.

Every test drives the OFFICIAL composition (D036 seam -> apply_shocks ->
compiled SensitivityPnlCalculator -> FX plan -> compute_var_es). The invariant
under test is: a production risk result is inseparable from the immutable
identity of the computation that produced it, and a stale input, a foreign
environment or an unreproducible configuration is refused BEFORE any numerics.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    NonReproducibleRunError,
    RiskRun,
)
from riskquant.solutions.run_identity.api import (
    EnvironmentMismatchError,
    InputMismatchError,
)
from riskquant.solutions.run_identity.schemas import RunManifest
from riskquant.solutions.scenario_models.api import parse_scenario_model_config

sys.path.insert(0, str(Path(__file__).parent))
from _fixture import FACTOR_IDS, K, S, calculator, derived, inputs, reporting_spec


def _remat(m: RiskFactorMatrix, **over) -> RiskFactorMatrix:
    parts = {"dates": m.dates, "factor_ids": m.factor_ids, "values": m.values}
    parts.update(over)
    return RiskFactorMatrix(**parts)


def _numbers(r: RiskRun) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(float(v).hex() for v in r.var), tuple(float(v).hex() for v in r.es)


# ---- result always carries its identity --------------------------------------


def test_result_is_inseparable_from_identity() -> None:
    r = run_scenario_var_es(**inputs())
    assert r.identity == r.manifest.identity()
    assert len(r.identity) == 64
    assert r.var.shape == (r.position_uids.shape[0] + 1,)  # positions + portfolio


def test_identity_cannot_be_detached_or_tampered() -> None:
    r = run_scenario_var_es(**inputs())
    with pytest.raises(ValueError):
        RiskRun(
            position_uids=r.position_uids,
            var=r.var,
            es=r.es,
            reporting_ccy=r.reporting_ccy,
            manifest=r.manifest,
            identity="0" * 64,
            reproducible=True,
            pnl_provenance=r.pnl_provenance,
        )
    payload = json.loads(r.to_json())
    payload["identity"] = "0" * 64
    with pytest.raises(ValueError):
        RiskRun.from_json(json.dumps(payload))


def test_json_roundtrip_is_bitwise_and_self_identifying() -> None:
    r = run_scenario_var_es(**inputs())
    back = RiskRun.from_json(r.to_json())
    assert _numbers(back) == _numbers(r)
    assert back.identity == r.identity == back.manifest.identity()
    assert np.array_equal(back.position_uids, r.position_uids)


def test_manifest_is_reparsable_scenario_config() -> None:
    r = run_scenario_var_es(**inputs())
    cfg = parse_scenario_model_config(r.manifest.scenario_config)
    assert cfg.model_dump() == inputs()["scenario_config"].model_dump()


# ---- seed policy ---------------------------------------------------------------


def test_reproducible_mode_refuses_seed_none_before_numerics() -> None:
    with pytest.raises(NonReproducibleRunError):
        run_scenario_var_es(**inputs(seed=None))


def test_exploratory_mode_marks_result_non_reproducible() -> None:
    r = run_scenario_var_es(**inputs(seed=None), mode="exploratory")
    assert r.reproducible is False
    assert r.manifest.scenario_config["seed"] is None


def test_deterministic_model_is_reproducible_without_a_seed() -> None:
    """seed != run identity: Historical carries no seed yet is fully identified."""
    r = run_scenario_var_es(**inputs(model="historical"))
    assert r.reproducible is True
    assert "seed" not in r.manifest.scenario_config


def test_batch_size_is_operational_not_identity() -> None:
    a = run_scenario_var_es(**{**inputs(), "batch_size": 3})
    b = run_scenario_var_es(**{**inputs(), "batch_size": S + 5})
    assert a.identity == b.identity
    assert _numbers(a) == _numbers(b)


# ---- stale-input protection at the production boundary -------------------------


def test_reproduce_against_matching_manifest_succeeds() -> None:
    r = run_scenario_var_es(**inputs())
    again = run_scenario_var_es(**inputs(), reproduce=r.manifest)
    assert again.identity == r.identity and _numbers(again) == _numbers(r)


def test_stale_data_is_refused_before_execution() -> None:
    r = run_scenario_var_es(**inputs())
    inp = inputs()
    revised = np.array(inp["data"].values, copy=True)
    revised[30, 1] *= 1.01
    inp["data"] = _remat(inp["data"], values=np.asfortranarray(revised))
    with pytest.raises(InputMismatchError):
        run_scenario_var_es(**inp, reproduce=r.manifest)


def test_stale_book_is_refused_before_execution() -> None:
    r = run_scenario_var_es(**inputs())
    inp = inputs()
    # a different compiled book: same axis, different sensitivity
    import polars as pl

    from riskquant.sensitivity_pnl.conventions import DisplacementConvention
    from riskquant.solutions.sensitivity_pnl.api import (
        FactorAxis,
        SensitivityBookSpec,
        SensitivityPnlCalculator,
    )

    abs_id = int(DisplacementConvention.ABSOLUTE)
    sens = pl.DataFrame({
        "position_id": [1, 1, 2],
        "factor_id": [10, 30, 20],
        "first_order": [1001.0, -4115.0, -500.0],
        "second_order": [0.0, 0.0, 0.0],
    })
    meta = pl.DataFrame({
        "factor_id": FACTOR_IDS.tolist(),
        "displacement_convention_id": [abs_id] * K,
        "quote_base": [1.0] * K,
    })
    inp["taylor_derived"] = derived(
        SensitivityPnlCalculator.compile(
            SensitivityBookSpec(
                sensitivities=sens,
                factor_meta=meta,
                factor_axis=FactorAxis.from_keys(FACTOR_IDS.tolist()),
                position_uids=[1, 2],
            )
        )
    )
    with pytest.raises(InputMismatchError):
        run_scenario_var_es(**inp, reproduce=r.manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shock_scale", np.array([0.9, 1.0, 1.0], np.float32)),
        ("source_factor_ids", np.array([20, 10, 30], np.int32)),
        ("confidence_level", 0.95),
        ("base_levels", None),
    ],
)
def test_substituted_input_is_refused_before_execution(field, value) -> None:
    r = run_scenario_var_es(**inputs())
    inp = inputs()
    inp[field] = inp["base_levels"] * 1.001 if value is None else value
    with pytest.raises(InputMismatchError):
        run_scenario_var_es(**inp, reproduce=r.manifest)


def test_foreign_environment_is_refused_before_execution() -> None:
    r = run_scenario_var_es(**inputs())
    payload = json.loads(r.manifest.to_json())
    payload["engine"]["numpy_version"] = "0.0.0-not-this"
    foreign = RunManifest.model_validate(payload)
    with pytest.raises(EnvironmentMismatchError):
        run_scenario_var_es(**inputs(), reproduce=foreign)


# ---- mutation matrix: result-defining -> result AND identity change ------------


def _rebased(base, levels):
    return {
        "base_levels": levels,
        "taylor_derived": derived(
            calculator(base_levels=levels),
            v0=np.array(base["taylor_derived"].base_settlement_values),
        ),
    }


def test_greeks_from_another_base_fail_closed() -> None:
    """P1-A case H at the production entry point: same book, moved levels."""
    from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorContractError

    inp = inputs()
    inp["base_levels"] = inp["base_levels"] * 1.001
    with pytest.raises(TaylorContractError, match="base_levels_digest"):
        run_scenario_var_es(**inp)


def test_reporting_fx_base_from_another_base_fails_closed() -> None:
    """P1-A case L: identical Greeks and V_0, but q_0 (the USDEUR base level the
    reporting layer reads from base_levels) taken from another base."""
    from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorContractError

    inp = inputs()
    levels = inp["base_levels"].copy()
    levels[2] *= 1.05  # FACTOR_IDS[2] = 30 = USDEUR, the reporting FX row
    inp["base_levels"] = levels
    with pytest.raises(TaylorContractError, match="base_levels_digest"):
        run_scenario_var_es(**inp)


def test_uncertified_book_is_refused_by_the_production_entry_point() -> None:
    import dataclasses

    from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorContractError

    inp = inputs()
    inp["taylor_derived"] = derived(
        dataclasses.replace(inp["taylor_derived"].calculator, taylor_declaration=None)
    )
    with pytest.raises(TaylorContractError, match="UNCERTIFIED"):
        run_scenario_var_es(**inp)


def _mutations():
    base = inputs()
    revised = np.array(base["data"].values, copy=True)
    revised[30, 1] *= 1.01
    permuted = np.asfortranarray(np.array(base["data"].values)[:, [1, 0, 2]])
    return {
        "seed": {"scenario_config": inputs(seed=8)["scenario_config"]},
        "historical_value": {
            "data": _remat(base["data"], values=np.asfortranarray(revised))
        },
        "factor_order": {"data": _remat(base["data"], values=permuted)},
        "source_factor_ids": {"source_factor_ids": np.array([20, 10, 30], np.int32)},
        "shift_type": {"required_shift_type_id": np.array([2] * K, np.int8)},
        "shock_scale": {"shock_scale": np.array([0.9, 1.0, 1.0], np.float32)},
        # P1-A: a new base is a new Taylor representation. Moving the levels
        # under a book certified at the OLD base now fails closed (see
        # test_greeks_from_another_base_fail_closed); the result-defining
        # mutation re-certifies the book and the V_0 binding at the new base.
        "base_levels": _rebased(base, base["base_levels"] * 1.001),
        "confidence_level": {"confidence_level": 0.95},
        "reporting_currency": {"reporting": reporting_spec(reporting_ccy="EUR")},
        "model": {
            "scenario_config": inputs(model="filtered_bootstrap")["scenario_config"]
        },
    }


@pytest.mark.parametrize("name", list(_mutations()))
def test_result_defining_mutation_changes_result_and_identity(name) -> None:
    base = run_scenario_var_es(**inputs())
    mutated = run_scenario_var_es(**{**inputs(), **_mutations()[name]})
    assert _numbers(mutated) != _numbers(base), f"{name} did not change the result"
    assert mutated.identity != base.identity, f"{name} did not change the identity"


def test_equivalent_objects_do_not_change_identity() -> None:
    base = run_scenario_var_es(**inputs())
    inp = inputs()
    inp["data"] = _remat(
        inp["data"], values=np.asfortranarray(np.array(inp["data"].values, copy=True))
    )
    inp["base_levels"] = inp["base_levels"].copy()
    inp["taylor_derived"] = derived(calculator())  # freshly compiled, same content
    other = run_scenario_var_es(**inp)
    assert other.identity == base.identity and _numbers(other) == _numbers(base)


# ---- fresh-process reproduction through the production API ---------------------


_CHILD = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {tests!r})
    from _fixture import inputs
    from riskquant.solutions.risk_run.api import run_scenario_var_es
    r = run_scenario_var_es(**inputs())
    print(json.dumps({{"id": r.identity, "var": [float(v).hex() for v in r.var],
                       "es": [float(v).hex() for v in r.es], "run": r.to_json()}}))
    """
)


def _child(tests_dir: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(tests=tests_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_processes_reproduce_identity_and_numbers(request) -> None:
    d = str(request.path.parent)
    a, b = _child(d), _child(d)
    assert a["id"] == b["id"] and a["var"] == b["var"] and a["es"] == b["es"]
    assert RiskRun.from_json(a["run"]).identity == a["id"]


def test_fresh_process_matches_in_process_and_reproduces_against_manifest(
    request,
) -> None:
    child = _child(str(request.path.parent))
    local = run_scenario_var_es(**inputs())
    assert child["id"] == local.identity
    assert tuple(child["var"]) == _numbers(local)[0]
    persisted = RiskRun.from_json(child["run"])
    again = run_scenario_var_es(**inputs(), reproduce=persisted.manifest)
    assert _numbers(again) == _numbers(persisted)
