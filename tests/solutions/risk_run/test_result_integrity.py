# tests/solutions/risk_run/test_result_integrity.py
"""
D083: persisted-result integrity of `RiskRun`, separate from run identity.

Contract under test:

* `RunManifest.identity()` is the identity of the CALCULATION DEFINITION. It is
  unchanged by this contract and commits to no output (D040, D081 §7).
* `RiskRun.result_digest` binds identity, `position_uids`, `var`, `es` and
  `reproducible`. `from_json` REJECTS a payload whose stored result does not
  match its digest, or that carries no digest.
* The digest is unkeyed: integrity, not authentication. A fully self-consistent
  forgery is accepted, and a test pins that non-claim.
* Non-finite VaR/ES is not a result and is refused on construction and on load.

Every tamper test proves first that the edit moves the payload (D059), then
that the persisted artifact is REJECTED. Comparisons are exact: `float.hex()`,
array equality on int64, string equality on digests. No tolerance is used.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import _resultfix as rf
import numpy as np
import pytest

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import ResultIntegrityError, RiskRun
from riskquant.solutions.run_identity.schemas import RunManifest

_V4_MANIFEST_KEYS = {
    "manifest_schema_version",
    "engine",
    "scenario_config",
    "data",
    "shocks",
    "pnl",
    "reporting",
    "risk_measure",
}


def _hexes(values: np.ndarray) -> list[str]:
    return [float(v).hex() for v in values]


def _payload(run: RiskRun) -> dict[str, Any]:
    return json.loads(run.to_json())


def _edited(run: RiskRun, edit: Callable[[dict[str, Any]], None]) -> str:
    """Apply one edit; prove it moved the payload (D059) before it is used."""
    original = run.to_json()
    payload = json.loads(original)
    edit(payload)
    edited = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert edited != original, "tamper did not move the payload; the test is void"
    return edited


def _ulp_moves(value_hex: str) -> list[str]:
    v = float.fromhex(value_hex)
    return [math.nextafter(v, math.inf).hex(), math.nextafter(v, -math.inf).hex()]


@pytest.fixture(scope="module")
def run() -> RiskRun:
    return run_scenario_var_es(**rf.inputs())


@pytest.fixture(scope="module")
def other_run() -> RiskRun:
    """A genuinely different calculation (stochastic model, explicit seed)."""
    return run_scenario_var_es(**rf.inputs(scenario_config=rf.sbb(7)))


# ---- A. authentic round trip ---------------------------------------------------


def test_authentic_round_trip_is_bitwise(run: RiskRun) -> None:
    back = RiskRun.from_json(run.to_json())
    assert _hexes(back.var) == _hexes(run.var)
    assert _hexes(back.es) == _hexes(run.es)
    assert back.position_uids.dtype == np.int64
    assert np.array_equal(back.position_uids, run.position_uids)
    assert back.reproducible is run.reproducible
    assert back.identity == run.identity
    assert back.result_digest == run.result_digest
    assert back.to_json() == run.to_json()


def test_persisted_artifact_carries_result_digest(run: RiskRun) -> None:
    payload = _payload(run)
    assert payload["result_digest"] == run.result_digest
    assert len(run.result_digest) == 64
    assert run.result_digest != run.identity


# ---- B. one-ULP result tampering ---------------------------------------------------


def _ulp_cases(run: RiskRun) -> list[tuple[str, int, str]]:
    cases = []
    for field, values in (("var_hex", run.var), ("es_hex", run.es)):
        for i, v in enumerate(values):
            for moved in _ulp_moves(float(v).hex()):
                cases.append((field, i, moved))
    return cases


def test_every_var_and_es_entry_one_ulp_is_rejected(run: RiskRun) -> None:
    cases = _ulp_cases(run)
    assert len(cases) == 2 * 2 * run.var.shape[0]
    for field, i, moved in cases:
        edited = _edited(run, lambda p, f=field, j=i, m=moved: p[f].__setitem__(j, m))
        with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
            RiskRun.from_json(edited)


def test_swapped_var_entries_are_rejected(run: RiskRun) -> None:
    assert _hexes(run.var)[0] != _hexes(run.var)[1], "fixture must separate entries"
    edited = _edited(
        run,
        lambda p: p.__setitem__(
            "var_hex", [p["var_hex"][1], p["var_hex"][0], *p["var_hex"][2:]]
        ),
    )
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


# ---- C. position UID tampering --------------------------------------------------


@pytest.mark.parametrize(
    "uids",
    [
        pytest.param([102, 101], id="permutation"),
        pytest.param([101, 103], id="replacement"),
        pytest.param([101, 101], id="duplicate"),
    ],
)
def test_position_uid_tampering_is_rejected(run: RiskRun, uids: list[int]) -> None:
    edited = _edited(run, lambda p: p.__setitem__("position_uids", uids))
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


# ---- D. reproducible flag ----------------------------------------------------------


def test_flipped_reproducible_flag_is_rejected(run: RiskRun) -> None:
    edited = _edited(
        run, lambda p: p.__setitem__("reproducible", not p["reproducible"])
    )
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


# ---- E / F. digest tampering and absence ---------------------------------------------


@pytest.mark.parametrize(
    "replace",
    [
        pytest.param(lambda d: "0" * 64, id="zeroed"),
        pytest.param(lambda d: ("1" if d[0] == "0" else "0") + d[1:], id="one-char"),
        pytest.param(lambda d: d.upper(), id="case"),
    ],
)
def test_replaced_digest_is_rejected(run: RiskRun, replace) -> None:
    edited = _edited(
        run, lambda p: p.__setitem__("result_digest", replace(p["result_digest"]))
    )
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


def test_missing_digest_is_rejected(run: RiskRun) -> None:
    edited = _edited(run, lambda p: p.pop("result_digest"))
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


# ---- G. cross-run transplant -------------------------------------------------------


def test_digest_transplanted_from_another_run_is_rejected(
    run: RiskRun, other_run: RiskRun
) -> None:
    assert other_run.identity != run.identity
    assert other_run.result_digest != run.result_digest
    edited = _edited(
        other_run, lambda p: p.__setitem__("result_digest", run.result_digest)
    )
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


def test_whole_result_transplanted_onto_another_identity_is_rejected(
    run: RiskRun, other_run: RiskRun
) -> None:
    """Run A's numbers, UIDs, flag AND digest under run B's manifest/identity."""

    def graft(p: dict[str, Any]) -> None:
        a = _payload(run)
        for key in (
            "var_hex",
            "es_hex",
            "position_uids",
            "reproducible",
            "result_digest",
        ):
            p[key] = a[key]

    edited = _edited(other_run, graft)
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


# ---- H. D2-style manifest relabelling -----------------------------------------------


def test_relabelled_manifest_with_recomputed_identity_is_rejected(run: RiskRun) -> None:
    """Before D083 this payload LOADED: the identity check is consistency only."""

    def relabel(p: dict[str, Any]) -> None:
        p["manifest"]["risk_measure"]["confidence_level"] = 0.95
        p["identity"] = RunManifest.model_validate(p["manifest"]).identity()

    edited = _edited(run, relabel)
    assert json.loads(edited)["identity"] != run.identity
    with pytest.raises(ResultIntegrityError, match="RESULT INTEGRITY"):
        RiskRun.from_json(edited)


# ---- I. negative control: integrity != authentication --------------------------------


def test_fully_self_consistent_forgery_is_accepted(run: RiskRun) -> None:
    """
    Pins the NON-claim. A forger who rebuilds manifest, identity, numbers and
    digest consistently produces an artifact that loads. `result_digest` is
    not a signature; numerical authority needs reproduction (D041/D042).
    """
    manifest_dict = json.loads(run.manifest.to_json())
    manifest_dict["risk_measure"]["confidence_level"] = 0.95
    forged_manifest = RunManifest.model_validate(manifest_dict)
    forged_var = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    forged = RiskRun(
        position_uids=run.position_uids,
        var=forged_var,
        es=forged_var,
        reporting_ccy=run.reporting_ccy,
        manifest=forged_manifest,
        identity=forged_manifest.identity(),
        reproducible=run.reproducible,
        pnl_provenance=run.pnl_provenance,
    )
    back = RiskRun.from_json(forged.to_json())
    assert back.identity != run.identity
    assert back.manifest.risk_measure.confidence_level == 0.95
    assert _hexes(back.var) == _hexes(forged_var)
    assert back.result_digest == forged.result_digest != run.result_digest


# ---- J. non-finite values ------------------------------------------------------------


_NON_FINITE = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="+inf"),
    pytest.param(-math.inf, id="-inf"),
]


@pytest.mark.parametrize("bad", _NON_FINITE)
@pytest.mark.parametrize("field", ["var", "es"])
def test_non_finite_result_is_refused_on_construction(
    run: RiskRun, field: str, bad: float
) -> None:
    values = np.array(getattr(run, field), copy=True)
    values[-1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        dataclasses.replace(run, **{field: values})


@pytest.mark.parametrize("bad", _NON_FINITE)
@pytest.mark.parametrize("field", ["var_hex", "es_hex"])
def test_non_finite_result_is_refused_on_load(
    run: RiskRun, field: str, bad: float
) -> None:
    edited = _edited(run, lambda p: p[field].__setitem__(0, bad.hex()))
    with pytest.raises(ValueError, match="non-finite"):
        RiskRun.from_json(edited)


# ---- construction contract that keeps the digest well defined ------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("var", "float32"),
        ("es", "float32"),
        ("position_uids", "int32"),
    ],
)
def test_result_arrays_require_the_exact_dtype(
    run: RiskRun, field: str, value: str
) -> None:
    with pytest.raises(TypeError, match=field):
        dataclasses.replace(run, **{field: getattr(run, field).astype(value)})


def test_reproducible_must_be_a_bool(run: RiskRun) -> None:
    with pytest.raises(TypeError, match="reproducible"):
        dataclasses.replace(run, reproducible=np.bool_(True))


def test_stored_arrays_are_owned_and_read_only(run: RiskRun) -> None:
    var = np.array(run.var, copy=True)
    rebuilt = dataclasses.replace(run, var=var)
    var[-1] = 0.0  # a later write to the caller's buffer
    assert _hexes(rebuilt.var) == _hexes(run.var)
    assert rebuilt.result_digest == run.result_digest
    for name in ("position_uids", "var", "es"):
        assert not getattr(run, name).flags.writeable


# ---- malformed payloads fail deliberately (ADJ-6) ------------------------------------


@pytest.mark.parametrize(
    "key",
    ["var_hex", "es_hex", "position_uids", "identity", "manifest", "reproducible"],
)
def test_missing_key_is_a_malformed_payload_not_a_key_error(
    run: RiskRun, key: str
) -> None:
    edited = _edited(run, lambda p: p.pop(key))
    with pytest.raises(ValueError, match="malformed persisted RiskRun") as info:
        RiskRun.from_json(edited)
    assert not isinstance(info.value, KeyError)


def test_non_object_payload_is_malformed() -> None:
    with pytest.raises(ValueError, match="malformed persisted RiskRun"):
        RiskRun.from_json("[]")


def test_non_bool_reproducible_is_malformed(run: RiskRun) -> None:
    edited = _edited(run, lambda p: p.__setitem__("reproducible", "false"))
    with pytest.raises(ValueError, match="reproducible"):
        RiskRun.from_json(edited)


# ---- K. calculation identity is unchanged --------------------------------------------


def test_calculation_identity_commits_to_no_output(run: RiskRun) -> None:
    assert run.identity == run.manifest.identity()
    assert RunManifest.from_json(run.manifest.to_json()).identity() == run.identity
    assert set(json.loads(run.manifest.to_json())) == _V4_MANIFEST_KEYS
    assert run.result_digest not in run.manifest.to_json()
    moved = np.array(run.var, copy=True)
    moved[-1] = math.nextafter(float(moved[-1]), math.inf)
    other = dataclasses.replace(run, var=moved)
    assert other.identity == run.identity  # the output never enters identity
    assert other.manifest.identity() == run.manifest.identity()
    assert other.result_digest != run.result_digest  # the artifact digest does


def test_identity_is_independent_of_the_result_digest_field(run: RiskRun) -> None:
    again = run_scenario_var_es(**rf.inputs())
    assert again.identity == run.identity
    assert again.result_digest == run.result_digest
    assert _hexes(again.var) == _hexes(run.var) and _hexes(again.es) == _hexes(run.es)


# ---- L. exploratory seed=None --------------------------------------------------------


def test_exploratory_seed_none_keeps_identity_and_digests_its_own_result() -> None:
    a = run_scenario_var_es(
        **rf.inputs(scenario_config=rf.sbb(None)), mode="exploratory"
    )
    b = run_scenario_var_es(
        **rf.inputs(scenario_config=rf.sbb(None)), mode="exploratory"
    )
    assert a.reproducible is False and b.reproducible is False
    assert a.identity == b.identity  # D040/D042 semantics unchanged
    same_numbers = _hexes(a.var) == _hexes(b.var) and _hexes(a.es) == _hexes(b.es)
    assert (a.result_digest == b.result_digest) is same_numbers
    for r in (a, b):
        back = RiskRun.from_json(r.to_json())
        assert back.result_digest == r.result_digest and back.reproducible is False


# ---- M. fresh process ----------------------------------------------------------------


_CHILD = textwrap.dedent(
    """
    import json
    import _resultfix as rf
    from riskquant.solutions.risk_run.api import run_scenario_var_es
    r = run_scenario_var_es(**rf.inputs(scenario_config=rf.sbb(7)))
    print(json.dumps({"run": r.to_json(), "digest": r.result_digest}))
    """
)


def _child(env_extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(env_extra or {})
    env["PYTHONPATH"] = os.pathsep.join([str(_HERE), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_processes_agree_on_result_digest(other_run: RiskRun) -> None:
    a, b = _child(), _child()
    assert a["digest"] == b["digest"] == other_run.result_digest
    assert a["run"] == b["run"]
    assert RiskRun.from_json(a["run"]).result_digest == a["digest"]


# ---- N. batch-size and thread-configuration invariance -------------------------------


@pytest.mark.parametrize("batch_size", [1, 3, 64, rf.S_SBB, rf.S_SBB + 1])
def test_result_digest_is_invariant_to_batch_size(
    other_run: RiskRun, batch_size: int
) -> None:
    got = run_scenario_var_es(
        **rf.inputs(scenario_config=rf.sbb(7), batch_size=batch_size)
    )
    assert got.identity == other_run.identity
    assert got.result_digest == other_run.result_digest


def test_result_digest_is_invariant_to_numba_thread_configuration(
    other_run: RiskRun,
) -> None:
    """Configured thread counts only; genuine multi-thread execution is proven
    by `test_release_certification.py`, not here."""
    digests = {_child({"NUMBA_NUM_THREADS": str(t)})["digest"] for t in (1, 4)}
    assert digests == {other_run.result_digest}
