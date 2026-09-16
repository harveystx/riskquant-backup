# tests/solutions/run_identity/test_run_identity.py
"""
Run-identity contract tests.

The manifest is only production-grade if it tracks the numerical dependency
boundary, so the suite is built from three adversarial directions:

  * MUTATION — change a result-defining input: the risk result changes AND the
    run identity changes;
  * FALSE POSITIVE — change something the result does not depend on: both the
    result and the identity stay put;
  * STALE INPUT — present a substituted snapshot against a persisted manifest:
    it is detected and raised, never silently computed.

Hashing is checked against an oracle constructed independently in the test, not
by calling the implementation twice. The end-to-end reproduction oracle runs in
a genuinely fresh interpreter.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.scenario_metrics.api.var_es import compute_var_es
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from riskquant.solutions.run_identity.api import (
    InputMismatchError,
    build_run_manifest,
    scenario_config_from_manifest,
    verify_run_inputs,
)
from riskquant.solutions.run_identity.hashing import digest_array, digest_named_arrays
from riskquant.solutions.run_identity.schemas import RunManifest
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)

_K, _T, _S = 4, 80, 64


# --------------------------------------------------------------------------
# the risk path under test: config -> scenarios -> shocks -> P&L -> FX -> VaR/ES
# --------------------------------------------------------------------------


def _inputs() -> dict:
    rng = np.random.default_rng(0)
    vals = np.asfortranarray(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (_T, _K)), axis=0))
    )
    matrix = RiskFactorMatrix(
        dates=np.arange(_T, dtype=np.int32),
        factor_ids=np.arange(_K, dtype=np.int32),
        values=vals,
    )
    return {
        "scenario_cfg": {
            "model": "sbb",
            "horizon": 1,
            "total_scenarios": _S,
            "expected_block_size": 5.0,
            "seed": 7,
        },
        "data": matrix,
        "required_shift_type_id": np.array([SHIFT_LOG_RETURN_ID] * _K, np.int8),
        "source_factor_ids": np.arange(_K, dtype=np.int32),
        "shock_scale": np.ones(_K, np.float32),
        "base_levels": np.ascontiguousarray(vals[-1].astype(np.float64)),
        "sensitivities": np.array([1000.0, -500.0, 250.0, 750.0], np.float64),
        "reporting_currency": "USD",
        "fx": np.array([1.25], np.float64),
        "confidence_level": 0.99,
        "batch_size": 16,  # execution knob only (D041): not part of identity
    }


def _run_risk(inp: dict) -> tuple[float, float, np.ndarray]:
    gen = build_scenario_generator(
        parse_scenario_model_config(inp["scenario_cfg"]), data=inp["data"]
    )
    n = inp["base_levels"].shape[0]
    chunks = []
    for batch in gen.iter_batches(
        required_shift_type_id=inp["required_shift_type_id"],
        source_factor_ids=inp["source_factor_ids"],
        shock_scale=inp["shock_scale"],
        batch_size=inp["batch_size"],
    ):
        out = np.empty((n, batch.shocks.shape[0]), dtype=np.float64, order="C")
        apply_shocks_to_base_levels(
            base_levels=inp["base_levels"], batch=batch, out=out
        )
        chunks.append(inp["sensitivities"] @ (out - inp["base_levels"][:, None]))
    pnl = np.concatenate(chunks) * float(inp["fx"][0])
    res = compute_var_es(
        np.ascontiguousarray(pnl.reshape(1, -1)), inp["confidence_level"]
    )
    return (
        float(np.asarray(res.var).ravel()[0]),
        float(np.asarray(res.es).ravel()[0]),
        pnl,
    )


def _manifest(inp: dict) -> RunManifest:
    return build_run_manifest(
        scenario_config=parse_scenario_model_config(inp["scenario_cfg"]),
        data=inp["data"],
        required_shift_type_id=inp["required_shift_type_id"],
        source_factor_ids=inp["source_factor_ids"],
        shock_scale=inp["shock_scale"],
        base_levels=inp["base_levels"],
        sensitivities_digest=digest_array(inp["sensitivities"]),
        reporting_currency=inp["reporting_currency"],
        fx_digest=digest_array(inp["fx"]),
        confidence_level=inp["confidence_level"],
    )


def _remat(matrix: RiskFactorMatrix, **over) -> RiskFactorMatrix:
    parts = {
        "dates": matrix.dates,
        "factor_ids": matrix.factor_ids,
        "values": matrix.values,
    }
    parts.update(over)
    return RiskFactorMatrix(**parts)


# --------------------------------------------------------------------------
# hashing contract (independent oracle)
# --------------------------------------------------------------------------


def test_digest_matches_independent_oracle() -> None:
    """Oracle built here from hashlib, not by calling the implementation twice."""
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode("utf-8"))
    h.update(b"|")
    h.update(str(a.shape).encode("utf-8"))
    h.update(b"|")
    h.update(np.ascontiguousarray(a).tobytes(order="C"))
    assert digest_array(a) == h.hexdigest()


def test_content_identity_not_object_identity() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert digest_array(a) == digest_array(a.copy())
    assert digest_array(np.asfortranarray(a)) == digest_array(a)


@pytest.mark.parametrize(
    "transform",
    [
        lambda a: a.T,  # different logical matrix
        lambda a: a.reshape(3, 2),  # shape folded in
        lambda a: a.astype(np.float32),  # dtype folded in
        lambda a: a + 1.0,
    ],
    ids=["transpose", "reshape", "dtype", "values"],
)
def test_digest_distinguishes_different_interpretations(transform) -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert digest_array(transform(a)) != digest_array(a)


def test_one_ulp_difference_changes_digest() -> None:
    """Floats are hashed as exact binary; never rounded or stringified."""
    a = np.array([1.0, 2.0], dtype=np.float64)
    b = a.copy()
    b[0] = np.nextafter(b[0], 2.0)
    assert digest_array(a) != digest_array(b)


def test_named_bundle_is_order_and_name_sensitive() -> None:
    a = np.arange(3, dtype=np.float64)
    b = a + 1.0
    assert digest_named_arrays(x=a, y=b) != digest_named_arrays(y=b, x=a)
    assert digest_named_arrays(x=a, y=b) != digest_named_arrays(p=a, q=b)


# --------------------------------------------------------------------------
# manifest determinism, serialization, validation
# --------------------------------------------------------------------------


def test_identity_is_deterministic_across_rebuilds() -> None:
    assert _manifest(_inputs()).identity() == _manifest(_inputs()).identity()


def test_canonical_bytes_stable_under_key_insertion_order() -> None:
    inp = _inputs()
    a = _manifest(inp)
    reordered = dict(reversed(list(inp["scenario_cfg"].items())))
    b = _manifest({**inp, "scenario_cfg": reordered})
    assert a.canonical_bytes() == b.canonical_bytes()
    assert a.identity() == b.identity()


def test_json_roundtrip_preserves_identity() -> None:
    m = _manifest(_inputs())
    assert RunManifest.from_json(m.to_json()).identity() == m.identity()


def test_scenario_config_reconstructs_through_the_frozen_seam() -> None:
    inp = _inputs()
    m = _manifest(inp)
    rebuilt = scenario_config_from_manifest(m)
    assert (
        rebuilt.model_dump()
        == parse_scenario_model_config(inp["scenario_cfg"]).model_dump()
    )


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: {**d, "unexpected_field": 1},
        lambda d: {k: v for k, v in d.items() if k != "risk_measure"},
        lambda d: {**d, "manifest_schema_version": "1"},
        lambda d: {**d, "data": {**d["data"], "kind": "not_a_kind"}},
        lambda d: {**d, "execution": {"batch_size": 4}},  # v1 field must be rejected
    ],
    ids=[
        "foreign_field",
        "missing_section",
        "superseded_schema_version",
        "bad_data_kind",
        "bad_batch_size",
    ],
)
def test_malformed_manifest_rejected(corrupt) -> None:
    payload = json.loads(_manifest(_inputs()).to_json())
    with pytest.raises(ValidationError):
        RunManifest.from_json(json.dumps(corrupt(payload)))


def test_manifest_is_immutable() -> None:
    m = _manifest(_inputs())
    with pytest.raises(ValidationError):
        m.risk_measure = None


def test_manifest_carries_no_operational_metadata() -> None:
    """Identity must not drift with time, host or operator."""
    fields = set(json.loads(_manifest(_inputs()).to_json()))
    assert not fields & {"timestamp", "created_at", "host", "operator", "job_id"}


# --------------------------------------------------------------------------
# MUTATION: result-defining input -> result changes AND identity changes
# --------------------------------------------------------------------------


def _revised_cell(values: np.ndarray) -> np.ndarray:
    """One historical observation silently revised by 1%."""
    revised = np.array(values, copy=True)
    revised[30, 1] *= 1.01
    return revised


def _mutations():
    base = _inputs()
    return {
        # A SINGLE revised observation. A uniform rescale of the whole history
        # would be return-neutral for a log-return model and is covered
        # separately by the conservative-direction test below.
        "historical_values": {
            "data": _remat(
                base["data"],
                values=np.asfortranarray(_revised_cell(base["data"].values)),
            )
        },
        "column_ordering": {
            "data": _remat(
                base["data"],
                values=np.asfortranarray(
                    np.array(base["data"].values)[:, [1, 0, 2, 3]]
                ),
            )
        },
        "shift_type": {"required_shift_type_id": np.array([2] * _K, np.int8)},
        "source_factor_ids": {"source_factor_ids": np.array([1, 1, 2, 3], np.int32)},
        "shock_scale": {"shock_scale": np.array([0.95, 1, 1, 1], np.float32)},
        "base_levels": {"base_levels": base["base_levels"] * 1.0001},
        "sensitivities": {"sensitivities": base["sensitivities"] * 1.01},
        "reporting_fx": {"fx": np.array([1.26], np.float64)},
        "confidence_level": {"confidence_level": 0.95},
        "seed": {"scenario_cfg": {**base["scenario_cfg"], "seed": 8}},
        "model": {
            "scenario_cfg": {
                "model": "filtered_bootstrap",
                "horizon": 1,
                "total_scenarios": _S,
                "seed": 7,
            }
        },
    }


@pytest.mark.parametrize("name", list(_mutations()))
def test_result_defining_mutation_changes_result_and_identity(name) -> None:
    base = _inputs()
    mutated = {**base, **_mutations()[name]}
    var0, es0, pnl0 = _run_risk(base)
    var1, es1, pnl1 = _run_risk(mutated)
    # the risk result really does depend on this input ...
    assert not np.array_equal(pnl0, pnl1) or (var0, es0) != (var1, es1)
    # ... and the identity tracks it, so the change cannot masquerade as the
    # same run.
    assert _manifest(base).identity() != _manifest(mutated).identity()


def test_identity_errs_on_the_conservative_side() -> None:
    """A uniform rescale of the whole history is return-neutral for a
    log-return model, so the risk result is unchanged, yet the data digest
    changes and the run is reported as a different run.

    That direction is safe and deliberate: the contract forbids "same identity,
    different result", not "different identity, same result".
    """
    base = _inputs()
    rescaled = {
        **base,
        "data": _remat(
            base["data"],
            values=np.asfortranarray(np.array(base["data"].values) * 1.01),
        ),
    }
    assert np.array_equal(_run_risk(base)[2], _run_risk(rescaled)[2])
    assert _manifest(base).identity() != _manifest(rescaled).identity()


# --------------------------------------------------------------------------
# FALSE POSITIVE: result-independent change -> identity stable
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rebuild",
    [
        lambda inp: {
            **inp,
            "data": _remat(
                inp["data"],
                values=np.asfortranarray(np.array(inp["data"].values, copy=True)),
                dates=np.array(inp["data"].dates, copy=True),
                factor_ids=np.array(inp["data"].factor_ids, copy=True),
            ),
        },
        lambda inp: {**inp, "sensitivities": inp["sensitivities"].copy()},
        lambda inp: {
            **inp,
            "base_levels": np.ascontiguousarray(inp["base_levels"].copy()),
        },
    ],
    ids=["data_copied", "sensitivities_copied", "base_levels_copied"],
)
def test_object_identity_does_not_affect_run_identity(rebuild) -> None:
    base = _inputs()
    other = rebuild(base)
    assert _run_risk(base)[0] == _run_risk(other)[0]
    assert _manifest(base).identity() == _manifest(other).identity()


def test_batch_size_is_not_part_of_run_identity() -> None:
    """D041: the official risk path is bitwise invariant to batch partitioning,
    so the batch size is an execution knob and must not perturb the identity."""
    inp = _inputs()
    a = _manifest(inp)
    b = _manifest({**inp, "batch_size": 7})
    assert a.identity() == b.identity()


# --------------------------------------------------------------------------
# STALE INPUT detection
# --------------------------------------------------------------------------


def test_verify_accepts_the_identified_inputs() -> None:
    inp = _inputs()
    verify_run_inputs(
        _manifest(inp),
        data=inp["data"],
        sensitivities=inp["sensitivities"],
        shock_scale=inp["shock_scale"],
        confidence_level=inp["confidence_level"],
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("sensitivities", np.array([1000.0, -500.0, 250.0, 751.0], np.float64)),
        ("shock_scale", np.array([0.9, 1, 1, 1], np.float32)),
        ("source_factor_ids", np.array([3, 2, 1, 0], np.int32)),
        ("base_levels", np.arange(4, dtype=np.float64)),
        ("fx", np.array([9.99], np.float64)),
        ("confidence_level", 0.90),
        ("reporting_currency", "EUR"),
    ],
)
def test_substituted_input_is_detected(name, value) -> None:
    with pytest.raises(InputMismatchError):
        verify_run_inputs(_manifest(_inputs()), **{name: value})


def test_stale_data_snapshot_is_detected() -> None:
    """Manifest from run A + data snapshot from run B must fail fast rather than
    compute a plausible different VaR."""
    inp = _inputs()
    revised = np.array(inp["data"].values, copy=True)
    revised[30, 1] *= 1.01
    with pytest.raises(InputMismatchError):
        verify_run_inputs(
            _manifest(inp),
            data=_remat(inp["data"], values=np.asfortranarray(revised)),
        )


def test_wrong_data_kind_is_detected() -> None:
    parametric = ParametricRiskData(
        factor_ids=np.arange(_K, dtype=np.int32),
        cholesky_matrix=np.ascontiguousarray(np.eye(_K, dtype=np.float64)),
        volatilities=np.ascontiguousarray(np.full(_K, 0.1)),
    )
    with pytest.raises(InputMismatchError):
        verify_run_inputs(_manifest(_inputs()), data=parametric)


def test_unknown_input_name_rejected() -> None:
    with pytest.raises(TypeError):
        verify_run_inputs(_manifest(_inputs()), not_an_input=1)


# --------------------------------------------------------------------------
# FRESH-PROCESS end-to-end reproduction oracle
# --------------------------------------------------------------------------


_CHILD = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {tests!r})
    from test_run_identity import _inputs, _run_risk, _manifest
    inp = _inputs()
    var, es, pnl = _run_risk(inp)
    m = _manifest(inp)
    print(json.dumps({{"var": var.hex(), "es": es.hex(),
                       "identity": m.identity(),
                       "manifest": m.to_json()}}))
    """
)


def _run_child(tests_dir: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(tests=tests_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_process_reproduces_identity_and_risk_numbers(request) -> None:
    """Process A computes and dies; process B recomputes. Bitwise comparison of
    VaR and ES via exact float hex, plus identical run identity and manifest."""
    tests_dir = str(request.path.parent)
    a = _run_child(tests_dir)
    b = _run_child(tests_dir)
    assert a["identity"] == b["identity"]
    assert a["manifest"] == b["manifest"]
    assert a["var"] == b["var"]  # exact IEEE-754 hex
    assert a["es"] == b["es"]


def test_fresh_process_matches_in_process_result(request) -> None:
    child = _run_child(str(request.path.parent))
    var, es, _ = _run_risk(_inputs())
    assert child["var"] == var.hex() and child["es"] == es.hex()
    assert child["identity"] == _manifest(_inputs()).identity()
    # and the persisted manifest reloads to the same identity
    assert RunManifest.from_json(child["manifest"]).identity() == child["identity"]
