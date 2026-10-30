# tests/solutions/credit_var/test_result_contract.py
"""
WS_CREDIT_VAR_API S1: result contract of `run_credit_var` (C.1 to C.11).

Tests S1.T1 to S1.T12 of the workstream document. Each test states the clause
it pins. The oracles are independent of the code under test:

* expected orderings, keys and versions are written out here;
* the build identity (S1.T10) and the input hash (S1.T11) are recomputed here
  from the text of P-CREDIT_VAR_API-001 clause 5 and A_CONTRACT section 4.2,
  not by a call of the production helper.

Books are tiny (N <= 5, S = 2_000): these tests pin shapes, keys, order and
serialization, not model numbers (S4 owns the numerical oracles).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
import math
import re
import typing
from importlib import metadata
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
import scipy
from pydantic import BaseModel, ValidationError

import riskquant
from riskquant.credit.jtd.models import JtdModel
from riskquant.solutions.credit_var import provenance
from riskquant.solutions.credit_var import run as credit_var_run
from riskquant.solutions.credit_var.api import (
    MODEL_CATALOG,
    CreditVarRequest,
    CreditVarResult,
    LossHistogram,
    ModelTail,
    ObligorContribution,
    RiskSummary,
    RunProvenance,
    StressTestObligor,
    StressTestResult,
    run_credit_var,
)
from riskquant.solutions.credit_var.errors import CreditVarEngineError
from riskquant.solutions.credit_var.schemas import AlphaRisk, PercentileLoss

_S = 2_000
_SEED = 4242
_ENGINE_VERSION_RE = re.compile(r"^[^+]+\+src\.[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JTD_NAMES = frozenset(m.name for m in JtdModel)


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------


def _obligor(
    uid: int, ead: float, pd: float, lgd: float, **flags: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {"obligor_uid": uid, "ead": ead, "pd": pd, "lgd": lgd}
    row.update(flags)
    return row


def _obligors() -> list[dict[str, Any]]:
    """Four names, request order deliberately not sorted by uid. 20 is forced
    to default inside the simulation; 30 and 40 are in the stress test."""
    return [
        _obligor(30, 1.0e6, 0.03, 0.45, in_stress_test=True),
        _obligor(10, 2.5e6, 0.05, 0.60),
        _obligor(20, 3.0e5, 0.04, 1.00, force_default=True),
        _obligor(40, 7.0e5, 0.06, 0.25, in_stress_test=True),
    ]


def _book(
    model: str, *, alphas: tuple[float, ...] = (0.999, 0.95, 0.99)
) -> dict[str, Any]:
    """One valid request per catalog key (N = 4, S = 2_000)."""
    common: dict[str, Any] = {
        "model": model,
        "n_scenarios": _S,
        "alphas": list(alphas),
        "seed": _SEED,
    }
    obligors = _obligors()
    grouped = [
        {**row, "group": "A" if i % 2 == 0 else "B"} for i, row in enumerate(obligors)
    ]
    loadings = [
        {"obligor_uid": row["obligor_uid"], "factor": factor, "loading": loading}
        for row in obligors
        for factor, loading in (("F1", 0.5), ("F2", 0.3))
    ]
    with_rho = copy.deepcopy(obligors)
    with_rho[1]["rho"] = 0.3
    books: dict[str, dict[str, Any]] = {
        "forced_default": {"obligors": obligors},
        "independent_bernoulli": {"obligors": obligors},
        "stress_regime_bernoulli": {
            "obligors": obligors,
            "params": {"stress_prob": 0.1, "stress_pd_multiplier": 3.0},
        },
        "stress_regime_bernoulli_grouped": {
            "obligors": grouped,
            "groups": [
                {
                    "name": "A",
                    "params": {"stress_prob": 0.1, "stress_pd_multiplier": 3.0},
                },
                {
                    "name": "B",
                    "params": {"stress_prob": 0.2, "stress_pd_multiplier": 4.0},
                },
            ],
        },
        "beta_binomial": {"obligors": obligors, "params": {"alpha": 2.0, "beta": 38.0}},
        "beta_binomial_grouped": {
            "obligors": grouped,
            "groups": [
                {"name": "A", "params": {"alpha": 1.0, "beta": 99.0}},
                {"name": "B", "params": {"alpha": 4.0, "beta": 16.0}},
            ],
        },
        "sf_gaussian": {"obligors": with_rho, "params": {"rho": 0.15}},
        "sf_gaussian_basel": {"obligors": obligors},
        "sf_students_t": {"obligors": with_rho, "params": {"rho": 0.15, "nu": 7.0}},
        "sf_students_t_basel": {"obligors": obligors, "params": {"nu": 7.0}},
        "mf_gaussian": {"obligors": obligors, "loadings": loadings},
        "mf_students_t": {
            "obligors": obligors,
            "loadings": loadings,
            "params": {"nu": 7.0},
        },
        "marshall_olkin": {"obligors": obligors, "params": {"systemic_pd": 0.02}},
    }
    return {**common, **books[model]}


_BOOK_KEYS: tuple[str, ...] = (
    "forced_default",
    "independent_bernoulli",
    "stress_regime_bernoulli",
    "stress_regime_bernoulli_grouped",
    "beta_binomial",
    "beta_binomial_grouped",
    "sf_gaussian",
    "sf_gaussian_basel",
    "sf_students_t",
    "sf_students_t_basel",
    "mf_gaussian",
    "mf_students_t",
    "marshall_olkin",
)


def _small_bernoulli(alphas: tuple[float, ...], pd: float = 0.05) -> dict[str, Any]:
    return {
        "model": "independent_bernoulli",
        "obligors": [
            _obligor(1, 1.0e6, pd, 0.45),
            _obligor(2, 2.0e6, pd, 0.60),
            _obligor(3, 5.0e5, pd, 1.00),
        ],
        "n_scenarios": _S,
        "alphas": list(alphas),
        "seed": _SEED,
    }


@pytest.fixture(scope="module")
def sf_result() -> CreditVarResult:
    """One result with every optional part present (stress test, overlay)."""
    return run_credit_var(_book("sf_gaussian"))


def _walk_floats(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [f for item in value.values() for f in _walk_floats(item)]
    if isinstance(value, (list, tuple)):
        return [f for item in value for f in _walk_floats(item)]
    return []


# ---------------------------------------------------------------------------
# S1.T1 (C.1): no NaN or infinity in any float of any result model
# ---------------------------------------------------------------------------

_RESULT_MODELS: tuple[type[BaseModel], ...] = (
    CreditVarResult,
    RiskSummary,
    AlphaRisk,
    PercentileLoss,
    ModelTail,
    ObligorContribution,
    LossHistogram,
    StressTestObligor,
    StressTestResult,
    RunProvenance,
)


def _annotation_leaves(annotation: object) -> list[object]:
    args = typing.get_args(annotation)
    if not args:
        return [annotation]
    return [leaf for arg in args for leaf in _annotation_leaves(arg)]


def _float_fields(model: type[BaseModel]) -> tuple[list[tuple[str, str]], list[str]]:
    """(name, kind) of each float-bearing field, and the names of fields that
    hold a float in a form this test does not inject into."""
    handled: list[tuple[str, str]] = []
    unhandled: list[str] = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        origin, args = typing.get_origin(annotation), typing.get_args(annotation)
        if annotation is float:
            handled.append((name, "scalar"))
        elif origin is dict and args == (str, float):
            handled.append((name, "dict"))
        elif origin is tuple and args == (float, Ellipsis):
            handled.append((name, "tuple"))
        elif float in _annotation_leaves(annotation):
            unhandled.append(name)
    return handled, unhandled


_FLOAT_CASES = [
    (model, name, kind)
    for model in _RESULT_MODELS
    for name, kind in _float_fields(model)[0]
]


def _payload_for(model: type[BaseModel], full: dict[str, Any]) -> dict[str, Any]:
    sub: dict[str, Any]
    if model is CreditVarResult:
        sub = full
    elif model is RiskSummary:
        sub = full["summary"]
    elif model is AlphaRisk:
        sub = full["summary"]["by_alpha"][0]
    elif model is PercentileLoss:
        sub = full["loss_percentiles"][0]
    elif model is ModelTail:
        sub = next(iter(full["per_model_tail"].values()))
    elif model is ObligorContribution:
        sub = full["obligor_contributions"][0]
    elif model is LossHistogram:
        sub = full["loss_histogram"]
    elif model is StressTestObligor:
        sub = full["stress_test"]["obligors"][0]
    elif model is StressTestResult:
        sub = full["stress_test"]
    elif model is RunProvenance:
        sub = full["provenance"]
    else:  # pragma: no cover - the case list holds only the models above
        raise AssertionError(model)
    return sub


def test_s1_t1_every_float_field_is_enumerated() -> None:
    """Guard of the guard: every float in a result model is in a form that
    S1.T1 injects into, so no float field escapes the NaN/inf check."""
    for model in _RESULT_MODELS:
        _, unhandled = _float_fields(model)
        assert unhandled == [], f"{model.__name__}: untested float fields {unhandled}"
    assert len(_FLOAT_CASES) == 34


@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"]
)
@pytest.mark.parametrize(
    ("model", "name", "kind"),
    _FLOAT_CASES,
    ids=[f"{m.__name__}.{n}" for m, n, _ in _FLOAT_CASES],
)
def test_s1_t1_non_finite_float_is_rejected(
    sf_result: CreditVarResult,
    model: type[BaseModel],
    name: str,
    kind: str,
    bad: float,
) -> None:
    payload = copy.deepcopy(_payload_for(model, sf_result.model_dump(mode="json")))
    model.model_validate(payload)  # the unmodified payload is valid
    if kind == "scalar":
        payload[name] = bad
    elif kind == "dict":
        assert payload[name], f"{name}: need a non-empty dict to inject into"
        payload[name][next(iter(payload[name]))] = bad
    else:
        assert payload[name], f"{name}: need a non-empty tuple to inject into"
        payload[name][0] = bad
    with pytest.raises(ValidationError) as excinfo:
        model.model_validate(payload)
    assert [e["type"] for e in excinfo.value.errors()] == ["finite_number"]


# ---------------------------------------------------------------------------
# S1.T2 (C.2): foreign result schema version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["0.9", "1.1"])
def test_s1_t2_foreign_result_schema_version_is_rejected(
    sf_result: CreditVarResult, version: str
) -> None:
    assert sf_result.result_schema_version == "1.0"
    payload = sf_result.model_dump(mode="json")
    payload["result_schema_version"] = version
    with pytest.raises(ValidationError) as from_python:
        CreditVarResult.model_validate(payload)
    assert from_python.value.errors()[0]["loc"] == ("result_schema_version",)
    with pytest.raises(ValidationError):
        CreditVarResult.model_validate_json(json.dumps(payload))


def test_s1_t2_missing_result_schema_version_is_rejected(
    sf_result: CreditVarResult,
) -> None:
    payload = sf_result.model_dump(mode="json")
    del payload["result_schema_version"]
    with pytest.raises(ValidationError) as excinfo:
        CreditVarResult.model_validate(payload)
    assert excinfo.value.errors()[0]["type"] == "missing"


# ---------------------------------------------------------------------------
# S1.T3, S1.T4 (C.3): levels are data, not label keys
# ---------------------------------------------------------------------------


def test_s1_t3_close_levels_give_distinct_ascending_records() -> None:
    requested = (0.999, 0.95, 0.9999, 0.99991)
    result = run_credit_var(_small_bernoulli(requested))
    records = result.summary.by_alpha
    assert len(records) == 4
    assert [r.alpha.hex() for r in records] == [a.hex() for a in sorted(requested)]
    assert all(lo.alpha < hi.alpha for lo, hi in itertools.pairwise(records))
    # The provenance keeps the caller's order; only the result sorts.
    assert [a.hex() for a in result.provenance.alphas] == [a.hex() for a in requested]


def test_s1_t3_no_result_dict_is_keyed_by_a_level_label(
    sf_result: CreditVarResult,
) -> None:
    payload = sf_result.model_dump(mode="json")
    assert isinstance(payload["loss_percentiles"], list)
    assert isinstance(payload["summary"]["by_alpha"], list)
    for removed in ("var", "es", "economic_capital_var", "economic_capital_es"):
        assert removed not in payload["summary"]
    assert set(payload["summary"]["expected_loss_by_model"]) <= _JTD_NAMES
    assert set(payload["per_model_tail"]) <= _JTD_NAMES
    percentiles = [p["percentile"] for p in payload["loss_percentiles"]]
    assert percentiles == [
        0.1, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0,
        90.0, 95.0, 97.5, 99.0, 99.5, 99.9, 99.95, 99.99,
    ]  # fmt: skip


def test_s1_t4_levels_near_one_are_not_saturated() -> None:
    requested = (0.99995, 0.999999)
    records = run_credit_var(_small_bernoulli(requested)).summary.by_alpha
    assert len(records) == 2
    assert all(r.alpha != 1.0 for r in records)
    assert [r.alpha.hex() for r in records] == [a.hex() for a in requested]


# ---------------------------------------------------------------------------
# S1.T5 (C.4): exact lookup, KeyError otherwise
# ---------------------------------------------------------------------------


def test_s1_t5_at_is_exact_and_raises_key_error() -> None:
    summary = run_credit_var(_small_bernoulli((0.95, 0.999))).summary
    record = summary.at(0.999)
    assert record.alpha.hex() == (0.999).hex()
    with pytest.raises(KeyError):
        summary.at(float(np.nextafter(0.999, 1.0)))
    with pytest.raises(KeyError):
        summary.at(0.5)
    for removed in ("max_alpha", "get_var", "get_es"):
        assert not hasattr(summary, removed)


def test_s1_t5_by_alpha_must_be_strictly_ascending(sf_result: CreditVarResult) -> None:
    payload = sf_result.summary.model_dump(mode="json")
    payload["by_alpha"] = list(reversed(payload["by_alpha"]))
    with pytest.raises(ValidationError):
        RiskSummary.model_validate(payload)
    payload["by_alpha"] = [payload["by_alpha"][0], payload["by_alpha"][0]]
    with pytest.raises(ValidationError):
        RiskSummary.model_validate(payload)


# ---------------------------------------------------------------------------
# S1.T6 (C.5): VaR = 0, no ratio member, no NaN
# ---------------------------------------------------------------------------


def test_s1_t6_zero_var_gives_no_ratio_no_exception_no_nan() -> None:
    result = run_credit_var(_small_bernoulli((0.95,), pd=0.001))
    assert result.summary.at(0.95).var == 0.0  # precondition of the case
    for removed in ("es_to_var_ratio", "get_es_to_var_ratio"):
        assert not hasattr(result.summary, removed)
        assert not hasattr(RiskSummary, removed)
    for tail in result.per_model_tail.values():
        assert not hasattr(tail, "es_to_var_ratio")
    assert not hasattr(ModelTail, "es_to_var_ratio")
    floats = _walk_floats(result.model_dump(mode="python"))
    assert floats
    assert all(math.isfinite(f) for f in floats)


# ---------------------------------------------------------------------------
# S1.T7 (C.6): obligor order, request ead, missing ead is an engine fault
# ---------------------------------------------------------------------------


def _three_name_book() -> dict[str, Any]:
    return {
        "model": "sf_gaussian",
        "obligors": [
            _obligor(30, 1234567.125, 0.03, 0.45, in_stress_test=True),
            _obligor(10, 7654321.25, 0.05, 0.60, in_stress_test=True),
            _obligor(20, 333333.5, 0.04, 1.00, force_default=True),
        ],
        "params": {"rho": 0.15},
        "n_scenarios": _S,
        "alphas": [0.99],
        "seed": _SEED,
    }


def test_s1_t7_contributions_ascending_with_request_ead() -> None:
    book = _three_name_book()
    result = run_credit_var(book)
    assert [c.obligor_uid for c in result.obligor_contributions] == [10, 20, 30]
    request_ead = {row["obligor_uid"]: row["ead"] for row in book["obligors"]}
    for contribution in result.obligor_contributions:
        assert contribution.ead.hex() == request_ead[contribution.obligor_uid].hex()
    assert result.stress_test is not None
    assert [o.obligor_uid for o in result.stress_test.obligors] == [10, 30]


def test_s1_t7_row_without_request_ead_raises_engine_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = credit_var_run._build_loss_matrix

    def foreign_row_uid(
        request: CreditVarRequest, alphas: tuple[float, ...]
    ) -> credit_var_run._LossMatrix:
        matrix = original(request, alphas)
        uids = matrix.uids.copy()
        uids[0] = 99  # not a request obligor
        return dataclasses.replace(matrix, uids=uids)

    monkeypatch.setattr(credit_var_run, "_build_loss_matrix", foreign_row_uid)
    with pytest.raises(CreditVarEngineError, match="99"):
        run_credit_var(_three_name_book())
    assert not issubclass(CreditVarEngineError, ValueError)


# ---------------------------------------------------------------------------
# S1.T8 (C.7): one key vocabulary, "FORCED_DEFAULT"
# ---------------------------------------------------------------------------


def test_s1_t8_forced_default_model_key() -> None:
    result = run_credit_var(_book("forced_default"))
    assert set(result.per_model_tail) == {"FORCED_DEFAULT"}
    assert set(result.summary.expected_loss_by_model) == {"FORCED_DEFAULT"}


def test_s1_t8_forced_default_overlay_key() -> None:
    book = _book("sf_gaussian")
    result = run_credit_var(book)
    assert set(result.per_model_tail) == {"COPULA_SF_GAUSSIAN", "FORCED_DEFAULT"}
    assert set(result.summary.expected_loss_by_model) == {
        "COPULA_SF_GAUSSIAN",
        "FORCED_DEFAULT",
    }
    assert "forced_default" not in result.per_model_tail
    assert "forced_default" not in result.summary.expected_loss_by_model
    forced = sum(
        r["ead"] * r["lgd"] for r in book["obligors"] if r.get("force_default")
    )
    tail = result.per_model_tail["FORCED_DEFAULT"]
    assert tail.var == tail.es == forced
    assert result.summary.expected_loss_by_model["FORCED_DEFAULT"] == forced


# ---------------------------------------------------------------------------
# S1.T9 (C.8): numerical environment and construction id
# ---------------------------------------------------------------------------


def test_s1_t9_provenance_records_the_numerical_environment(
    sf_result: CreditVarResult,
) -> None:
    stamp = sf_result.provenance
    assert stamp.numpy_version == np.__version__
    assert stamp.scipy_version == scipy.__version__
    assert stamp.numba_version == numba.__version__
    assert stamp.scenario_construction_id == "MixedCreditFactorGenerator/D-C15"
    payload = stamp.model_dump(mode="json")
    payload["scenario_construction_id"] = "MixedCreditFactorGenerator/pre-D-C15"
    with pytest.raises(ValidationError):
        RunProvenance.model_validate(payload)


# ---------------------------------------------------------------------------
# S1.T10 (C.9): build identity, no fallback
# ---------------------------------------------------------------------------


def _clause_5_digest(root: Path) -> str:
    """P-CREDIT_VAR_API-001 clause 5, written out from the text."""
    files = sorted(
        (p.relative_to(root).as_posix(), p)
        for p in root.rglob("*.py")
        if "__pycache__" not in p.relative_to(root).parts
    )
    digest = hashlib.sha256()
    for relative, path in files:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def test_s1_t10_engine_version_format_and_value(sf_result: CreditVarResult) -> None:
    stamp = sf_result.provenance.engine_version
    assert _ENGINE_VERSION_RE.fullmatch(stamp)
    assert riskquant.__file__ is not None
    expected = (
        f"{metadata.version('riskquant')}+src."
        f"{_clause_5_digest(Path(riskquant.__file__).parent)}"
    )
    assert stamp == expected
    assert provenance.engine_version() == stamp
    assert provenance.compute_engine_version() == stamp


def test_s1_t10_missing_metadata_raises_engine_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_distribution(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", no_distribution)
    with pytest.raises(CreditVarEngineError) as excinfo:
        provenance.compute_engine_version()
    assert isinstance(excinfo.value.__cause__, metadata.PackageNotFoundError)


def test_s1_t10_local_version_segment_raises_engine_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata, "version", lambda name: "0.2.0+x")
    with pytest.raises(CreditVarEngineError):
        provenance.compute_engine_version()


def test_s1_t10_run_fails_before_simulation_without_build_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold cache and missing metadata: the run raises E and never reaches
    the simulation. The failure is not cached."""

    def no_distribution(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    def must_not_run(request: CreditVarRequest, alphas: tuple[float, ...]) -> None:
        raise AssertionError("simulation reached without a build identity")

    monkeypatch.setattr(provenance, "_engine_version_value", None)
    monkeypatch.setattr(metadata, "version", no_distribution)
    monkeypatch.setattr(credit_var_run, "_build_loss_matrix", must_not_run)
    with pytest.raises(CreditVarEngineError):
        run_credit_var(_small_bernoulli((0.99,)))
    assert provenance._engine_version_value is None


def test_s1_t10_source_digest_rules(tmp_path: Path) -> None:
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (tmp_path / "b.py").write_bytes(b"z = 3\n")
    (tmp_path / "notes.md").write_bytes(b"not source")
    (tmp_path / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00")
    reference = provenance.source_tree_digest(tmp_path)
    assert reference == _clause_5_digest(tmp_path)

    (tmp_path / "pkg" / "a.py").write_bytes(b"x = 1\ny = 2\n")  # CRLF -> LF
    (tmp_path / "notes.md").write_bytes(b"changed")  # non-.py file
    (tmp_path / "pkg" / "k.nbi").write_bytes(b"numba index")  # cache file
    assert provenance.source_tree_digest(tmp_path) == reference

    (tmp_path / "b.py").write_bytes(b"z = 4\n")
    assert provenance.source_tree_digest(tmp_path) != reference


# ---------------------------------------------------------------------------
# S1.T11 (C.10): input hash, D040 canonical form
# ---------------------------------------------------------------------------


def _canonical_hash(request: CreditVarRequest) -> str:
    """A_CONTRACT section 4.2, written out from the text."""

    def canon(value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return {"__f64__": value.hex()}
        if isinstance(value, dict):
            return {str(k): canon(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [canon(v) for v in value]
        return value

    text = json.dumps(
        canon(request.model_dump(mode="python")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_s1_t11_input_hash_identifies_bit_equal_requests() -> None:
    book = _book("sf_gaussian")
    from_dict = CreditVarRequest.model_validate(book)
    from_json = CreditVarRequest.model_validate_json(json.dumps(book))
    digest = provenance.request_input_hash(from_dict)
    assert _SHA256_RE.fullmatch(digest)
    assert provenance.request_input_hash(from_json) == digest
    assert digest == _canonical_hash(from_dict)

    one_ulp = copy.deepcopy(book)
    one_ulp["obligors"][0]["ead"] = math.nextafter(book["obligors"][0]["ead"], math.inf)
    assert (
        provenance.request_input_hash(CreditVarRequest.model_validate(one_ulp))
        != digest
    )

    positive, negative = copy.deepcopy(book), copy.deepcopy(book)
    positive["obligors"][0]["ead"] = 0.0
    negative["obligors"][0]["ead"] = -0.0
    assert provenance.request_input_hash(
        CreditVarRequest.model_validate(positive)
    ) != provenance.request_input_hash(CreditVarRequest.model_validate(negative))


def test_s1_t11_result_carries_the_request_hash(sf_result: CreditVarResult) -> None:
    request = CreditVarRequest.model_validate(_book("sf_gaussian"))
    assert sf_result.provenance.input_hash == _canonical_hash(request)


# ---------------------------------------------------------------------------
# S1.T12 (C.11): lossless JSON for every catalog key
# ---------------------------------------------------------------------------


def test_s1_t12_books_cover_the_catalog() -> None:
    assert sorted(_BOOK_KEYS) == sorted(MODEL_CATALOG)


@pytest.mark.parametrize("model", _BOOK_KEYS)
def test_s1_t12_json_round_trip_is_lossless(model: str) -> None:
    result = run_credit_var(_book(model))
    first = result.model_dump_json()
    restored = CreditVarResult.model_validate_json(first)
    assert restored == result
    assert restored.model_dump_json() == first
