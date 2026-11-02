# tests/solutions/credit_var/test_result_contract.py
"""
WS_CREDIT_VAR_API S1: result contract of `run_credit_var` (C.1 to C.11).

Tests S1.T1 to S1.T12, the S1.T16 bounds and S1.T17 of the workstream
document. Each test states the clause it pins. The oracles are independent of
the code under test:

* expected orderings, keys, versions and the sequence fields of each result
  model (A_CONTRACT section 3) are written out here;
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
from collections.abc import Callable, Collection, Mapping, Sequence, Set
from importlib import metadata
from pathlib import Path
from typing import Any, TypeGuard

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
# S1.T1 (C.1, R-002): frozen, no extra field, sequence fields are tuples
# ---------------------------------------------------------------------------

#: A_CONTRACT section 3, written out from the text: the sequence fields of each
#: result model. No other field of the model holds a sequence.
_SEQUENCE_FIELDS: dict[type[BaseModel], frozenset[str]] = {
    CreditVarResult: frozenset({"loss_percentiles", "obligor_contributions"}),
    RiskSummary: frozenset({"by_alpha"}),
    AlphaRisk: frozenset(),
    PercentileLoss: frozenset(),
    ModelTail: frozenset(),
    ObligorContribution: frozenset(),
    LossHistogram: frozenset({"bin_edges", "counts"}),
    StressTestObligor: frozenset(),
    StressTestResult: frozenset({"obligors"}),
    RunProvenance: frozenset({"alphas"}),
}

_MODEL_IDS = [m.__name__ for m in _RESULT_MODELS]
_UNKNOWN_FIELD = "not_a_result_field"


def _is_sequence(value: object) -> TypeGuard[Collection[object]]:
    """A list, a tuple, a set or another sequence. A str, bytes or a mapping is
    not a sequence here."""
    return isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes))


def _model_instances(
    result: CreditVarResult,
) -> dict[type[BaseModel], list[BaseModel]]:
    """Each model instance in `result`, grouped by type, in walk order.

    The walk enters model fields, dict values and the elements of every
    sequence, so a sequence field of the wrong type does not hide the models
    that it holds.
    """
    found: dict[type[BaseModel], list[BaseModel]] = {}

    def walk(value: object) -> None:
        if isinstance(value, BaseModel):
            found.setdefault(type(value), []).append(value)
            for name in type(value).model_fields:
                walk(getattr(value, name))
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif _is_sequence(value):
            for item in value:
                walk(item)

    walk(result)
    return found


def test_s1_t1_real_result_holds_every_result_model(
    sf_result: CreditVarResult,
) -> None:
    """Guard of the guard: the real result that the three C.1 tests below read
    holds at least one instance of each of the 10 result models and of no other
    model, and the sequence-field oracle names each of the 10 models."""
    assert len(_RESULT_MODELS) == 10
    assert set(_model_instances(sf_result)) == set(_RESULT_MODELS)
    assert set(_SEQUENCE_FIELDS) == set(_RESULT_MODELS)


@pytest.mark.parametrize("model", _RESULT_MODELS, ids=_MODEL_IDS)
def test_s1_t1_attribute_assignment_is_rejected(
    sf_result: CreditVarResult, model: type[BaseModel]
) -> None:
    """`frozen=True`: assignment to each field of a real instance raises.

    The test assigns the current value of the field. If the model is not
    frozen, the assignment succeeds and the test fails, but the module-scoped
    result does not change, so no other test sees a changed fixture.
    """
    instance = _model_instances(sf_result)[model][0]
    before = instance.model_dump_json()
    accepted: list[str] = []
    for name in model.model_fields:
        try:
            setattr(instance, name, getattr(instance, name))
        except ValidationError as exc:
            errors = [(e["type"], e["loc"]) for e in exc.errors()]
            assert errors == [("frozen_instance", (name,))], name
        else:
            accepted.append(name)
    assert accepted == [], f"{model.__name__}: assignment accepted for {accepted}"
    assert instance.model_dump_json() == before


@pytest.mark.parametrize("model", _RESULT_MODELS, ids=_MODEL_IDS)
def test_s1_t1_unknown_field_is_rejected(
    sf_result: CreditVarResult, model: type[BaseModel]
) -> None:
    """`extra="forbid"`: one unknown key in a valid payload raises one
    extra_forbidden error at that key, from Python input and from JSON input."""
    assert _UNKNOWN_FIELD not in model.model_fields
    payload = copy.deepcopy(_payload_for(model, sf_result.model_dump(mode="json")))
    model.model_validate(payload)  # the unmodified payload is valid
    payload[_UNKNOWN_FIELD] = 0.0
    with pytest.raises(ValidationError) as from_python:
        model.model_validate(payload)
    with pytest.raises(ValidationError) as from_json:
        model.model_validate_json(json.dumps(payload))
    expected = [("extra_forbidden", (_UNKNOWN_FIELD,))]
    for excinfo in (from_python, from_json):
        assert [(e["type"], e["loc"]) for e in excinfo.value.errors()] == expected


@pytest.mark.parametrize("model", _RESULT_MODELS, ids=_MODEL_IDS)
def test_s1_t1_sequence_fields_are_tuples(
    sf_result: CreditVarResult, model: type[BaseModel]
) -> None:
    """Each sequence field of each instance in a real `run_credit_var` result is
    a `tuple`. `frozen=True` blocks assignment only, so a list field stays
    mutable in place (A_CONTRACT section 1)."""
    for instance in _model_instances(sf_result)[model]:
        values = {name: getattr(instance, name) for name in model.model_fields}
        sequences = {name for name, value in values.items() if _is_sequence(value)}
        assert sequences == _SEQUENCE_FIELDS[model], model.__name__
        wrong = {
            name: type(values[name]).__name__
            for name in sorted(sequences)
            if type(values[name]) is not tuple
        }
        assert wrong == {}, f"{model.__name__}: sequence fields not tuple: {wrong}"


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


# ===========================================================================
# OD-1 (b) result bounds and the ZV zero-variance rule (owner decisions
# 2026-10-31). Each case starts from a valid run result, changes one field
# in its python dump, and validates again, so the payload holds exactly one
# fault. The imports are local so that this block does not depend on the
# import block of the module.
# ===========================================================================

_OD1_BASE_PAYLOAD: dict[str, object] = {}


def _od1_payload() -> dict[str, object]:
    """A deep copy of the python dump of one valid independent_bernoulli run."""
    import copy

    from riskquant.solutions.credit_var.api import run_credit_var

    if not _OD1_BASE_PAYLOAD:
        result = run_credit_var({
            "model": "independent_bernoulli",
            "obligors": [
                {"obligor_uid": 1, "ead": 1.0, "pd": 0.05, "lgd": 1.0},
                {"obligor_uid": 2, "ead": 2.0, "pd": 0.10, "lgd": 1.0},
                {"obligor_uid": 3, "ead": 4.0, "pd": 0.20, "lgd": 1.0},
            ],
            "n_scenarios": 2_000,
            "alphas": [0.95, 0.99],
            "seed": 20261031,
        })
        _OD1_BASE_PAYLOAD.update(result.model_dump(mode="python"))
    return copy.deepcopy(_OD1_BASE_PAYLOAD)


def _od1_single_error(payload: dict[str, object]) -> dict[str, object]:
    """Validate `payload` and return its only ValidationError entry."""
    import pydantic
    import pytest

    from riskquant.solutions.credit_var.api import CreditVarResult

    with pytest.raises(pydantic.ValidationError) as exc_info:
        CreditVarResult.model_validate(payload)
    errors = exc_info.value.errors(include_url=False)
    assert len(errors) == 1, errors
    return dict(errors[0])


def test_od1_base_payload_is_valid() -> None:
    from riskquant.solutions.credit_var.api import CreditVarResult

    CreditVarResult.model_validate(_od1_payload())


def test_od1_loss_std_dev_below_zero_raises() -> None:
    import math

    payload = _od1_payload()
    summary = payload["summary"]
    assert isinstance(summary, dict)
    summary["loss_std_dev"] = math.nextafter(0.0, -1.0)
    error = _od1_single_error(payload)
    assert error["loc"] == ("summary", "loss_std_dev")
    assert error["type"] == "greater_than_equal"


def test_od1_loss_probability_below_zero_raises() -> None:
    import math

    payload = _od1_payload()
    summary = payload["summary"]
    assert isinstance(summary, dict)
    summary["loss_probability"] = math.nextafter(0.0, -1.0)
    error = _od1_single_error(payload)
    assert error["loc"] == ("summary", "loss_probability")
    assert error["type"] == "greater_than_equal"


def test_od1_loss_probability_above_one_raises() -> None:
    import math

    payload = _od1_payload()
    summary = payload["summary"]
    assert isinstance(summary, dict)
    summary["loss_probability"] = math.nextafter(1.0, 2.0)
    error = _od1_single_error(payload)
    assert error["loc"] == ("summary", "loss_probability")
    assert error["type"] == "less_than_equal"


def test_od1_histogram_edge_count_mismatch_raises() -> None:
    payload = _od1_payload()
    histogram = payload["loss_histogram"]
    assert isinstance(histogram, dict)
    histogram["bin_edges"] = tuple(histogram["bin_edges"])[:-1]
    error = _od1_single_error(payload)
    assert error["loc"] == ("loss_histogram",)
    assert "len(counts) + 1" in str(error["msg"])


def test_od1_histogram_edges_not_ascending_raises() -> None:
    payload = _od1_payload()
    histogram = payload["loss_histogram"]
    assert isinstance(histogram, dict)
    edges = list(histogram["bin_edges"])
    edges[1], edges[2] = edges[2], edges[1]
    histogram["bin_edges"] = tuple(edges)
    error = _od1_single_error(payload)
    assert error["loc"] == ("loss_histogram",)
    assert "bin_edges must be strictly ascending" in str(error["msg"])


def test_od1_histogram_counts_sum_not_n_scenarios_raises() -> None:
    payload = _od1_payload()
    histogram = payload["loss_histogram"]
    assert isinstance(histogram, dict)
    counts = list(histogram["counts"])
    counts[0] += 1
    histogram["counts"] = tuple(counts)
    error = _od1_single_error(payload)
    assert error["loc"] == ()
    assert "sum to provenance.n_scenarios" in str(error["msg"])


def test_zv_forced_default_moments_are_exact_zero() -> None:
    """ZV: a constant loss vector gives exactly 0.0 moments.

    The book is one obligor with ead 1.0 and lgd 0.1 over 2000 scenarios: the
    floating-point mean of 2000 copies of 0.1 is 1.4e-17 away from 0.1, so
    without the rule the std is about 1.4e-17 and the skewness about -1.
    """
    from riskquant.solutions.credit_var.api import run_credit_var

    result = run_credit_var({
        "model": "forced_default",
        "obligors": [{"obligor_uid": 1, "ead": 1.0, "pd": 0.01, "lgd": 0.1}],
        "n_scenarios": 2_000,
        "alphas": [0.95],
        "seed": 7,
    })
    summary = result.summary
    assert summary.loss_min == summary.loss_max
    assert summary.loss_std_dev == 0.0
    assert summary.loss_skewness == 0.0
    assert summary.loss_excess_kurtosis == 0.0


# ===========================================================================
# S1.T16 cases 4 to 6, the C.7 per_model_tail keys and S1.T17 (A_CONTRACT
# sections 3.1 and 3.2; owner decisions OD-1, ZV, OD-B; closure review R-004
# to R-007). The fault cases reuse `_od1_payload` and `_od1_single_error`:
# each payload is the dump of one valid run with exactly one changed value,
# and `test_od1_base_payload_is_valid` is the control.
# ===========================================================================


def _od1_summary(payload: dict[str, object]) -> dict[str, object]:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    return summary


def _od1_provenance(payload: dict[str, object]) -> dict[str, object]:
    stamp = payload["provenance"]
    assert isinstance(stamp, dict)
    return stamp


def _od1_tuple(value: object) -> tuple[object, ...]:
    assert isinstance(value, tuple)
    return value


def test_s1_t16_case4_by_alpha_not_strictly_ascending_raises() -> None:
    """S1.T16 case 4: by_alpha with its two records swapped is rejected.

    The swap keeps the alpha set, so only the order rule of RiskSummary can
    fire. `test_s1_t5_by_alpha_must_be_strictly_ascending` validates a
    RiskSummary payload alone and does not count the errors, so it does not
    meet the S1.T16 one-fault form.
    """
    payload = _od1_payload()
    summary = _od1_summary(payload)
    records = _od1_tuple(summary["by_alpha"])
    assert len(records) == 2
    summary["by_alpha"] = (records[1], records[0])
    error = _od1_single_error(payload)
    assert error["loc"] == ("summary",)
    assert error["type"] == "value_error"
    assert "by_alpha alpha must be strictly ascending" in str(error["msg"])


def _replace_one_provenance_alpha(payload: dict[str, object]) -> None:
    stamp = _od1_provenance(payload)
    alphas = list(_od1_tuple(stamp["alphas"]))
    assert 0.995 not in alphas
    alphas[-1] = 0.995
    stamp["alphas"] = tuple(alphas)


def _add_one_provenance_alpha(payload: dict[str, object]) -> None:
    stamp = _od1_provenance(payload)
    alphas = _od1_tuple(stamp["alphas"])
    assert 0.9999 not in alphas
    stamp["alphas"] = (*alphas, 0.9999)


def _drop_last_by_alpha_record(payload: dict[str, object]) -> None:
    summary = _od1_summary(payload)
    summary["by_alpha"] = _od1_tuple(summary["by_alpha"])[:-1]


@pytest.mark.parametrize(
    "mutate",
    [
        _replace_one_provenance_alpha,
        _add_one_provenance_alpha,
        _drop_last_by_alpha_record,
    ],
    ids=["provenance-alpha-replaced", "provenance-alpha-added", "by-alpha-dropped"],
)
def test_s1_t16_case5_by_alpha_set_differs_from_provenance_alphas_raises(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    """S1.T16 case 5: the by_alpha alpha set differs from provenance.alphas.

    Each variant keeps by_alpha strictly ascending, so only the set rule of
    CreditVarResult can fire.
    """
    payload = _od1_payload()
    mutate(payload)
    error = _od1_single_error(payload)
    assert error["loc"] == ()
    assert error["type"] == "value_error"
    assert "summary.by_alpha alpha set must equal" in str(error["msg"])


def _rename_model_key_to_lowercase(by_model: dict[str, object]) -> None:
    # Same concept, non-JtdModel spelling (the F-11 defect class).
    by_model["independent_bernoulli"] = by_model.pop("INDEPENDENT_BERNOULLI")


def _add_unknown_model_key(by_model: dict[str, object]) -> None:
    by_model["NOT_A_JTD_MODEL"] = 0.0


@pytest.mark.parametrize(
    "mutate",
    [_rename_model_key_to_lowercase, _add_unknown_model_key],
    ids=["lowercase-name", "unknown-name-added"],
)
def test_s1_t16_case6_expected_loss_key_not_a_jtd_model_name_raises(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    """S1.T16 case 6: an expected_loss_by_model key that is not a JtdModel name."""
    payload = _od1_payload()
    by_model = _od1_summary(payload)["expected_loss_by_model"]
    assert isinstance(by_model, dict)
    assert set(by_model) == {"INDEPENDENT_BERNOULLI"}
    mutate(by_model)
    error = _od1_single_error(payload)
    assert error["loc"] == ("summary", "expected_loss_by_model")
    assert error["type"] == "value_error"
    assert "expected_loss_by_model keys must be JtdModel member names" in str(
        error["msg"]
    )


def test_c7_per_model_tail_key_not_a_jtd_model_name_raises() -> None:
    """A_CONTRACT 3.1 / C.7: a per_model_tail key that is not a JtdModel name."""
    payload = _od1_payload()
    tails = payload["per_model_tail"]
    assert isinstance(tails, dict)
    assert set(tails) == {"INDEPENDENT_BERNOULLI"}
    tails["independent_bernoulli"] = tails.pop("INDEPENDENT_BERNOULLI")
    error = _od1_single_error(payload)
    assert error["loc"] == ("per_model_tail",)
    assert error["type"] == "value_error"
    assert "per_model_tail keys must be JtdModel member names" in str(error["msg"])


# S1.T17 row: forced_default, one obligor, ead 450000.3, lgd 1.0, S = 2_000.
# Every scenario loses 450000.3. The floating-point mean of 2_000 equal values
# is not exactly 450000.3, so a direct moment computation gives a nonzero std
# and skewness (WS S1.T17, sandbox numpy 2.4.4: std 1.16e-10, skewness 0.999).
_ZV_REQUEST: dict[str, Any] = {
    "model": "forced_default",
    "obligors": [{"obligor_uid": 1, "ead": 450000.3, "pd": 0.01, "lgd": 1.0}],
    "n_scenarios": 2_000,
    "seed": 12345,
}


@pytest.fixture(scope="module")
def zv_result() -> CreditVarResult:
    """The valid S1.T17 run result."""
    return run_credit_var(_ZV_REQUEST)


def test_s1_t17_constant_loss_run_gives_zero_moments(
    zv_result: CreditVarResult,
) -> None:
    """S1.T17 case 1: the S1.T17 run reports exact zero moments (ZV)."""
    summary = zv_result.summary
    assert summary.loss_min == summary.loss_max == 450000.3 * 1.0
    assert summary.loss_std_dev == 0.0
    assert summary.loss_skewness == 0.0
    assert summary.loss_excess_kurtosis == 0.0


def test_s1_t17_zero_variance_payload_with_nonzero_skewness_raises(
    zv_result: CreditVarResult,
) -> None:
    """S1.T17 case 2: loss_min == loss_max and loss_skewness 0.5 is rejected."""
    payload: dict[str, object] = copy.deepcopy(zv_result.model_dump(mode="python"))
    CreditVarResult.model_validate(payload)  # the unmodified payload is valid
    summary = _od1_summary(payload)
    assert summary["loss_min"] == summary["loss_max"]
    summary["loss_skewness"] = 0.5
    error = _od1_single_error(payload)
    assert error["loc"] == ("summary",)
    assert error["type"] == "value_error"
    assert "zero-variance rule (ZV)" in str(error["msg"])
