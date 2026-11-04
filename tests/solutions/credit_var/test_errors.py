# tests/solutions/credit_var/test_errors.py
"""
WS_CREDIT_VAR_API S2: error taxonomy of the public entry points (C.15, C.20).

Tests S2.T2, S2.T10 and S2.T11 of the workstream document.

* R = ``pydantic.ValidationError``: a request fault, raised before the first
  engine call, with exactly one entry at the loc of A_CONTRACT section 2.5.
* E = `CreditVarEngineError`: any fault after admission, with the original
  exception as ``__cause__``. E does not subclass ``ValueError``.

"No engine call" is observed through recorders on the first engine step of
each entry point (`engine_version` for `run_credit_var`, `_build_loss_matrix`
for both) and on the scenario generator. The expected locs are written out
from the table of A_CONTRACT section 2.5, not taken from the code.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from riskquant.exceptions import RiskQuantError
from riskquant.solutions.credit_var import run as credit_var_run
from riskquant.solutions.credit_var.api import (
    CreditVarRequest,
    run_credit_var,
    simulate_portfolio_losses,
)
from riskquant.solutions.credit_var.errors import CreditVarEngineError

_Loc = tuple[str | int, ...]
_ENTRY_POINTS: tuple[Callable[[Any], object], ...] = (
    run_credit_var,
    simulate_portfolio_losses,
)


# ---------------------------------------------------------------------------
# Books: one valid request per model family (N = 3, S = 1_000)
# ---------------------------------------------------------------------------


def _obligors() -> list[dict[str, Any]]:
    return [
        {"obligor_uid": 10, "ead": 1.0e6, "pd": 0.03, "lgd": 0.45},
        {"obligor_uid": 20, "ead": 2.5e6, "pd": 0.05, "lgd": 0.60},
        {"obligor_uid": 30, "ead": 7.0e5, "pd": 0.04, "lgd": 0.25},
    ]


def _common(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "obligors": _obligors(),
        "n_scenarios": 1_000,
        "alphas": [0.95, 0.99],
        "seed": 11,
    }


def _sf() -> dict[str, Any]:
    return {**_common("sf_gaussian"), "params": {"rho": 0.15}}


def _ib() -> dict[str, Any]:
    return _common("independent_bernoulli")


def _mf() -> dict[str, Any]:
    return {
        **_common("mf_gaussian"),
        "loadings": [
            {"obligor_uid": 10, "factor": "F1", "loading": 0.5},
            {"obligor_uid": 20, "factor": "F1", "loading": 0.4},
            {"obligor_uid": 30, "factor": "F1", "loading": 0.3},
        ],
    }


def _mft() -> dict[str, Any]:
    return {**_mf(), "model": "mf_students_t", "params": {"nu": 7.0}}


def _grouped() -> dict[str, Any]:
    book = _common("beta_binomial_grouped")
    for row, group in zip(book["obligors"], ("A", "B", "A"), strict=True):
        row["group"] = group
    book["groups"] = [
        {"name": "A", "params": {"alpha": 1.0, "beta": 99.0}},
        {"name": "B", "params": {"alpha": 4.0, "beta": 16.0}},
    ]
    return book


def _mo() -> dict[str, Any]:
    return {**_common("marshall_olkin"), "params": {"systemic_pd": 0.02}}


_BASES: dict[str, Callable[[], dict[str, Any]]] = {
    "sf": _sf,
    "ib": _ib,
    "mf": _mf,
    "mft": _mft,
    "grouped": _grouped,
    "mo": _mo,
}


@pytest.mark.parametrize("base", sorted(_BASES))
def test_base_books_are_valid(base: str) -> None:
    """Each fault case below changes one value of a valid book."""
    assert CreditVarRequest.model_validate(_BASES[base]()).model


# ---------------------------------------------------------------------------
# Fault cases: one per rule of A_CONTRACT section 2.5, each with one fault
# ---------------------------------------------------------------------------


def _with(base: str, edit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    book = _BASES[base]()
    edit(book)
    return book


def _obligor(i: int, **values: object) -> Callable[[dict[str, Any]], None]:
    def edit(book: dict[str, Any]) -> None:
        book["obligors"][i].update(values)

    return edit


def _field(name: str, value: object) -> Callable[[dict[str, Any]], None]:
    def edit(book: dict[str, Any]) -> None:
        book[name] = value

    return edit


def _drop(name: str) -> Callable[[dict[str, Any]], None]:
    def edit(book: dict[str, Any]) -> None:
        del book[name]

    return edit


def _append(name: str, row: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
    def edit(book: dict[str, Any]) -> None:
        book[name].append(row)

    return edit


def _loading(j: int, **values: object) -> Callable[[dict[str, Any]], None]:
    def edit(book: dict[str, Any]) -> None:
        book["loadings"][j].update(values)

    return edit


def _group_params(g: int, params: dict[str, float]) -> Callable[[dict[str, Any]], None]:
    def edit(book: dict[str, Any]) -> None:
        book["groups"][g]["params"] = params

    return edit


def _sum_sq_over(book: dict[str, Any]) -> None:
    book["loadings"][1]["loading"] = 0.8
    book["loadings"].append({"obligor_uid": 20, "factor": "F2", "loading": 0.7})


_CASES: list[tuple[str, dict[str, Any], _Loc]] = [
    # Rule 1: field constraints of sections 2.1 to 2.4.
    ("r01 ead < 0", _with("sf", _obligor(1, ead=-1.0)), ("obligors", 1, "ead")),
    ("r01 ead +inf", _with("sf", _obligor(2, ead=math.inf)), ("obligors", 2, "ead")),
    ("r01 pd > 1", _with("sf", _obligor(0, pd=1.5)), ("obligors", 0, "pd")),
    ("r01 lgd < 0", _with("sf", _obligor(0, lgd=-0.1)), ("obligors", 0, "lgd")),
    ("r01 rho == 1", _with("sf", _obligor(2, rho=1.0)), ("obligors", 2, "rho")),
    (
        "r01 group empty",
        _with("grouped", _obligor(1, group="")),
        ("obligors", 1, "group"),
    ),
    (
        "r01 uid > int64",
        _with("sf", _obligor(0, obligor_uid=2**63)),
        ("obligors", 0, "obligor_uid"),
    ),
    (
        "r01 extra field",
        _with("sf", _obligor(0, sector="X")),
        ("obligors", 0, "sector"),
    ),
    ("r01 seed < 0", _with("sf", _field("seed", -1)), ("seed",)),
    ("r01 n_scenarios 0", _with("sf", _field("n_scenarios", 0)), ("n_scenarios",)),
    ("r01 obligors empty", _with("sf", _field("obligors", [])), ("obligors",)),
    ("r01 obligors absent", _with("sf", _drop("obligors")), ("obligors",)),
    (
        "r01 loading > 1",
        _with("mf", _loading(0, loading=1.5)),
        ("loadings", 0, "loading"),
    ),
    (
        "r01 factor empty",
        _with("mf", _loading(2, factor="")),
        ("loadings", 2, "factor"),
    ),
    (
        "r01 group name empty",
        _with("grouped", lambda b: b["groups"][1].update(name="")),
        ("groups", 1, "name"),
    ),
    (
        "r01 params value type",
        _with("sf", _field("params", {"rho": "0.2"})),
        ("params", "rho"),
    ),
    # Rules 2 to 4: alphas.
    ("r02 alpha == 1", _with("sf", _field("alphas", [0.95, 1.0])), ("alphas", 1)),
    ("r03 alphas empty", _with("sf", _field("alphas", [])), ("alphas",)),
    (
        "r04 alphas duplicate",
        _with("sf", _field("alphas", [0.9, 0.99, 0.9])),
        ("alphas", 2),
    ),
    # Rules 5 to 7: model and params.
    ("r05 unknown model", _with("sf", _field("model", "sf_gauss")), ("model",)),
    (
        "r06 unknown param",
        _with("sf", _field("params", {"rho": 0.15, "zeta": 1.0})),
        ("params", "zeta"),
    ),
    ("r07 param range", _with("sf", _field("params", {"rho": 1.0})), ("params", "rho")),
    # Rule 8: duplicate obligor_uid, second occurrence.
    (
        "r08 duplicate uid",
        _with("sf", _obligor(2, obligor_uid=10)),
        ("obligors", 2, "obligor_uid"),
    ),
    # Rules 9 to 15: loadings.
    (
        "r09 loadings for sf",
        _with(
            "sf",
            _field("loadings", [{"obligor_uid": 10, "factor": "F", "loading": 0.1}]),
        ),
        ("loadings",),
    ),
    ("r10 loadings absent", _with("mf", _drop("loadings")), ("loadings",)),
    ("r10 loadings empty", _with("mf", _field("loadings", [])), ("loadings",)),
    (
        "r11 stray loadings uid",
        _with(
            "mf",
            _append("loadings", {"obligor_uid": 99, "factor": "F1", "loading": 0.1}),
        ),
        ("loadings", 3, "obligor_uid"),
    ),
    (
        "r12 obligor without loadings",
        _with("mf", _loading(1, obligor_uid=10, factor="F2")),
        ("obligors", 1, "obligor_uid"),
    ),
    (
        "r13 duplicate uid factor",
        _with(
            "mf",
            _append("loadings", {"obligor_uid": 20, "factor": "F1", "loading": 0.1}),
        ),
        ("loadings", 3, "factor"),
    ),
    ("r14 sum loading^2 > 1", _with("mf", _sum_sq_over), ("loadings", 3, "loading")),
    (
        "r15 reserved IDIO",
        _with("mf", _loading(1, factor="IDIO_020")),
        ("loadings", 1, "factor"),
    ),
    (
        "r15 reserved CHI",
        _with("mft", _loading(0, factor="CHI_GLOBAL")),
        ("loadings", 0, "factor"),
    ),
    # Rules 16 to 23: groups.
    (
        "r16 groups for sf",
        _with("sf", _field("groups", [{"name": "A", "params": {}}])),
        ("groups",),
    ),
    (
        "r17 obligor group for sf",
        _with("sf", _obligor(1, group="A")),
        ("obligors", 1, "group"),
    ),
    (
        "r18 params for grouped",
        _with("grouped", _field("params", {"alpha": 1.0})),
        ("params",),
    ),
    ("r19 groups absent", _with("grouped", _drop("groups")), ("groups",)),
    ("r19 groups empty", _with("grouped", _field("groups", [])), ("groups",)),
    (
        "r20 obligor without group",
        _with("grouped", _obligor(1, group=None)),
        ("obligors", 1, "group"),
    ),
    (
        "r21 duplicate group name",
        _with(
            "grouped",
            _append("groups", {"name": "A", "params": {"alpha": 2.0, "beta": 8.0}}),
        ),
        ("groups", 2, "name"),
    ),
    (
        "r22 undefined group",
        _with("grouped", _obligor(2, group="C")),
        ("obligors", 2, "group"),
    ),
    (
        "r23 group param unknown",
        _with("grouped", _group_params(0, {"alpha": 1.0, "beta": 99.0, "gamma": 1.0})),
        ("groups", 0, "params", "gamma"),
    ),
    (
        "r23 group param missing",
        _with("grouped", _group_params(1, {"alpha": 4.0})),
        ("groups", 1, "params", "beta"),
    ),
    (
        "r23 group param range",
        _with("grouped", _group_params(1, {"alpha": 0.0, "beta": 16.0})),
        ("groups", 1, "params", "alpha"),
    ),
    # Rule 24: marshall_olkin pd >= systemic_pd, first offender.
    (
        "r24 pd below systemic_pd",
        _with("mo", _obligor(1, pd=0.01)),
        ("obligors", 1, "pd"),
    ),
    # Rule 25: per-obligor rho for a model without it.
    ("r25 rho for ib", _with("ib", _obligor(2, rho=0.3)), ("obligors", 2, "rho")),
]

_RULES_COVERED = {int(case[1:3]) for case, _, _ in _CASES}


def test_s2_t10_every_rule_has_a_case() -> None:
    assert set(range(1, 26)) == _RULES_COVERED


def _json_expressible(book: dict[str, Any]) -> bool:
    try:
        json.dumps(book, allow_nan=False)
    except ValueError:
        return False
    return True


@pytest.fixture
def engine_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Recorders on the first engine step of each entry point and on the
    scenario generator. A recorder raises, so a reached engine also turns
    into a wrong exception type."""
    calls: list[str] = []

    def recorder(name: str) -> Callable[..., Any]:
        def record(*args: object, **kwargs: object) -> Any:
            calls.append(name)
            raise AssertionError(f"engine step '{name}' reached")

        return record

    class _Generator:
        from_universe = staticmethod(recorder("MixedCreditFactorGenerator"))

    monkeypatch.setattr(credit_var_run, "engine_version", recorder("engine_version"))
    monkeypatch.setattr(
        credit_var_run, "_build_loss_matrix", recorder("_build_loss_matrix")
    )
    monkeypatch.setattr(credit_var_run, "MixedCreditFactorGenerator", _Generator)
    return calls


def _only_entry(excinfo: pytest.ExceptionInfo[ValidationError]) -> dict[str, Any]:
    errors = excinfo.value.errors()
    assert len(errors) == 1, errors
    return dict(errors[0])


@pytest.mark.parametrize(
    ("book", "loc"),
    [(book, loc) for _, book, loc in _CASES],
    ids=[case for case, _, _ in _CASES],
)
def test_s2_t10_one_fault_gives_one_entry_at_its_loc(
    book: dict[str, Any], loc: _Loc, engine_calls: list[str]
) -> None:
    with pytest.raises(ValidationError) as direct:
        CreditVarRequest.model_validate(copy.deepcopy(book))
    assert _only_entry(direct)["loc"] == loc

    if _json_expressible(book):
        with pytest.raises(ValidationError) as from_json:
            CreditVarRequest.model_validate_json(json.dumps(book))
        assert _only_entry(from_json)["loc"] == loc

    for entry_point in _ENTRY_POINTS:
        with pytest.raises(ValidationError) as excinfo:
            entry_point(copy.deepcopy(book))
        assert type(excinfo.value) is ValidationError
        assert _only_entry(excinfo)["loc"] == loc
    assert engine_calls == []


@pytest.mark.parametrize("bad", ["not a request", 42, None, [1, 2]])
def test_s2_t10_non_mapping_input_is_a_request_fault(
    bad: object, engine_calls: list[str]
) -> None:
    for entry_point in _ENTRY_POINTS:
        with pytest.raises(ValidationError):
            entry_point(bad)
    assert engine_calls == []


@pytest.mark.parametrize(
    "bad",
    [-1, 1.5, 10.0, "10", True, math.nan],
    ids=["negative", "float", "integral float", "str", "bool", "nan"],
)
def test_s2_t10_bad_max_samples_is_a_request_fault(
    bad: object, engine_calls: list[str]
) -> None:
    """Owner decision on S2: a bad `max_samples` is a caller fault, R at loc
    ("max_samples",) with one entry, before the engine runs. A valid request
    makes the argument the only fault."""
    with pytest.raises(ValidationError) as excinfo:
        simulate_portfolio_losses(_ib(), max_samples=bad)  # type: ignore[arg-type]
    assert type(excinfo.value) is ValidationError
    assert _only_entry(excinfo)["loc"] == ("max_samples",)
    assert engine_calls == []


@pytest.mark.parametrize("limit", [None, 0, 1, 500, 1_000, 10**12])
def test_s2_t10_valid_max_samples_is_accepted(limit: int | None) -> None:
    """None and every int >= 0 pass admission; 0 gives an empty sample."""
    losses = simulate_portfolio_losses(_ib(), max_samples=limit)
    expected = 1_000 if limit is None else min(limit, 1_000)
    assert losses.shape == (expected,)


# ---------------------------------------------------------------------------
# S2.T2 (C.15): call-time validation of an instance mutated in place
# ---------------------------------------------------------------------------


def _mutations() -> list[tuple[str, CreditVarRequest, _Loc]]:
    out_of_range = CreditVarRequest.model_validate(_sf())
    out_of_range.params["rho"] = 1.5
    wrong_type = CreditVarRequest.model_validate(_sf())
    wrong_type.params["rho"] = "0.15"  # type: ignore[assignment]
    unknown_key = CreditVarRequest.model_validate(_sf())
    unknown_key.params["zeta"] = 0.5
    group_range = CreditVarRequest.model_validate(_grouped())
    assert group_range.groups is not None
    group_range.groups[0].params["alpha"] = -1.0
    return [
        ("params out of range", out_of_range, ("params", "rho")),
        ("params wrong type", wrong_type, ("params", "rho")),
        ("params unknown key", unknown_key, ("params", "zeta")),
        ("group params out of range", group_range, ("groups", 0, "params", "alpha")),
    ]


@pytest.mark.parametrize(
    "index", range(len(_mutations())), ids=[m[0] for m in _mutations()]
)
def test_s2_t2_mutated_instance_is_validated_at_call_time(
    index: int, engine_calls: list[str]
) -> None:
    _, request, loc = _mutations()[index]
    for entry_point in _ENTRY_POINTS:
        with pytest.raises(ValidationError) as excinfo:
            entry_point(request)
        assert _only_entry(excinfo)["loc"] == loc
    assert engine_calls == []


def test_s2_t2_run_uses_a_validated_copy_not_the_callers_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = CreditVarRequest.model_validate(_sf())
    seen: list[CreditVarRequest] = []

    def capture(req: CreditVarRequest, alphas: tuple[float, ...]) -> None:
        seen.append(req)
        raise RuntimeError("stop after admission")

    monkeypatch.setattr(credit_var_run, "_build_loss_matrix", capture)
    with pytest.raises(CreditVarEngineError):
        run_credit_var(request)
    assert len(seen) == 1
    assert seen[0] is not request
    assert seen[0] == request


# ---------------------------------------------------------------------------
# S2.T11 (C.20): engine faults become E, chained, not a ValueError
# ---------------------------------------------------------------------------


def test_s2_t11_engine_error_is_not_a_value_error() -> None:
    assert not issubclass(CreditVarEngineError, ValueError)
    assert issubclass(CreditVarEngineError, RiskQuantError)
    assert issubclass(ValidationError, ValueError)  # F-14: why E must not be one


@pytest.mark.parametrize(
    "fault",
    [ValueError("engine invariant"), TypeError("bad dtype"), RuntimeError("abort")],
    ids=["ValueError", "TypeError", "RuntimeError"],
)
@pytest.mark.parametrize(
    "entry_point", _ENTRY_POINTS, ids=lambda f: getattr(f, "__name__", "entry")
)
def test_s2_t11_engine_exception_becomes_engine_error(
    fault: Exception,
    entry_point: Callable[[Any], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_engine(req: CreditVarRequest, alphas: tuple[float, ...]) -> None:
        raise fault

    monkeypatch.setattr(credit_var_run, "_build_loss_matrix", failing_engine)
    with pytest.raises(CreditVarEngineError) as excinfo:
        entry_point(_ib())
    assert excinfo.value.__cause__ is fault
    assert not isinstance(excinfo.value, ValueError)


def test_s2_t11_nan_in_a_result_model_becomes_engine_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = credit_var_run._distribution_metrics

    def nan_metrics(losses: Any, expected_loss: float) -> dict[str, float]:
        metrics = original(losses, expected_loss)
        metrics["loss_mean"] = math.nan
        return metrics

    monkeypatch.setattr(credit_var_run, "_distribution_metrics", nan_metrics)
    with pytest.raises(CreditVarEngineError) as excinfo:
        run_credit_var(_ib())
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValidationError)
    assert [error["type"] for error in cause.errors()] == ["finite_number"]


def test_s2_t11_engine_error_passes_through_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CreditVarEngineError("already an engine fault")

    def raise_engine_error(req: CreditVarRequest, alphas: tuple[float, ...]) -> None:
        raise original

    monkeypatch.setattr(credit_var_run, "_build_loss_matrix", raise_engine_error)
    for entry_point in _ENTRY_POINTS:
        with pytest.raises(CreditVarEngineError) as excinfo:
            entry_point(_ib())
        assert excinfo.value is original


def test_s2_t11_valid_request_runs_on_both_entry_points() -> None:
    """The mapping does not change a valid run."""
    result = run_credit_var(_ib())
    assert [record.alpha for record in result.summary.by_alpha] == [0.95, 0.99]
    losses = simulate_portfolio_losses(_ib())
    assert losses.shape == (1_000,)
