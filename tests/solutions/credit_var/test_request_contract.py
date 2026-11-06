# tests/solutions/credit_var/test_request_contract.py
"""
WS_CREDIT_VAR_API S2: request admission rules (C.14 to C.19, C.21).

Tests S2.T1, S2.T3 to S2.T9 and S2.T12 of the workstream document. The error
taxonomy, the error locations of A_CONTRACT section 2.5 and the call-time
validation (S2.T2, S2.T10, S2.T11) are in `test_errors.py`.

The oracles are independent of the code under test: the expected locs,
coercion outcomes and bounds are written out here from A_CONTRACT sections 1
and 2. The loading-sum and reserved-name cases (S2.T6, S2.T7) also run the
engine's own check on the same input, so a drift between the request
predicate and the engine predicate fails here.
"""

from __future__ import annotations

import gc
import inspect
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import BaseModel, ValidationError

from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad
from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
from riskquant.credit.masters.netting_set.validate import (
    validate_uid_sorted_unique as validate_netting_set_uids,
)
from riskquant.credit.masters.obligors.validate import (
    validate_uid_sorted_unique as validate_obligor_uids,
)
from riskquant.solutions.credit_var import run as credit_var_run
from riskquant.solutions.credit_var.api import (
    CreditVarRequest,
    GroupRow,
    LoadingRow,
    ObligorRow,
    run_credit_var,
)
from riskquant.solutions.credit_var.recipes import assemble_inputs

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_REQUEST_MODELS: tuple[type[BaseModel], ...] = (
    ObligorRow,
    LoadingRow,
    GroupRow,
    CreditVarRequest,
)


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------


def _obligors() -> list[dict[str, Any]]:
    return [
        {"obligor_uid": 10, "ead": 1.0e6, "pd": 0.03, "lgd": 0.45},
        {"obligor_uid": 20, "ead": 2.5e6, "pd": 0.05, "lgd": 0.60},
        {"obligor_uid": 30, "ead": 7.0e5, "pd": 0.04, "lgd": 0.25},
    ]


def _sf_book() -> dict[str, Any]:
    return {
        "model": "sf_gaussian",
        "obligors": _obligors(),
        "params": {"rho": 0.15},
        "n_scenarios": 1_000,
        "alphas": [0.99],
        "seed": 7,
    }


def _mf_book(model: str = "mf_gaussian") -> dict[str, Any]:
    book: dict[str, Any] = {
        "model": model,
        "obligors": _obligors(),
        "loadings": [
            {"obligor_uid": 10, "factor": "F1", "loading": 0.5},
            {"obligor_uid": 20, "factor": "F1", "loading": 0.4},
            {"obligor_uid": 30, "factor": "F1", "loading": 0.3},
        ],
        "n_scenarios": 1_000,
        "alphas": [0.99],
        "seed": 7,
    }
    if model == "mf_students_t":
        book["params"] = {"nu": 7.0}
    return book


def _full_book() -> dict[str, Any]:
    """Every request field set to a non-default value, and every obligor
    field present, so that a round trip covers each type of the contract."""
    obligors = _obligors()
    obligors[0].update(in_stress_test=True, rho=0.3)
    obligors[1].update(force_default=True)
    return {
        "model": "sf_students_t",
        "obligors": obligors,
        "params": {"rho": 0.2, "nu": 6.0},
        "n_scenarios": 1_234,
        "alphas": [0.999, 0.95],
        "seed": 0,
    }


def _grouped_book() -> dict[str, Any]:
    obligors = _obligors()
    for row, group in zip(obligors, ("A", "B", "A"), strict=True):
        row["group"] = group
    return {
        "model": "beta_binomial_grouped",
        "obligors": obligors,
        "groups": [
            {"name": "A", "params": {"alpha": 1.0, "beta": 99.0}},
            {"name": "B", "params": {"alpha": 4.0, "beta": 16.0}},
        ],
        "n_scenarios": 1_000,
        "alphas": [0.99],
        "seed": 7,
    }


def _single_error(excinfo: pytest.ExceptionInfo[ValidationError]) -> dict[str, Any]:
    errors = excinfo.value.errors()
    assert len(errors) == 1, errors
    return dict(errors[0])


# ---------------------------------------------------------------------------
# S2.T1 (C.14): frozen, strict configuration; lossless JSON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", _REQUEST_MODELS, ids=lambda m: m.__name__)
def test_s2_t1_request_model_configuration(model: type[BaseModel]) -> None:
    config = model.model_config
    assert config.get("extra") == "forbid"
    assert config.get("frozen") is True
    assert config.get("allow_inf_nan") is False
    assert config.get("strict") is True


def _instances() -> dict[type[BaseModel], BaseModel]:
    mf = CreditVarRequest.model_validate(_mf_book())
    grouped = CreditVarRequest.model_validate(_grouped_book())
    assert mf.loadings is not None
    assert grouped.groups is not None
    return {
        ObligorRow: mf.obligors[0],
        LoadingRow: mf.loadings[0],
        GroupRow: grouped.groups[0],
        CreditVarRequest: mf,
    }


@pytest.mark.parametrize("model", _REQUEST_MODELS, ids=lambda m: m.__name__)
def test_s2_t1_assignment_to_each_field_raises(model: type[BaseModel]) -> None:
    instance = _instances()[model]
    for name in model.model_fields:
        value = getattr(instance, name)
        with pytest.raises(ValidationError) as excinfo:
            setattr(instance, name, value)
        error = _single_error(excinfo)
        assert error["type"] == "frozen_instance", (model.__name__, name)
        assert getattr(instance, name) is value


@pytest.mark.parametrize(
    "book",
    [_full_book(), _mf_book("mf_students_t"), _grouped_book()],
    ids=["sf_students_t_full", "mf_students_t", "beta_binomial_grouped"],
)
def test_s2_t1_request_json_round_trip_is_lossless(book: dict[str, Any]) -> None:
    request = CreditVarRequest.model_validate(book)
    first = request.model_dump_json()
    restored = CreditVarRequest.model_validate_json(first)
    assert restored == request
    assert restored.model_dump_json() == first
    for name in ("obligors", "alphas", "loadings", "groups"):
        value = getattr(restored, name)
        assert value is None or isinstance(value, tuple), name


# ---------------------------------------------------------------------------
# S2.T3 (C.16): alphas non-empty, in (0, 1), distinct under ==
# ---------------------------------------------------------------------------


def test_s2_t3_duplicate_alphas_raise() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate({**_sf_book(), "alphas": [0.99, 0.99]})
    assert _single_error(excinfo)["loc"] == ("alphas", 1)


def test_s2_t3_alphas_one_ulp_apart_are_accepted_bit_for_bit() -> None:
    upper = math.nextafter(0.99, 1.0)
    assert upper != 0.99
    request = CreditVarRequest.model_validate({**_sf_book(), "alphas": [0.99, upper]})
    assert [a.hex() for a in request.alphas] == [(0.99).hex(), upper.hex()]


def test_s2_t3_empty_alphas_raise() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate({**_sf_book(), "alphas": []})
    assert _single_error(excinfo)["loc"] == ("alphas",)


@pytest.mark.parametrize("bad", [1.0, 0.0, -0.5, 1.5])
def test_s2_t3_alpha_outside_open_unit_interval_raises(bad: float) -> None:
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate({**_sf_book(), "alphas": [bad]})
    assert _single_error(excinfo)["loc"] == ("alphas", 0)


# ---------------------------------------------------------------------------
# S2.T4 (C.17): seed >= 0 (D039 precedent)
# ---------------------------------------------------------------------------


def test_s2_t4_negative_seed_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate({**_sf_book(), "seed": -1})
    assert _single_error(excinfo)["loc"] == ("seed",)


def test_s2_t4_zero_seed_is_accepted() -> None:
    assert CreditVarRequest.model_validate({**_sf_book(), "seed": 0}).seed == 0


# ---------------------------------------------------------------------------
# S2.T5 (C.17): obligor_uid in the int64 range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("uid", [2**63, -(2**63) - 1, 2**70])
def test_s2_t5_obligor_uid_outside_int64_raises(uid: int) -> None:
    book = _sf_book()
    book["obligors"][1]["obligor_uid"] = uid
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("obligors", 1, "obligor_uid")
    with pytest.raises(ValidationError) as json_excinfo:
        CreditVarRequest.model_validate_json(json.dumps(book))
    assert _single_error(json_excinfo)["loc"] == ("obligors", 1, "obligor_uid")


def test_s2_t5_loading_uid_outside_int64_raises() -> None:
    book = _mf_book()
    book["loadings"][2]["obligor_uid"] = 2**63
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 2, "obligor_uid")


@pytest.mark.parametrize("uid", [_INT64_MAX, _INT64_MIN])
def test_s2_t5_int64_bounds_are_accepted(uid: int) -> None:
    book = _sf_book()
    book["obligors"][0]["obligor_uid"] = uid
    request = CreditVarRequest.model_validate(book)
    assert request.obligors[0].obligor_uid == uid
    assert np.int64(uid) == uid  # representable without wrap


#: Ascending uid pairs whose int64 difference exceeds the int64 range: np.diff
#: wraps to a negative value (S2 review R-001, OD-S2-1).
_WRAPPING_ASCENDING_UIDS: list[tuple[int, int]] = [
    (-1, _INT64_MAX),
    (-(2**62), 2**62),
]


@pytest.mark.parametrize("uids", _WRAPPING_ASCENDING_UIDS)
@pytest.mark.parametrize(
    ("model", "params"),
    [("sf_gaussian", {"rho": 0.15}), ("independent_bernoulli", {})],
)
def test_s2_t5_wrapping_uid_pair_runs_end_to_end(
    uids: tuple[int, int], model: str, params: dict[str, float]
) -> None:
    """R-001: a request with valid int64 uids runs; the engine ascending
    checks (obligor master, CompiledObligorEad) do not wrap."""
    book: dict[str, Any] = {
        "model": model,
        "obligors": [
            {"obligor_uid": uid, "ead": 1.0e6, "pd": 0.03, "lgd": 0.45} for uid in uids
        ],
        "params": params,
        "n_scenarios": 1_000,
        "alphas": [0.99],
        "seed": 7,
    }
    result = run_credit_var(book)
    assert result.provenance.n_obligors == 2


@pytest.mark.parametrize("validate", [validate_obligor_uids, validate_netting_set_uids])
def test_s2_t5_uid_ascending_check_does_not_wrap(
    validate: Callable[[np.ndarray], None],
) -> None:
    """R-001: the master validators accept a wrapping ascending pair and
    reject a descending pair whose np.diff wraps to a positive value. The
    netting-set master is not on the credit_var path, so the end-to-end test
    does not reach it."""
    for uids in _WRAPPING_ASCENDING_UIDS:
        validate(np.array(uids, dtype=np.int64))
    with pytest.raises(ValueError, match="not sorted ascending"):
        validate(np.array([_INT64_MAX, -2], dtype=np.int64))


def test_s2_t5_compiled_obligor_ead_ascending_check_does_not_wrap() -> None:
    """R-001: CompiledObligorEad accepts a wrapping ascending pair and rejects
    a descending pair whose np.diff wraps to a positive value."""
    ead = np.array([1.0, 2.0], dtype=np.float64)
    for uids in _WRAPPING_ASCENDING_UIDS:
        CompiledObligorEad(uid=np.array(uids, dtype=np.int64), ead=ead)
    with pytest.raises(ValueError, match="ascending-sorted and unique"):
        CompiledObligorEad(uid=np.array([_INT64_MAX, -2], dtype=np.int64), ead=ead)


# ---------------------------------------------------------------------------
# S2.T6 (C.17): loadings rules agree with the engine predicate
# ---------------------------------------------------------------------------


def _two_factor_loadings(sum_sq: float) -> tuple[float, float]:
    """Loadings (0.6, b) on obligor 20 with 0.36 + b*b == `sum_sq` up to the
    rounding of b. The test asserts the exact sum separately."""
    return 0.6, math.sqrt(sum_sq - 0.36)


def _book_with_obligor_20_loadings(first: float, second: float) -> dict[str, Any]:
    book = _mf_book()
    book["loadings"][1] = {"obligor_uid": 20, "factor": "F1", "loading": first}
    book["loadings"].append({"obligor_uid": 20, "factor": "F2", "loading": second})
    return book


#: Engine rejection messages (fragments) of the rules that S2.T6 and S2.T7
#: compare with the engine.
_ENGINE_DUPLICATE_LOADING_20 = "obligor(s) [20] load on the same systemic factor twice."
_ENGINE_SUM_SQ_20 = "sum of squared loadings exceeds 1 for obligor(s) [20]."


def _engine_duplicate_factor_name(factor: str) -> str:
    return f"credit factor universe: duplicate factor_name(s) [{factor!r}]."


def _engine_accepts(book: dict[str, Any], *, target: str) -> bool:
    """The engine's own loadings check (`credit.jtd.ingest.validate`), run on
    the recipe frames of `book`, without the request validator.

    Returns False only for a ValueError whose message contains `target`, the
    engine message of the rule under test. Any other ValueError fails the
    test: an engine rejection for a different reason is not evidence that the
    engine applies the rule (S2 review R-005).
    """
    obligor_table = pl.DataFrame({
        "uid": [row["obligor_uid"] for row in book["obligors"]],
        "ead": [row["ead"] for row in book["obligors"]],
        "pd": [row["pd"] for row in book["obligors"]],
        "lgd": [row["lgd"] for row in book["obligors"]],
    })
    loadings_table = pl.DataFrame({
        "uid": [row["obligor_uid"] for row in book["loadings"]],
        "factor": [row["factor"] for row in book["loadings"]],
        "loading": [row["loading"] for row in book["loadings"]],
    })
    try:
        inputs = assemble_inputs(
            model_key=book["model"],
            obligor_table=obligor_table,
            params=book.get("params", {}),
            loadings_table=loadings_table,
        )
        compile_jtd_portfolio(bundle=inputs.bundle, masters=inputs.masters)
    except ValueError as exc:
        if target not in str(exc):
            pytest.fail(
                f"engine raised a ValueError without the target message "
                f"{target!r}: {exc}"
            )
        return False
    return True


def test_s2_t6_duplicate_uid_factor_raises_as_in_the_engine() -> None:
    book = _mf_book()
    book["loadings"].append({"obligor_uid": 20, "factor": "F1", "loading": 0.1})
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 3, "factor")
    assert not _engine_accepts(book, target=_ENGINE_DUPLICATE_LOADING_20)


def test_s2_t6_sum_of_squares_above_tolerance_raises_as_in_the_engine() -> None:
    first, second = _two_factor_loadings(1.0 + 2e-9)
    assert math.fsum([first * first, second * second]) > 1.0 + 1e-9
    book = _book_with_obligor_20_loadings(first, second)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 3, "loading")
    assert not _engine_accepts(book, target=_ENGINE_SUM_SQ_20)


def test_s2_t6_sum_of_squares_inside_tolerance_is_accepted_as_in_the_engine() -> None:
    first, second = _two_factor_loadings(1.0 + 0.5e-9)
    total = math.fsum([first * first, second * second])
    assert 1.0 < total <= 1.0 + 1e-9
    book = _book_with_obligor_20_loadings(first, second)
    request = CreditVarRequest.model_validate(book)
    assert request.loadings is not None
    assert len(request.loadings) == 4
    assert _engine_accepts(book, target=_ENGINE_SUM_SQ_20)


def test_s2_t6_loc_is_the_row_where_the_running_sum_first_exceeds() -> None:
    """Rows of obligor 20 at j = 1 (0.8^2), 3 (0.5^2), 4 (0.5^2): the running
    sum is 0.64, 0.89, 1.14, so the loc is j = 4."""
    book = _mf_book()
    book["loadings"][1] = {"obligor_uid": 20, "factor": "F1", "loading": 0.8}
    book["loadings"].append({"obligor_uid": 20, "factor": "F2", "loading": 0.5})
    book["loadings"].append({"obligor_uid": 20, "factor": "F3", "loading": 0.5})
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 4, "loading")


#: Loadings of obligor 20 (20 rows each, exact as float.hex) where
#: ``math.fsum`` of the squares and the engine grouped sum round to adjacent
#: floats on the two sides of 1 + 1e-9. Source: a seeded search
#: (numpy default_rng(20261103)) over unit-norm vectors scaled by
#: sqrt(1 + 1e-9) and shifted by k * 2.2e-16, k in -4..4. In each vector only
#: the full sum can exceed the bound, so a rejection has its loc at the last
#: row.
_FSUM_OVER_ENGINE_WITHIN: tuple[str, ...] = (
    "0x1.894d071d29e45p-2", "0x1.f1ceb9b5eae51p-5", "-0x1.a40fb3a265e57p-2",
    "0x1.627aedb1ed591p-4", "0x1.2e802aa1c7859p-4", "-0x1.c8465e4a18d83p-3",
    "-0x1.081577ef9eef8p-4", "-0x1.04772fcc287adp-1", "0x1.32075f6209449p-2",
    "-0x1.576e1b522ee46p-6", "0x1.768ad8b7136ccp-4", "0x1.d33ea977b1e21p-4",
    "0x1.336671ad08239p-3", "-0x1.20d46da08aecdp-3", "0x1.7261dd889a405p-4",
    "0x1.e473c743c6247p-4", "0x1.e044fd6c72427p-3", "-0x1.18f412d91f253p-2",
    "-0x1.9c25bcb8dc35fp-4", "-0x1.93c19f642a7b2p-3",
)  # fmt: skip
_FSUM_WITHIN_ENGINE_OVER: tuple[str, ...] = (
    "-0x1.6cc5827925b65p-4", "0x1.0f5116a0b32b9p-2", "-0x1.000634a2f7561p-3",
    "-0x1.56b19925e80cfp-5", "0x1.25ff2bb345a49p-2", "-0x1.7cccc8d73e5a7p-4",
    "-0x1.46ee6d8fcbe5ep-5", "0x1.9cffee69712a8p-3", "-0x1.65ea934b88a58p-5",
    "0x1.a68d0e828b47ap-4", "0x1.7c9ff34151068p-2", "-0x1.12b5b4ba27978p-1",
    "0x1.818b06c57d125p-3", "-0x1.3aa90242953d8p-5", "-0x1.6f0f16288f8ebp-2",
    "-0x1.52d4a472950b7p-4", "0x1.a366ceff4cfeap-7", "0x1.5fdf2eb0a7616p-2",
    "0x1.5d30e064cb28ap-5", "-0x1.9ae843f492d1fp-3",
)  # fmt: skip
_SUM_SQ_BOUND = 1.0 + 1e-9


def _book_with_obligor_20_vector(hex_loadings: tuple[str, ...]) -> dict[str, Any]:
    """Rows 0 and 1 load obligors 10 and 30; rows 2 onward are the vector on
    obligor 20, one systemic factor per row."""
    book = _mf_book()
    del book["loadings"][1]
    book["loadings"] += [
        {"obligor_uid": 20, "factor": f"G{n:02d}", "loading": float.fromhex(h)}
        for n, h in enumerate(hex_loadings, start=1)
    ]
    return book


def _engine_grouped_sum(loadings: list[float]) -> float:
    """The v0.2.0 engine sum of `loadings` on one obligor: the pandas 3.0.x
    group sum of the float64 squares, as recorded at tag v0.2.0 in the
    WS_CREDIT_POLARS C.6 record (``named`` entries). The oracle reads the
    record; it does not depend on pandas or on the code under test (S1.T6)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "credit"
        / "jtd"
        / "golden"
        / "loading_sum_kahan.json"
    )
    named = json.loads(path.read_text(encoding="utf-8"))["named"]
    wanted = [x.hex() for x in loadings]
    for entry in named.values():
        if entry["loading"] == wanted:
            return float.fromhex(entry["total"])
    pytest.fail(f"no recorded engine sum for loadings {wanted}")


def test_s2_t6_fsum_over_engine_within_is_accepted_as_in_the_engine() -> None:
    """R-003: math.fsum says above 1 + 1e-9, the engine says within. The
    request decision follows the engine."""
    loadings = [float.fromhex(h) for h in _FSUM_OVER_ENGINE_WITHIN]
    squares = [x * x for x in loadings]
    assert math.fsum(squares) > _SUM_SQ_BOUND
    assert _engine_grouped_sum(loadings) <= _SUM_SQ_BOUND
    book = _book_with_obligor_20_vector(_FSUM_OVER_ENGINE_WITHIN)
    assert _engine_accepts(book, target=_ENGINE_SUM_SQ_20)
    request = CreditVarRequest.model_validate(book)
    assert request.loadings is not None
    assert len(request.loadings) == 2 + len(loadings)


def test_s2_t6_fsum_within_engine_over_raises_as_in_the_engine() -> None:
    """R-003: math.fsum says within 1 + 1e-9, the engine says above. The
    request decision follows the engine; the loc is the last row of obligor
    20, the first row at which the running sum exceeds the bound."""
    loadings = [float.fromhex(h) for h in _FSUM_WITHIN_ENGINE_OVER]
    squares = [x * x for x in loadings]
    assert math.fsum(squares) <= _SUM_SQ_BOUND
    assert _engine_grouped_sum(loadings) > _SUM_SQ_BOUND
    book = _book_with_obligor_20_vector(_FSUM_WITHIN_ENGINE_OVER)
    assert not _engine_accepts(book, target=_ENGINE_SUM_SQ_20)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 1 + len(loadings), "loading")


def _single_obligor_loadings(n_rows: int) -> tuple[LoadingRow, ...]:
    """`n_rows` loadings of 0.001 on obligor 20: sum of squares n_rows * 1e-6,
    inside the bound for every size used below."""
    return tuple(
        LoadingRow(obligor_uid=20, factor=f"F{n}", loading=0.001) for n in range(n_rows)
    )


def _best_time(loadings: tuple[LoadingRow, ...], repeats: int = 5) -> float:
    check = CreditVarRequest._check_loading_sum_sq
    best = math.inf
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            check(loadings)
            best = min(best, time.perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return best


def test_s2_t6_rule_14_time_is_linear_in_the_row_count() -> None:
    """R-004: 16 times the rows of one obligor take about 16 times the time
    (linear), not about 256 times (a prefix re-sum per row). The limit 64 is
    the geometric mean of the two, so the test does not depend on the speed
    of the machine."""
    small = _single_obligor_loadings(2_000)
    large = _single_obligor_loadings(32_000)
    _best_time(small, repeats=1)  # warm-up
    ratio = _best_time(large) / _best_time(small)
    assert ratio < 64.0, ratio


def test_s2_t6_engine_helper_rejects_a_different_engine_message() -> None:
    """R-005: an engine ValueError without the target message fails the
    caller; it is not counted as the target rejection."""
    book = _mf_book()
    book["loadings"].append({"obligor_uid": 20, "factor": "F1", "loading": 0.1})
    with pytest.raises(pytest.fail.Exception, match="without the target message"):
        _engine_accepts(book, target=_ENGINE_SUM_SQ_20)


# ---------------------------------------------------------------------------
# S2.T7 (C.17, F-36): reserved factor names
# ---------------------------------------------------------------------------


def _reserved_name_book(model: str, factor: str) -> dict[str, Any]:
    book = _mf_book(model)
    book["obligors"][0]["obligor_uid"] = 1
    book["loadings"][0] = {"obligor_uid": 1, "factor": "F1", "loading": 0.5}
    book["loadings"][2] = {"obligor_uid": 30, "factor": factor, "loading": 0.3}
    return book


@pytest.mark.parametrize(
    ("model", "factor"),
    [("mf_gaussian", "IDIO_001"), ("mf_students_t", "CHI_GLOBAL")],
)
def test_s2_t7_reserved_factor_name_raises_before_the_engine(
    model: str, factor: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = _reserved_name_book(model, factor)
    # Without the request rule, the engine fails on the same input (F-36).
    assert not _engine_accepts(book, target=_engine_duplicate_factor_name(factor))

    reached: list[str] = []

    def engine_entry(*args: object, **kwargs: object) -> None:
        reached.append("engine")
        raise AssertionError("engine reached with a reserved factor name")

    monkeypatch.setattr(credit_var_run, "engine_version", engine_entry)
    monkeypatch.setattr(credit_var_run, "_build_loss_matrix", engine_entry)
    with pytest.raises(ValidationError) as excinfo:
        run_credit_var(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 2, "factor")
    assert reached == []


def test_s2_t7_chi_name_is_not_reserved_for_mf_gaussian() -> None:
    """A_CONTRACT section 2.5: CHI_GLOBAL is reserved for mf_students_t only."""
    book = _mf_book("mf_gaussian")
    book["loadings"][2]["factor"] = "CHI_GLOBAL"
    assert CreditVarRequest.model_validate(book).model == "mf_gaussian"
    assert _engine_accepts(book, target=_engine_duplicate_factor_name("CHI_GLOBAL"))


def test_s2_t7_idio_name_of_an_absent_uid_is_not_reserved() -> None:
    book = _mf_book("mf_gaussian")
    book["loadings"][2]["factor"] = "IDIO_001"  # no obligor has uid 1
    assert CreditVarRequest.model_validate(book).model == "mf_gaussian"


# ---------------------------------------------------------------------------
# OD-S2-2 (a), A_CONTRACT sections 2.1 to 2.3: no NUL character in a label
# ---------------------------------------------------------------------------


def _nul_in_loading_factor() -> tuple[dict[str, Any], tuple[str | int, ...]]:
    book = _mf_book()
    book["loadings"][2]["factor"] = "F\x001"
    return book, ("loadings", 2, "factor")


def _nul_in_obligor_group() -> tuple[dict[str, Any], tuple[str | int, ...]]:
    book = _grouped_book()
    book["obligors"][1]["group"] = "B\x00"
    return book, ("obligors", 1, "group")


def _nul_in_group_name() -> tuple[dict[str, Any], tuple[str | int, ...]]:
    book = _grouped_book()
    book["groups"][0]["name"] = "\x00A"
    return book, ("groups", 0, "name")


_NulCase = Callable[[], tuple[dict[str, Any], tuple[str | int, ...]]]


@pytest.mark.parametrize(
    "build",
    [_nul_in_loading_factor, _nul_in_obligor_group, _nul_in_group_name],
    ids=["LoadingRow.factor", "ObligorRow.group", "GroupRow.name"],
)
@pytest.mark.parametrize("path", ["dict", "json"])
def test_od_s2_2_nul_character_in_a_label_raises_at_the_field_path(
    build: _NulCase, path: str
) -> None:
    """S2 review R-002: one errors() entry at the field path (rule 1), on the
    dict path and on the JSON path."""
    book, loc = build()
    with pytest.raises(ValidationError) as excinfo:
        if path == "dict":
            CreditVarRequest.model_validate(book)
        else:
            CreditVarRequest.model_validate_json(json.dumps(book))
    error = _single_error(excinfo)
    assert error["loc"] == loc
    assert error["type"] == "string_pattern_mismatch"


# ---------------------------------------------------------------------------
# S2.T8 (C.18, O-8): strict coercion
# ---------------------------------------------------------------------------


def _set(book: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    target: Any = book
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


_REJECTED: list[tuple[str, tuple[str | int, ...], object]] = [
    ("ead str", ("obligors", 0, "ead"), "100"),
    ("obligor_uid float", ("obligors", 0, "obligor_uid"), 7.0),
    ("in_stress_test int", ("obligors", 0, "in_stress_test"), 1),
    ("seed str", ("seed",), "7"),
    ("group int", ("obligors", 0, "group"), 5),
    ("seed bool", ("seed",), True),
    ("ead bool", ("obligors", 0, "ead"), True),
    ("params value str", ("params", "rho"), "0.15"),
    ("model int", ("model",), 5),
]


@pytest.mark.parametrize(
    ("path", "value"),
    [(path, value) for _, path, value in _REJECTED],
    ids=[case for case, _, _ in _REJECTED],
)
def test_s2_t8_lax_coercion_is_rejected(
    path: tuple[str | int, ...], value: object
) -> None:
    book = _sf_book()
    _set(book, path, value)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == path
    with pytest.raises(ValidationError) as json_excinfo:
        CreditVarRequest.model_validate_json(json.dumps(book))
    assert _single_error(json_excinfo)["loc"] == path


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("obligors", 0, "obligor_uid"), np.int64(7)),
        (("obligors", 0, "in_stress_test"), np.bool_(True)),
    ],
    ids=["numpy int", "numpy bool"],
)
def test_s2_t8_numpy_scalars_are_rejected(
    path: tuple[str | int, ...], value: object
) -> None:
    book = _sf_book()
    _set(book, path, value)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == path


def test_s2_t8_int_for_float_is_stored_as_float() -> None:
    book = _sf_book()
    book["obligors"][0]["ead"] = 100
    from_dict = CreditVarRequest.model_validate(book)
    from_json = CreditVarRequest.model_validate_json(json.dumps(book))
    for request in (from_dict, from_json):
        assert type(request.obligors[0].ead) is float
        assert request.obligors[0].ead == 100.0


@pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
def test_s2_t8_list_or_tuple_is_stored_as_tuple(container: type) -> None:
    book = _mf_book()
    book["alphas"] = container([0.95, 0.99])
    book["obligors"] = container(book["obligors"])
    book["loadings"] = container(book["loadings"])
    request = CreditVarRequest.model_validate(book)
    assert request.alphas == (0.95, 0.99)
    assert type(request.alphas) is tuple
    assert type(request.obligors) is tuple
    assert type(request.loadings) is tuple
    grouped = _grouped_book()
    grouped["groups"] = container(grouped["groups"])
    assert type(CreditVarRequest.model_validate(grouped).groups) is tuple


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alphas", {0.99}),
        ("alphas", frozenset({0.99})),
        ("alphas", (a for a in (0.99,))),
        ("alphas", "0.99"),
        ("obligors", "obligors"),
    ],
    ids=[
        "alphas set",
        "alphas frozenset",
        "alphas generator",
        "alphas str",
        "obligors str",
    ],
)
def test_s2_t8_other_sequence_types_raise(field: str, value: object) -> None:
    book = _sf_book()
    book[field] = value
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == (field,)


def test_s2_t8_dict_key_and_value_types_are_strict() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate({**_sf_book(), "params": {1: 0.15}})
    assert _single_error(excinfo)["loc"] == ("params", 1, "[key]")
    stored = CreditVarRequest.model_validate({**_sf_book(), "params": {"rho": 0}})
    assert type(stored.params["rho"]) is float


def test_s2_t8_non_finite_float_raises() -> None:
    for value in (math.inf, -math.inf, math.nan):
        book = _sf_book()
        book["obligors"][2]["ead"] = value
        with pytest.raises(ValidationError) as excinfo:
            CreditVarRequest.model_validate(book)
        error = _single_error(excinfo)
        assert error["loc"] == ("obligors", 2, "ead")
        assert error["type"] == "finite_number"
    with pytest.raises(ValidationError) as params_excinfo:
        CreditVarRequest.model_validate({**_sf_book(), "params": {"rho": math.nan}})
    assert _single_error(params_excinfo)["loc"] == ("params", "rho")


# ---------------------------------------------------------------------------
# S2.T9 (C.19, O-12): no size limit
# ---------------------------------------------------------------------------


def test_s2_t9_no_scenario_limit_at_validation() -> None:
    request = CreditVarRequest.model_validate({
        "model": "independent_bernoulli",
        "obligors": [{"obligor_uid": 1, "ead": 1.0, "pd": 0.01, "lgd": 1.0}],
        "n_scenarios": 10**12,
    })
    assert request.n_scenarios == 10**12


# ---------------------------------------------------------------------------
# S2.T12 (C.21): from_dataframe docstring and default
# ---------------------------------------------------------------------------


def test_s2_t12_from_dataframe_default_scenarios_equal_the_request_default() -> None:
    default = (
        inspect
        .signature(CreditVarRequest.from_dataframe)
        .parameters["n_scenarios"]
        .default
    )
    assert default == CreditVarRequest.model_fields["n_scenarios"].default
    table = pl.DataFrame({
        "obligor_uid": [1],
        "ead": [1.0],
        "pd": [0.01],
        "lgd": [1.0],
    })
    request = CreditVarRequest.from_dataframe(table, model="independent_bernoulli")
    assert request.n_scenarios == default
    direct = CreditVarRequest.model_validate({
        "model": "independent_bernoulli",
        "obligors": [{"obligor_uid": 1, "ead": 1.0, "pd": 0.01, "lgd": 1.0}],
    })
    assert direct.n_scenarios == default


def _docstring_column_lists(doc: str) -> list[tuple[str, ...]]:
    """Every double-backticked comma list in `doc`, as a tuple of names."""
    return [
        tuple(name.strip() for name in match.split(","))
        for match in re.findall(r"``([a-z_]+(?:, [a-z_]+)+)``", doc)
    ]


def test_s2_t12_from_dataframe_docstring_names_the_required_columns() -> None:
    doc = inspect.getdoc(CreditVarRequest.from_dataframe)
    assert doc is not None
    lists = _docstring_column_lists(doc)
    assert ("obligor_uid", "ead", "pd", "lgd") in lists
    assert ("obligor_uid", "factor", "loading") in lists
    assert not any("uid" in names for names in lists)

    table = pl.DataFrame({"obligor_uid": [1], "ead": [1.0], "pd": [0.01], "lgd": [1.0]})
    loadings = pl.DataFrame({"obligor_uid": [1], "factor": ["F1"], "loading": [0.5]})
    request = CreditVarRequest.from_dataframe(
        table, model="mf_gaussian", loadings_table=loadings, n_scenarios=1_000
    )
    assert request.loadings is not None
    for name in ("obligor_uid", "ead", "pd", "lgd"):
        with pytest.raises(ValueError, match=name):
            CreditVarRequest.from_dataframe(
                table.drop(name), model="independent_bernoulli"
            )
    for name in ("obligor_uid", "factor", "loading"):
        with pytest.raises(ValueError, match=name):
            CreditVarRequest.from_dataframe(
                table, model="mf_gaussian", loadings_table=loadings.drop(name)
            )


def test_s2_t12_from_dataframe_non_integral_uid_raises() -> None:
    """A float uid with a fractional part raises R; it is not cast to 7. An
    integral float uid (a float column) is still accepted as an int."""
    table = pl.DataFrame({
        "obligor_uid": [1.0, 7.9],
        "ead": [1.0, 1.0],
        "pd": [0.01, 0.01],
        "lgd": [1.0, 1.0],
    })
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(table, model="independent_bernoulli")
    error = _single_error(excinfo)
    assert error["loc"] == ("obligor_uid",)
    assert error["input"] == 7.9

    table = table.with_columns(pl.Series("obligor_uid", [1.0, 7.0]))
    request = CreditVarRequest.from_dataframe(table, model="independent_bernoulli")
    assert [o.obligor_uid for o in request.obligors] == [1, 7]
    assert all(type(o.obligor_uid) is int for o in request.obligors)


def test_s2_t12_from_dataframe_non_integral_loadings_uid_raises() -> None:
    """The same rule for the loadings_table obligor_uid column."""
    table = pl.DataFrame({"obligor_uid": [1], "ead": [1.0], "pd": [0.01], "lgd": [1.0]})
    loadings = pl.DataFrame({
        "obligor_uid": [1.5],
        "factor": ["F1"],
        "loading": [0.5],
    })
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            table, model="mf_gaussian", loadings_table=loadings, n_scenarios=1_000
        )
    error = _single_error(excinfo)
    assert error["loc"] == ("obligor_uid",)
    assert error["input"] == 1.5
