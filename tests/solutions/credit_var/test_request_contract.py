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

import inspect
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
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


def _engine_accepts(book: dict[str, Any]) -> bool:
    """The engine's own loadings check (`credit.jtd.ingest.validate`), run on
    the recipe frames of `book`, without the request validator."""
    obligor_table = pd.DataFrame({
        "uid": [row["obligor_uid"] for row in book["obligors"]],
        "ead": [row["ead"] for row in book["obligors"]],
        "pd": [row["pd"] for row in book["obligors"]],
        "lgd": [row["lgd"] for row in book["obligors"]],
    })
    loadings_table = pd.DataFrame({
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
    except ValueError:
        return False
    return True


def test_s2_t6_duplicate_uid_factor_raises_as_in_the_engine() -> None:
    book = _mf_book()
    book["loadings"].append({"obligor_uid": 20, "factor": "F1", "loading": 0.1})
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 3, "factor")
    assert not _engine_accepts(book)


def test_s2_t6_sum_of_squares_above_tolerance_raises_as_in_the_engine() -> None:
    first, second = _two_factor_loadings(1.0 + 2e-9)
    assert math.fsum([first * first, second * second]) > 1.0 + 1e-9
    book = _book_with_obligor_20_loadings(first, second)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    assert _single_error(excinfo)["loc"] == ("loadings", 3, "loading")
    assert not _engine_accepts(book)


def test_s2_t6_sum_of_squares_inside_tolerance_is_accepted_as_in_the_engine() -> None:
    first, second = _two_factor_loadings(1.0 + 0.5e-9)
    total = math.fsum([first * first, second * second])
    assert 1.0 < total <= 1.0 + 1e-9
    book = _book_with_obligor_20_loadings(first, second)
    request = CreditVarRequest.model_validate(book)
    assert request.loadings is not None
    assert len(request.loadings) == 4
    assert _engine_accepts(book)


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
    assert not _engine_accepts(book)

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
    assert _engine_accepts(book)


def test_s2_t7_idio_name_of_an_absent_uid_is_not_reserved() -> None:
    book = _mf_book("mf_gaussian")
    book["loadings"][2]["factor"] = "IDIO_001"  # no obligor has uid 1
    assert CreditVarRequest.model_validate(book).model == "mf_gaussian"


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
    table = pd.DataFrame({
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

    table = pd.DataFrame({"obligor_uid": [1], "ead": [1.0], "pd": [0.01], "lgd": [1.0]})
    loadings = pd.DataFrame({"obligor_uid": [1], "factor": ["F1"], "loading": [0.5]})
    request = CreditVarRequest.from_dataframe(
        table, model="mf_gaussian", loadings_table=loadings, n_scenarios=1_000
    )
    assert request.loadings is not None
    for name in ("obligor_uid", "ead", "pd", "lgd"):
        with pytest.raises(ValueError, match=name):
            CreditVarRequest.from_dataframe(
                table.drop(columns=[name]), model="independent_bernoulli"
            )
    for name in ("obligor_uid", "factor", "loading"):
        with pytest.raises(ValueError, match=name):
            CreditVarRequest.from_dataframe(
                table, model="mf_gaussian", loadings_table=loadings.drop(columns=[name])
            )
