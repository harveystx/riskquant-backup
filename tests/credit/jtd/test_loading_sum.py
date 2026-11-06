# tests/credit/jtd/test_loading_sum.py
"""
WS_CREDIT_POLARS S1: the one admission loading sum (C.5, C.6).

S1.T1 compares `loading_sum_sq` with the pandas 3.0.x group sum recorded at
v0.2.0 (`golden/loading_sum_kahan.json`, written by
`_gates/record_s1_goldens.py`). The record is the independent oracle: pandas
made it before the function existed. S1.T2 runs the WS_CREDIT_VAR_API S2.T6
cases through the request rule and through the engine semantic check. S1.T4
reads the source of both callers.
"""

from __future__ import annotations

import ast
import functools
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.credit.jtd.ingest.loading_sum import LoadingSumSq, loading_sum_sq
from riskquant.solutions.credit_var import run as credit_var_run
from riskquant.solutions.credit_var.api import (
    CreditVarRequest,
    LoadingRow,
    ObligorRow,
)

_GOLDEN = Path(__file__).parent / "golden" / "loading_sum_kahan.json"
_SRC = Path(__file__).resolve().parents[3] / "src" / "riskquant"
_CALLERS = (
    _SRC / "credit" / "jtd" / "ingest" / "validate.py",
    _SRC / "solutions" / "credit_var" / "schemas.py",
)
_BOUND = 1.0 + 1e-9
_ENGINE_MESSAGE = "sum of squared loadings exceeds 1 for obligor(s) {uids}."


@functools.cache
def _golden() -> dict[str, Any]:
    with _GOLDEN.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _golden_rows() -> tuple[np.ndarray, np.ndarray]:
    rows = _golden()["rows"]
    uids = np.array(rows["obligor_uid"], dtype=np.int64)
    loadings = np.array([float.fromhex(h) for h in rows["loading"]], dtype=np.float64)
    return uids, loadings


def _sums(uids: list[int], loadings: list[float]) -> LoadingSumSq:
    return loading_sum_sq(
        obligor_uid=np.array(uids, dtype=np.int64),
        loading=np.array(loadings, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# S1.T1 (C.6): equal to the pandas record under float.hex()
# ---------------------------------------------------------------------------


def test_s1_t1_every_total_equals_the_pandas_record() -> None:
    golden = _golden()
    uids, loadings = _golden_rows()
    sums = loading_sum_sq(obligor_uid=uids, loading=loadings)
    got = {
        str(uid): total.hex()
        for uid, total in zip(sums.uids.tolist(), sums.totals.tolist(), strict=True)
    }
    expected: dict[str, str] = golden["totals"]
    assert got.keys() == expected.keys()
    diff = sorted(uid for uid in expected if got[uid] != expected[uid])
    assert not diff, f"{len(diff)} of {len(expected)} totals differ: {diff[:10]}"
    assert len(expected) == golden["n_groups"] + golden["n_twins"] + len(
        golden["edge_uids"]
    )


def test_s1_t1_record_holds_groups_whose_two_orders_give_two_totals() -> None:
    """The twin of a group holds the same values in another row order. Some
    pairs must give two totals, else the record cannot catch an order-free
    sum (F-4: a different summation order gives other bits)."""
    golden = _golden()
    uids, loadings = _golden_rows()
    values: dict[int, list[str]] = {}
    for uid, x in zip(uids.tolist(), loadings.tolist(), strict=True):
        values.setdefault(uid, []).append(x.hex())
    totals = golden["totals"]
    n_two = 0
    for base, twin in golden["twin_pairs"]:
        assert sorted(values[base]) == sorted(values[twin]), (base, twin)
        n_two += totals[str(base)] != totals[str(twin)]
    assert len(golden["twin_pairs"]) == golden["n_twins"] == 200
    assert n_two == golden["n_twin_pairs_with_two_totals"]
    assert n_two > 0


def test_s1_t1_record_holds_both_sides_of_the_bound() -> None:
    totals = [float.fromhex(h) for h in _golden()["totals"].values()]
    near = [t for t in totals if abs(t - _BOUND) <= 16 * 2.0**-52]
    assert any(t > _BOUND for t in near)
    assert any(t <= _BOUND for t in near)


def test_s1_t1_running_total_of_each_last_row_is_the_final_total() -> None:
    uids, loadings = _golden_rows()
    sums = loading_sum_sq(obligor_uid=uids, loading=loadings)
    last: dict[int, int] = {}
    for row, uid in enumerate(uids.tolist()):
        last[uid] = row
    final = dict(zip(sums.uids.tolist(), sums.totals.tolist(), strict=True))
    for uid, row in last.items():
        assert sums.running[row].hex() == final[uid].hex(), uid
    assert sums.uids.tolist() == list(dict.fromkeys(uids.tolist()))


def test_s1_t1_overflow_keeps_infinity_as_pandas_3_0_does() -> None:
    """Hand oracle for the compensation reset. Rows 1e154, 1e154, 0.1: the
    second row overflows the total to +inf with compensation +inf; the third
    row then gives compensation (inf - inf) - 0.01 = NaN. pandas 3.0.x sets a
    non-finite compensation to 0, so the total stays +inf and the obligor is
    rejected. Without the reset the total is NaN, and NaN > bound is False."""
    sums = _sums([7, 7, 7], [1e154, 1e154, 0.1])
    assert sums.totals.tolist() == [math.inf]
    assert sums.running.tolist() == [1e308, math.inf, math.inf]


# ---------------------------------------------------------------------------
# C.5 input contract
# ---------------------------------------------------------------------------


def test_c5_rows_of_one_obligor_need_not_be_adjacent() -> None:
    # Dyadic loadings: every square and sum is exact, so the oracle is exact.
    sums = _sums([30, 10, 30, 20, 10], [0.5, 0.75, 0.5, 0.25, 0.5])
    assert sums.uids.tolist() == [30, 10, 20]
    assert sums.totals.tolist() == [0.5, 0.8125, 0.0625]
    assert sums.running.tolist() == [0.25, 0.5625, 0.5, 0.0625, 0.8125]


def test_c5_empty_input_gives_empty_arrays() -> None:
    sums = _sums([], [])
    for arr, dtype in (
        (sums.running, np.float64),
        (sums.uids, np.int64),
        (sums.totals, np.float64),
    ):
        assert arr.shape == (0,)
        assert arr.dtype == dtype


def test_c5_outputs_are_c_contiguous_writeable_and_inputs_unchanged() -> None:
    uids = np.array([1, 2, 1], dtype=np.int64)
    loadings = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    uids.flags.writeable = False
    loadings.flags.writeable = False
    sums = loading_sum_sq(obligor_uid=uids, loading=loadings)
    for arr in (sums.running, sums.uids, sums.totals):
        assert arr.flags.c_contiguous
        assert arr.flags.writeable
    assert uids.tolist() == [1, 2, 1]
    assert loadings.tolist() == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    ("uids", "loadings"),
    [
        (np.array([1], dtype=np.int32), np.array([0.5])),
        (np.array([1], dtype=np.uint64), np.array([0.5])),
        (np.array([1.0]), np.array([0.5])),
        (np.array([1], dtype=np.int64), np.array([0.5], dtype=np.float32)),
        ([1], np.array([0.5])),
        (np.array([1], dtype=np.int64), [0.5]),
    ],
    ids=["int32", "uint64", "float_uid", "float32", "list_uid", "list_loading"],
)
def test_c5_wrong_type_raises_type_error(uids: Any, loadings: Any) -> None:
    with pytest.raises(TypeError, match="loading_sum_sq"):
        loading_sum_sq(obligor_uid=uids, loading=loadings)


def test_c5_wrong_shape_raises_value_error() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        loading_sum_sq(
            obligor_uid=np.array([[1]], dtype=np.int64),
            loading=np.array([[0.5]]),
        )
    with pytest.raises(ValueError, match="one value per row"):
        loading_sum_sq(
            obligor_uid=np.array([1, 2], dtype=np.int64), loading=np.array([0.5])
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_c5_non_finite_loading_raises_value_error(bad: float) -> None:
    with pytest.raises(ValueError, match="row 1"):
        _sums([1, 1], [0.5, bad])


# ---------------------------------------------------------------------------
# S1.T2: one decision in the request rule and in the engine check
# ---------------------------------------------------------------------------


def _obligor_rows(uids: tuple[int, ...]) -> list[dict[str, Any]]:
    return [{"obligor_uid": uid, "ead": 1.0e6, "pd": 0.03, "lgd": 0.45} for uid in uids]


def _mf_book(
    obligor_uids: tuple[int, ...], loadings: list[tuple[int, str, float]]
) -> dict[str, Any]:
    return {
        "model": "mf_gaussian",
        "obligors": _obligor_rows(obligor_uids),
        "loadings": [
            {"obligor_uid": uid, "factor": factor, "loading": loading}
            for uid, factor, loading in loadings
        ],
        "n_scenarios": 1_000,
        "alphas": [0.99],
        "seed": 7,
    }


def _s2_t6_book(sum_sq: float) -> dict[str, Any]:
    """Obligor 20 loads (0.6, sqrt(sum_sq - 0.36)) on F1 and F2, as in
    WS_CREDIT_VAR_API S2.T6; obligors 10 and 30 stay far inside the bound."""
    second = math.sqrt(sum_sq - 0.36)
    return _mf_book(
        (10, 20, 30),
        [(10, "F1", 0.5), (20, "F1", 0.6), (30, "F1", 0.3), (20, "F2", second)],
    )


def _engine_rejection(book: dict[str, Any]) -> ValueError | None:
    """Run the engine path of `run._simulate` on `book` without the request
    rules: `model_construct` skips request validation, so only the engine
    semantic check can reject the loadings. Returns the ValueError, or None."""
    request = CreditVarRequest.model_construct(
        model=book["model"],
        obligors=tuple(ObligorRow.model_validate(o) for o in book["obligors"]),
        loadings=tuple(LoadingRow.model_validate(lo) for lo in book["loadings"]),
        n_scenarios=book["n_scenarios"],
        alphas=tuple(book["alphas"]),
        seed=book["seed"],
    )
    try:
        credit_var_run._simulate(request, list(request.obligors))
    except ValueError as exc:
        frames = {frame.name for frame in traceback.extract_tb(exc.__traceback__)}
        assert {"validate_ingest_bundle_semantics", "_semantic_mf_gaussian"} <= frames
        return exc
    return None


def _request_loc(book: dict[str, Any]) -> tuple[Any, ...] | None:
    try:
        CreditVarRequest.model_validate(book)
    except ValidationError as exc:
        errors = exc.errors()
        assert len(errors) == 1, errors
        return tuple(errors[0]["loc"])
    return None


def test_s1_t2_above_tolerance_is_rejected_by_both() -> None:
    book = _s2_t6_book(1.0 + 2e-9)
    assert _request_loc(book) == ("loadings", 3, "loading")
    rejection = _engine_rejection(book)
    assert rejection is not None
    assert _ENGINE_MESSAGE.format(uids=[20]) in str(rejection)


def test_s1_t2_inside_tolerance_is_accepted_by_both() -> None:
    book = _s2_t6_book(1.0 + 0.5e-9)
    assert _request_loc(book) is None
    assert _engine_rejection(book) is None


def test_s1_t2_engine_lists_rejected_uids_in_ascending_order() -> None:
    """Obligors given as 30, 10, 20, each above the bound (0.8^2 + 0.8^2).
    The engine message lists [10, 20, 30] (F-28). The request loc is row 3,
    where the running sum of obligor 30 first exceeds the bound."""
    book = _mf_book(
        (30, 10, 20),
        [
            (30, "F1", 0.8),
            (10, "F1", 0.8),
            (20, "F1", 0.8),
            (30, "F2", 0.8),
            (10, "F2", 0.8),
            (20, "F2", 0.8),
        ],
    )
    assert _request_loc(book) == ("loadings", 3, "loading")
    rejection = _engine_rejection(book)
    assert rejection is not None
    assert _ENGINE_MESSAGE.format(uids=[10, 20, 30]) in str(rejection)


# ---------------------------------------------------------------------------
# S1.T4 (C.5): the two callers hold one call and no other sum
# ---------------------------------------------------------------------------


def _mentions_loading(node: ast.AST) -> bool:
    return "loading" in ast.unparse(node)


@pytest.mark.parametrize("path", _CALLERS, ids=lambda p: p.name)
def test_s1_t4_caller_calls_the_function_once_and_sums_nothing_else(
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "loading_sum_sq"
    ]
    assert len(calls) == 1, f"{path.name}: {len(calls)} calls to loading_sum_sq"
    assert "groupby" not in source, f"{path.name} names groupby"
    squares = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and _mentions_loading(node)
        and (
            (
                isinstance(node.op, ast.Mult)
                and ast.dump(node.left) == ast.dump(node.right)
            )
            or (
                isinstance(node.op, ast.Pow)
                and isinstance(node.right, ast.Constant)
                and node.right.value == 2
            )
        )
    ]
    assert not squares, f"{path.name} squares loadings outside C.5: {squares}"
