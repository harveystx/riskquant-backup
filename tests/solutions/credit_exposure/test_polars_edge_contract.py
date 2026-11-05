# tests/solutions/credit_exposure/test_polars_edge_contract.py
"""
WS_CREDIT_POLARS S3: the credit exposure chain on Polars (C.7, C.8, C.12, C.4).

S3.T1  `compile_add_on_factors` and `compile_saccr_config` take ``pl.DataFrame``
       only, refuse a wrong table before any column read, and apply the C.8
       column rules (no truncation, no wrap, null and NaN refused).
S3.T2  `CreditExposureRequest.from_dataframe` and `attach_exposure_ead` on the lab
       book; a 120-trade book whose first 100 trades have no volume (F-23).
S3.T3  Each catalog method on the two S3 FIRST books gives the result JSON,
       without ``provenance.engine_version``, of the v0.2.0 record that
       ``_gates/record_s3_goldens.py`` made. The record also holds each request
       JSON, so a book change fails as a book change. The comparison runs only
       at the numpy, scipy and numba versions of the record (D-C15).
S3.T4  `from_dataframe` column admission by dtype (C.12, the C.11 rule of each
       field type).
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy
from pydantic import ValidationError

from riskquant.credit.exposure.add_on.ingest import compile_add_on_factors
from riskquant.credit.exposure.saccr.ingest import compile_saccr_config
from riskquant.solutions.credit_exposure import golden_master_credit_exposure as gm
from riskquant.solutions.credit_exposure.api import (
    CreditExposureRequest,
    CreditExposureResult,
    attach_exposure_ead,
    exposure_model_entry,
    run_credit_exposure,
    supported_exposure_models,
)
from riskquant_lab.fixtures.credit_exposure import generate_canonical_exposure_book

_HERE = Path(__file__).resolve().parent
_GOLDEN = _HERE / "golden" / "credit_exposure_results_v0_2_0.json"
_EXCLUDE: dict[str, Any] = {"provenance": {"engine_version"}}
_BOOKS = ("frozen_book", "lab_canonical_book")

# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_TRADE_CASTS: dict[str, type] = {
    "trade_uid": int,
    "netting_set_uid": int,
    "mtm": float,
    "volume": float,
    "current_price": float,
    "asset_subclass_uix": int,
    "direction": str,
    "remaining_maturity_years": float,
}
_NETTING_SET_CASTS: dict[str, type] = {
    "netting_set_uid": int,
    "obligor_uid": int,
    "net_collateral": float,
}
_ADD_ON_CASTS: dict[str, type] = {"subclass_uix": int, "factor": float}
_HEDGING_SETS = (10, 20)
_HEDGING_SET_RHO = ((10, 0.40), (20, 0.30))


def _add_on_frame() -> pl.DataFrame:
    return pl.DataFrame({"subclass_uix": [3, 1, 2], "factor": [0.3, 0.1, 0.2]})


def _hedging_frame() -> pl.DataFrame:
    return pl.DataFrame({"subclass_uix": [2, 1], "hedging_set_uix": [20, 10]})


def _rho_frame() -> pl.DataFrame:
    return pl.DataFrame({"hedging_set_uix": [20, 10], "rho": [0.3, 0.4]})


def _lab_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The lab canonical book as Polars tables, with the SA-CCR trade inputs
    that golden_master_credit_exposure attaches (long, M = 1y)."""
    book = generate_canonical_exposure_book()
    n = len(book.trade_uid)
    trades = pl.DataFrame({
        "trade_uid": book.trade_uid,
        "netting_set_uid": book.trade_netting_set_id,
        "mtm": book.mtm,
        "volume": book.volume,
        "current_price": book.current_price,
        "asset_subclass_uix": book.asset_subclass_uix,
        "direction": ["long"] * n,
        "remaining_maturity_years": [1.0] * n,
    })
    netting_sets = pl.DataFrame({
        "netting_set_uid": book.netting_set_uid,
        "obligor_uid": book.netting_set_obligor_uid,
        "net_collateral": book.netting_set_collateral,
    })
    add_on = pl.DataFrame({
        "subclass_uix": book.subclass_uix,
        "factor": book.subclass_factor,
    })
    return trades, netting_sets, add_on


def _rows(frame: pl.DataFrame, casts: dict[str, type]) -> list[dict[str, Any]]:
    columns = {name: frame.get_column(name).to_list() for name in casts}
    return [
        {name: casts[name](columns[name][i]) for name in casts}
        for i in range(frame.height)
    ]


def _tables(
    trades: pl.DataFrame, netting_sets: pl.DataFrame, add_on: pl.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    """Request tables as dicts, built as `_gates/record_s3_goldens.py` builds them."""
    subclasses = sorted(int(u) for u in add_on.get_column("subclass_uix").to_list())
    return {
        "trades": _rows(trades, _TRADE_CASTS),
        "netting_sets": _rows(netting_sets, _NETTING_SET_CASTS),
        "add_on_factors": _rows(add_on, _ADD_ON_CASTS),
        "subclass_hedging_sets": [
            {"subclass_uix": u, "hedging_set_uix": _HEDGING_SETS[i % 2]}
            for i, u in enumerate(subclasses)
        ],
        "hedging_set_correlations": [
            {"hedging_set_uix": h, "rho": r} for h, r in _HEDGING_SET_RHO
        ],
    }


def _book_tables(book: str) -> dict[str, list[dict[str, Any]]]:
    frames = gm._frozen_book() if book == "frozen_book" else _lab_frames()
    return _tables(*frames)


def _request_dict(
    tables: dict[str, list[dict[str, Any]]], method: str
) -> dict[str, Any]:
    entry = exposure_model_entry(method)
    payload: dict[str, Any] = {
        "model": method,
        "trades": tables["trades"],
        "netting_sets": tables["netting_sets"],
    }
    if entry.requires_add_on:
        payload["add_on_factors"] = tables["add_on_factors"]
    if entry.requires_saccr_config:
        payload["subclass_hedging_sets"] = tables["subclass_hedging_sets"]
        payload["hedging_set_correlations"] = tables["hedging_set_correlations"]
    return payload


def _from_dataframe(
    trades: pl.DataFrame,
    netting_sets: pl.DataFrame,
    add_on: pl.DataFrame,
    method: str,
) -> CreditExposureRequest:
    entry = exposure_model_entry(method)
    hedging, rho = gm._saccr_config_for(add_on)
    return CreditExposureRequest.from_dataframe(
        trades,
        netting_sets,
        model=method,
        add_on_factors=add_on if entry.requires_add_on else None,
        subclass_hedging_sets=hedging if entry.requires_saccr_config else None,
        hedging_set_correlations=rho if entry.requires_saccr_config else None,
    )


# ---------------------------------------------------------------------------
# S3.T1 (C.7, C.8): exposure ingest entries
# ---------------------------------------------------------------------------

_NOT_POLARS: dict[str, Any] = {
    "pandas": pd.DataFrame({"no_such_column": [1]}),
    "lazy": pl.DataFrame({"no_such_column": [1]}).lazy(),
    "dict": {"subclass_uix": [1], "factor": [0.1]},
    "none": None,
}


@pytest.mark.parametrize("kind", sorted(_NOT_POLARS))
def test_s3_t1_add_on_refuses_a_table_that_is_not_a_polars_frame(kind: str) -> None:
    # The pandas and lazy frames miss the required columns: a column read
    # before the type check would raise ValueError, not TypeError.
    with pytest.raises(TypeError, match=r"pl\.DataFrame"):
        compile_add_on_factors(_NOT_POLARS[kind])


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("kind", sorted(_NOT_POLARS))
def test_s3_t1_saccr_refuses_a_table_that_is_not_a_polars_frame(
    kind: str, position: int
) -> None:
    # The other table is a Polars frame without columns: both type checks run
    # before any column check.
    tables: list[Any] = [pl.DataFrame(), pl.DataFrame()]
    tables[position] = _NOT_POLARS[kind]
    with pytest.raises(TypeError, match=r"pl\.DataFrame"):
        compile_saccr_config(tables[0], tables[1])


@pytest.mark.parametrize(
    ("column", "values", "dtype", "error"),
    [
        ("subclass_uix", [1.0, 1.5, 3.0], pl.Float64, TypeError),
        ("subclass_uix", [1.0, math.inf, 3.0], pl.Float64, TypeError),
        ("subclass_uix", [1, 2**40, 3], pl.Int64, TypeError),
        ("subclass_uix", [1, None, 3], pl.Int64, ValueError),
        ("subclass_uix", [1.0, math.nan, 3.0], pl.Float64, ValueError),
        ("subclass_uix", [True, False, True], pl.Boolean, TypeError),
        ("subclass_uix", ["1", "2", "3"], pl.Utf8, TypeError),
        ("factor", [0.1, None, 0.3], pl.Float64, ValueError),
        ("factor", [0.1, math.nan, 0.3], pl.Float64, ValueError),
        ("factor", [True, False, True], pl.Boolean, TypeError),
    ],
)
def test_s3_t1_add_on_column_rules(
    column: str, values: list[Any], dtype: pl.DataType, error: type[Exception]
) -> None:
    frame = _add_on_frame().with_columns(pl.Series(column, values, dtype=dtype))
    with pytest.raises(error, match=column):
        compile_add_on_factors(frame)


@pytest.mark.parametrize(
    ("table", "column", "values", "dtype", "error"),
    [
        ("hedging", "subclass_uix", [2.0, 1.5], pl.Float64, TypeError),
        ("hedging", "hedging_set_uix", [20, 2**40], pl.Int64, TypeError),
        ("hedging", "hedging_set_uix", [20, None], pl.Int64, ValueError),
        ("rho", "hedging_set_uix", [20.0, 10.5], pl.Float64, TypeError),
        ("rho", "rho", [0.3, None], pl.Float64, ValueError),
        ("rho", "rho", [0.3, math.nan], pl.Float64, ValueError),
        ("rho", "rho", [True, False], pl.Boolean, TypeError),
    ],
)
def test_s3_t1_saccr_column_rules(
    table: str,
    column: str,
    values: list[Any],
    dtype: pl.DataType,
    error: type[Exception],
) -> None:
    hedging, rho = _hedging_frame(), _rho_frame()
    series = pl.Series(column, values, dtype=dtype)
    if table == "hedging":
        hedging = hedging.with_columns(series)
    else:
        rho = rho.with_columns(series)
    with pytest.raises(error, match=column):
        compile_saccr_config(hedging, rho)


def test_s3_t1_integral_floats_compile_to_contiguous_writeable_arrays() -> None:
    add_on = _add_on_frame().with_columns(pl.col("subclass_uix").cast(pl.Float64))
    before = add_on.clone()
    compiled = compile_add_on_factors(add_on)
    assert add_on.equals(before)
    np.testing.assert_array_equal(compiled.subclass_uix, [1, 2, 3])
    np.testing.assert_array_equal(compiled.factor, [0.1, 0.2, 0.3])

    hedging = _hedging_frame().with_columns(pl.col("subclass_uix").cast(pl.Int8))
    config = compile_saccr_config(hedging, _rho_frame())
    np.testing.assert_array_equal(config.subclass_uix, [1, 2])
    np.testing.assert_array_equal(config.subclass_hedging_set, [10, 20])
    np.testing.assert_array_equal(config.hedging_set_uix, [10, 20])
    np.testing.assert_array_equal(config.hedging_set_rho, [0.4, 0.3])

    arrays: dict[str, tuple[np.ndarray[Any, Any], type[np.generic]]] = {
        "subclass_uix": (compiled.subclass_uix, np.int32),
        "factor": (compiled.factor, np.float64),
        "config.subclass_uix": (config.subclass_uix, np.int32),
        "config.subclass_hedging_set": (config.subclass_hedging_set, np.int32),
        "config.hedging_set_uix": (config.hedging_set_uix, np.int32),
        "config.hedging_set_rho": (config.hedging_set_rho, np.float64),
    }
    for name, (array, dtype) in arrays.items():
        assert array.dtype == dtype, name
        assert array.flags.c_contiguous, name
        assert array.flags.writeable, name


# ---------------------------------------------------------------------------
# S3.T2 (C.12): from_dataframe, attach_exposure_ead, the 120-trade book
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", supported_exposure_models())
def test_s3_t2_from_dataframe_equals_the_dict_built_request(method: str) -> None:
    frames = _lab_frames()
    from_frames = _from_dataframe(*frames, method)
    from_dicts = CreditExposureRequest.model_validate(
        _request_dict(_tables(*frames), method)
    )
    assert from_frames.model_dump_json() == from_dicts.model_dump_json()
    assert run_credit_exposure(from_frames).model_dump_json() == (
        run_credit_exposure(from_dicts).model_dump_json()
    )


def _lab_result() -> CreditExposureResult:
    return run_credit_exposure(_from_dataframe(*_lab_frames(), "replacement_cost"))


def test_s3_t2_attach_exposure_ead_returns_a_new_frame() -> None:
    result = _lab_result()
    expected = result.ead_by_obligor_uid()
    table = pl.DataFrame({"obligor_uid": [3, 1, 5], "name": ["c", "a", "e"]})
    before = table.clone()
    out = attach_exposure_ead(table, result)
    assert table.equals(before)
    assert out.columns == ["obligor_uid", "name", "ead"]
    assert out.schema["ead"] == pl.Float64
    assert out.get_column("ead").to_list() == [expected[3], expected[1], expected[5]]

    renamed = table.rename({"obligor_uid": "uid"})
    out_uid = attach_exposure_ead(renamed, result, uid_col="uid")
    assert out_uid.get_column("ead").to_list() == out.get_column("ead").to_list()

    with pytest.raises(ValueError, match=r"\[99\]"):
        attach_exposure_ead(pl.DataFrame({"obligor_uid": [1, 99]}), result)
    with pytest.raises(ValueError, match="missing 'uid'"):
        attach_exposure_ead(table, result, uid_col="uid")
    with pytest.raises(TypeError, match=r"pl\.DataFrame"):
        attach_exposure_ead(pd.DataFrame(table.to_dict(as_series=False)), result)  # type: ignore[arg-type]


def test_s3_t2_book_of_120_trades_with_no_volume_in_the_first_100() -> None:
    n = 120
    volume = [None] * 100 + [float(v) for v in range(1, n - 99)]
    trades = pl.DataFrame(
        {
            "trade_uid": list(range(1, n + 1)),
            "netting_set_uid": [1 + i % 3 for i in range(n)],
            "mtm": [float(10 * (i % 7) - 30) for i in range(n)],
            "volume": volume,
        },
        schema={
            "trade_uid": pl.Int64,
            "netting_set_uid": pl.Int64,
            "mtm": pl.Float64,
            "volume": pl.Float64,
        },
    )
    netting_sets = pl.DataFrame({
        "netting_set_uid": [1, 2, 3],
        "obligor_uid": [7, 7, 8],
        "net_collateral": [0.0, 5.0, 0.0],
    })
    request = CreditExposureRequest.from_dataframe(
        trades, netting_sets, model="replacement_cost"
    )
    assert [t.volume for t in request.trades] == volume
    result = run_credit_exposure(request)
    # Independent oracle: RC per netting set = max(sum(mtm) - collateral, 0).
    mtm = trades.get_column("mtm").to_list()
    ns = trades.get_column("netting_set_uid").to_list()
    collateral = {1: 0.0, 2: 5.0, 3: 0.0}
    expected = {
        k: max(math.fsum(m for m, s in zip(mtm, ns, strict=True) if s == k) - c, 0.0)
        for k, c in collateral.items()
    }
    got = {row.netting_set_uid: row.ead for row in result.netting_set_ead}
    assert got == expected


# ---------------------------------------------------------------------------
# S3.T3 (C.4, C.12): results equal the v0.2.0 record
# ---------------------------------------------------------------------------


@functools.cache
def _golden() -> dict[str, Any]:
    with _GOLDEN.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _version_differences() -> list[str]:
    golden = _golden()
    runtime = {
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numba_version": numba.__version__,
    }
    return [
        f"{name}: recorded {golden[name]}, runtime {value}"
        for name, value in runtime.items()
        if golden[name] != value
    ]


_CASES = sorted(
    f"{book}:{method}" for book in _BOOKS for method in supported_exposure_models()
)


def test_s3_t3_record_holds_each_book_and_method() -> None:
    assert set(_golden()["cases"]) == set(_CASES)
    assert len(_CASES) == 6
    assert _golden()["excluded"] == ["provenance.engine_version"]


@pytest.mark.parametrize("case", _CASES)
def test_s3_t3_result_json_equals_the_v0_2_0_record(case: str) -> None:
    recorded = _golden()["cases"][case]
    book, method = case.split(":")
    request = CreditExposureRequest.model_validate(
        _request_dict(_book_tables(book), method)
    )
    assert request.model_dump_json() == recorded["request"], (
        f"{case}: the request differs from the recorded request"
    )
    differences = _version_differences()
    if differences:
        pytest.fail(
            "C.4 compares only at the recorded numpy, scipy and numba versions "
            f"(D-C15): {'; '.join(differences)}"
        )
    result = run_credit_exposure(request)
    assert result.model_dump_json(exclude=_EXCLUDE) == recorded["result"], case


def test_s3_t3_version_check_names_each_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__version__", "0.0.1")
    monkeypatch.setattr(numba, "__version__", "0.0.3")
    differences = _version_differences()
    assert len(differences) == 2
    assert differences[0].startswith("numpy_version: recorded ")
    assert differences[0].endswith("runtime 0.0.1")
    assert differences[1].startswith("numba_version: recorded ")


# ---------------------------------------------------------------------------
# S3.T4 (C.12): from_dataframe column admission by dtype
# ---------------------------------------------------------------------------


def _small_trades(**columns: pl.Series) -> pl.DataFrame:
    base = pl.DataFrame({
        "trade_uid": [7, 8],
        "netting_set_uid": [1, 1],
        "mtm": [10.0, -4.0],
        "volume": [100.0, 50.0],
    })
    return base.with_columns(*columns.values())


_NETTING_SETS = pl.DataFrame({
    "netting_set_uid": [1],
    "obligor_uid": [9],
    "net_collateral": [0.0],
})


def _rc(trades: pl.DataFrame) -> CreditExposureRequest:
    return CreditExposureRequest.from_dataframe(
        trades, _NETTING_SETS, model="replacement_cost"
    )


def test_s3_t4_integral_float_uid_is_admitted_and_fractional_uid_raises() -> None:
    admitted = _rc(_small_trades(t=pl.Series("trade_uid", [7.0, 8.0])))
    assert [t.trade_uid for t in admitted.trades] == [7, 8]
    assert all(type(t.trade_uid) is int for t in admitted.trades)

    for bad in (7.9, math.nan, math.inf):
        with pytest.raises(ValidationError) as excinfo:
            _rc(_small_trades(t=pl.Series("trade_uid", [bad, 8.0])))
        assert excinfo.value.errors()[0]["loc"] == ("trade_uid",)


@pytest.mark.parametrize(
    ("column", "series"),
    [
        ("volume", pl.Series("volume", [True, False], dtype=pl.Boolean)),
        ("mtm", pl.Series("mtm", [True, False], dtype=pl.Boolean)),
        ("mtm", pl.Series("mtm", ["7.9", "1"], dtype=pl.Utf8)),
        ("mtm", pl.Series("mtm", [1, 2], dtype=pl.Decimal(10, 2))),
        ("trade_uid", pl.Series("trade_uid", ["7", "8"], dtype=pl.Utf8)),
        ("trade_uid", pl.Series("trade_uid", [None, None], dtype=pl.Null)),
        ("direction", pl.Series("direction", ["long", "short"], dtype=pl.Categorical)),
        ("direction", pl.Series("direction", [1, 2], dtype=pl.Int64)),
    ],
)
def test_s3_t4_a_column_of_a_refused_dtype_raises_type_error(
    column: str, series: pl.Series
) -> None:
    with pytest.raises(TypeError, match=f"'{column}'"):
        _rc(_small_trades(c=series))


def test_s3_t4_null_cells() -> None:
    # Optional field: a null (in a float column: null or NaN) means absent; a
    # pl.Null column means absent for all rows.
    request = _rc(
        _small_trades(
            v=pl.Series("volume", [None, math.nan], dtype=pl.Float64),
            d=pl.Series("direction", [None, None], dtype=pl.Null),
            s=pl.Series("asset_subclass_uix", [3.0, None], dtype=pl.Float64),
        )
    )
    assert [t.volume for t in request.trades] == [None, None]
    assert [t.direction for t in request.trades] == [None, None]
    assert [t.asset_subclass_uix for t in request.trades] == [3, None]
    # Required field: a null cell raises R at the field path.
    for column, series in (
        ("trade_uid", pl.Series("trade_uid", [7, None], dtype=pl.Int64)),
        ("mtm", pl.Series("mtm", [None, 1.0], dtype=pl.Float64)),
    ):
        with pytest.raises(ValidationError) as excinfo:
            _rc(_small_trades(c=series))
        assert excinfo.value.errors()[0]["loc"] == (column,)


@pytest.mark.parametrize("position", ["trades", "netting_sets", "add_on_factors"])
@pytest.mark.parametrize("kind", ["pandas", "lazy", "dict"])
def test_s3_t4_from_dataframe_refuses_a_table_that_is_not_a_polars_frame(
    kind: str, position: str
) -> None:
    trades, netting_sets, add_on = _lab_frames()
    bad = {
        "pandas": pd.DataFrame(trades.to_dict(as_series=False)),
        "lazy": trades.lazy(),
        "dict": trades.to_dict(as_series=False),
    }[kind]
    tables: dict[str, Any] = {
        "trades": trades,
        "netting_sets": netting_sets,
        "add_on_factors": add_on,
    }
    tables[position] = bad
    with pytest.raises(TypeError, match=position):
        CreditExposureRequest.from_dataframe(
            tables["trades"],
            tables["netting_sets"],
            model="current_exposure_method",
            add_on_factors=tables["add_on_factors"],
        )
