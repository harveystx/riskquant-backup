# tests/solutions/credit_var/test_polars_edge_contract.py
"""
WS_CREDIT_POLARS S6: `CreditVarRequest.from_dataframe` on ``pl.DataFrame`` and
release 0.2.1.

S6.T1 (C.11)  A table that is not a ``pl.DataFrame`` (pandas, ``pl.LazyFrame``,
              dict) raises TypeError before any column read, in
              ``obligor_table`` and in ``loadings_table``.
S6.T2 (C.11)  ``pl.DataFrame`` tables for each of the 13 catalog keys give the
              request that the dict path gives, field for field, with an equal
              ``input_hash``.
S6.T3 (C.11)  Column admission by dtype and the null and NaN rules (B-4 (ii),
              B-5, B-8, S3 OD-1 and OD-C, S3 OD-A (b), S4+S5 OD-1 (a) and R-002).
S6.T4 (C.17)  The project version and the installed distribution version are
              0.2.1; the pandas requirement is still ``pandas>=2.0``.
S3 OD-2       The handoff example of `solutions/credit_exposure/handoff.py`
              (`attach_exposure_ead` -> `from_dataframe` -> `run_credit_var`)
              runs as its docstring states it.

The negative pandas case imports pandas inside the test (`pytest.importorskip`),
so this file collects in an environment without pandas (WS_PANDAS_REMOVAL).
"""

from __future__ import annotations

import importlib.metadata
import math
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest
from pydantic import ValidationError

from riskquant.solutions.credit_exposure.api import (
    CreditExposureRequest,
    attach_exposure_ead,
    run_credit_exposure,
)
from riskquant.solutions.credit_var.api import (
    CreditVarRequest,
    CreditVarResult,
    model_entry,
    run_credit_var,
    supported_model_keys,
)
from riskquant.solutions.credit_var.provenance import request_input_hash
from riskquant_lab.fixtures.credit_exposure import generate_canonical_exposure_book

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_PYPROJECT: Final[Path] = _REPO_ROOT / "pyproject.toml"
_RELEASE: Final[str] = "0.2.1"

_UIDS: Final[list[int]] = [11, 12, 13, 14]
_EADS: Final[list[float]] = [1_000_000.0, 2_500_000.0, 400_000.0, 750_000.0]
_PDS: Final[list[float]] = [0.03, 0.05, 0.10, 0.25]
_LGDS: Final[list[float]] = [0.45, 0.60, 0.40, 0.55]
_GROUPS: Final[list[str]] = ["IG", "HY", "HY", "IG"]
#: Explicit rho on two obligors; the others take the fill (null cell).
_RHOS: Final[list[float | None]] = [0.20, None, 0.35, None]
_LOADINGS: Final[list[tuple[int, str, float]]] = [
    (11, "MARKET", 0.30),
    (11, "ENERGY", 0.50),
    (12, "MARKET", 0.25),
    (13, "MARKET", 0.40),
    (13, "TECH", 0.45),
    (14, "MARKET", 0.20),
]
_GROUP_PARAMS: Final[dict[str, dict[str, dict[str, float]]]] = {
    "beta_binomial_grouped": {
        "IG": {"alpha": 1.0, "beta": 80.0},
        "HY": {"alpha": 2.0, "beta": 18.0},
    },
    "stress_regime_bernoulli_grouped": {
        "IG": {"stress_prob": 0.05, "stress_pd_multiplier": 2.0},
        "HY": {"stress_prob": 0.10, "stress_pd_multiplier": 3.0},
    },
}
_RUN: Final[dict[str, Any]] = {"n_scenarios": 1_000, "alphas": [0.99], "seed": 7}


def _obligor_frame(**columns: pl.Series) -> pl.DataFrame:
    """The base obligor table; a keyword replaces or adds one column."""
    frame = pl.DataFrame({
        "obligor_uid": pl.Series(_UIDS, dtype=pl.Int64),
        "ead": pl.Series(_EADS, dtype=pl.Float64),
        "pd": pl.Series(_PDS, dtype=pl.Float64),
        "lgd": pl.Series(_LGDS, dtype=pl.Float64),
    })
    return frame.with_columns(series.alias(name) for name, series in columns.items())


def _loadings_frame(**columns: pl.Series) -> pl.DataFrame:
    frame = pl.DataFrame({
        "obligor_uid": pl.Series([u for u, _, _ in _LOADINGS], dtype=pl.Int64),
        "factor": pl.Series([f for _, f, _ in _LOADINGS], dtype=pl.Utf8),
        "loading": pl.Series([v for _, _, v in _LOADINGS], dtype=pl.Float64),
    })
    return frame.with_columns(series.alias(name) for name, series in columns.items())


def _single_error(excinfo: pytest.ExceptionInfo[ValidationError]) -> dict[str, Any]:
    errors = excinfo.value.errors()
    assert len(errors) == 1, errors
    return dict(errors[0])


# ---------------------------------------------------------------------------
# S6.T1: a table that is not a pl.DataFrame raises TypeError before any read
# ---------------------------------------------------------------------------


def _wrong_tables() -> list[tuple[str, object]]:
    pd = pytest.importorskip("pandas")
    frame = _obligor_frame()
    return [
        (
            "pandas",
            pd.DataFrame({c: frame.get_column(c).to_list() for c in frame.columns}),
        ),
        ("lazyframe", frame.lazy()),
        ("dict", {c: frame.get_column(c).to_list() for c in frame.columns}),
    ]


@pytest.mark.parametrize("kind", ["pandas", "lazyframe", "dict"])
def test_s6_t1_obligor_table_of_another_type_raises_type_error(kind: str) -> None:
    table = dict(_wrong_tables())[kind]
    with pytest.raises(TypeError, match=r"'obligor_table' must be a pl\.DataFrame"):
        CreditVarRequest.from_dataframe(table, model="independent_bernoulli")  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["pandas", "lazyframe", "dict"])
def test_s6_t1_loadings_table_of_another_type_raises_before_any_read(
    kind: str,
) -> None:
    """The obligor table has a refused ``ead`` dtype. The TypeError names
    ``loadings_table``, so the type check ran before the first column read."""
    obligors = _obligor_frame(ead=pl.Series([True, False, True, False]))
    loadings = dict(_wrong_tables())[kind]
    with pytest.raises(TypeError, match=r"'loadings_table' must be a pl\.DataFrame"):
        CreditVarRequest.from_dataframe(
            obligors,
            model="mf_gaussian",
            loadings_table=loadings,  # type: ignore[arg-type]
            **_RUN,
        )


# ---------------------------------------------------------------------------
# S6.T2: Polars tables equal the dict-built request for each catalog key
# ---------------------------------------------------------------------------


def _dict_request(key: str) -> CreditVarRequest:
    entry = model_entry(key)
    obligors: list[dict[str, Any]] = []
    for i, uid in enumerate(_UIDS):
        row: dict[str, Any] = {
            "obligor_uid": uid,
            "ead": _EADS[i],
            "pd": _PDS[i],
            "lgd": _LGDS[i],
        }
        if entry.requires_groups:
            row["group"] = _GROUPS[i]
        if entry.accepts_obligor_rho and _RHOS[i] is not None:
            row["rho"] = _RHOS[i]
        obligors.append(row)
    payload: dict[str, Any] = {"model": key, "obligors": obligors, **_RUN}
    if entry.requires_groups:
        payload["groups"] = [
            {"name": name, "params": params}
            for name, params in _GROUP_PARAMS[key].items()
        ]
    if entry.requires_loadings:
        payload["loadings"] = [
            {"obligor_uid": u, "factor": f, "loading": v} for u, f, v in _LOADINGS
        ]
    return CreditVarRequest.model_validate(payload)


def _frame_request(key: str) -> CreditVarRequest:
    entry = model_entry(key)
    columns: dict[str, pl.Series] = {}
    if entry.requires_groups:
        columns["group"] = pl.Series(_GROUPS, dtype=pl.Utf8)
    if entry.accepts_obligor_rho:
        columns["rho"] = pl.Series(_RHOS, dtype=pl.Float64)
    return CreditVarRequest.from_dataframe(
        _obligor_frame(**columns),
        model=key,
        loadings_table=_loadings_frame() if entry.requires_loadings else None,
        group_params=_GROUP_PARAMS.get(key),
        **_RUN,
    )


def test_s6_t2_covers_the_13_catalog_keys() -> None:
    assert len(supported_model_keys()) == 13


@pytest.mark.parametrize("key", supported_model_keys())
def test_s6_t2_polars_tables_equal_the_dict_built_request(key: str) -> None:
    from_frame = _frame_request(key)
    from_dict = _dict_request(key)
    assert from_frame == from_dict
    assert from_frame.model_dump(mode="json") == from_dict.model_dump(mode="json")
    assert request_input_hash(from_frame) == request_input_hash(from_dict)


# ---------------------------------------------------------------------------
# S6.T3: column admission by dtype; null and NaN rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "series",
    [
        pl.Series([True, False, True, False], dtype=pl.Boolean),
        pl.Series([Decimal("1.5")] * 4, dtype=pl.Decimal(10, 2)),
        pl.Series(["7.9"] * 4, dtype=pl.Utf8),
        pl.Series([None] * 4, dtype=pl.Null),
    ],
    ids=["boolean", "decimal", "utf8", "null_dtype"],
)
def test_s6_t3_ead_dtype_not_admitted_raises_type_error(series: pl.Series) -> None:
    with pytest.raises(TypeError, match=r"column 'ead'"):
        CreditVarRequest.from_dataframe(
            _obligor_frame(ead=series), model="independent_bernoulli"
        )


def test_s6_t3_integral_float_uid_is_admitted_as_int() -> None:
    request = CreditVarRequest.from_dataframe(
        _obligor_frame(obligor_uid=pl.Series([11.0, 7.0, 13.0, 14.0])),
        model="independent_bernoulli",
    )
    assert [o.obligor_uid for o in request.obligors] == [11, 7, 13, 14]
    assert all(type(o.obligor_uid) is int for o in request.obligors)


@pytest.mark.parametrize("cell", [7.9, math.nan, math.inf, -math.inf])
def test_s6_t3_non_integral_float_uid_raises_at_obligor_uid(cell: float) -> None:
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            _obligor_frame(obligor_uid=pl.Series([11.0, cell, 13.0, 14.0])),
            model="independent_bernoulli",
        )
    assert _single_error(excinfo)["loc"] == ("obligor_uid",)


@pytest.mark.parametrize("cell", [7.9, math.nan, math.inf])
def test_s6_t3_non_integral_float_loadings_uid_raises_at_obligor_uid(
    cell: float,
) -> None:
    uids = [float(u) for u, _, _ in _LOADINGS]
    uids[2] = cell
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            _obligor_frame(),
            model="mf_gaussian",
            loadings_table=_loadings_frame(obligor_uid=pl.Series(uids)),
            **_RUN,
        )
    assert _single_error(excinfo)["loc"] == ("obligor_uid",)


def test_s6_t3_null_dtype_group_and_rho_mean_absent_for_every_row() -> None:
    absent = pl.Series([None] * 4, dtype=pl.Null)
    request = CreditVarRequest.from_dataframe(
        _obligor_frame(group=absent, rho=absent), model="sf_gaussian", **_RUN
    )
    assert all(o.group is None for o in request.obligors)
    assert all(o.rho is None for o in request.obligors)


def test_s6_t3_null_group_cell_means_no_group_for_that_obligor() -> None:
    """A grouped model refuses an obligor without a group (rule 18) at that
    obligor, which proves that the null cell reached the row as no group."""
    groups = pl.Series(["IG", None, "HY", "IG"], dtype=pl.Utf8)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            _obligor_frame(group=groups),
            model="beta_binomial_grouped",
            group_params=_GROUP_PARAMS["beta_binomial_grouped"],
            **_RUN,
        )
    error = _single_error(excinfo)
    assert error["loc"] == ("obligors", 1, "group")
    assert error["input"] is None


@pytest.mark.parametrize("cell", [None, math.nan], ids=["null", "nan"])
def test_s6_t3_null_or_nan_rho_cell_means_the_fill(cell: float | None) -> None:
    rhos = pl.Series([0.20, cell, 0.35, None], dtype=pl.Float64)
    request = CreditVarRequest.from_dataframe(
        _obligor_frame(rho=rhos), model="sf_gaussian", **_RUN
    )
    assert [o.rho for o in request.obligors] == [0.20, None, 0.35, None]


@pytest.mark.parametrize("field", ["ead", "pd", "lgd"])
@pytest.mark.parametrize("cell", [None, math.nan], ids=["null", "nan"])
def test_s6_t3_null_or_nan_in_a_required_float_raises_at_the_field(
    field: str, cell: float | None
) -> None:
    values = {"ead": _EADS, "pd": _PDS, "lgd": _LGDS}[field][:]
    values_with_fault: list[float | None] = list(values)
    values_with_fault[2] = cell
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            _obligor_frame(**{field: pl.Series(values_with_fault, dtype=pl.Float64)}),
            model="independent_bernoulli",
        )
    assert _single_error(excinfo)["loc"] == (field,)


@pytest.mark.parametrize("cell", [None, math.nan], ids=["null", "nan"])
def test_s6_t3_null_or_nan_loading_raises_at_loading(cell: float | None) -> None:
    values: list[float | None] = [v for _, _, v in _LOADINGS]
    values[3] = cell
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            _obligor_frame(),
            model="mf_gaussian",
            loadings_table=_loadings_frame(loading=pl.Series(values, dtype=pl.Float64)),
            **_RUN,
        )
    assert _single_error(excinfo)["loc"] == ("loading",)


def test_s6_t3_null_int_uid_cell_raises_at_obligor_uid() -> None:
    uids = pl.Series([11, None, 13, 14], dtype=pl.Int64)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            _obligor_frame(obligor_uid=uids), model="independent_bernoulli"
        )
    assert _single_error(excinfo)["loc"] == ("obligor_uid",)


def test_s6_t3_null_factor_cell_raises_at_factor() -> None:
    factors = pl.Series([f for _, f, _ in _LOADINGS], dtype=pl.Utf8)
    factors = factors.scatter(4, None)
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.from_dataframe(
            _obligor_frame(),
            model="mf_gaussian",
            loadings_table=_loadings_frame(factor=factors),
            **_RUN,
        )
    assert _single_error(excinfo)["loc"] == ("factor",)


@pytest.mark.parametrize(
    ("table", "column", "series"),
    [
        ("obligor_table", "obligor_uid", pl.Series([None] * 4, dtype=pl.Null)),
        ("loadings_table", "factor", pl.Series([None] * 6, dtype=pl.Null)),
        ("loadings_table", "loading", pl.Series([None] * 6, dtype=pl.Null)),
        ("obligor_table", "group", pl.Series(_GROUPS, dtype=pl.Categorical)),
        ("loadings_table", "factor", pl.Series([1, 2, 3, 4, 5, 6])),
    ],
    ids=[
        "uid_null_dtype",
        "factor_null_dtype",
        "loading_null_dtype",
        "group_categorical",
        "factor_int64",
    ],
)
def test_s6_t3_column_dtype_not_admitted_names_the_column(
    table: str, column: str, series: pl.Series
) -> None:
    obligors = _obligor_frame(**({column: series} if table == "obligor_table" else {}))
    loadings = _loadings_frame(
        **({column: series} if table == "loadings_table" else {})
    )
    with pytest.raises(TypeError, match=rf"{table} column '{column}'"):
        CreditVarRequest.from_dataframe(
            obligors, model="mf_gaussian", loadings_table=loadings, **_RUN
        )


# ---------------------------------------------------------------------------
# S6.T4: release 0.2.1
# ---------------------------------------------------------------------------


def test_s6_t4_version_is_0_2_1_and_pandas_requirement_is_unchanged() -> None:
    with _PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["version"] == _RELEASE
    pandas_entries = [d for d in project["dependencies"] if d.startswith("pandas")]
    assert "pandas>=2.0" in pandas_entries
    assert importlib.metadata.version("riskquant") == _RELEASE


# ---------------------------------------------------------------------------
# S3 OD-2: the credit_exposure -> credit_var handoff example
# ---------------------------------------------------------------------------


def test_s3_od2_handoff_example_runs_into_credit_var() -> None:
    """The handoff.py docstring example, as written: the obligor table gets its
    ``ead`` column from the exposure run, then feeds `from_dataframe` with the
    request defaults."""
    book = generate_canonical_exposure_book()
    exposure = run_credit_exposure(
        CreditExposureRequest.from_dataframe(
            pl.DataFrame({
                "trade_uid": book.trade_uid,
                "netting_set_uid": book.trade_netting_set_id,
                "mtm": book.mtm,
                "volume": book.volume,
                "current_price": book.current_price,
                "asset_subclass_uix": book.asset_subclass_uix,
            }),
            pl.DataFrame({
                "netting_set_uid": book.netting_set_uid,
                "obligor_uid": book.netting_set_obligor_uid,
                "net_collateral": book.netting_set_collateral,
            }),
            model="current_exposure_method",
            add_on_factors=pl.DataFrame({
                "subclass_uix": book.subclass_uix,
                "factor": book.subclass_factor,
            }),
        )
    )
    uids = sorted({int(u) for u in book.netting_set_obligor_uid})
    credit_obligors = pl.DataFrame({
        "obligor_uid": pl.Series(uids, dtype=pl.Int64),
        "pd": pl.Series([0.02] * len(uids), dtype=pl.Float64),
        "lgd": pl.Series([0.45] * len(uids), dtype=pl.Float64),
    })

    obligors = attach_exposure_ead(credit_obligors, exposure)  # fills `ead`
    result = run_credit_var(
        CreditVarRequest.from_dataframe(obligors, model="sf_gaussian")
    )

    assert isinstance(result, CreditVarResult)
    ead_by_uid = exposure.ead_by_obligor_uid()
    assert {c.obligor_uid for c in result.obligor_contributions} == set(uids)
    assert obligors.get_column("ead").to_list() == [ead_by_uid[u] for u in uids]
    assert "ead" not in credit_obligors.columns  # the caller's frame is unchanged
