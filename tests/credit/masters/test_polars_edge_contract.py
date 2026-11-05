# tests/credit/masters/test_polars_edge_contract.py
"""
WS_CREDIT_POLARS S4: the credit masters, `compile_obligor_ead` and the
obligor-table side of `solutions.credit_var` on Polars.

Contract: C.7 (frame type), C.8 (column rules), C.9 (compiled arrays equal to
the v0.2.0 record, C-contiguous and writeable), C.10 (recipe and run frames).

The npz record is made by `_gates/record_s45_goldens.py` on a v0.2.0 worktree.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from riskquant.constants import (
    NO_CHI_NU,
    NO_COUNTRY_ID,
    NO_PARENT_OBLIGOR,
    NO_RATING_ID,
    NO_SECTOR_ID,
)
from riskquant.credit.exposure.obligor_ead.ingest import compile_obligor_ead
from riskquant.credit.masters.factor_universe.ingest import (
    compile_credit_factor_universe,
)
from riskquant.credit.masters.netting_set.ingest import compile_netting_set_master
from riskquant.credit.masters.obligors.ingest import compile_obligor_master
from riskquant.ingest_fixtures.credit.credit_factors.minimal_factor_universe_master import (
    CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DF,
)
from riskquant.ingest_fixtures.credit.jtd.forced_default.minimal_forced_default_jtd import (
    CANONICAL_FORCED_DEFAULT_JTD_EAD_OVERLAY,
)
from riskquant.ingest_fixtures.credit.netting_set.minimal_netting_set_master import (
    CANONICAL_NETTING_SET_MASTER_INGEST_DF,
)
from riskquant.ingest_fixtures.credit.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)
from riskquant.solutions.credit_var import run as credit_var_run
from riskquant.solutions.credit_var.recipes import assemble_inputs
from riskquant.solutions.credit_var.schemas import ObligorRow

_GOLDEN = Path(__file__).resolve().parent / "golden" / "compiled_masters_v0_2_0.npz"

_ENTRIES = {
    "compile_obligor_master": compile_obligor_master,
    "compile_netting_set_master": compile_netting_set_master,
    "compile_credit_factor_universe": compile_credit_factor_universe,
    "compile_obligor_ead": compile_obligor_ead,
}


def _ead_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "uid": list(CANONICAL_FORCED_DEFAULT_JTD_EAD_OVERLAY),
        "ead": list(CANONICAL_FORCED_DEFAULT_JTD_EAD_OVERLAY.values()),
    })


_CANONICAL = {
    "obligor": (compile_obligor_master, CANONICAL_OBLIGOR_CREDIT_INGEST_DF),
    "netting_set": (compile_netting_set_master, CANONICAL_NETTING_SET_MASTER_INGEST_DF),
    "factor_universe": (
        compile_credit_factor_universe,
        CANONICAL_CREDIT_FACTOR_UNIVERSE_INGEST_DF,
    ),
    "obligor_ead": (compile_obligor_ead, _ead_frame()),
}


def _arrays(container: object) -> dict[str, np.ndarray]:
    return {
        f.name: getattr(container, f.name)
        for f in dataclasses.fields(container)  # type: ignore[arg-type]
        if isinstance(getattr(container, f.name), np.ndarray)
    }


def _obligors(**over: Any) -> pl.DataFrame:
    cols: dict[str, Any] = {
        "uid": pl.Series([1, 2], dtype=pl.Int64),
        "pd": pl.Series([0.01, 0.02], dtype=pl.Float64),
        "lgd": pl.Series([0.5, 0.5], dtype=pl.Float64),
    }
    cols.update(over)
    return pl.DataFrame(cols)


# ==========================================================================
# S4.T1: frame type (C.7)
# ==========================================================================
_NOT_POLARS = {
    "pandas": lambda: pd.DataFrame({"x": [1]}),
    "lazy": lambda: pl.DataFrame({"x": [1]}).lazy(),
    "dict": lambda: {"uid": [1], "pd": [0.1], "lgd": [0.5]},
    "none": lambda: None,
}


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
@pytest.mark.parametrize("kind", sorted(_NOT_POLARS))
def test_s4_t1_entry_refuses_a_non_polars_frame(entry: str, kind: str) -> None:
    """The frame has no required column: a TypeError (not a missing-column
    ValueError) proves the type check runs before any column is read."""
    with pytest.raises(TypeError, match=r"pl\.DataFrame"):
        _ENTRIES[entry](_NOT_POLARS[kind]())


# ==========================================================================
# S4.T2: column rules (C.8, F-11)
# ==========================================================================
@pytest.mark.parametrize(
    "values,dtype,expected",
    [
        ([3.0, 4.0], pl.Float64, [3, 4]),
        ([3, None], pl.Int64, [3, int(NO_RATING_ID)]),
        ([3.0, float("nan")], pl.Float64, [3, int(NO_RATING_ID)]),
        ([None, None], pl.Null, [int(NO_RATING_ID)] * 2),
    ],
    ids=["integral_float", "null_cell", "nan_cell", "pl_null_column"],
)
def test_s4_t2_rating_id_admitted_values(
    values: list[Any], dtype: pl.DataType, expected: list[int]
) -> None:
    master = compile_obligor_master(_obligors(rating_id=pl.Series(values, dtype=dtype)))
    assert master.rating_id.dtype == np.int8
    assert master.rating_id.tolist() == expected


def test_s4_t2_absent_optional_columns_are_all_sentinel() -> None:
    master = compile_obligor_master(_obligors())
    assert master.rating_id.tolist() == [int(NO_RATING_ID)] * 2
    assert master.sector_id.tolist() == [int(NO_SECTOR_ID)] * 2
    assert master.country_id.tolist() == [int(NO_COUNTRY_ID)] * 2
    assert master.parent_uid.tolist() == [int(NO_PARENT_OBLIGOR)] * 2


@pytest.mark.parametrize(
    "values,dtype",
    [([3.5, 4.0], pl.Float64), ([200, 4], pl.Int64), ([True, False], pl.Boolean)],
    ids=["fractional", "int8_overflow", "boolean"],
)
def test_s4_t2_rating_id_refused_values(values: list[Any], dtype: pl.DataType) -> None:
    with pytest.raises(TypeError, match="rating_id"):
        compile_obligor_master(_obligors(rating_id=pl.Series(values, dtype=dtype)))


@pytest.mark.parametrize(
    "values,dtype,error",
    [
        ([7.5, 2.0], pl.Float64, TypeError),
        ([1, None], pl.Int64, ValueError),
        ([2**63, 2], pl.UInt64, TypeError),
        ([None, None], pl.Null, TypeError),
    ],
    ids=["fractional", "null", "uint64_above_int64", "pl_null_column"],
)
def test_s4_t2_uid_refused_values(
    values: list[Any], dtype: pl.DataType, error: type[Exception]
) -> None:
    with pytest.raises(error, match="'uid'"):
        compile_obligor_master(_obligors(uid=pl.Series(values, dtype=dtype)))


@pytest.mark.parametrize(
    "values,dtype,error",
    [
        ([float("nan"), 0.02], pl.Float64, ValueError),
        ([None, 0.02], pl.Float64, ValueError),
        ([True, False], pl.Boolean, TypeError),
    ],
    ids=["nan", "null", "boolean"],
)
def test_s4_t2_pd_refused_values(
    values: list[Any], dtype: pl.DataType, error: type[Exception]
) -> None:
    with pytest.raises(error, match="'pd'"):
        compile_obligor_master(_obligors(pd=pl.Series(values, dtype=dtype)))


def _universe(**over: Any) -> pl.DataFrame:
    cols: dict[str, Any] = {
        "factor_id": pl.Series([0, 1], dtype=pl.Int64),
        "factor_name": ["GLOBAL", "CHI_A"],
        "factor_kind": ["SYSTEMIC", "CHI"],
        "nu": pl.Series([None, 5.0], dtype=pl.Float64),
    }
    cols.update(over)
    return pl.DataFrame(cols)


def test_s4_t2_factor_id_int32_overflow_raises() -> None:
    with pytest.raises(TypeError, match="factor_id"):
        compile_credit_factor_universe(
            _universe(factor_id=pl.Series([0, 2**40], dtype=pl.Int64))
        )


def test_s4_t2_null_nu_maps_to_the_sentinel() -> None:
    universe = compile_credit_factor_universe(_universe())
    assert universe.factor_chi_nu.tolist() == [float(NO_CHI_NU), 5.0]


@pytest.mark.parametrize(
    "column,error",
    [
        (pl.Series([0, 1], dtype=pl.Int64), TypeError),
        (pl.Series(["GLOBAL", None], dtype=pl.String), ValueError),
    ],
    ids=["integer_names", "null_name"],
)
def test_s4_t2_factor_name_string_rule(
    column: pl.Series, error: type[Exception]
) -> None:
    with pytest.raises(error, match="factor_name"):
        compile_credit_factor_universe(_universe(factor_name=column))


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["CHI_A", "GLOBAL"])])
def test_s4_t2_factor_names_admit_categorical_and_enum(dtype: pl.DataType) -> None:
    reference = compile_credit_factor_universe(_universe())
    names = pl.Series(["GLOBAL", "CHI_A"], dtype=dtype)
    universe = compile_credit_factor_universe(_universe(factor_name=names))
    assert universe.factor_names.tolist() == reference.factor_names.tolist()
    assert universe.factor_names.dtype == reference.factor_names.dtype


# ==========================================================================
# S4.T3: compiled arrays equal the v0.2.0 record (C.9, F-25)
# ==========================================================================
@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(_GOLDEN, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("reverse", [False, True], ids=["authored", "reversed"])
@pytest.mark.parametrize("name", sorted(_CANONICAL))
def test_s4_t3_compiled_arrays_equal_the_v0_2_0_record(
    golden: dict[str, np.ndarray], name: str, reverse: bool
) -> None:
    entry, frame = _CANONICAL[name]
    arrays = _arrays(entry(frame.reverse() if reverse else frame))
    recorded = {
        k.split("__", 1)[1]: v for k, v in golden.items() if k.startswith(f"{name}__")
    }
    assert set(arrays) == set(recorded), name
    for field, value in arrays.items():
        expected = recorded[field]
        assert value.dtype == expected.dtype, (name, field)
        np.testing.assert_array_equal(value, expected, err_msg=f"{name}.{field}")
        assert value.flags.c_contiguous, (name, field)
        assert value.flags.writeable, (name, field)


def test_s4_t3_caller_frame_is_not_changed() -> None:
    frame = CANONICAL_OBLIGOR_CREDIT_INGEST_DF
    before = frame.clone()
    compile_obligor_master(frame)
    assert frame.equals(before, null_equal=True)


# ==========================================================================
# S4.T4: the obligor table of the recipes (C.10, F-12, F-13, F-26)
# ==========================================================================
def _rows(n: int, rho: dict[int, float] | None = None) -> list[ObligorRow]:
    rho = rho or {}
    return [
        ObligorRow(
            obligor_uid=i,
            ead=1_000.0 + i,
            pd=0.02,
            lgd=0.45,
            rho=rho.get(i),
        )
        for i in range(1, n + 1)
    ]


def test_s4_t4_explicit_rho_wins_and_null_rows_take_the_fill() -> None:
    table = credit_var_run._obligor_frame(_rows(5, {2: 0.05, 4: 0.33}))
    assert table.schema["rho"] == pl.Float64
    inputs = assemble_inputs(
        model_key="sf_gaussian", obligor_table=table, params={"rho": 0.2}
    )
    frame = inputs.bundle.sf_gaussian_copula_df
    assert frame is not None
    rho = np.asarray(frame["factor_loading"].to_numpy(), dtype=np.float64) ** 2
    np.testing.assert_allclose(rho, [0.2, 0.05, 0.2, 0.33, 0.2], rtol=1e-15)


def test_s4_t4_pl_null_group_and_rho_columns_are_accepted() -> None:
    table = pl.DataFrame({
        "uid": [1, 2],
        "ead": [10.0, 20.0],
        "pd": [0.1, 0.2],
        "lgd": [0.5, 0.5],
        "group": pl.Series([None, None], dtype=pl.Null),
        "rho": pl.Series([None, None], dtype=pl.Null),
    })
    forced = assemble_inputs(model_key="forced_default", obligor_table=table)
    assert forced.obligor_ead.uid.tolist() == [1, 2]
    sf = assemble_inputs(
        model_key="sf_gaussian", obligor_table=table, params={"rho": 0.25}
    )
    frame = sf.bundle.sf_gaussian_copula_df
    assert frame is not None
    assert frame["factor_loading"].to_list() == [0.5, 0.5]


@pytest.mark.parametrize(
    "model_key,group_params",
    [
        (
            "beta_binomial_grouped",
            {"A": {"alpha": 2.0, "beta": 8.0}, "B": {"alpha": 1.0, "beta": 9.0}},
        ),
        (
            "stress_regime_bernoulli_grouped",
            {
                "A": {"stress_prob": 0.1, "stress_pd_multiplier": 2.0},
                "B": {"stress_prob": 0.2, "stress_pd_multiplier": 3.0},
            },
        ),
    ],
)
def test_s4_t4_grouped_assemblers_read_a_utf8_group_column(
    model_key: str, group_params: dict[str, dict[str, float]]
) -> None:
    table = pl.DataFrame({
        "uid": [1, 2, 3],
        "ead": [10.0, 20.0, 30.0],
        "pd": [0.1, 0.2, 0.3],
        "lgd": [0.5, 0.5, 0.5],
        "group": pl.Series(["B", "A", "B"], dtype=pl.String),
    })
    inputs = assemble_inputs(
        model_key=model_key, obligor_table=table, group_params=group_params
    )
    names = inputs.masters.factor_universe.factor_names.tolist()
    # one factor per group, in first-seen order (B before A)
    group_factors = [n for n in names if "::" in n]
    assert [n.split("::")[1] for n in group_factors] == ["B", "A"]


def test_s4_t4_grouped_assembler_refuses_a_null_group() -> None:
    table = pl.DataFrame({
        "uid": [1, 2],
        "ead": [10.0, 20.0],
        "pd": [0.1, 0.2],
        "lgd": [0.5, 0.5],
        "group": pl.Series(["A", None], dtype=pl.String),
    })
    with pytest.raises(ValueError, match=r"obligor\(s\) \[2\] have no group"):
        assemble_inputs(
            model_key="beta_binomial_grouped",
            obligor_table=table,
            group_params={"A": {"alpha": 2.0, "beta": 8.0}},
        )


def test_s4_t4_assemble_inputs_refuses_a_pandas_table() -> None:
    with pytest.raises(TypeError, match="obligor_table"):
        assemble_inputs(
            model_key="forced_default",
            obligor_table=pd.DataFrame({"uid": [1]}),  # type: ignore[arg-type]
        )


# ==========================================================================
# S4.T5: run._obligor_frame has a stated schema (C.10, F-23)
# ==========================================================================
def test_s4_t5_obligor_frame_schema_does_not_depend_on_the_first_rows() -> None:
    rows = _rows(150, {i: 0.01 * (i % 50) for i in range(121, 151)})
    table = credit_var_run._obligor_frame(rows)
    assert dict(table.schema) == {
        "uid": pl.Int64,
        "ead": pl.Float64,
        "pd": pl.Float64,
        "lgd": pl.Float64,
        "group": pl.String,
        "rho": pl.Float64,
    }
    assert table["uid"].to_list() == [r.obligor_uid for r in rows]
    assert table["ead"].to_list() == [r.ead for r in rows]
    assert table["rho"].to_list() == [r.rho for r in rows]
    assert table["group"].null_count() == 150
