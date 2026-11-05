# tests/credit/jtd/test_polars_edge_contract.py
"""
WS_CREDIT_POLARS S5: the JTD ingest bundle, the structural and semantic
validators, the shared column readers and the builders on Polars.

Contract: C.7 (frame type and JTD frame admission), C.9 (compiled arrays), and
the S5 test table. The records are made by `_gates/record_s45_goldens.py` on a
v0.2.0 worktree:

* golden/jtd_ingest_v0_2_0.npz: S5.T2 arrays of the mixed canonical fixture.
* golden/jtd_ingest_errors_v0_2_0.json: S5.T3 case inputs, exception types and
  messages. Each case is rebuilt here as Polars frames from the same plain
  data.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
from riskquant.credit.jtd.ingest.validate import (
    validate_ingest_bundle,
    validate_ingest_bundle_semantics,
)
from riskquant.credit.masters.cross_master.compiled import CompiledCreditMasters
from riskquant.credit.masters.obligors.ingest import compile_obligor_master
from riskquant.ingest_fixtures.credit.credit_factors.minimal_factor_universe_master import (
    build_canonical_credit_factor_universe,
)
from riskquant.ingest_fixtures.credit.jtd.mixed.minimal_mixed import (
    CANONICAL_MIXED_JTD_INGEST_DF,
    CANONICAL_MIXED_JTD_LOADINGS_DF,
    build_canonical_mixed_jtd_bundle,
)
from riskquant.ingest_fixtures.credit.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_ARRAYS = _GOLDEN_DIR / "jtd_ingest_v0_2_0.npz"
_ERRORS = _GOLDEN_DIR / "jtd_ingest_errors_v0_2_0.json"

_SLOTS = tuple(f.name for f in dataclasses.fields(JtdIngestBundle))


def _error_cases() -> list[dict[str, Any]]:
    with _ERRORS.open(encoding="utf-8") as handle:
        return list(json.load(handle)["cases"])


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(_ARRAYS, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _mixed_portfolio(golden: dict[str, np.ndarray]) -> CompiledJtdPortfolio:
    master = compile_obligor_master(
        pl.DataFrame({
            "uid": golden["master__uid"],
            "pd": golden["master__pd"],
            "lgd": golden["master__lgd"],
        })
    )
    return compile_jtd_portfolio(
        bundle=build_canonical_mixed_jtd_bundle(),
        masters=CompiledCreditMasters(
            obligor_master=master,
            factor_universe=build_canonical_credit_factor_universe(),
        ),
    )


def _block_arrays(portfolio: CompiledJtdPortfolio) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, block in enumerate(portfolio.blocks):
        for f in dataclasses.fields(block):
            value = getattr(block, f.name)
            if isinstance(value, np.ndarray):
                out[f"block{i}__{f.name}"] = value
    return out


# ==========================================================================
# S5.T1: frame type (C.7, R-11)
# ==========================================================================
_NOT_POLARS = {
    "pandas": lambda: pd.DataFrame({"uid": [1], "instrument": ["FORCED_DEFAULT"]}),
    "lazy": lambda: pl.DataFrame({"uid": [1], "instrument": ["FORCED_DEFAULT"]}).lazy(),
    "dict": lambda: {"uid": [1], "instrument": ["FORCED_DEFAULT"]},
}
_OK_FRAME = pl.DataFrame({"uid": [1], "instrument": ["FORCED_DEFAULT"]})


@pytest.mark.parametrize("kind", sorted(_NOT_POLARS))
def test_s5_t1_from_instrument_frame_refuses_a_non_polars_frame(kind: str) -> None:
    with pytest.raises(TypeError, match=r"frame must be a pl\.DataFrame"):
        JtdIngestBundle.from_instrument_frame(_NOT_POLARS[kind]())  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", sorted(_NOT_POLARS))
def test_s5_t1_from_instrument_frame_refuses_non_polars_loadings(kind: str) -> None:
    with pytest.raises(TypeError, match=r"mf_loadings_df must be a pl\.DataFrame"):
        JtdIngestBundle.from_instrument_frame(
            _OK_FRAME,
            mf_loadings_df=_NOT_POLARS[kind](),  # type: ignore[arg-type]
        )


def test_s5_t1_from_instrument_frame_refuses_none() -> None:
    with pytest.raises(TypeError, match="got None"):
        JtdIngestBundle.from_instrument_frame(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("slot", _SLOTS)
def test_s5_t1_each_bundle_slot_refuses_a_pandas_frame(slot: str) -> None:
    with pytest.raises(TypeError, match=f"JtdIngestBundle.{slot} must be"):
        JtdIngestBundle(**{slot: pd.DataFrame({"uid": [1]})})


def test_s5_t1_from_instrument_frame_keeps_row_order_and_the_caller_frame() -> None:
    before = CANONICAL_MIXED_JTD_INGEST_DF.clone()
    bundle = build_canonical_mixed_jtd_bundle()
    assert CANONICAL_MIXED_JTD_INGEST_DF.equals(before, null_equal=True)
    expected = CANONICAL_MIXED_JTD_INGEST_DF.filter(
        pl.col("instrument") == "COPULA_SF_GAUSSIAN"
    ).drop("instrument")
    frame = bundle.sf_gaussian_copula_df
    assert frame is not None
    assert frame.equals(expected, null_equal=True)
    assert "instrument" not in frame.columns
    assert bundle.mf_loadings_df is CANONICAL_MIXED_JTD_LOADINGS_DF


# ==========================================================================
# S5.T2: mixed fixture arrays equal the v0.2.0 record
# ==========================================================================
def test_s5_t2_mixed_fixture_arrays_equal_the_v0_2_0_record(
    golden: dict[str, np.ndarray],
) -> None:
    portfolio = _mixed_portfolio(golden)
    order = [type(b).__name__ for b in portfolio.blocks]
    assert order == golden["block_order"].tolist()
    uids = np.asarray(portfolio.obligor_uids)
    assert uids.dtype == golden["obligor_uids"].dtype
    np.testing.assert_array_equal(uids, golden["obligor_uids"])

    arrays = _block_arrays(portfolio)
    recorded = {k: v for k, v in golden.items() if k.startswith("block") and "__" in k}
    assert set(arrays) == set(recorded)
    for key, value in arrays.items():
        assert value.dtype == recorded[key].dtype, key
        np.testing.assert_array_equal(value, recorded[key], err_msg=key)
    # The packed loadings and the residual weight of the multi-factor block.
    for field in ("loading_indptr", "loading_values", "idio_weight"):
        assert any(k.endswith(f"__{field}") for k in arrays), field


# ==========================================================================
# S5.T3: each validator check with one fault (F-24, B-8)
# ==========================================================================
def _polars_bundle(case: dict[str, Any]) -> JtdIngestBundle:
    slots = {k: pl.DataFrame(v) for k, v in case["slots"].items()}
    loadings = None if case["loadings"] is None else pl.DataFrame(case["loadings"])
    return JtdIngestBundle(mf_loadings_df=loadings, **slots)


def _run_stage(stage: str, bundle: JtdIngestBundle) -> None:
    if stage == "structural":
        validate_ingest_bundle(
            bundle, compile_obligor_master(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)
        )
    else:
        validate_ingest_bundle_semantics(
            bundle, build_canonical_credit_factor_universe()
        )


@pytest.mark.parametrize("case", _error_cases(), ids=lambda c: c["id"])
def test_s5_t3_check_raises_the_v0_2_0_type_and_message(case: dict[str, Any]) -> None:
    with pytest.raises(Exception) as excinfo:
        _run_stage(case["stage"], _polars_bundle(case))
    assert type(excinfo.value).__name__ == case["exception_type"]
    message = str(excinfo.value)
    if case["compare"] == "prefix":
        assert message.split(", got ")[0] == case["message"].split(", got ")[0]
    else:
        assert message == case["message"]


def test_s5_t3_record_holds_nan_cells_as_float_nan() -> None:
    """The NaN cells survive the JSON record, so the NaN cases test NaN."""
    case = next(c for c in _error_cases() if c["id"] == "S04_nan_required_float")
    assert math.isnan(
        case["slots"]["stress_regime_bernoulli_df"]["stress_pd_multiplier"][0]
    )
    frame = _polars_bundle(case).stress_regime_bernoulli_df
    assert frame is not None
    assert frame["stress_pd_multiplier"].dtype == pl.Float64
    assert frame["stress_pd_multiplier"].null_count() == 0


def _mf_bundle(loadings: pl.DataFrame) -> JtdIngestBundle:
    return JtdIngestBundle(
        mf_gaussian_copula_df=pl.DataFrame({
            "uid": [1],
            "idiosyncratic_factor": ["IDIO_01"],
        }),
        mf_loadings_df=loadings,
    )


def test_s5_t3_boolean_factor_loading_is_not_numeric() -> None:
    bundle = _mf_bundle(
        pl.DataFrame({
            "obligor_uid": [1],
            "factor_name": ["GLOBAL_CREDIT"],
            "factor_loading": [True],
        })
    )
    with pytest.raises(
        ValueError, match="'factor_loading' must be numeric, got Boolean"
    ):
        _run_stage("structural", bundle)


def test_s5_t3_null_factor_loading_reads_as_nan() -> None:
    bundle = _mf_bundle(
        pl.DataFrame({
            "obligor_uid": [1, 1],
            "factor_name": ["GLOBAL_CREDIT", "SECTOR_ENERGY"],
            "factor_loading": pl.Series([0.3, None], dtype=pl.Float64),
        })
    )
    with pytest.raises(ValueError, match="contains NaN/inf"):
        _run_stage("structural", bundle)


def test_s5_t3_int64_factor_name_column_raises() -> None:
    bundle = JtdIngestBundle(
        independent_bernoulli_df=pl.DataFrame({
            "uid": [1],
            "uniform_factor": pl.Series([8], dtype=pl.Int64),
        })
    )
    with pytest.raises(ValueError, match="must be string-like, got Int64"):
        _run_stage("structural", bundle)


def test_s5_t3_all_null_required_column_raises_type_error() -> None:
    """Owner decision OD-1 of S3: a pl.Null required column is a TypeError."""
    bundle = JtdIngestBundle(
        independent_bernoulli_df=pl.DataFrame({
            "uid": [1],
            "uniform_factor": pl.Series([None], dtype=pl.Null),
        })
    )
    with pytest.raises(TypeError, match="all null"):
        _run_stage("structural", bundle)


def test_s5_t3_uint64_uid_above_int64_raises_type_error() -> None:
    """A uint64 uid >= 2**63 would wrap in the int64 uid lists (S1 review A-3)."""
    bundle = JtdIngestBundle(
        independent_bernoulli_df=pl.DataFrame({
            "uid": pl.Series([2**63], dtype=pl.UInt64),
            "uniform_factor": ["U_01"],
        })
    )
    with pytest.raises(TypeError, match="above the int64 range"):
        _run_stage("structural", bundle)


def _string_typed(frame: pl.DataFrame, dtype: pl.DataType) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(c).cast(pl.String).cast(dtype)
        for c, t in frame.schema.items()
        if t == pl.String and c != "instrument"
    )


@pytest.mark.parametrize("kind", ["categorical", "enum"])
def test_s5_t3_categorical_and_enum_factor_names_compile_equal(
    golden: dict[str, np.ndarray], kind: str
) -> None:
    names = sorted(build_canonical_credit_factor_universe().factor_name_to_uix)
    dtype: pl.DataType = pl.Categorical() if kind == "categorical" else pl.Enum(names)
    reference = _block_arrays(_mixed_portfolio(golden))
    bundle = JtdIngestBundle.from_instrument_frame(
        _string_typed(CANONICAL_MIXED_JTD_INGEST_DF, dtype),
        mf_loadings_df=_string_typed(CANONICAL_MIXED_JTD_LOADINGS_DF, dtype),
    )
    master = compile_obligor_master(
        pl.DataFrame({
            "uid": golden["master__uid"],
            "pd": golden["master__pd"],
            "lgd": golden["master__lgd"],
        })
    )
    portfolio = compile_jtd_portfolio(
        bundle=bundle,
        masters=CompiledCreditMasters(
            obligor_master=master,
            factor_universe=build_canonical_credit_factor_universe(),
        ),
    )
    arrays = _block_arrays(portfolio)
    assert set(arrays) == set(reference)
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, reference[key], err_msg=key)


# ==========================================================================
# S5.T4: every compiled block array is C-contiguous and writeable (F-25)
# ==========================================================================
def test_s5_t4_every_block_array_is_c_contiguous_and_writeable(
    golden: dict[str, np.ndarray],
) -> None:
    arrays = _block_arrays(_mixed_portfolio(golden))
    assert arrays
    bad = [
        key
        for key, value in arrays.items()
        if not (value.flags.c_contiguous and value.flags.writeable)
    ]
    assert not bad, bad
