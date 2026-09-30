# tests/portfolio/ingest/test_polars_edge_contract.py
"""
Edge contract of the Polars ingest entry point (D094, D095).

Each test pins one behaviour that the post-migration audit found wrong or
unproven. Old-pipeline behaviour was measured on the pre-migration ingest
under pandas 2.2.3 and 3.0.2; both versions agree on every case here.

- A pandas frame is refused with TypeError before any stage runs.
- The caller's frame is not mutated (the pandas entry mutated it by default).
- NaN is missing. Polars keeps NaN distinct from null; pandas ``isna`` did
  not. A NaN quantity or strike must not reach the SoA.
- Text-like dtypes (all-null ``pl.Null``, Categorical) are strings, as
  pandas ``astype("string")`` accepted them.
- Column names that collide after strip + lower raise PrettyKeyError.
- A pandas-era oracle for an option + forward book pins dates, strikes,
  vol ids, discount curves and cp_mult (the SPOT oracle covers none of them).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import NOT_OPTION_MULT
from riskquant.exceptions import PrettyKeyError
from riskquant.ingest_fixtures.config.risk_static_config import (
    canonical_risk_static_config as _canonical,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

_ORACLE = Path(__file__).parent / "_pandas_era_oracle_node_book.json"
_AS_OF = dt.date(2026, 1, 15)
_PILLAR = "BRENT_CRUDE_FWD[M+0]"


@pytest.fixture(scope="module")
def rsc() -> RiskStaticConfig:
    return compile_risk_static_config_from_dict_of_dfs(
        _canonical.CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )


def _node_book(**cols: list[Any]) -> dict[str, list[Any]]:
    """One option row and one forward row; the frame used for the oracle."""
    data: dict[str, list[Any]] = {
        "uid": [1, 2],
        "instrument": ["VANILLA_EUROPEAN_OPTION_NODE", "VANILLA_FORWARD_NODE"],
        "quantity": [1000.0, 1000.0],
        "underlying": [_PILLAR, _PILLAR],
        "settlement_ccy": ["USD", "USD"],
        "strike_price": [70.0, 70.0],
        "strike_ccy": ["USD", "USD"],
        "is_call": [1.0, None],
        "expiry_date": [dt.date(2026, 2, 18), None],
        "settlement_date": [dt.date(2026, 2, 20), dt.date(2026, 2, 17)],
        "delivery_date": [None, dt.date(2026, 2, 15)],
        "volatility_container_name": ["BRENT_CRUDE_ATM_VOL", None],
        "under_discount_curve": ["USD_SOFR_OIS", "USD_SOFR_OIS"],
        "strike_discount_curve": ["USD_SOFR_OIS", "USD_SOFR_OIS"],
    }
    data.update(cols)
    return data


def _ingest(rsc: RiskStaticConfig, frame: pl.DataFrame) -> Any:
    return compile_raw_df_into_ingest_batch(
        raw_df=frame, static_config=rsc, as_of_date=_AS_OF
    )


# ---- D094: entry type ------------------------------------------------------


def test_pandas_frame_is_refused_before_any_stage(rsc: RiskStaticConfig) -> None:
    with pytest.raises(TypeError, match=r"must be a polars\.DataFrame.*pandas"):
        _ingest(rsc, pd.DataFrame(_node_book()))  # type: ignore[arg-type]


def test_lazy_frame_is_refused(rsc: RiskStaticConfig) -> None:
    with pytest.raises(TypeError, match=r"must be a polars\.DataFrame"):
        _ingest(rsc, pl.from_dict(_node_book()).lazy())  # type: ignore[arg-type]


# ---- D095: no mutation -----------------------------------------------------


def test_caller_frame_is_not_mutated(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book(settlement_ccy=[" usd ", "Usd"]))
    snapshot = frame.clone()
    _ingest(rsc, frame)
    assert frame.equals(snapshot)
    assert frame.columns == snapshot.columns
    assert frame.schema == snapshot.schema


# ---- NaN is missing --------------------------------------------------------


@pytest.mark.parametrize("col", ["quantity", "strike_price"])
def test_native_nan_float_is_refused_as_missing(
    rsc: RiskStaticConfig, col: str
) -> None:
    frame = pl.from_dict(_node_book(**{col: [1000.0, float("nan")]}))
    with pytest.raises(ValueError, match=rf"missing '{col}' at rows: \[1\]"):
        _ingest(rsc, frame)


def test_nan_string_quantity_is_refused(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book(quantity=["1000", "nan"]))
    with pytest.raises(ValueError, match=r"missing 'quantity' at rows: \[1\]"):
        _ingest(rsc, frame)


def test_native_nan_flag_on_non_option_row_is_absent(rsc: RiskStaticConfig) -> None:
    batch = _ingest(rsc, pl.from_dict(_node_book(is_call=[1.0, float("nan")])))
    assert int(batch.cp_mult[1]) == int(NOT_OPTION_MULT)


# ---- text-like dtypes ------------------------------------------------------


def test_all_null_optional_string_column_is_accepted(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict({
        k: v[1:] for k, v in _node_book().items() if k not in ("is_call", "expiry_date")
    })
    assert frame["volatility_container_name"].dtype == pl.Null
    batch = _ingest(rsc, frame)
    assert batch.size == 1


def test_categorical_string_column_is_accepted(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book()).with_columns(
        pl.col("settlement_ccy").cast(pl.Categorical)
    )
    batch = _ingest(rsc, frame)
    assert batch.size == 2


def test_numeric_string_column_is_refused(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book(settlement_ccy=[1, 2]))
    with pytest.raises(TypeError, match=r"String column 'settlement_ccy'"):
        _ingest(rsc, frame)


# ---- duplicate names after normalisation ----------------------------------


def test_names_colliding_after_normalisation_raise_pretty_key_error(
    rsc: RiskStaticConfig,
) -> None:
    data = _node_book()
    data[" UID "] = data["uid"]
    with pytest.raises(PrettyKeyError, match=r"Duplicate ingest columns detected"):
        _ingest(rsc, pl.from_dict(data))


# ---- pandas-era oracle: option + forward book ------------------------------


def test_node_book_matches_pandas_era_oracle(rsc: RiskStaticConfig) -> None:
    oracle = json.loads(_ORACLE.read_text())
    batch = _ingest(rsc, pl.from_dict(_node_book()))
    assert batch.size == oracle["size"]
    assert int(batch.as_of_ord32) == oracle["as_of_ord32"]
    for field, want in oracle["fields"].items():
        arr = np.asarray(getattr(batch, field))
        assert str(arr.dtype) == want["dtype"], field
        assert arr.flags.c_contiguous, field
        got = hashlib.sha256(arr.tobytes()).hexdigest()
        assert got == want["sha256"], (field, arr.tolist(), want["values"])


# ---- numeric and date text (pre-migration runtime; pandas 2.2 == 3.0) -------

_ISO = {
    "expiry_date": ["2026-02-18", None],
    "settlement_date": ["2026-02-20", "2026-02-17"],
    "delivery_date": [None, "2026-02-15"],
}


def test_padded_numeric_text_is_a_number(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book(quantity=[" 1000 ", "\t1000\n"]))
    batch = _ingest(rsc, frame)
    assert batch.qty.tolist() == [1000.0, 1000.0]


def test_empty_numeric_text_is_missing(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book(quantity=["1000", ""]))
    with pytest.raises(ValueError, match=r"missing 'quantity' at rows: \[1\]"):
        _ingest(rsc, frame)


def test_whitespace_only_numeric_text_is_refused(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book(quantity=["1000", "   "]))
    with pytest.raises(TypeError, match=r"Failed to materialize float column"):
        _ingest(rsc, frame)


def test_iso_and_unpadded_date_text_match_typed_dates(rsc: RiskStaticConfig) -> None:
    typed = _ingest(rsc, pl.from_dict(_node_book()))
    text = _ingest(rsc, pl.from_dict(_node_book(**_ISO)))
    unpadded = _ingest(
        rsc, pl.from_dict(_node_book(**{**_ISO, "expiry_date": ["2026-2-18", None]}))
    )
    for batch in (text, unpadded):
        for field in ("settle_ord", "delivery_ord", "expiry_ord"):
            assert getattr(batch, field).tolist() == getattr(typed, field).tolist()


@pytest.mark.parametrize(
    "bad", [" 2026-02-18", "2026-02-18 ", "+2026-02-18", "20260218", "2026-02-18T00:00"]
)
def test_date_text_must_be_exactly_iso(rsc: RiskStaticConfig, bad: str) -> None:
    frame = pl.from_dict(_node_book(**{**_ISO, "expiry_date": [bad, None]}))
    with pytest.raises(ValueError, match=r"Dates must be exactly 'YYYY-MM-DD'"):
        _ingest(rsc, frame)


def test_empty_date_text_is_missing(rsc: RiskStaticConfig) -> None:
    frame = pl.from_dict(_node_book(**{**_ISO, "expiry_date": ["", None]}))
    with pytest.raises(ValueError, match=r"missing 'expiry_date' at rows: \[0\]"):
        _ingest(rsc, frame)


# ---- D098: netting_set_id corrections --------------------------------------


@pytest.mark.parametrize("bad", [1.5, float("inf"), 1e19])
def test_netting_set_id_refuses_non_integer_and_out_of_range(
    rsc: RiskStaticConfig, bad: float
) -> None:
    frame = pl.from_dict(_node_book(netting_set_id=[1.0, bad]))
    with pytest.raises(TypeError, match=r"(?i)nullable-int column 'netting_set_id'"):
        _ingest(rsc, frame)
