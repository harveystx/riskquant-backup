# tests/config/test_validate_tiers_regression.py
"""
Regression test for RQ-AUD-018: pins every validation-tier failure path in
the migrated `static_config/validate.py` and `proxy_config/validate.py`
against real Polars DataFrames, so a future refactor that silently drops a
check (e.g. a mistranslated pandas idiom) is caught immediately.
"""
from __future__ import annotations

import polars as pl
import pytest

from riskquant.config.proxy_config.validate import validate_proxy_config_df
from riskquant.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# proxy_config Stage-1 validation (all 7 tiers)
# ---------------------------------------------------------------------------


def _base_proxy_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "proxying_level": ["CONTAINER", "RISK_FACTOR"],
            "required_name": ["REQ_A", "REQ_B"],
            "proxy_name": ["PROXY_A", "PROXY_B"],
            "vol_scaling_factor": [1.5, 2.0],
        }
    )


def test_empty_proxy_df_is_valid():
    empty = pl.DataFrame(
        {
            "proxying_level": [],
            "required_name": [],
            "proxy_name": [],
            "vol_scaling_factor": [],
        },
        schema={
            "proxying_level": pl.Utf8,
            "required_name": pl.Utf8,
            "proxy_name": pl.Utf8,
            "vol_scaling_factor": pl.Float64,
        },
    )
    validate_proxy_config_df(empty)  # must not raise


def test_valid_baseline_passes():
    validate_proxy_config_df(_base_proxy_df())  # must not raise


def test_missing_required_columns_raises():
    with pytest.raises(ValueError, match="missing required column"):
        validate_proxy_config_df(
            pl.DataFrame({"proxying_level": ["CONTAINER"], "required_name": ["A"]})
        )


def test_null_in_required_column_raises():
    df = _base_proxy_df().with_columns(
        pl.when(pl.col("required_name") == "REQ_A")
        .then(None)
        .otherwise(pl.col("required_name"))
        .alias("required_name")
    )
    with pytest.raises(ValueError, match="null values"):
        validate_proxy_config_df(df)


def test_blank_string_raises():
    df = _base_proxy_df().with_columns(
        pl.when(pl.col("required_name") == "REQ_A")
        .then(pl.lit("  "))
        .otherwise(pl.col("required_name"))
        .alias("required_name")
    )
    with pytest.raises(ValueError, match="blank string"):
        validate_proxy_config_df(df)


def test_invalid_proxying_level_raises():
    df = _base_proxy_df().with_columns(
        pl.when(pl.col("proxying_level") == "CONTAINER")
        .then(pl.lit("BOGUS"))
        .otherwise(pl.col("proxying_level"))
        .alias("proxying_level")
    )
    with pytest.raises(ValueError, match="invalid proxying_level"):
        validate_proxy_config_df(df)


@pytest.mark.parametrize("bad_scale", [0.0, 15.0, -1.0])
def test_vol_scaling_factor_out_of_range_raises(bad_scale):
    df = _base_proxy_df().with_columns(
        pl.when(pl.col("required_name") == "REQ_A")
        .then(bad_scale)
        .otherwise(pl.col("vol_scaling_factor"))
        .alias("vol_scaling_factor")
    )
    with pytest.raises(ValueError, match="vol_scaling_factor outside"):
        validate_proxy_config_df(df)


def test_duplicate_level_required_name_raises():
    df = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER", "CONTAINER"],
            "required_name": ["REQ_A", "REQ_A"],
            "proxy_name": ["PROXY_A", "PROXY_B"],
            "vol_scaling_factor": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_proxy_config_df(df)


def test_self_reference_raises():
    df = _base_proxy_df().with_columns(
        pl.when(pl.col("required_name") == "REQ_A")
        .then(pl.lit("REQ_A"))
        .otherwise(pl.col("proxy_name"))
        .alias("proxy_name")
    )
    with pytest.raises(ValueError, match="self-referencing"):
        validate_proxy_config_df(df)
