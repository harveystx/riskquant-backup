# tests/config/proxy_model/conftest.py
"""
Static configurations for the Proxy V1 compiler tests (Slice 1).

``canonical_static_config`` is the canonical fixture, compiled unchanged.

``extended_static_config`` is a test-owned extension of the canonical tables.
It adds exactly what the canonical fixture cannot express, and nothing else:

* ``TEST_BRENT_WTI_SPREAD_FWD`` (rf 14000-14001): a SPREAD curve whose
  configured shift type is DIFFERENCE, for LOG-target <- DIFF-driver cases.
* ``TEST_ANNUAL_CAL_FWD`` (rf 15000-15003): OFFSET_Y / CALENDAR_PERIOD,
  anchor 0. Its EXACT pairing against ``CO2_EUA_ANNUAL_FWD`` (ROLL_ON_MMDD,
  roll 04-01) changes across the roll date, which exercises as-of sensitivity.
* ``TEST_ANCHOR2_FWD`` (rf 16000-16003): OFFSET_M / CALENDAR_PERIOD with
  anchor 2 against BRENT's anchor 1, so equal tokens denote different
  delivery months.
* ``AAPL_EQ`` reconfigured to ABSOLUTE_BASE_RETURN and ``AAPL_ATM_VOL`` to
  NO_SHIFT, the two native (non-V1) shift types of D025.

The canonical table objects are never mutated; new frames are concatenated.
"""

from __future__ import annotations

import polars as pl
import pytest

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS as CANONICAL_DFS,
)


def build_extended_static_config(
    *, wti_shift_type: str = "LOG_RETURN"
) -> RiskStaticConfig:
    """Canonical tables plus the test-owned curves and native-node overrides."""
    cm = CANONICAL_DFS["curve_meta"]
    cp = CANONICAL_DFS["curve_point_meta"]
    nm = CANONICAL_DFS["node_meta"]

    brent = cm.filter(pl.col("curve_name") == "BRENT_CRUDE_FWD")
    co2 = cm.filter(pl.col("curve_name") == "CO2_EUA_ANNUAL_FWD")
    spread = brent.with_columns(
        pl.lit(250_013_000).alias("curve_id"),
        pl.lit("TEST_BRENT_WTI_SPREAD_FWD").alias("curve_name"),
        pl.lit("SPREAD").alias("quote_type"),
        pl.lit("DIFFERENCE").alias("shift_type"),
    )
    annual = co2.with_columns(
        pl.lit(250_014_000).alias("curve_id"),
        pl.lit("TEST_ANNUAL_CAL_FWD").alias("curve_name"),
        pl.lit("CALENDAR_PERIOD").alias("roll_method"),
        pl.lit(0).cast(cm.schema["anchor_offset"]).alias("anchor_offset"),
        pl.lit(None).cast(cm.schema["roll_month"]).alias("roll_month"),
        pl.lit(None).cast(cm.schema["roll_day"]).alias("roll_day"),
    )
    anchor2 = brent.with_columns(
        pl.lit(250_015_000).alias("curve_id"),
        pl.lit("TEST_ANCHOR2_FWD").alias("curve_name"),
        pl.lit(2).cast(cm.schema["anchor_offset"]).alias("anchor_offset"),
    )
    curves = pl.concat([cm, spread, annual, anchor2], how="vertical_relaxed")
    curves = curves.with_columns(
        pl.when(pl.col("curve_name") == "WTI_CRUDE_FWD")
        .then(pl.lit(wti_shift_type))
        .otherwise(pl.col("shift_type"))
        .alias("shift_type")
    )
    points = pl.DataFrame(
        {
            "rf_id": [
                14000,
                14001,
                15000,
                15001,
                15002,
                15003,
                16000,
                16001,
                16002,
                16003,
            ],
            "curve_id": [250_013_000] * 2 + [250_014_000] * 4 + [250_015_000] * 4,
            "tenor_token": [
                "[M+0]",
                "[M+1]",
                "[Y+0]",
                "[Y+1]",
                "[Y+2]",
                "[Y+3]",
                "[M+0]",
                "[M+1]",
                "[M+2]",
                "[M+3]",
            ],
        }
    ).cast(cp.schema)
    nodes = nm.with_columns(
        pl.when(pl.col("node_name") == "AAPL_EQ")
        .then(pl.lit("ABSOLUTE_BASE_RETURN"))
        .when(pl.col("node_name") == "AAPL_ATM_VOL")
        .then(pl.lit("NO_SHIFT"))
        .otherwise(pl.col("shift_type"))
        .alias("shift_type")
    )
    tables = dict(CANONICAL_DFS)
    tables["curve_meta"] = curves
    tables["curve_point_meta"] = pl.concat([cp, points], how="vertical_relaxed")
    tables["node_meta"] = nodes
    return compile_risk_static_config_from_dict_of_dfs(tables)


@pytest.fixture(scope="session")
def canonical_static_config() -> RiskStaticConfig:
    return compile_risk_static_config_from_dict_of_dfs(dict(CANONICAL_DFS))


@pytest.fixture(scope="session")
def extended_static_config() -> RiskStaticConfig:
    return build_extended_static_config()


@pytest.fixture(scope="session")
def static_config_builder():
    """The builder itself, for tests that vary one configured shift type."""
    return build_extended_static_config
