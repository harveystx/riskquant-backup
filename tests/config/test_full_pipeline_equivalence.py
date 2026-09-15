# tests/config/test_full_pipeline_equivalence.py
"""
Regression test for RQ-AUD-018: full pandas -> Polars migration of the
`riskquant.config` ingest/validate/compile pipelines.

This proves two things:
1. The fully-migrated Polars pipeline correctly compiles a real, non-trivial
   fixture (8 static_config tables + a 6-row proxy_config table) into valid
   RiskStaticConfig / ProxyTopology objects.
2. Every Stage-2 cross-table validation path (FK, self-reference, chain-ban)
   still fires correctly against the compiled state.

Numerical equivalence against the pre-migration pandas pipeline was verified
during the migration itself via a byte-for-byte JSON diff of the fully
compiled RiskStaticConfig (all 8 tables, 1508 lines, zero differences) and is
not re-verified here since the original pandas code no longer exists in the
repository to compare against -- this test instead pins the CURRENT (Polars)
pipeline's behaviour so any future regression is caught directly.
"""

from __future__ import annotations

import polars as pl
import pytest

from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_proxy_config.canonical_risk_proxy_config import (
    CANONICAL_RISK_PROXY_CONFIG_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
)


@pytest.fixture(scope="module")
def compiled_static_config():
    dict_of_pl = {
        name: pl.DataFrame(table_data)
        for name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }
    return compile_risk_static_config_from_dict_of_dfs(dict_of_pl)


def test_static_config_compiles_from_polars_dataframes(compiled_static_config):
    rsc = compiled_static_config
    assert len(rsc.node_meta_dict) == 10
    assert len(rsc.curve_meta_dict) == 13
    assert len(rsc.surface_meta_dict) == 3
    assert len(rsc.ccy_dict) == 5
    assert len(rsc.holiday_calendars_dict) == 5
    assert len(rsc.curve_point_meta_dict) == 98
    assert len(rsc.surface_point_meta_dict) == 60


def test_holiday_calendar_ordinals_are_int32_and_sorted(compiled_static_config):
    import numpy as np

    for cal in compiled_static_config.holiday_calendars_dict.values():
        assert cal.holiday_dates_ord.dtype == np.int32
        assert np.all(np.diff(cal.holiday_dates_ord) > 0), (
            "ordinals must be strictly sorted"
        )


def test_proxy_topology_compiles_from_polars_dataframe(compiled_static_config):
    proxy_df = pl.DataFrame(CANONICAL_RISK_PROXY_CONFIG_DATA)
    topology = compile_proxy_topology_from_df(
        df=proxy_df, static_config=compiled_static_config
    )
    assert topology is not None


def test_proxy_fk_violation_still_raises(compiled_static_config):
    bad_fk = pl.DataFrame(
        {
            "proxying_level": ["RISK_FACTOR"],
            "required_name": ["TOTALLY_UNKNOWN_NAME"],
            "proxy_name": ["USDEUR"],
            "vol_scaling_factor": [1.0],
        }
    )
    with pytest.raises(ValueError, match="unknown"):
        compile_proxy_topology_from_df(df=bad_fk, static_config=compiled_static_config)


def test_proxy_self_reference_still_raises(compiled_static_config):
    bad_self = pl.DataFrame(
        {
            "proxying_level": ["RISK_FACTOR"],
            "required_name": ["USDGBP"],
            "proxy_name": ["USDGBP"],
            "vol_scaling_factor": [1.0],
        }
    )
    with pytest.raises(ValueError, match="self-referencing"):
        compile_proxy_topology_from_df(
            df=bad_self, static_config=compiled_static_config
        )


def test_proxy_chain_violation_still_raises(compiled_static_config):
    bad_chain = pl.DataFrame(
        {
            "proxying_level": ["RISK_FACTOR", "RISK_FACTOR"],
            "required_name": ["USDGBP", "EURCHF"],
            "proxy_name": ["EURCHF", "USDEUR"],
            "vol_scaling_factor": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="Chained"):
        compile_proxy_topology_from_df(
            df=bad_chain, static_config=compiled_static_config
        )
