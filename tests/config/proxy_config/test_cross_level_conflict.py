# tests/config/proxy_config/test_cross_level_conflict.py
"""
Regression test for `validate.py::_check_cross_level_conflict`
(RQ-AUD-027 remediation) -- the one Stage-2 validation tier not covered
by the existing FK/self-reference/chain tests in
`test_full_pipeline_equivalence.py`.
"""
from __future__ import annotations

import polars as pl
import pytest

from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
)


@pytest.fixture(scope="module")
def real_static_config():
    dict_of_pl = {
        name: pl.DataFrame(table_data)
        for name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }
    return compile_risk_static_config_from_dict_of_dfs(dict_of_pl)


def test_curve_container_proxy_and_its_own_point_conflict(real_static_config):
    """BRENT_CRUDE_FWD proxied at container level AND one of its own
    curve points proxied at rf level -- must raise."""
    conflict_df = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER", "RISK_FACTOR"],
            "required_name": ["BRENT_CRUDE_FWD", "BRENT_CRUDE_FWD[M+0]"],
            "proxy_name": ["WTI_CRUDE_FWD", "WTI_CRUDE_FWD[M+0]"],
            "vol_scaling_factor": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="cross-level conflicts"):
        compile_proxy_topology_from_df(df=conflict_df, static_config=real_static_config)


def test_surface_container_proxy_and_its_own_point_conflict(real_static_config):
    """Same conflict, but for a SURFACE container rather than a CURVE --
    confirms both branches of _check_cross_level_conflict's topology-type
    dispatch are exercised, not just the curve path."""
    conflict_df = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER", "RISK_FACTOR"],
            "required_name": ["USDEUR_VOL_SURF", "USDEUR_VOL_SURF[1M,0.25]"],
            "proxy_name": ["AAPL_VOL_SURF", "AAPL_VOL_SURF[1M,0.25]"],
            "vol_scaling_factor": [1.0, 1.0],
        }
    )
    # The surface point token grammar may not literally accept this exact
    # relative name; fall back to a real point if the literal one above
    # doesn't parse. Resolve the actual point name from static config.
    surf_id = real_static_config.get_container_id("USDEUR_VOL_SURF")
    real_point = next(
        p
        for p in real_static_config.surface_point_meta_dict.values()
        if p.surface_id == surf_id
    )
    proxy_surf_id = real_static_config.get_container_id("AAPL_VOL_SURF")
    proxy_point = next(
        p
        for p in real_static_config.surface_point_meta_dict.values()
        if p.surface_id == proxy_surf_id
    )
    # x_tenor_token is stored WITH its own brackets already (e.g. "[1W]");
    # the relative name is "<surface><x_tenor_token>[<y_label>]" -- verified
    # empirically against static_config.rel_name_to_rf_id.
    real_point_rel_name = f"USDEUR_VOL_SURF{real_point.x_tenor_token}[{real_point.y_label}]"
    proxy_point_rel_name = f"AAPL_VOL_SURF{proxy_point.x_tenor_token}[{proxy_point.y_label}]"

    conflict_df2 = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER", "RISK_FACTOR"],
            "required_name": ["USDEUR_VOL_SURF", real_point_rel_name],
            "proxy_name": ["AAPL_VOL_SURF", proxy_point_rel_name],
            "vol_scaling_factor": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="cross-level conflicts"):
        compile_proxy_topology_from_df(df=conflict_df2, static_config=real_static_config)


def test_container_proxy_alone_does_not_conflict(real_static_config):
    """A container-level-only proxy set (no rf-level proxies at all) must
    not trigger the cross-level check -- it short-circuits when either
    side is empty."""
    container_only = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER"],
            "required_name": ["WTI_CRUDE_FWD"],
            "proxy_name": ["BRENT_CRUDE_FWD"],
            "vol_scaling_factor": [1.0],
        }
    )
    compile_proxy_topology_from_df(
        df=container_only, static_config=real_static_config
    )  # must not raise


def test_unrelated_container_and_rf_proxies_do_not_conflict(real_static_config):
    """A container-level proxy on one curve and an rf-level proxy on a
    DIFFERENT curve's point -- no shared parent, must not raise."""
    unrelated = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER", "RISK_FACTOR"],
            "required_name": ["WTI_CRUDE_FWD", "USDGBP"],
            "proxy_name": ["BRENT_CRUDE_FWD", "USDEUR"],
            "vol_scaling_factor": [1.0, 1.0],
        }
    )
    compile_proxy_topology_from_df(
        df=unrelated, static_config=real_static_config
    )  # must not raise
