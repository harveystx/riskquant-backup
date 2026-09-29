# tests/simlab/shocks/conftest.py
"""
Subject: the Proxy V1 simlab shock path over the spot universe.

Fixtures come from `tests/_support`: one definition, imported by every suite
that needs it. Transitive fixture dependencies are imported too, because
pytest resolves a fixture argument by name in the requesting conftest scope.
"""

from __future__ import annotations

from tests._support.spot_scenario import (
    proxy_terms,
    raw_shocks,
    resolved,
)
from tests._support.spot_universe import (
    as_of_date,
    golden_levels,
    level_snapshot,
    portfolio_mapper_market,
    raw_positions,
    rsc,
    static_config_dfs,
)

__all__ = [
    "as_of_date",
    "golden_levels",
    "level_snapshot",
    "portfolio_mapper_market",
    "proxy_terms",
    "raw_positions",
    "raw_shocks",
    "resolved",
    "rsc",
    "static_config_dfs",
]
