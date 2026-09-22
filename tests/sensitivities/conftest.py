# tests/sensitivities/conftest.py
"""
Subject: `riskquant.sensitivities` on a compiled SPOT portfolio.

Fixtures come from `tests/_support`: one definition, imported by every suite
that needs it. Transitive fixture dependencies are imported too, because
pytest resolves a fixture argument by name in the requesting conftest scope.
"""

from __future__ import annotations

from tests._support.spot_universe import (
    as_of_date,
    golden_levels,
    level_snapshot,
    portfolio_mapper_market,
    position_uids,
    raw_positions,
    rsc,
    sensitivities,
    static_config_dfs,
)

__all__ = [
    "as_of_date",
    "golden_levels",
    "level_snapshot",
    "portfolio_mapper_market",
    "position_uids",
    "raw_positions",
    "rsc",
    "sensitivities",
    "static_config_dfs",
]
