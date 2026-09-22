# tests/simlab/scenarios/conftest.py
"""
Subject: the historical generator, whose frames resolve factor ids through
the spot-universe static config.

Fixtures come from `tests/_support`: one definition, imported by every suite
that needs it. Transitive fixture dependencies are imported too, because
pytest resolves a fixture argument by name in the requesting conftest scope.
"""

from __future__ import annotations

from tests._support.spot_universe import (
    rsc,
    static_config_dfs,
)

__all__ = [
    "rsc",
    "static_config_dfs",
]
