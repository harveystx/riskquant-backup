# tests/test_proxy_v1_exclusivity.py
"""
D088 Stage E: Proxy V1 is the only proxy architecture in RiskQuant.

Structural checks over the source trees (not behaviour): the D013 packages
and modules do not exist, nothing imports them, and no code names the D013
protocol concepts. Historical records (DECISIONS.md, comments) are text, not
code, and are out of scope for this test.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TREES = ("src", "scripts")

_REMOVED_PATHS = (
    "src/riskquant/proxy_resolver",
    "src/riskquant/config/proxy_config",
    "src/riskquant/ingest_fixtures/config/risk_proxy_config",
    "src/riskquant/simlab/shocks/apply.py",
    "src/riskquant/simlab/shocks/shock_kernel.py",
    "src/riskquant/simlab/builder/shock_universe.py",
    "src/riskquant/simlab/scenarios/world_column_stream.py",
)
_REMOVED_MODULES = (
    "riskquant.proxy_resolver",
    "riskquant.config.proxy_config",
    "riskquant.ingest_fixtures.config.risk_proxy_config",
    "riskquant.simlab.shocks.apply",
    "riskquant.simlab.shocks.shock_kernel",
    "riskquant.simlab.builder.shock_universe",
    "riskquant.simlab.scenarios.world_column_stream",
)
_D013_NAMES = frozenset({
    "shock_scale",
    "source_factor_ids",
    "required_to_shock_uix",
    "ProxyTopology",
    "ProxyResolution",
    "build_shock_universe",
    "ShockUniverse",
    "apply_shocks_to_base_levels",
    "resolve_required_factor_ids_to_market_data_factor_ids",
    "compile_proxy_topology_from_df",
    "compile_proxy_topology_from_xlsx",
    "RiskProxyConfigData",
    "proxying_level",
    "vol_scaling_factor",
})


def _python_files() -> list[pathlib.Path]:
    return sorted(p for t in _TREES for p in (_ROOT / t).rglob("*.py"))


@pytest.mark.parametrize("path", _REMOVED_PATHS)
def test_the_d013_module_does_not_exist(path: str) -> None:
    assert not (_ROOT / path).exists(), path


def test_no_code_imports_a_d013_module() -> None:
    offenders = []
    for f in _python_files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if any(n == m or n.startswith(m + ".") for m in _REMOVED_MODULES):
                    offenders.append(f"{f.relative_to(_ROOT)}: {n}")
    assert offenders == []


def test_no_code_names_a_d013_protocol_concept() -> None:
    offenders = []
    for f in _python_files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            found = (
                (isinstance(node, ast.Name) and node.id)
                or (isinstance(node, ast.Attribute) and node.attr)
                or (isinstance(node, ast.arg) and node.arg)
                or (isinstance(node, ast.keyword) and node.arg)
                or (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name)
                or (isinstance(node, ast.alias) and node.name.split(".")[-1])
            )
            if found in _D013_NAMES:
                offenders.append(f"{f.relative_to(_ROOT)}:{node.lineno}: {found}")
    assert offenders == []
