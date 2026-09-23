# _context_gen/tests/test_closure.py
from __future__ import annotations

from _context_gen.ast_model import ModuleInfo
from _context_gen.closure import compute_closure
from _context_gen.depgraph import RepoGraph

ROOT = "riskquant"


def _mk(dotted: str) -> ModuleInfo:
    return ModuleInfo(dotted=dotted, repo_relative=f"src/{dotted.replace('.', '/')}.py", docstring=None)


def _graph(edges: dict[str, frozenset[str]], costs: dict[str, int]) -> RepoGraph:
    modules = {d: _mk(d) for d in set(edges) | {t for ts in edges.values() for t in ts}}
    importer_domains: dict[str, set[str]] = {}
    for src, targets in edges.items():
        for t in targets:
            importer_domains.setdefault(t, set()).add(src.split(".")[1])
    return RepoGraph(modules=modules, token_costs=costs, edges=edges, importer_domains=importer_domains)


def test_closure_resolves_to_core_and_stops_expanding() -> None:
    edges = {
        "riskquant.valuation.engine": frozenset({"riskquant.jit", "riskquant.valuation.blocks"}),
        "riskquant.valuation.blocks": frozenset(),
        "riskquant.jit": frozenset({"riskquant.something_jit_would_never_actually_import"}),
    }
    costs = {"riskquant.jit": 100}
    graph = _graph(edges, costs)
    result = compute_closure(
        domain_own=frozenset({"riskquant.valuation.engine", "riskquant.valuation.blocks"}),
        graph=graph,
        core_names=frozenset({"riskquant.jit"}),
        root_package=ROOT,
    )
    assert "riskquant.jit" in result.tier1_core
    # jit's own (hypothetical) further import must NOT be expanded into --
    # core terminates the BFS at that node.
    assert "riskquant.something_jit_would_never_actually_import" not in result.tier1_core
    assert "riskquant.something_jit_would_never_actually_import" not in result.tier2_declared
    assert "riskquant.something_jit_would_never_actually_import" not in result.tier3_cross_reference


def test_small_foreign_dependency_becomes_tier2() -> None:
    edges = {
        "riskquant.simlab.runner": frozenset({"riskquant.config.static_config"}),
    }
    costs = {"riskquant.config.static_config": 5_000}
    graph = _graph(edges, costs)
    result = compute_closure(
        domain_own=frozenset({"riskquant.simlab.runner"}),
        graph=graph,
        core_names=frozenset(),
        root_package=ROOT,
    )
    assert result.tier2_declared == frozenset({"riskquant.config.static_config"})
    assert result.tier3_cross_reference == frozenset()


def test_large_foreign_package_becomes_tier3_not_embedded() -> None:
    edges = {
        "riskquant.sensitivity_pnl.engine": frozenset(
            {f"riskquant.credit.module_{i}" for i in range(10)}
        ),
    }
    costs = {f"riskquant.credit.module_{i}": 5_000 for i in range(10)}  # 50,000 tok, 10 files -- too big
    graph = _graph(edges, costs)
    result = compute_closure(
        domain_own=frozenset({"riskquant.sensitivity_pnl.engine"}),
        graph=graph,
        core_names=frozenset(),
        root_package=ROOT,
    )
    assert result.tier2_declared == frozenset()
    assert len(result.tier3_cross_reference) == 10
    assert "credit" in result.foreign_domains_touched


def test_unresolved_target_reported_not_silently_dropped() -> None:
    edges = {
        "riskquant.dependence_stress.engine": frozenset({"riskquant.observations.compile"}),
    }
    graph = _graph(edges, costs={})
    # riskquant.observations.compile is NOT in graph.modules (simulating a
    # renamed/missing/not-yet-indexed file) -- this must surface, not vanish.
    graph.modules.pop("riskquant.observations.compile", None)
    result = compute_closure(
        domain_own=frozenset({"riskquant.dependence_stress.engine"}),
        graph=graph,
        core_names=frozenset(),
        root_package=ROOT,
    )
    assert "riskquant.observations.compile" in result.unresolved_external


def test_declared_core_member_missing_from_snapshot_is_unresolved_not_silent() -> None:
    # Regression test for a real bug caught via failure-injection testing
    # against the reconstructed RiskQuant repository: a core module name
    # can be declared without actually being present in this run's
    # snapshot (renamed, deleted, unreadable). That must surface as a
    # completeness failure, never be silently treated as "resolved."
    edges = {"riskquant.valuation.engine": frozenset({"riskquant.jit"})}
    graph = _graph(edges, costs={})
    graph.modules.pop("riskquant.jit", None)  # simulate jit.py missing from this snapshot
    result = compute_closure(
        domain_own=frozenset({"riskquant.valuation.engine"}),
        graph=graph,
        core_names=frozenset({"riskquant.jit"}),
        root_package=ROOT,
    )
    assert "riskquant.jit" in result.unresolved_external
    assert "riskquant.jit" not in result.tier1_core
    # Regression test encoding the real finding: these two packages have
    # zero import edges between them. A synthetic graph mirroring that
    # shape must produce an empty closure in both directions.
    edges = {
        "riskquant.sensitivities.engine": frozenset({"riskquant.jit"}),
        "riskquant.sensitivity_pnl.engine": frozenset({"riskquant.jit"}),
    }
    costs = {"riskquant.jit": 100}
    graph = _graph(edges, costs)
    result_a = compute_closure(
        domain_own=frozenset({"riskquant.sensitivities.engine"}),
        graph=graph,
        core_names=frozenset({"riskquant.jit"}),
        root_package=ROOT,
    )
    result_b = compute_closure(
        domain_own=frozenset({"riskquant.sensitivity_pnl.engine"}),
        graph=graph,
        core_names=frozenset({"riskquant.jit"}),
        root_package=ROOT,
    )
    assert "sensitivity_pnl" not in result_a.foreign_domains_touched
    assert "sensitivities" not in result_b.foreign_domains_touched
