# _context_gen/closure.py
"""
Production dependency closure.

Recursively resolves a domain's internal imports (per the implementation
contract, §5), but expansion terminates at two boundaries so the closure
cannot silently absorb the whole repository:

  1. reaching a CORE module (core is substrate by definition and is not
     further expanded);
  2. a foreign domain's touched footprint exceeding the Tier-2 budget, at
     which point that whole foreign domain is classified Tier 3
     (cross-reference only, never embedded, never expanded further).

A bounded BFS depth is kept as a defensive limit only; real measurement on
the RiskQuant repository (see the implementation report) shows the graph
terminates in 1-2 hops in every case tested, so the limit is not expected
to bind in practice — it exists so a future, unrelated change to the
codebase cannot silently turn this into an unbounded crawl.
"""

from __future__ import annotations

from dataclasses import dataclass

from _context_gen.depgraph import RepoGraph, domain_of

MAX_BFS_DEPTH = 6
# Token cost, not file count, is the primary Tier-2/Tier-3 gate: a small
# foreign package that happens to span several files (e.g. `observations`,
# 6 files / ~9K tokens, discovered by real recursive closure against the
# RiskQuant repository -- see the implementation report) is still cheap
# enough to embed directly. File count is kept only as a coarse sanity
# ceiling against a foreign domain trickling in one file at a time forever.
TIER2_MAX_FILES_PER_FOREIGN_DOMAIN = 10
TIER2_MAX_TOKENS_PER_FOREIGN_DOMAIN = 20_000


@dataclass(frozen=True, slots=True, kw_only=True)
class ClosureResult:
    domain_own: frozenset[str]
    tier1_core: frozenset[str]
    tier2_declared: frozenset[str]
    tier3_cross_reference: frozenset[str]  # not embedded — index-only pointer
    unresolved_external: frozenset[str]  # internal-looking target with no known source — completeness risk
    foreign_domains_touched: frozenset[str]


def compute_closure(
    *,
    domain_own: frozenset[str],
    graph: RepoGraph,
    core_names: frozenset[str],
    root_package: str,
    tier2_max_files: int = TIER2_MAX_FILES_PER_FOREIGN_DOMAIN,
    tier2_max_tokens: int = TIER2_MAX_TOKENS_PER_FOREIGN_DOMAIN,
) -> ClosureResult:
    visited: set[str] = set(domain_own)
    frontier: set[str] = set(domain_own)
    core_touched: set[str] = set()
    foreign_touched: dict[str, set[str]] = {}
    unresolved: set[str] = set()

    depth = 0
    while frontier and depth < MAX_BFS_DEPTH:
        next_frontier: set[str] = set()
        for m in frontier:
            for target in graph.edges.get(m, frozenset()):
                if target in visited:
                    continue
                if target in core_names:
                    if target not in graph.modules:
                        # A declared core member has no actual source in
                        # this snapshot (e.g. deleted/renamed/unreadable).
                        # This is worse than an ordinary missing dependency
                        # -- core is claimed to always be available -- so
                        # it is surfaced the same way: never silently
                        # treated as "resolved."
                        unresolved.add(target)
                        visited.add(target)
                        continue
                    core_touched.add(target)
                    visited.add(target)
                    continue  # core terminates expansion — substrate, not further explored
                if target not in graph.modules:
                    unresolved.add(target)
                    visited.add(target)
                    continue
                visited.add(target)
                fd = domain_of(target, root_package)
                foreign_touched.setdefault(fd, set()).add(target)
                next_frontier.add(target)
        frontier = next_frontier
        depth += 1

    tier2: set[str] = set()
    tier3: set[str] = set()
    tier3_domains: set[str] = set()
    for fd, mods in foreign_touched.items():
        total_tokens = sum(graph.token_costs.get(m, 0) for m in mods)
        if len(mods) <= tier2_max_files and total_tokens <= tier2_max_tokens:
            tier2 |= mods
        else:
            tier3 |= mods
            tier3_domains.add(fd)

    return ClosureResult(
        domain_own=domain_own,
        tier1_core=frozenset(core_touched),
        tier2_declared=frozenset(tier2),
        tier3_cross_reference=frozenset(tier3),
        unresolved_external=frozenset(unresolved),
        foreign_domains_touched=frozenset(foreign_touched.keys()),
    )
