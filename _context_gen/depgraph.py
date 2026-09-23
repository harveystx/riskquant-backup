# _context_gen/depgraph.py
"""
Repository-wide dependency graph built from ModuleInfo objects.

Answers exactly one question, mechanically: for module M, which other
internal modules does M's own import syntax name? No call-graph inference,
no transitive-closure-by-default — closure.py decides what to do with the
edges this module exposes.
"""

from __future__ import annotations

from dataclasses import dataclass

from _context_gen.ast_model import ModuleInfo

APPROX_CHARS_PER_TOKEN = 3.7


def estimate_tokens(text: str) -> int:
    """
    Same heuristic used throughout the audit trail: chars/3.7. No offline
    Claude tokenizer is available in this environment; this is disclosed
    everywhere the figure is reported, never presented as exact.
    """
    return round(len(text) / APPROX_CHARS_PER_TOKEN) if text else 0


def domain_of(dotted: str, root_package: str) -> str:
    """
    The top-level domain a module belongs to, e.g. "riskquant.valuation.
    blocks.spot_block" -> "valuation" (root_package="riskquant"). A module
    that *is* the root package itself, or a package with no further
    component, maps to the root package name.
    """
    if not dotted.startswith(root_package + "."):
        return dotted.split(".")[0]
    rest = dotted[len(root_package) + 1 :]
    return rest.split(".")[0] if rest else root_package


@dataclass(frozen=True, slots=True, kw_only=True)
class RepoGraph:
    modules: dict[str, ModuleInfo]  # dotted -> ModuleInfo
    token_costs: dict[str, int]  # dotted -> estimated tokens of that module's own source
    edges: dict[str, frozenset[str]]  # dotted -> internal targets it imports (module granularity)
    importer_domains: dict[str, set[str]]  # dotted target -> set of DISTINCT domains that import it


def _resolve_raw_import(base: str, names: tuple[str, ...], known_modules: frozenset[str]) -> list[str]:
    """
    Resolve one raw `from base import names` (or plain `import base`)
    statement against the real module registry.

    `from pkg import name`: prefer the submodule interpretation
    (`pkg.name`) whenever `pkg.name` is a real known module (this is the
    correct fix for the case discovered against the real RiskQuant
    repository: `from riskquant.observations import _validate`, where
    `_validate.py` is a genuine submodule, not an attribute of
    `observations/__init__.py`). Fall back to `base` itself when no name
    resolves to a real submodule, or when there are no names at all
    (`import base`).
    """
    if not names:
        return [base]
    resolved: list[str] = []
    any_submodule_hit = False
    for name in names:
        candidate = f"{base}.{name}"
        if candidate in known_modules:
            resolved.append(candidate)
            any_submodule_hit = True
    if not any_submodule_hit:
        resolved.append(base)
    return resolved


def build_graph(module_infos: list[ModuleInfo], texts: dict[str, str], root_package: str) -> RepoGraph:
    modules = {m.dotted: m for m in module_infos}
    known = frozenset(modules.keys())
    token_costs = {m.dotted: estimate_tokens(texts.get(m.dotted, "")) for m in module_infos}
    edges: dict[str, frozenset[str]] = {}
    importer_domains: dict[str, set[str]] = {}
    internal_prefixes = frozenset({root_package}) | frozenset(
        d.split(".")[0] for d in modules if d.split(".")[0] != root_package
    )

    for m in module_infos:
        targets: set[str] = set()
        for raw in m.raw_imports:
            for resolved in _resolve_raw_import(raw.base, raw.names, known):
                is_internal = any(resolved == p or resolved.startswith(p + ".") for p in internal_prefixes)
                if is_internal:
                    targets.add(resolved)
        edges[m.dotted] = frozenset(targets)
        src_domain = domain_of(m.dotted, root_package)
        for target in targets:
            importer_domains.setdefault(target, set()).add(src_domain)

    return RepoGraph(modules=modules, token_costs=token_costs, edges=edges, importer_domains=importer_domains)
