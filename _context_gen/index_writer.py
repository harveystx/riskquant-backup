# _context_gen/index_writer.py
"""
Renders context_riskquant_index.txt.

Every field here passed the test in the architecture contract §8: does it
add navigation capability per token the domain profiles cannot provide
themselves? No prose summaries, no repeated governance doctrine (one
pointer line only), no directory tree (superseded by the flat manifest —
same information, cheaper, more greppable).
"""

from __future__ import annotations

from _context_gen.ast_model import ModuleInfo
from _context_gen.core import CoreEvaluation
from _context_gen.depgraph import RepoGraph
from _context_gen.snapshot import RepositoryIdentity


def render_snapshot(identity: RepositoryIdentity, content_hash: str) -> str:
    commit = identity.commit or "UNKNOWN (not a git repository)"
    dirty = "unknown" if identity.dirty is None else str(identity.dirty).lower()
    return f"SNAPSHOT commit={commit} dirty={dirty} content_hash={content_hash}"


def render_manifest(modules: list[ModuleInfo], other_files: list[str]) -> str:
    lines = ["MANIFEST"]
    paths = sorted([m.repo_relative for m in modules] + other_files)
    lines.extend(paths)
    return "\n".join(lines)


def render_symbols(modules: list[ModuleInfo]) -> str:
    lines = ["SYMBOLS"]
    for m in sorted(modules, key=lambda mi: mi.dotted):
        if not m.symbols and not m.is_disabled:
            continue
        lines.append(f"MODULE {m.dotted}" + ("  [DISABLED/COMMENT-ONLY]" if m.is_disabled else ""))
        for s in m.symbols:
            indent = "  " if s.kind in ("class", "function") else "    "
            label = {"class": "CLASS", "function": "FUNCTION", "method": "METHOD", "nested_class": "CLASS"}[s.kind]
            bases = f"({', '.join(s.bases)})" if s.bases else ""
            decos = f" {' '.join(s.decorators)}" if s.decorators else ""
            sig = f" {s.signature}" if s.signature else ""
            loc = f"{m.repo_relative.rsplit('/', 1)[-1]}:{s.line}"
            lines.append(f"{indent}{label} {s.name}{bases}{sig}{decos}  {loc}")
    return "\n".join(lines)


def render_dependencies(modules: list[ModuleInfo]) -> str:
    lines = ["DEPENDENCIES"]
    for m in sorted(modules, key=lambda mi: mi.dotted):
        internal = sorted({e.target for e in m.imports if e.is_internal})
        if not internal:
            continue
        lines.append(f"{m.dotted} -> {', '.join(internal)}")
    return "\n".join(lines)


def render_core(evaluations: list[CoreEvaluation]) -> str:
    lines = ["CORE"]
    for ev in sorted(evaluations, key=lambda e: e.module):
        if not ev.currently_approved and not ev.needs_review:
            continue  # not core, and not a drift candidate -- not worth a line
        flag = "  [NEEDS REVIEW: measured evidence disagrees with approval]" if ev.needs_review else ""
        lines.append(
            f"{ev.module}  domains={ev.distinct_domains} tokens={ev.token_cost} "
            f"approved={str(ev.currently_approved).lower()}{flag}"
        )
    return "\n".join(lines)


def render_profiles(profile_summaries: list[tuple[str, str, int]]) -> str:
    # (name, status, token_estimate)
    lines = ["PROFILES"]
    for name, status, tokens in sorted(profile_summaries):
        lines.append(f"{name}  status={status}  tokens~={tokens}")
    return "\n".join(lines)


def render_index(
    *,
    identity: RepositoryIdentity,
    content_hash: str,
    modules: list[ModuleInfo],
    other_files: list[str],
    graph: RepoGraph,
    core_evaluations: list[CoreEvaluation],
    profile_summaries: list[tuple[str, str, int]],
) -> str:
    sections = [
        render_snapshot(identity, content_hash),
        "GOVERNANCE  See CLAUDE.md for architecture, Numba, and DOD doctrine — not restated here.",
        render_manifest(modules, other_files),
        render_symbols(modules),
        render_dependencies(modules),
        render_core(core_evaluations),
        render_profiles(profile_summaries),
    ]
    return "\n\n".join(sections) + "\n"
