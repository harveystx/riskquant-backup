# _context_gen/core.py
"""
Shared CORE membership registry.

Core membership is a curated decision backed by mechanical evidence, not a
bare fan-out calculation (a module imported by many files within one
package is not core; a module imported by few files across many distinct
packages can be). See the architecture contract, §5, for the full test:

    substrate (no domain identity of its own)
    AND breadth >= MIN_DISTINCT_DOMAINS
    AND cost <= MAX_SUBSTRATE_TOKENS

Each entry below is an explicit approval, not an automatic inclusion. The
generator recomputes each entry's breadth/cost on every run (see
`evaluate_candidates`) and flags drift for re-approval — it never silently
adds or drops a member because the measured numbers moved.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_DISTINCT_DOMAINS = 3
MAX_SUBSTRATE_TOKENS = 5_000


@dataclass(frozen=True, slots=True, kw_only=True)
class CoreMember:
    module: str  # dotted module name
    approved: bool  # explicit human approval, not inferred
    rationale: str  # one line — why this is substrate, not domain


# Explicit, reviewable core set. Extending this list is a deliberate edit,
# not a side effect of the codebase growing.
APPROVED_CORE: tuple[CoreMember, ...] = (
    CoreMember(
        module="riskquant.constants",
        approved=True,
        rationale="Sentinel/constant definitions; no domain identity of its own.",
    ),
    CoreMember(
        module="riskquant._arrays",
        approved=True,
        rationale="Array-aliasing/dtype helpers used at the SoA boundary; substrate.",
    ),
    CoreMember(
        module="riskquant.jit",
        approved=True,
        rationale="Canonical Numba decorator/type-alias source of truth; substrate by estate doctrine.",
    ),
    CoreMember(
        module="riskquant.exceptions",
        approved=True,
        rationale="Exception hierarchy; no domain identity, trivial cost.",
    ),
    CoreMember(
        module="riskquant.universe",
        approved=True,
        rationale="Self-documented as a deliberately leaf UIX-identity contract module.",
    ),
    CoreMember(
        module="riskquant.portfolio.compiled",
        approved=True,
        rationale="Compiled-portfolio contract type, not the portfolio engine itself.",
    ),
    CoreMember(
        module="riskquant.market_data.state",
        approved=True,
        rationale="Run-time market-state contract type, not market_data compute logic.",
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CoreEvaluation:
    module: str
    distinct_domains: int
    token_cost: int
    passes_breadth: bool
    passes_cost: bool
    currently_approved: bool
    needs_review: bool  # True when measured evidence and approval status disagree


def evaluate_candidates(
    *,
    importer_domains: dict[str, set[str]],
    token_costs: dict[str, int],
) -> list[CoreEvaluation]:
    """
    Recompute breadth/cost for every currently-approved core member (and
    report drift), plus any other module that newly clears the mechanical
    bar and is NOT yet approved (a candidate for human review, never
    auto-added).
    """
    approved_by_name = {m.module: m for m in APPROVED_CORE}
    all_names = set(approved_by_name) | {
        mod
        for mod, domains in importer_domains.items()
        if len(domains) >= MIN_DISTINCT_DOMAINS and token_costs.get(mod, 10**9) <= MAX_SUBSTRATE_TOKENS
    }

    out: list[CoreEvaluation] = []
    for mod in sorted(all_names):
        domains = importer_domains.get(mod, set())
        cost = token_costs.get(mod, 0)
        passes_breadth = len(domains) >= MIN_DISTINCT_DOMAINS
        passes_cost = cost <= MAX_SUBSTRATE_TOKENS
        currently_approved = mod in approved_by_name and approved_by_name[mod].approved
        mechanically_qualifies = passes_breadth and passes_cost
        out.append(
            CoreEvaluation(
                module=mod,
                distinct_domains=len(domains),
                token_cost=cost,
                passes_breadth=passes_breadth,
                passes_cost=passes_cost,
                currently_approved=currently_approved,
                needs_review=(mechanically_qualifies != currently_approved),
            )
        )
    return out


def core_module_names() -> frozenset[str]:
    return frozenset(m.module for m in APPROVED_CORE if m.approved)
