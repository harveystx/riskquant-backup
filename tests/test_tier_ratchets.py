# tests/test_tier_ratchets.py
"""
Ratchets for the tier model in ``CLAUDE.md`` section 3.

Two legacy deviations are frozen rather than fixed. A ratchet fails in both
directions: a new offender breaks the build, and so does an allowlist entry
that no longer offends. The allowlists can therefore only shrink.

The kernel rule has no allowlist. Every kernel module is clean, so any
edge-tier import into a kernel is a regression.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "riskquant"

#: Edge-tier import roots that must never reach a kernel module.
KERNEL_FORBIDDEN_ROOTS = frozenset(
    {"polars", "pandas", "pydantic", "pandera", "datetime"}
)

#: Frozen: the only module outside ``solutions/`` that imports the solution tier.
SOLUTIONS_IMPORT_ALLOWLIST = frozenset({"sensitivities/book_adapter.py"})

#: Frozen: legacy pandas importers. Delete an entry when its module migrates.
PANDAS_IMPORT_ALLOWLIST = frozenset(
    {
        "config/proxy_config/audit.py",
        "config/static_config/audit.py",
        "credit/exposure/add_on/ingest.py",
        "credit/exposure/obligor_ead/ingest.py",
        "credit/exposure/saccr/ingest.py",
        "credit/jtd/ingest/_shared.py",
        "credit/jtd/ingest/builders/beta_binomial.py",
        "credit/jtd/ingest/builders/forced_default.py",
        "credit/jtd/ingest/builders/independent_bernoulli.py",
        "credit/jtd/ingest/builders/marshall_olkin.py",
        "credit/jtd/ingest/builders/mf_gaussian.py",
        "credit/jtd/ingest/builders/mf_students_t.py",
        "credit/jtd/ingest/builders/sf_gaussian.py",
        "credit/jtd/ingest/builders/sf_students_t.py",
        "credit/jtd/ingest/builders/stress_regime_bernoulli.py",
        "credit/jtd/ingest/bundle.py",
        "credit/jtd/ingest/validate.py",
        "credit/masters/factor_universe/ingest.py",
        "credit/masters/netting_set/ingest.py",
        "credit/masters/obligors/ingest.py",
        "ingest_fixtures/credit/credit_exposure/minimal_spot_with_netting_sets.py",
        "ingest_fixtures/credit/credit_factors/minimal_factor_data.py",
        "ingest_fixtures/credit/credit_factors/minimal_factor_universe_master.py",
        "ingest_fixtures/credit/jtd/beta_binomial/minimal_beta_binomial_jtd.py",
        "ingest_fixtures/credit/jtd/forced_default/minimal_forced_default_jtd.py",
        "ingest_fixtures/credit/jtd/independent_bernoulli/minimal_independent_bernoulli_jtd.py",
        "ingest_fixtures/credit/jtd/mf_copula/minimal_mf_copula_gaussian.py",
        "ingest_fixtures/credit/jtd/mixed/minimal_mixed.py",
        "ingest_fixtures/credit/jtd/sf_copula_jtd/minimal_sf_copula_gaussian.py",
        "ingest_fixtures/credit/jtd/sf_copula_jtd/minimal_sf_copula_students_t_global_chi.py",
        "ingest_fixtures/credit/jtd/stress_regime_bernoulli/minimal_stress_regime_bernoulli_jtd.py",
        "ingest_fixtures/credit/netting_set/minimal_netting_set_master.py",
        "ingest_fixtures/credit/obligors/minimal_obligors_master.py",
        "ingest_fixtures/market_data/canonical_market_data_history.py",
        "ingest_fixtures/portfolio/exhaustive.py",
        "ingest_fixtures/portfolio/spot.py",
        "ingest_fixtures/portfolio/vanilla_european_option_curve.py",
        "ingest_fixtures/portfolio/vanilla_european_option_node.py",
        "ingest_fixtures/portfolio/vanilla_forward_curve.py",
        "ingest_fixtures/portfolio/vanilla_forward_node.py",
        "ingest_fixtures/portfolio/vanilla_irs.py",
        "market_data/risk_factors/builder_matrix.py",
        "market_data/risk_factors/builder_snapshot.py",
        "market_data/risk_factors/matrix.py",
        "market_data/risk_factors/snapshot.py",
        "portfolio/ingest/batch.py",
        "portfolio/ingest/canonicalize.py",
        "portfolio/ingest/compile.py",
        "portfolio/ingest/dispatcher.py",
        "portfolio/ingest/fx_resolver.py",
        "portfolio/ingest/materialize.py",
        "portfolio/ingest/parser.py",
        "portfolio/ingest/pipeline.py",
        "portfolio/ingest/rf_resolver.py",
        "portfolio/ingest/utils.py",
        "portfolio/ingest/validate.py",
        "proxy_resolver/proxy_audit.py",
        "solutions/credit_exposure/golden_master_credit_exposure.py",
        "solutions/credit_exposure/handoff.py",
        "solutions/credit_exposure/recipes.py",
        "solutions/credit_exposure/run.py",
        "solutions/credit_exposure/schemas.py",
        "solutions/credit_var/recipes.py",
        "solutions/credit_var/run.py",
        "solutions/credit_var/schemas.py",
        "utils/dataframes.py",
        "utils/dates.py",
        "utils/printer.py",
    }
)


def _source_files() -> list[Path]:
    files = sorted(SOURCE_ROOT.rglob("*.py"))
    assert files, f"no source files under {SOURCE_ROOT}"
    return files


def _relative(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).as_posix()


def _absolute_imports(path: Path) -> frozenset[str]:
    """
    Return every absolute module name a file imports, at any nesting depth.

    ``ast.walk`` also reaches imports inside functions and ``TYPE_CHECKING``
    blocks, so a deferred import cannot hide a tier breach.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return frozenset(names)


def _is_kernel_module(relative_path: str) -> bool:
    parts = relative_path.split("/")
    return any("kernel" in part for part in parts) or "core_math" in parts[:-1]


def _imports_root(imports: frozenset[str], root: str) -> bool:
    return any(name == root or name.startswith(f"{root}.") for name in imports)


def _ratchet_report(
    *, rule: str, offenders: set[str], allowlist: frozenset[str]
) -> list[str]:
    new = sorted(offenders - allowlist)
    stale = sorted(allowlist - offenders)
    report: list[str] = []
    if new:
        report.append(f"{rule}: new offenders (fix them; never extend the allowlist):")
        report.extend(f"  {name}" for name in new)
    if stale:
        report.append(f"{rule}: stale allowlist entries (delete them):")
        report.extend(f"  {name}" for name in stale)
    return report


def test_kernel_modules_import_no_edge_or_solution_tier() -> None:
    offenders: list[str] = []
    checked = 0
    for path in _source_files():
        relative_path = _relative(path)
        if not _is_kernel_module(relative_path):
            continue
        checked += 1
        banned = sorted(
            name
            for name in _absolute_imports(path)
            if name.split(".")[0] in KERNEL_FORBIDDEN_ROOTS
            or _imports_root(frozenset({name}), "riskquant.solutions")
        )
        if banned:
            offenders.append(f"{relative_path}: {banned}")
    assert checked > 0, "the kernel guard scanned nothing"
    assert not offenders, "kernel modules import a higher tier:\n" + "\n".join(
        offenders
    )


def test_solution_tier_import_ratchet() -> None:
    offenders = {
        _relative(path)
        for path in _source_files()
        if not _relative(path).startswith("solutions/")
        and _imports_root(_absolute_imports(path), "riskquant.solutions")
    }
    report = _ratchet_report(
        rule="riskquant.solutions imported below the solution tier",
        offenders=offenders,
        allowlist=SOLUTIONS_IMPORT_ALLOWLIST,
    )
    assert not report, "\n".join(report)


def test_pandas_import_ratchet() -> None:
    offenders = {
        _relative(path)
        for path in _source_files()
        if _imports_root(_absolute_imports(path), "pandas")
    }
    report = _ratchet_report(
        rule="pandas import",
        offenders=offenders,
        allowlist=PANDAS_IMPORT_ALLOWLIST,
    )
    assert not report, "\n".join(report)
