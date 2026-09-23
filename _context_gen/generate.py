# _context_gen/generate.py
"""
Orchestration.

One filesystem walk, one set of reads, one in-memory snapshot -- every
artifact this module writes is derived from that single pass (implementation
contract §2.2 / §29: one repository snapshot -> one deterministic
generation -> all derived indexes and source artifacts). Publication is
staged and atomic (§29): nothing at the final output path is touched until
every requested artifact has been rendered and validated as COMPLETE (or
the caller has explicitly opted into a disclosed incomplete write).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from _context_gen import ast_model, completeness, core, depgraph, fs, index_writer, profiles, snapshot, source_writer
from _context_gen.ast_model import ModuleInfo
from _context_gen.closure import compute_closure
from _context_gen.depgraph import RepoGraph, domain_of
from _context_gen.fs import TEXT_SUFFIXES
from _context_gen.source_read import ReadOutcome, read_source

ROOT_PACKAGE = profiles.ROOT_PACKAGE
INTERNAL_PREFIXES = frozenset({"riskquant", "riskquant_lab"})
SRC_ROOTS = ("src/riskquant", "src/riskquant_lab")


@dataclass(frozen=True, slots=True, kw_only=True)
class RepoSnapshot:
    module_infos: list[ModuleInfo]
    texts: dict[str, str]  # dotted -> exact source text
    read_failures: list[tuple[str, str]]  # (repo_relative, ReadOutcome.value) for UNREADABLE/PARSE_FAILED
    other_files: list[fs.DiscoveredFile]  # non-.py text files (pyproject.toml, *.md, ...)
    aux_texts: dict[str, str]  # repo_relative -> exact text, for every file with no dotted identity
    #                            (tests/, scripts/, non-.py text). Read in the same single pass as
    #                            package source, so test/demo artifacts never re-read the disk.
    graph: RepoGraph
    identity: snapshot.RepositoryIdentity


def take_snapshot(repo_root: Path) -> RepoSnapshot:
    """One filesystem walk + one read/parse pass over every source file."""
    py_files: list[fs.DiscoveredFile] = []
    for root in SRC_ROOTS:
        py_files.extend(fs.sweep(repo_root / root, repo_root, suffixes=frozenset({".py"})))
    for root in ("scripts", "tests"):
        py_files.extend(fs.sweep(repo_root / root, repo_root, suffixes=frozenset({".py"})))
    py_files.sort(key=lambda f: f.repo_relative)

    other_files: list[fs.DiscoveredFile] = []
    for root in SRC_ROOTS + ("scripts", "tests"):
        other_files.extend(
            fs.sweep(repo_root / root, repo_root, suffixes=frozenset(TEXT_SUFFIXES) - {".py"})
        )
    pyproject = fs.single_file(repo_root / "pyproject.toml", repo_root)
    if pyproject is not None:
        other_files.append(pyproject)
    other_files.sort(key=lambda f: f.repo_relative)

    module_infos: list[ModuleInfo] = []
    texts: dict[str, str] = {}
    read_failures: list[tuple[str, str]] = []

    aux_texts: dict[str, str] = {}

    for f in py_files:
        result = read_source(f.absolute, f.kind)
        if result.outcome is ReadOutcome.UNREADABLE:
            read_failures.append((f.repo_relative, ReadOutcome.UNREADABLE.value))
            continue
        if result.outcome is ReadOutcome.PARSE_FAILED:
            read_failures.append((f.repo_relative, ReadOutcome.PARSE_FAILED.value))
            if result.text is not None:
                # Keep the exact text: a test/demo artifact still embeds it
                # verbatim, while the PARSE FAILURE is reported in its status.
                aux_texts[f.repo_relative] = result.text
            continue
        # OK -- a .py file always yields OK or PARSE_FAILED, never NOT_PYTHON here
        assert result.tree is not None and result.text is not None
        dotted = ast_model.dotted_name_for(f.repo_relative, "src/riskquant/", "riskquant") or ast_model.dotted_name_for(
            f.repo_relative, "src/riskquant_lab/", "riskquant_lab"
        )
        if dotted is None:
            # A .py file outside any known package root (e.g. under scripts/
            # or tests/) -- kept verbatim for whichever test/demo artifact
            # sweeps it, but it has no dotted identity and is not part of
            # the internal dependency graph.
            aux_texts[f.repo_relative] = result.text
            continue
        info = ast_model.build_module_info(
            repo_relative=f.repo_relative,
            module_dotted=dotted,
            text=result.text,
            tree=result.tree,
            internal_prefixes=INTERNAL_PREFIXES,
        )
        module_infos.append(info)
        texts[dotted] = result.text

    for f in other_files:
        result = read_source(f.absolute, f.kind)
        if result.outcome is ReadOutcome.UNREADABLE:
            read_failures.append((f.repo_relative, ReadOutcome.UNREADABLE.value))
            continue
        if result.text is not None:
            aux_texts[f.repo_relative] = result.text

    graph = depgraph.build_graph(module_infos, texts, ROOT_PACKAGE)
    identity = snapshot.repository_identity(repo_root)

    return RepoSnapshot(
        module_infos=module_infos,
        texts=texts,
        read_failures=read_failures,
        other_files=other_files,
        aux_texts=aux_texts,
        graph=graph,
        identity=identity,
    )


def discover_domains(repo_root: Path) -> list[str]:
    """
    Every direct subdirectory of src/riskquant/ containing >=1 .py file
    becomes a domain, except `solutions/` itself (its children are
    discovered separately -- see `discover_solution_domains`).
    """
    root = repo_root / "src" / "riskquant"
    domains: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "solutions":
            continue
        if any(child.rglob("*.py")):
            domains.append(child.name)
    return domains


def discover_solution_domains(repo_root: Path) -> list[str]:
    """
    Every direct subdirectory of src/riskquant/solutions/ becomes its own
    domain, named `solutions_<name>` to avoid colliding with a core-package
    domain of the same bare name (e.g. `sensitivity_pnl` core vs.
    `solutions_sensitivity_pnl`).

    This is a targeted fix, not a new architecture tier: the SOLUTION tier
    is the estate's own documented public import surface (`riskquant.
    solutions.*` -- "the only import surface for end users"), and a
    same-name-only attachment (a single `solutions/<domain>/api.py` file
    bolted onto a core domain of the identical name) silently owned
    nothing for facades whose name doesn't match a core package 1:1 --
    confirmed on the live-proxy repository: `solutions/credit_exposure/`,
    `solutions/credit_var/`, `solutions/risk_run/`, `solutions/
    run_identity/`, and `solutions/scenario_models/` have no core package
    of the same name at all, and most of `solutions/sensitivity_pnl/`
    beyond `api.py` was reachable only if some other domain's closure
    happened to pull it in. Each solutions subdirectory is now a full
    domain in its own right, closed the same way every other domain is
    (own source + dependency closure through core/Tier-2/Tier-3) --
    nothing new is invented, the existing model is just applied
    completely instead of partially.
    """
    root = repo_root / "src" / "riskquant" / "solutions"
    if not root.exists():
        return []
    domains: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.rglob("*.py")):
            domains.append(f"solutions_{child.name}")
    return domains


def build_profile(domain: str, repo_root: Path) -> profiles.Profile:
    if domain.startswith("solutions_"):
        bare = domain[len("solutions_") :]
        return profiles.Profile(
            name=domain,
            production_package_dirs=(f"src/riskquant/solutions/{bare}",),
        )
    extra: tuple[str, ...] = ()
    api_file = repo_root / "src" / "riskquant" / "solutions" / domain / "api.py"
    if api_file.exists():
        # Cheap, established convenience (report 4, §5/§9): a domain whose
        # name matches a solutions/ subdirectory also gets that facade's
        # single api.py file, so a standalone read of the core domain
        # alone still sees its public entry point -- harmless duplication
        # of one small file, on top of (not instead of) that facade's own
        # full `solutions_<name>` domain above.
        extra = (f"src/riskquant/solutions/{domain}/api.py",)
    return profiles.Profile(
        name=domain,
        production_package_dirs=(f"src/riskquant/{domain}",),
        extra_production_files=extra,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedProfile:
    name: str
    filename: str
    content: str
    completeness_report: completeness.CompletenessReport
    token_estimate: int


def render_domain_profile(
    profile: profiles.Profile,
    snap: RepoSnapshot,
    repo_root: Path,
    *,
    tests: bool,
    demos: bool,
) -> RenderedProfile:
    resolved = profiles.with_flags(profile, tests=tests, demos=demos)
    prod_files = profiles.resolve_production_files(resolved, repo_root)
    own_dotted = frozenset(
        m.dotted for m in snap.module_infos if any(m.repo_relative == f.repo_relative for f in prod_files)
    )

    closure = compute_closure(
        domain_own=own_dotted,
        graph=snap.graph,
        core_names=core.core_module_names(),
        root_package=ROOT_PACKAGE,
    )

    configured_dirs = list(profile.production_package_dirs)
    existing_dirs = {d for d in configured_dirs if (repo_root / d).exists()}
    read_failures_here = [
        (p, reason) for p, reason in snap.read_failures if any(p.startswith(d + "/") for d in configured_dirs)
    ]
    report = completeness.build_report(
        profile_name=profile.name,
        configured_dirs=configured_dirs,
        existing_dirs=existing_dirs,
        read_failures=read_failures_here,
        closure=closure,
    )

    embedded_dotted = own_dotted | closure.tier1_core | closure.tier2_declared
    files: list[tuple[str, str]] = []
    for m in snap.module_infos:
        if m.dotted in embedded_dotted:
            files.append((m.repo_relative, snap.texts[m.dotted]))

    test_files = profiles.resolve_test_files(resolved, repo_root) if tests else []
    demo_files = profiles.resolve_demo_files(resolved, repo_root) if demos else []
    for df in test_files + demo_files:
        # Taken from the snapshot, never re-read from disk: a second read
        # here would break the one-snapshot invariant, and the previous
        # `except OSError: continue` silently dropped unreadable files.
        if df.repo_relative in snap.aux_texts:
            files.append((df.repo_relative, snap.aux_texts[df.repo_relative]))
    optional_failures = tuple(
        (p, reason)
        for p, reason in snap.read_failures
        if any(p == df.repo_relative for df in test_files + demo_files)
    )
    if optional_failures:
        report = completeness.CompletenessReport(
            profile_name=report.profile_name,
            missing_configured_paths=report.missing_configured_paths,
            unreadable_files=report.unreadable_files
            + tuple(p for p, r in optional_failures if r == ReadOutcome.UNREADABLE.value),
            parse_failures=report.parse_failures
            + tuple(p for p, r in optional_failures if r == ReadOutcome.PARSE_FAILED.value),
            missing_required_dependencies=report.missing_required_dependencies,
        )
    files.sort(key=lambda pc: pc[0])

    tier3_note = None
    if closure.tier3_cross_reference:
        foreign = sorted({domain_of(t, ROOT_PACKAGE) for t in closure.tier3_cross_reference})
        tier3_note = "CROSS-REFERENCE (not embedded -- load the owning profile too): " + ", ".join(foreign)

    content = source_writer.render_source_artifact(
        profile_name=profile.name,
        completeness=report,
        header_note=tier3_note,
        files=files,
    )
    tokens = depgraph.estimate_tokens(content)
    return RenderedProfile(
        name=profile.name,
        filename=f"context_riskquant_{profile.name}.txt",
        content=content,
        completeness_report=report,
        token_estimate=tokens,
    )


TESTS_ROOT = "tests"
TESTS_ARTIFACT_NOTE = (
    "TESTS ONLY -- production source is not embedded here. Resolve imported "
    "riskquant.* symbols via context_riskquant_index.txt and the owning domain profile."
)


def render_tests_artifact(snap: RepoSnapshot, repo_root: Path) -> RenderedProfile:
    """
    One artifact containing every test file under tests/, verbatim.

    Tests are evidence, not production dependencies, so no closure is
    computed and no production source is embedded (embedding each test's
    imports would re-create the whole repository). Completeness here means:
    tests/ exists, and every file under it was read and, for .py, parsed.
    """
    swept = fs.sweep(repo_root / TESTS_ROOT, repo_root, suffixes=frozenset(TEXT_SUFFIXES))
    files = [(f.repo_relative, snap.aux_texts[f.repo_relative]) for f in swept if f.repo_relative in snap.aux_texts]
    files.sort(key=lambda pc: pc[0])

    prefix = TESTS_ROOT + "/"
    failures = [(p, r) for p, r in snap.read_failures if p.startswith(prefix)]
    report = completeness.CompletenessReport(
        profile_name="tests",
        missing_configured_paths=() if (repo_root / TESTS_ROOT).is_dir() else (TESTS_ROOT,),
        unreadable_files=tuple(p for p, r in failures if r == ReadOutcome.UNREADABLE.value),
        parse_failures=tuple(p for p, r in failures if r == ReadOutcome.PARSE_FAILED.value),
    )
    content = source_writer.render_source_artifact(
        profile_name="tests", completeness=report, header_note=TESTS_ARTIFACT_NOTE, files=files
    )
    return RenderedProfile(
        name="tests",
        filename="context_riskquant_tests.txt",
        content=content,
        completeness_report=report,
        token_estimate=depgraph.estimate_tokens(content),
    )


def render_core_artifact(snap: RepoSnapshot) -> RenderedProfile:
    core_names = core.core_module_names()
    files = [(m.repo_relative, snap.texts[m.dotted]) for m in snap.module_infos if m.dotted in core_names]
    files.sort(key=lambda pc: pc[0])
    report = completeness.CompletenessReport(profile_name="core")
    content = source_writer.render_source_artifact(
        profile_name="core", completeness=report, header_note=None, files=files
    )
    return RenderedProfile(
        name="core",
        filename="context_riskquant_core.txt",
        content=content,
        completeness_report=report,
        token_estimate=depgraph.estimate_tokens(content),
    )


def render_index_artifact(snap: RepoSnapshot, repo_root: Path, rendered_profiles: list[RenderedProfile]) -> RenderedProfile:
    importer_domains = snap.graph.importer_domains
    evaluations = core.evaluate_candidates(importer_domains=importer_domains, token_costs=snap.graph.token_costs)
    all_files = [(rp.filename, rp.content) for rp in rendered_profiles]
    content_hash = snapshot.artifact_content_hash(all_files)
    profile_summaries = [
        (rp.name, "COMPLETE" if rp.completeness_report.is_complete else "INCOMPLETE", rp.token_estimate)
        for rp in rendered_profiles
    ]
    content = index_writer.render_index(
        identity=snap.identity,
        content_hash=content_hash,
        modules=snap.module_infos,
        other_files=[f.repo_relative for f in snap.other_files],
        graph=snap.graph,
        core_evaluations=evaluations,
        profile_summaries=profile_summaries,
    )
    return RenderedProfile(
        name="index",
        filename="context_riskquant_index.txt",
        content=content,
        completeness_report=completeness.CompletenessReport(profile_name="index"),
        token_estimate=depgraph.estimate_tokens(content),
    )


def publish(rendered: list[RenderedProfile], output_dir: Path, *, allow_incomplete: bool = False) -> list[str]:
    """
    Atomic publication (implementation contract §29): render everything into
    a staging directory first; only if every artifact is COMPLETE (or the
    caller explicitly passed allow_incomplete=True) does the staging
    directory get moved over the final output path. A failed/incomplete
    run never touches a previously-published good snapshot.

    A single invocation may regenerate only a subset of artifacts (e.g.
    `--profile valuation`). Staging starts from a copy of whatever is
    already published, so an un-regenerated artifact from a previous run
    is carried forward untouched rather than deleted -- the atomic swap
    still guarantees that a failed run leaves the previously-published
    directory completely unchanged, byte for byte.
    """
    incomplete = [rp for rp in rendered if not rp.completeness_report.is_complete]
    if incomplete and not allow_incomplete:
        names = ", ".join(rp.name for rp in incomplete)
        raise RuntimeError(
            f"Refusing to publish: {len(incomplete)} profile(s) INCOMPLETE ({names}). "
            "Previous output left untouched. Pass allow_incomplete=True to publish anyway "
            "(each incomplete artifact still carries its own STATUS: INCOMPLETE block)."
        )

    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    if output_dir.exists():
        shutil.copytree(output_dir, staging)
    else:
        staging.mkdir(parents=True)
    for rp in rendered:
        (staging / rp.filename).write_text(rp.content, encoding="utf-8")

    if output_dir.exists():
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        output_dir.rename(backup)
    staging.rename(output_dir)

    return [rp.filename for rp in rendered]
