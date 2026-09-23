# _context_gen/profiles.py
"""
Declarative profile composition.

A profile describes intent (§4 of the implementation contract); the
generator derives the concrete file set from the real tree and the real
dependency graph. No profile in this module is a hand-typed path list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from _context_gen import fs

ROOT_PACKAGE = "riskquant"
SRC_PREFIX = f"src/{ROOT_PACKAGE}/"
SECONDARY_PACKAGES: tuple[tuple[str, str], ...] = (("riskquant_lab", f"src/riskquant_lab/"),)


@dataclass(frozen=True, slots=True, kw_only=True)
class Profile:
    """
    Declarative composition. `production_package_dirs` is one or more
    directories under src/<pkg>/ whose sweep defines this domain's OWN
    production source (§4) — resolved by rglob at generation time, never
    a hand-typed file list.
    """

    name: str
    production_package_dirs: tuple[str, ...]  # repo-relative dirs, e.g. ("src/riskquant/valuation",)
    include_core: bool = True
    include_tests: bool = True  # tests off by default is a policy choice made per-invocation, not here
    include_demos: bool = True
    extra_production_files: tuple[str, ...] = field(default_factory=tuple)  # single explicit files


def with_flags(profile: Profile, *, tests: bool, demos: bool) -> Profile:
    """Compose a variant of a profile with different optional-material flags,
    never a second hand-maintained profile (this directly replaces the
    `_light` / `_static_config` pattern, §12 of the implementation contract)."""
    return Profile(
        name=profile.name,
        production_package_dirs=profile.production_package_dirs,
        include_core=profile.include_core,
        include_tests=tests,
        include_demos=demos,
        extra_production_files=profile.extra_production_files,
    )


def resolve_production_files(profile: Profile, repo_root: Path) -> list[fs.DiscoveredFile]:
    found: list[fs.DiscoveredFile] = []
    for rel_dir in profile.production_package_dirs:
        found.extend(fs.sweep(repo_root / rel_dir, repo_root, suffixes=frozenset({".py"})))
    for rel_file in profile.extra_production_files:
        single = fs.single_file(repo_root / rel_file, repo_root)
        if single is not None:
            found.append(single)
    found.sort(key=lambda f: f.repo_relative)
    return found


def resolve_test_files(profile: Profile, repo_root: Path) -> list[fs.DiscoveredFile]:
    if not profile.include_tests:
        return []
    return fs.sweep(repo_root / "tests" / profile.name, repo_root, suffixes=frozenset({".py"}))


def resolve_demo_files(profile: Profile, repo_root: Path) -> list[fs.DiscoveredFile]:
    if not profile.include_demos:
        return []
    found: list[fs.DiscoveredFile] = []
    for kind in ("demos", "solutions"):
        found.extend(fs.sweep(repo_root / "scripts" / kind / profile.name, repo_root, suffixes=frozenset({".py"})))
    found.sort(key=lambda f: f.repo_relative)
    return found
