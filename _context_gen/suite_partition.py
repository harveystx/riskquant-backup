# _context_gen/suite_partition.py
"""
Partition of the tests/ tree into per-domain test buckets.

Pure functions over repo-relative POSIX paths and syntactic imports. This
module does no filesystem access: every input comes from the single
repository snapshot that generate.py takes, so the one-snapshot invariant
holds for test artifacts too.

Bucket rule (mechanical, path-only, mirrors the domain-profile names):

    tests/<name>/...            -> "<name>"            (e.g. "credit")
    tests/solutions/<name>/...  -> "solutions_<name>"  (e.g. "solutions_credit_var")
    tests/solutions/<file>      -> "solutions"
    tests/<file>                -> "root"

The key of a bucket is therefore identical to the name of the domain
profile that owns the production code under test, so `tests_credit` pairs
with `credit` and `tests_solutions_credit_var` pairs with
`solutions_credit_var` without a lookup table.

Support closure: pytest runs with `pythonpath = [".", "src"]` and
`--import-mode=importlib`, so test helpers are imported by absolute
`tests.*` names. A bucket embeds every `tests.*` module that its own files
import (recursively) from outside its directory, plus the `conftest.py`
files of its ancestor directories, which pytest applies without an
import. A `tests.*` import with no source in the snapshot is reported as
unresolved, never silently dropped.

Names in this module avoid the `test_` / `Test` prefixes on purpose: a
public function called `test_*` that a test module imports is collected
by pytest as a test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from _context_gen.ast_model import RawImport, dotted_name_for

TESTS_ROOT = "tests"
TESTS_PACKAGE = "tests"
SOLUTIONS_DIR = "solutions"
SOLUTIONS_PREFIX = "solutions_"
ROOT_BUCKET = "root"
CONFTEST = "conftest.py"


@dataclass(frozen=True, slots=True, kw_only=True)
class SuiteBucket:
    """
    One per-domain slice of the test suite.

    Attributes
    ----------
    key : str
        Bucket key; equal to the owning domain-profile name where one exists.
    directory : str
        Repo-relative POSIX directory that the bucket owns.
    own_files : tuple[str, ...]
        Every snapshot file under `directory` (all text kinds), sorted.
    support_files : tuple[str, ...]
        `.py` files outside `directory` that the bucket needs: imported
        `tests.*` helpers (transitive) and ancestor `conftest.py` files.
    unresolved_imports : tuple[str, ...]
        Dotted `tests.*` import targets with no source in the snapshot.
    """

    key: str
    directory: str
    own_files: tuple[str, ...]
    support_files: tuple[str, ...]
    unresolved_imports: tuple[str, ...]


def bucket_of(repo_relative: str) -> tuple[str, str]:
    """
    Return `(bucket_key, bucket_directory)` for one file under tests/.

    Raises
    ------
    ValueError
        If `repo_relative` is not a file path under tests/.
    """
    parts = repo_relative.split("/")
    if len(parts) < 2 or parts[0] != TESTS_ROOT or not parts[-1]:
        raise ValueError(f"not a file under {TESTS_ROOT}/: {repo_relative!r}")
    if len(parts) == 2:
        return ROOT_BUCKET, TESTS_ROOT
    if parts[1] == SOLUTIONS_DIR:
        if len(parts) == 3:
            return SOLUTIONS_DIR, f"{TESTS_ROOT}/{SOLUTIONS_DIR}"
        return f"{SOLUTIONS_PREFIX}{parts[2]}", "/".join(parts[:3])
    return parts[1], "/".join(parts[:2])


def suite_dir_for(profile_name: str) -> str:
    """
    Return the tests/ directory that belongs to a domain profile.

    Inverse of `bucket_of` for domain-profile names, so the `--tests` flag
    of a domain profile and the per-domain test artifacts agree on one rule.
    """
    if profile_name.startswith(SOLUTIONS_PREFIX):
        bare = profile_name[len(SOLUTIONS_PREFIX) :]
        return f"{TESTS_ROOT}/{SOLUTIONS_DIR}/{bare}"
    return f"{TESTS_ROOT}/{profile_name}"


def _module_registry(paths: Iterable[str]) -> tuple[dict[str, str], frozenset[str]]:
    """
    Map dotted `tests.*` module names to files, and list package names.

    Packages include namespace packages (directories without
    `__init__.py`), so `from tests._support import spot_universe` resolves
    through a directory that has no module of its own.
    """
    modules: dict[str, str] = {}
    packages: set[str] = set()
    prefix = TESTS_ROOT + "/"
    for rel in paths:
        dotted = dotted_name_for(rel, prefix, TESTS_PACKAGE)
        if dotted is None:
            continue
        modules[dotted] = rel
        dir_parts = rel.split("/")[:-1]
        for depth in range(1, len(dir_parts) + 1):
            packages.add(".".join(dir_parts[:depth]))
    return modules, frozenset(packages)


def _resolve(
    raw_imports: tuple[RawImport, ...],
    modules: Mapping[str, str],
    packages: frozenset[str],
) -> tuple[set[str], set[str]]:
    """
    Resolve the `tests.*` imports of one file.

    Returns
    -------
    tuple[set[str], set[str]]
        Repo-relative files the imports need, and unresolved dotted targets.
        `from base import name` takes the submodule `base.name` when it
        exists; otherwise `name` is an attribute and `base` itself must be
        a real module file.
    """
    targets: set[str] = set()
    unresolved: set[str] = set()
    for raw in raw_imports:
        base = raw.base
        if base != TESTS_PACKAGE and not base.startswith(TESTS_PACKAGE + "."):
            continue
        if not raw.names:
            if base in modules:
                targets.add(modules[base])
            elif base not in packages:
                unresolved.add(base)
            continue
        for name in raw.names:
            candidate = f"{base}.{name}"
            if candidate in modules:
                targets.add(modules[candidate])
            elif candidate in packages:
                continue
            elif base in modules:
                targets.add(modules[base])
            else:
                unresolved.add(candidate)
    return targets, unresolved


def _ancestor_conftests(directory: str, known: frozenset[str]) -> set[str]:
    parts = directory.split("/")
    found: set[str] = set()
    for depth in range(1, len(parts)):
        candidate = "/".join([*parts[:depth], CONFTEST])
        if candidate in known:
            found.add(candidate)
    return found


def plan_buckets(
    *,
    paths: Iterable[str],
    raw_imports: Mapping[str, tuple[RawImport, ...]],
) -> tuple[SuiteBucket, ...]:
    """
    Group every tests/ file into buckets and close each bucket's support set.

    Parameters
    ----------
    paths : Iterable[str]
        Every repo-relative file under tests/ in the snapshot, readable or
        not. Unreadable files must be present so that their bucket exists
        and reports the failure.
    raw_imports : Mapping[str, tuple[RawImport, ...]]
        Syntactic imports of each parsed tests/ `.py` file.

    Raises
    ------
    ValueError
        If two different directories map to the same bucket key (for
        example `tests/solutions_x/` and `tests/solutions/x/`). Merging
        them would hide a layout error.
    """
    all_paths = sorted(set(paths))
    known = frozenset(all_paths)
    members: dict[str, list[str]] = {}
    directories: dict[str, set[str]] = {}
    for rel in all_paths:
        key, directory = bucket_of(rel)
        members.setdefault(key, []).append(rel)
        directories.setdefault(key, set()).add(directory)

    collisions = {k: sorted(d) for k, d in directories.items() if len(d) > 1}
    if collisions:
        raise ValueError(f"test bucket key collision: {collisions}")

    py_paths = [p for p in all_paths if p.endswith(".py")]
    modules, packages = _module_registry(py_paths)
    resolved: dict[str, tuple[set[str], set[str]]] = {
        rel: _resolve(raw_imports.get(rel, ()), modules, packages) for rel in py_paths
    }

    buckets: list[SuiteBucket] = []
    for key in sorted(members):
        own = members[key]
        (directory,) = directories[key]
        own_set = set(own)
        support = _ancestor_conftests(directory, known)
        unresolved: set[str] = set()
        frontier = sorted({p for p in own if p.endswith(".py")} | support)
        visited = set(frontier)
        while frontier:
            next_frontier: set[str] = set()
            for rel in frontier:
                targets, missing = resolved.get(rel, (set(), set()))
                unresolved |= missing
                for target in targets - visited:
                    visited.add(target)
                    if target not in own_set:
                        support.add(target)
                    next_frontier.add(target)
            frontier = sorted(next_frontier)
        buckets.append(
            SuiteBucket(
                key=key,
                directory=directory,
                own_files=tuple(own),
                support_files=tuple(sorted(support)),
                unresolved_imports=tuple(sorted(unresolved)),
            )
        )
    return tuple(buckets)
