# _context_gen/fs.py
"""
Repository root resolution and deterministic file discovery.

No absolute machine-specific paths anywhere in this module. Every path is
derived from `repo_root()` at call time, and every path this package emits
is normalised to repository-relative POSIX form before it leaves this
module's boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES: dict[str, str] = {
    ".py": "python",
    ".md": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".txt": "text",
}

# Directories never swept, regardless of which profile is being resolved.
# This governs sweeps only; explicit single-file inclusion (e.g. pyproject.toml)
# is unaffected because it never goes through rglob.
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".idea",
        ".venv",
        ".env",
        "venv",
        ".pytest_cache",
        ".ruff_cache",
        "ruff_cache",
        "_deprecated",
        ".deprecated",
        "_context_gen",  # this tool's own package — never a content source
        "_contexts",  # this tool's own output directory
    }
)


def repo_root(start: Path | None = None) -> Path:
    """
    Resolve the repository root.

    The generator script lives at the repository root (confirmed: the
    previous generator's own first-line path comment, `# src/riskquant/
    _create_master_context.py`, was stale — the file actually lives at
    `<repo_root>/_create_master_context.py`). Root is therefore the parent
    of this package's own location, with no machine-specific fallback.
    """
    here = (start or Path(__file__)).resolve()
    # _context_gen/fs.py -> _context_gen -> repo root
    return here.parent.parent if here.name == "fs.py" else here.parent


def to_repo_relative_posix(path: Path, root: Path) -> str:
    """Normalise an absolute path to repository-relative POSIX form."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts) or any(
        part.startswith(".") and part not in {".", ".."} for part in path.parts
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DiscoveredFile:
    """A single file found during a sweep, with its repo-relative path and kind."""

    repo_relative: str  # POSIX, relative to repo root
    absolute: Path
    kind: str  # one of TEXT_SUFFIXES.values(), or "unknown"


def sweep(
    root_dir: Path,
    repo_root_dir: Path,
    suffixes: frozenset[str] = frozenset({".py"}),
) -> list[DiscoveredFile]:
    """
    Recursively discover files under `root_dir` whose suffix is in `suffixes`.

    Deterministic: results are sorted by repo-relative POSIX path. Directories
    in EXCLUDED_DIR_NAMES, and any dotfile/dotdir, are never descended into.
    Missing `root_dir` returns an empty list — the caller (completeness.py)
    is responsible for treating "configured root does not exist" as an
    explicit failure, not this function silently degrading.
    """
    if not root_dir.exists():
        return []

    found: list[DiscoveredFile] = []
    for candidate in root_dir.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix not in suffixes:
            continue
        if _is_excluded(candidate.relative_to(repo_root_dir)):
            continue
        rel = to_repo_relative_posix(candidate, repo_root_dir)
        kind = TEXT_SUFFIXES.get(candidate.suffix, "text")
        found.append(DiscoveredFile(repo_relative=rel, absolute=candidate, kind=kind))

    found.sort(key=lambda f: f.repo_relative)
    return found


def single_file(path: Path, repo_root_dir: Path) -> DiscoveredFile | None:
    """Resolve one explicitly-named file (e.g. pyproject.toml). None if absent."""
    if not path.exists() or not path.is_file():
        return None
    rel = to_repo_relative_posix(path, repo_root_dir)
    kind = TEXT_SUFFIXES.get(path.suffix, "text")
    return DiscoveredFile(repo_relative=rel, absolute=path, kind=kind)
