# tests/test_no_stale_path_comments.py
"""
Repo-wide guard (RQ-AUD-006): every non-``__init__.py`` module's first line
must be the exact ``# src/riskquant/<relative/path>.py`` comment, and every
``__init__.py`` must be empty.

Both checks already exist in ``tests/test_architecture.py``
(``test_every_module_declares_its_repository_relative_path`` and
``test_every_init_file_is_empty``); this file is a standalone, minimal
regression test scoped to exactly the RQ-AUD-006 finding, kept independent of
the broader architecture suite so it can be run in isolation and makes the
specific defect class explicit: refactors that move or rename a module
(e.g. ``portfolio/blocks/`` -> ``valuation/blocks/``, or a kernel file rename)
routinely leave the OLD path in the header comment behind. 28 such files were
found stale after the most recent refactor, plus one file with no path
comment at all and one non-empty ``__init__.py``.
"""
from __future__ import annotations

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "riskquant"


def test_every_init_file_is_empty():
    init_files = list(SRC_ROOT.rglob("__init__.py"))
    assert init_files, "expected at least one __init__.py under src/riskquant"
    non_empty = [p for p in init_files if p.stat().st_size > 0]
    assert not non_empty, (
        "__init__.py files must be empty (strings/logic live in named "
        "modules, not package initialisers):\n"
        + "\n".join(f"  {p.relative_to(SRC_ROOT)} ({p.stat().st_size} bytes)" for p in non_empty)
    )


def test_every_module_declares_its_repository_relative_path():
    offenders: list[tuple[str, str, str]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            continue
        expected = f"# src/riskquant/{path.relative_to(SRC_ROOT).as_posix()}"
        if lines[0] != expected:
            offenders.append((str(path.relative_to(SRC_ROOT)), lines[0], expected))

    assert not offenders, (
        "Modules with an incorrect or missing first-line path comment "
        "(commonly left stale by a file move/rename):\n"
        + "\n".join(f"  {f}\n    got:      {g!r}\n    expected: {e!r}" for f, g, e in offenders)
    )
