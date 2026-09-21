# tests/test_claude_md_contract.py
"""
Mechanical guard: ``CLAUDE.md`` and ``.claude/rules/*.md`` stay true to the repo.

Instruction files drift silently. A rule that cites a superseded decision or a
deleted test keeps steering the agent after the code has moved on. This module
turns every citation into a checked reference.

Checks
------
1. Every decision ID cited exists in ``DECISIONS.md``.
2. A superseded ID appears only on a line that says so ("supersed...").
3. Every backticked ``src/``, ``tests/`` or ``scripts/`` path exists.
4. Every ``path.py::name`` test node names a function defined in that file.
5. Every rules file declares ``paths`` frontmatter, and each glob matches a file.
6. The root ``CLAUDE.md`` stays at or under the 200-line adherence budget.

Assumed ``DECISIONS.md`` format (adjust the two heading patterns if it differs)
-----------------------------------------------------------------------------
- Each entry opens with a Markdown heading whose first token is its ID,
  for example ``## D081 — Ordinary mark-to-market reporting P&L``.
- A superseding entry says ``supersedes D014`` or ``superseding D014``
  (ranges such as ``D042-D044`` expand); or a superseded entry carries
  ``superseded`` in its own heading line.
"""

from __future__ import annotations

import ast
import glob
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
RULES_DIR = REPO_ROOT / ".claude" / "rules"
DECISIONS_MD = REPO_ROOT / "DECISIONS.md"

#: Claude Code guidance: adherence drops above ~200 lines per CLAUDE.md file.
MAX_ROOT_LINES = 200

_DECISION_ID = re.compile(r"\bD\d{3}\b")
_DECISION_RANGE = re.compile(r"\bD(\d{3})\s*[-–]\s*D(\d{3})\b")
_DECISION_HEADING = re.compile(r"^#{1,6}[ \t*]*(D\d{3})\b.*$", re.MULTILINE)
_SUPERSEDES = re.compile(
    r"\bsupersed(?:es|ing)\s+(D\d{3}(?:\s*(?:,|/|-|–|and)\s*D\d{3})*)",
    re.IGNORECASE,
)
_CITED_PATH = re.compile(r"`((?:src|tests|scripts)/[\w./-]+?\.py)(?:::(\w+))?`")


def _instruction_files() -> list[Path]:
    files = [CLAUDE_MD, *sorted(RULES_DIR.rglob("*.md"))]
    missing = [path for path in files if not path.is_file()]
    assert not missing, f"missing instruction files: {missing}"
    return files


def _expand_ids(text: str) -> set[str]:
    """Return every decision ID in ``text``, with ``D042-D044`` expanded."""
    ids = set(_DECISION_ID.findall(text))
    for low, high in _DECISION_RANGE.findall(text):
        ids.update(f"D{number:03d}" for number in range(int(low), int(high) + 1))
    return ids


def _decision_register() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(known_ids, superseded_ids)`` parsed from ``DECISIONS.md``."""
    assert DECISIONS_MD.is_file(), (
        "DECISIONS.md is the decision authority; it is missing"
    )
    text = DECISIONS_MD.read_text(encoding="utf-8")
    headings = list(_DECISION_HEADING.finditer(text))
    assert headings, "no decision headings parsed; update _DECISION_HEADING"

    known = {match.group(1) for match in headings}
    superseded = {
        match.group(1) for match in headings if "supersed" in match.group(0).lower()
    }
    for clause in _SUPERSEDES.findall(text):
        superseded |= _expand_ids(clause)
    return frozenset(known), frozenset(superseded)


def _function_names(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _frontmatter_paths(path: Path) -> list[str]:
    """Parse the ``paths`` list from YAML frontmatter without a YAML dependency."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0] == "---", f"{path.name}: frontmatter must open line 1"
    end = lines.index("---", 1)
    patterns: list[str] = []
    in_paths = False
    for line in lines[1:end]:
        if line.startswith("paths:"):
            in_paths = True
        elif in_paths and line.lstrip().startswith("- "):
            patterns.append(line.lstrip()[2:].strip().strip("\"'"))
        elif line.strip():
            in_paths = False
    return patterns


@pytest.mark.parametrize("path", _instruction_files(), ids=lambda p: p.name)
def test_cited_decisions_exist_and_are_live(path: Path) -> None:
    known, superseded = _decision_register()
    problems: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for decision in sorted(_expand_ids(line)):
            if decision not in known:
                problems.append(f"line {number}: {decision} is not in DECISIONS.md")
            elif decision in superseded and "supersed" not in line.lower():
                problems.append(
                    f"line {number}: {decision} is superseded but cited as live"
                )
    assert not problems, f"{path.name}:\n" + "\n".join(problems)


@pytest.mark.parametrize("path", _instruction_files(), ids=lambda p: p.name)
def test_cited_paths_and_test_nodes_exist(path: Path) -> None:
    problems: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for relative_path, node in _CITED_PATH.findall(line):
            target = REPO_ROOT / relative_path
            if not target.is_file():
                problems.append(f"line {number}: {relative_path} does not exist")
            elif node and node not in _function_names(target):
                problems.append(f"line {number}: {relative_path}::{node} not defined")
    assert not problems, f"{path.name}:\n" + "\n".join(problems)


@pytest.mark.parametrize("path", sorted(RULES_DIR.rglob("*.md")), ids=lambda p: p.name)
def test_every_rule_scope_matches_a_file(path: Path) -> None:
    patterns = _frontmatter_paths(path)
    assert patterns, f"{path.name}: declare `paths` so the rule loads on demand"
    dead = [
        pattern
        for pattern in patterns
        if not glob.glob(pattern, root_dir=REPO_ROOT, recursive=True)
    ]
    assert not dead, f"{path.name}: globs match nothing (stale after a move?): {dead}"


def test_root_claude_md_fits_the_adherence_budget() -> None:
    line_count = len(CLAUDE_MD.read_text(encoding="utf-8").splitlines())
    assert line_count <= MAX_ROOT_LINES, (
        f"CLAUDE.md has {line_count} lines; move detail into .claude/rules/"
    )
