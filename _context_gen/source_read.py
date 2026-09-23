# _context_gen/source_read.py
"""
Source reading with explicit error classification.

Replaces the previous generator's `except Exception: return empty_summary`.
A read failure, a non-Python file, and a genuine Python syntax error are
three different situations and must be reported as three different things —
never silently collapsed into "no symbols found."
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ReadOutcome(Enum):
    OK = "ok"
    UNREADABLE = "unreadable"  # OS/encoding failure reading the file at all
    NOT_PYTHON = "not_python"  # file kind is not .py — parsing is not attempted
    PARSE_FAILED = "parse_failed"  # .py file, read fine, ast.parse raised


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadResult:
    outcome: ReadOutcome
    text: str | None  # exact file content, unmodified, if OUTCOME allows it
    tree: ast.Module | None
    error: str | None  # human-readable reason, only set for UNREADABLE/PARSE_FAILED


def read_source(path: Path, kind: str) -> ReadResult:
    """
    Read one file and, if it is Python, parse it.

    `kind` comes from fs.DiscoveredFile.kind — it is never guessed here.
    A `.py` file that fails to parse is PARSE_FAILED, not NOT_PYTHON: this
    distinction is exactly what the previous bare `except Exception` erased,
    and it is a P0-class silent-failure risk for genuinely broken production
    source (§31).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ReadResult(outcome=ReadOutcome.UNREADABLE, text=None, tree=None, error=str(exc))

    if kind != "python":
        return ReadResult(outcome=ReadOutcome.NOT_PYTHON, text=text, tree=None, error=None)

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return ReadResult(
            outcome=ReadOutcome.PARSE_FAILED,
            text=text,
            tree=None,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    return ReadResult(outcome=ReadOutcome.OK, text=text, tree=tree, error=None)
