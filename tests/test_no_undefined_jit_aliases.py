# tests/test_no_undefined_jit_aliases.py
"""
Repo-wide guard (RQ-AUD-001): no module may import a name from `riskquant.jit`
that `jit.py` does not actually define.

This is the family-level regression test: it would have caught the original
defect (and would catch any future reintroduction of a layout-suffix-less
alias, e.g. `_F64_2D_RO` instead of `_F64_2D_C_RO`) anywhere in the source
tree, not just in the two files that happened to be audited.
"""
from __future__ import annotations

import re
from pathlib import Path

import riskquant.jit as jit_module

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "riskquant"

_BLOCK_IMPORT = re.compile(r"from riskquant\.jit import \(([^)]+)\)", re.DOTALL)
_LINE_IMPORT = re.compile(r"from riskquant\.jit import ([^\n(]+)")


def _imported_names(source: str) -> set[str]:
    names: set[str] = set()
    for m in _BLOCK_IMPORT.finditer(source):
        for token in m.group(1).split("\n"):
            token = token.split("#", 1)[0].strip().rstrip(",").strip()
            if token:
                names.add(token)
    # Only single-line (non-parenthesised) imports; strip trailing comments
    # before splitting on commas so `# type: ignore[...]` isn't mistaken for
    # part of a name.
    for m in _LINE_IMPORT.finditer(source):
        whole = m.group(0)
        if "(" in whole:
            continue
        clause = m.group(1).split("#", 1)[0]
        for token in clause.split(","):
            token = token.strip()
            if token:
                names.add(token)
    return names


def test_no_undefined_jit_aliases_anywhere_in_src():
    defined = set(dir(jit_module))
    offenders: dict[str, list[str]] = {}

    for path in SRC_ROOT.rglob("*.py"):
        if path.name == "jit.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "from riskquant.jit import" not in source:
            continue
        missing = sorted(n for n in _imported_names(source) if n not in defined)
        if missing:
            offenders[str(path.relative_to(SRC_ROOT))] = missing

    assert not offenders, (
        "Undefined riskquant.jit imports found:\n"
        + "\n".join(f"  {f}: {ms}" for f, ms in sorted(offenders.items()))
    )
