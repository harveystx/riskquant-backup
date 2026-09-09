# tests/test_no_local_numba_aliases.py
"""
Repo-wide guard (RQ-AUD-005): no module may define its own Numba array type
alias — every array type used in a kernel signature must be imported from
`riskquant.jit`, the single source of truth.

This is the family-level regression test: it would have caught the original
defect (12 files with local `_F64_2D`, `_I32_1D`, `_I8_1D`, etc. redefinitions)
anywhere in the source tree, not just in the files the audit happened to
inspect, and it prevents silent drift if `jit.py`'s canonical definitions
ever change without every local copy being updated in lockstep.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "riskquant"

_LOCAL_ALIAS_DEF = re.compile(
    r"^(_[A-Z0-9]+_[0-9]D(?:_[CF])?(?:_RO)?)\s*=\s*nb\.",
    re.MULTILINE,
)


def test_no_local_numba_array_alias_definitions():
    offenders: dict[str, list[str]] = {}

    for path in SRC_ROOT.rglob("*.py"):
        if path.name == "jit.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        found = sorted(set(_LOCAL_ALIAS_DEF.findall(source)))
        if found:
            offenders[str(path.relative_to(SRC_ROOT))] = found

    assert not offenders, (
        "Local Numba array type alias definitions found (must import the "
        "canonical alias from riskquant.jit instead):\n"
        + "\n".join(f"  {f}: {a}" for f, a in sorted(offenders.items()))
    )
