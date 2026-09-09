# tests/config/test_type_ignore_and_typing_hygiene.py
"""
Python engineering-baseline hygiene test for `riskquant.config`
(RQ-AUD-021, LOW).

`.github/copilot-instructions.md` §16 states plainly: "no `# type: ignore`".
`static_config/compile.py::_int` and `::_opt_int` each suppress a real
mypy `arg-type` error on `int(v)` where `v: object` -- the raw materialised
Excel-cell value. This is a legitimate but explicitly-prohibited pattern;
the root cause is that the cell-materialisation boundary uses the bare
`object` type rather than a narrower `int | float | str | None` union or a
`SupportsInt` protocol that would let `int(v)` type-check without a
suppression.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src" / "riskquant" / "config"


def test_no_type_ignore_comments():
    """EXPECTED TO FAIL TODAY (2 occurrences in static_config/compile.py)."""
    offenders = []
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(source.splitlines(), 1):
            if "# type: ignore" in line:
                offenders.append(f"{path.relative_to(CONFIG_ROOT)}:{lineno}")
    assert not offenders, (
        "riskquant.config contains '# type: ignore' suppressions, prohibited "
        f"by copilot-instructions.md §16:\n" + "\n".join(f"  {o}" for o in offenders)
    )
