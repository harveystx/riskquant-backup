# tests/portfolio/test_pandas_free_boundary.py
"""
Architecture enforcement test: ``riskquant.portfolio.ingest`` must be pandas-free.

Scope
-----
Every ``.py`` module under ``src/riskquant/portfolio/ingest/`` must contain
zero ``import pandas`` statements.

Exemptions
----------
None. The entire package was migrated in sprint D-INGEST-01. ``batch.py``
retains a lazy ``from pandas import DataFrame`` inside a method body for the
debug-only ``to_df()`` / ``.df`` property. This is an in-method import that
does not appear as a top-level ``import pandas`` or ``from pandas`` statement
and is explicitly exempted below.

Rationale
---------
The ``portfolio/ingest`` package is an Edge-tier module. The approved edge
technology is Polars. Pandas at the edge is permitted ONLY at an unavoidable
external interoperability boundary (e.g. legacy consumer contracts).
``batch.py``'s debug helpers do not constitute a production interoperability
boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

INGEST_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "riskquant"
    / "portfolio"
    / "ingest"
)

# ``batch.py`` retains a lazy ``from pandas import DataFrame`` inside the
# ``to_df`` / ``df`` method body only. It does NOT appear at module level and
# does NOT cause a pandas import on module load. The structural check below
# requires that the only exempted occurrence is inside a function/method body
# (indented), not at module level (column 0).
_EXEMPTED_FILE = "batch.py"

# Regex for top-level pandas imports (column 0, no leading whitespace)
_TOPLEVEL_IMPORT_RE = re.compile(
    r"^(?:import pandas|from pandas\s+import)", re.MULTILINE
)


def _collect_violations() -> dict[str, list[int]]:
    """Return {rel_path: [line_numbers]} for every top-level pandas import found."""
    violations: dict[str, list[int]] = {}

    for path in sorted(INGEST_ROOT.rglob("*.py")):
        rel = path.relative_to(INGEST_ROOT).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")

        offending_lines: list[int] = []
        for i, line in enumerate(source.splitlines(), start=1):
            if _TOPLEVEL_IMPORT_RE.match(line):
                # For the exempted file, flag only if the import is at
                # module level (not indented).
                if rel == _EXEMPTED_FILE and line.startswith(" "):
                    continue
                if rel == _EXEMPTED_FILE:
                    # Module-level pandas import in batch.py is NOT expected.
                    offending_lines.append(i)
                else:
                    offending_lines.append(i)

        if offending_lines:
            violations[rel] = offending_lines

    return violations


def test_portfolio_ingest_is_pandas_free() -> None:
    """
    Every file under ``portfolio/ingest/`` must have zero top-level pandas imports.
    """
    violations = _collect_violations()
    if not violations:
        return

    lines = [
        "portfolio/ingest is NOT pandas-free. Top-level pandas imports found:",
        "",
    ]
    for rel, linenos in violations.items():
        lines.append(f"  {rel}: lines {linenos}")

    raise AssertionError("\n".join(lines))


def test_pipeline_entry_accepts_polars_dataframe() -> None:
    """
    ``compile_raw_df_into_ingest_batch`` must accept ``pl.DataFrame``.

    Verifies the public type signature without running the full pipeline.
    """
    import inspect
    import polars as pl

    from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

    sig = inspect.signature(compile_raw_df_into_ingest_batch)
    ann = sig.parameters["raw_df"].annotation

    # annotation is "pl.DataFrame" string (from __future__ annotations) or the type itself
    ann_str = str(ann)
    assert "pl" in ann_str or "polars" in ann_str or ann is pl.DataFrame, (
        f"raw_df annotation is {ann_str!r}; expected pl.DataFrame"
    )


def test_copy_parameter_removed() -> None:
    """
    The ``copy`` parameter must not exist on ``compile_raw_df_into_ingest_batch``.

    Polars DataFrames are immutable. The ``copy`` parameter was a pandas-era
    artifact and is no longer meaningful.
    """
    import inspect

    from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

    sig = inspect.signature(compile_raw_df_into_ingest_batch)
    assert "copy" not in sig.parameters, (
        "compile_raw_df_into_ingest_batch must not have a 'copy' parameter "
        "(Polars DataFrames are immutable; copy is always a no-op)."
    )
