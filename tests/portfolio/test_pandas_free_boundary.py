# tests/portfolio/test_pandas_free_boundary.py
"""
Pandas boundary of ``riskquant.portfolio.ingest`` (D094, D096).

What this file proves, and what it does not
-------------------------------------------
1. No module in ``portfolio/ingest`` imports pandas, at any indentation,
   except the two ``batch.py`` lines that D096 keeps: the TYPE_CHECKING
   import and the local import inside the debug-only ``IngestBatch.to_df``.
2. The public entry annotation resolves to ``polars.DataFrame``.
3. ``copy`` is not a parameter (D095).
4. The package is NOT importable without pandas today. ``riskquant.utils.dates``
   imports pandas at module level, and ``parser``, ``compile`` and ``batch``
   import that module. That dependency is outside the migration scope. The
   strict xfail below records it: it turns into a failure when the dependency
   is removed, so the marker cannot outlive the defect.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import get_type_hints

import polars as pl
import pytest

INGEST_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "riskquant" / "portfolio" / "ingest"
)

# Any pandas import statement, at any indentation.
_PANDAS_IMPORT_RE = re.compile(r"^\s*(?:import\s+pandas\b|from\s+pandas\b)")

# D096: the only permitted occurrences (file, stripped line).
_ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {("batch.py", "from pandas import DataFrame")}
)
_ALLOWED_COUNT = {"batch.py": 2}  # TYPE_CHECKING import + to_df local import


def _pandas_import_lines() -> dict[str, list[tuple[int, str]]]:
    found: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(INGEST_ROOT.rglob("*.py")):
        rel = path.relative_to(INGEST_ROOT).as_posix()
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _PANDAS_IMPORT_RE.match(line):
                found.setdefault(rel, []).append((i, line.strip()))
    return found


def test_portfolio_ingest_has_no_pandas_import_except_d096() -> None:
    found = _pandas_import_lines()
    bad = {
        rel: hits
        for rel, hits in found.items()
        if any((rel, text) not in _ALLOWED for _, text in hits)
        or len(hits) > _ALLOWED_COUNT.get(rel, 0)
    }
    assert not bad, f"pandas imports outside D096 allowance: {bad}"


def test_pipeline_entry_annotation_is_polars_dataframe() -> None:
    from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

    hints = get_type_hints(compile_raw_df_into_ingest_batch)
    assert hints["raw_df"] is pl.DataFrame


def test_copy_parameter_removed() -> None:
    import inspect

    from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

    assert "copy" not in inspect.signature(compile_raw_df_into_ingest_batch).parameters


@pytest.mark.xfail(
    strict=True,
    reason=(
        "riskquant/utils/dates.py imports pandas at module level; parser.py, "
        "compile.py and batch.py import it. Outside the ingest migration scope."
    ),
)
def test_ingest_pipeline_imports_without_pandas(tmp_path: Path) -> None:
    blocker = tmp_path / "pandas"
    blocker.mkdir()
    (blocker / "__init__.py").write_text('raise ImportError("pandas blocked")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
    code = textwrap.dedent(
        """
        import riskquant.portfolio.ingest.pipeline  # noqa: F401
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
