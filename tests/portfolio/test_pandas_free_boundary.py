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
4. The production path needs no pandas: the pipeline imports, and a
   representative Polars ingest runs, in a process where ``import pandas``
   fails.
5. ``IngestBatch.to_df`` / ``.df`` is the one optional pandas surface (D096):
   in that same process it raises ImportError; nothing else does.
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

INGEST_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "riskquant" / "portfolio" / "ingest"
)

# Any pandas import statement, at any indentation.
_PANDAS_IMPORT_RE = re.compile(r"^\s*(?:import\s+pandas\b|from\s+pandas\b)")

# D096: the only permitted occurrences (file, stripped line).
_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    ("batch.py", "from pandas import DataFrame")
})
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


_BLOCKED_PANDAS_CHILD = """
import datetime as dt
import sys

import polars as pl

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_static_config import (
    canonical_risk_static_config as canonical,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

rsc = compile_risk_static_config_from_dict_of_dfs(
    canonical.CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
)
pillar = "BRENT_CRUDE_FWD[M+0]"
frame = pl.from_dict(
    {
        "uid": [1, 2, 3],
        "instrument": ["SPOT", "VANILLA_EUROPEAN_OPTION_NODE", "VANILLA_FORWARD_NODE"],
        "quantity": [" 100 ", "1000", "1000"],
        "underlying": ["AAPL_EQ", pillar, pillar],
        "settlement_ccy": ["usd", "USD", "USD"],
        "strike_price": [None, 70.0, 70.0],
        "strike_ccy": [None, "USD", "USD"],
        "is_call": [None, 1.0, None],
        "expiry_date": [None, "2026-02-18", None],
        "settlement_date": [None, "2026-02-20", "2026-2-17"],
        "delivery_date": [None, None, "2026-02-15"],
        "volatility_container_name": [None, "BRENT_CRUDE_ATM_VOL", None],
        "under_discount_curve": [None, "USD_SOFR_OIS", "USD_SOFR_OIS"],
        "strike_discount_curve": [None, "USD_SOFR_OIS", "USD_SOFR_OIS"],
        "netting_set_id": [7, None, 9],
    }
)
batch = compile_raw_df_into_ingest_batch(
    raw_df=frame, static_config=rsc, as_of_date=dt.date(2026, 1, 15)
)
assert batch.size == 3
assert batch.netting_set_id.tolist() == [7, -999_999, 9]
assert "pandas" not in sys.modules or sys.modules["pandas"] is None
try:
    batch.to_df()
except ImportError:
    print("OK to_df needs pandas")
else:
    raise SystemExit("to_df ran without pandas")
"""


def _run_without_pandas(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    blocker = tmp_path / "pandas"
    blocker.mkdir()
    (blocker / "__init__.py").write_text('raise ImportError("pandas blocked")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def test_ingest_pipeline_imports_without_pandas(tmp_path: Path) -> None:
    proc = _run_without_pandas(
        tmp_path, "import riskquant.portfolio.ingest.pipeline  # noqa: F401\n"
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_ingest_executes_without_pandas_and_to_df_is_the_only_pandas_surface(
    tmp_path: Path,
) -> None:
    proc = _run_without_pandas(tmp_path, textwrap.dedent(_BLOCKED_PANDAS_CHILD))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK to_df needs pandas" in proc.stdout
