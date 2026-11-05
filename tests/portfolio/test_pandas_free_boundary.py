# tests/portfolio/test_pandas_free_boundary.py
"""
Pandas boundary of ``riskquant.portfolio.ingest`` (D094; D096 superseded by
P-CREDIT_POLARS-001 clause 4, WS_CREDIT_POLARS C.14).

What this file proves, and what it does not
-------------------------------------------
1. No module in ``portfolio/ingest`` imports pandas, at any indentation.
2. The public entry annotation resolves to ``polars.DataFrame``.
3. ``copy`` is not a parameter (D095).
4. The production path needs no pandas: the pipeline imports, and a
   representative Polars ingest runs, in a process where ``import pandas``
   fails.
5. ``IngestBatch.to_df`` / ``.df`` return ``pl.DataFrame`` in that same
   process (S2.T2), with the v0.2.0 columns, one dtype per column and null
   at each SoA sentinel.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast, get_type_hints

import numpy as np
import polars as pl
import pytest

from riskquant.constants import NO_DATE_ORD32, NO_FX_RF_REQUIRED, NO_ID, NOT_OPTION_MULT

INGEST_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "riskquant" / "portfolio" / "ingest"
)

# Any pandas import statement, at any indentation.
_PANDAS_IMPORT_RE = re.compile(r"^\s*(?:import\s+pandas\b|from\s+pandas\b)")

# D096 is superseded (C.14): batch.py has no pandas import, so nothing is permitted.
_ALLOWED: frozenset[tuple[str, str]] = frozenset()
_ALLOWED_COUNT: dict[str, int] = {}


def _pandas_import_lines() -> dict[str, list[tuple[int, str]]]:
    found: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(INGEST_ROOT.rglob("*.py")):
        rel = path.relative_to(INGEST_ROOT).as_posix()
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _PANDAS_IMPORT_RE.match(line):
                found.setdefault(rel, []).append((i, line.strip()))
    return found


def test_portfolio_ingest_has_no_pandas_import() -> None:
    found = _pandas_import_lines()
    bad = {
        rel: hits
        for rel, hits in found.items()
        if any((rel, text) not in _ALLOWED for _, text in hits)
        or len(hits) > _ALLOWED_COUNT.get(rel, 0)
    }
    assert not bad, f"pandas imports in portfolio/ingest: {bad}"


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

import dataclasses
import datetime as dt
from typing import Any

import numpy as np
import polars as pl
import pytest

from riskquant.constants import NO_DATE_ORD32, NO_FX_RF_REQUIRED, NO_ID, NOT_OPTION_MULT

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
for debug_frame in (batch.to_df(), batch.df):
    assert isinstance(debug_frame, pl.DataFrame)
    assert debug_frame.height == 3
assert "pandas" not in sys.modules or sys.modules["pandas"] is None
print("OK to_df needs no pandas")
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


def test_ingest_executes_without_pandas_and_to_df_needs_no_pandas(
    tmp_path: Path,
) -> None:
    proc = _run_without_pandas(tmp_path, textwrap.dedent(_BLOCKED_PANDAS_CHILD))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK to_df needs no pandas" in proc.stdout


# ---------------------------------------------------------------------------
# S2.T2: IngestBatch.to_df / .df dtypes and sentinel nulls (C.14)
# ---------------------------------------------------------------------------

# Column -> (SoA field, Polars dtype, SoA sentinel or None). The column names
# and their order are the v0.2.0 adapter columns.
_TO_DF_CONTRACT: dict[str, tuple[str, pl.DataType, float | None]] = {
    "uid": ("uids", pl.Int64(), None),
    "inst_id": ("instrument_ids", pl.Int32(), None),
    "qty": ("qty", pl.Float64(), None),
    "k": ("strike_price", pl.Float64(), 0.0),
    "cp_mult": ("cp_mult", pl.Float64(), float(NOT_OPTION_MULT)),
    "und_ids": ("underlying_ids", pl.Int64(), float(NO_ID)),
    "vol_ids": ("volatility_ids", pl.Int64(), float(NO_ID)),
    "fx_u2s_ids": ("fx_u2s_ids", pl.Int64(), float(NO_FX_RF_REQUIRED)),
    "fx_u2s_inv": ("fx_u2s_inv", pl.Int8(), None),
    "fx_k2s_ids": ("fx_k2s_ids", pl.Int64(), float(NO_FX_RF_REQUIRED)),
    "fx_k2s_inv": ("fx_k2s_inv", pl.Int8(), None),
    "disc_und_id": ("under_discount_curve_ids", pl.Int64(), float(NO_ID)),
    "disc_k_id": ("strike_discount_curve_ids", pl.Int64(), float(NO_ID)),
    "settle_date": ("settle_ord", pl.Date(), float(NO_DATE_ORD32)),
    "delivery_date": ("delivery_ord", pl.Date(), float(NO_DATE_ORD32)),
    "expiry_date": ("expiry_ord", pl.Date(), float(NO_DATE_ORD32)),
    "vol_surface_y_axis_type_id": ("vol_surface_y_axis_type_ids", pl.Int8(), None),
    "netting_set_id": ("netting_set_id", pl.Int64(), float(NO_ID)),
}
_SENTINEL_COLUMNS = [c for c, (_, _, s) in _TO_DF_CONTRACT.items() if s is not None]
_EPOCH = dt.date(1970, 1, 1)


def _canonical_spot_frame() -> pl.DataFrame:
    """
    The canonical SPOT fixture as a Polars frame, minus two rows of input.

    Two defects outside WS_CREDIT_POLARS S2 stop the full fixture at ingest
    (S2 report): an all-null ``strike_ccy`` column truncates the fallback
    currency to one character (``compile.py``), and uid 11
    (``USD_LIBOR_3M[10Y]``) is refused by D063. An absent ``strike_ccy``
    means the same as an all-null one (fallback to the underlying currency).
    """
    from riskquant.ingest_fixtures.portfolio.spot import (
        CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA,
    )

    return (
        pl
        .from_dict(cast(dict[str, Any], CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA))
        .drop("strike_ccy")
        .filter(pl.col("uid") != 11)
    )


def _node_frame() -> pl.DataFrame:
    """Option and forward rows: strike, cp_mult, vol id, curves and dates."""
    pillar = "BRENT_CRUDE_FWD[M+0]"
    return pl.from_dict({
        "uid": [21, 22],
        "instrument": ["VANILLA_EUROPEAN_OPTION_NODE", "VANILLA_FORWARD_NODE"],
        "quantity": [1000.0, -500.0],
        "underlying": [pillar, pillar],
        "settlement_ccy": ["USD", "USD"],
        "strike_price": [70.0, 71.5],
        "strike_ccy": ["USD", "USD"],
        "is_call": [1.0, None],
        "expiry_date": [dt.date(2026, 2, 18), None],
        "settlement_date": [dt.date(2026, 2, 20), dt.date(2026, 2, 17)],
        "delivery_date": [None, dt.date(2026, 2, 15)],
        "volatility_container_name": ["BRENT_CRUDE_ATM_VOL", None],
        "under_discount_curve": ["USD_SOFR_OIS", "USD_SOFR_OIS"],
        "strike_discount_curve": ["USD_SOFR_OIS", "USD_SOFR_OIS"],
        "netting_set_id": [None, 7],
    })


@pytest.fixture(scope="module")
def debug_batches() -> dict[str, Any]:
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
    batches = {
        name: compile_raw_df_into_ingest_batch(
            raw_df=frame, static_config=rsc, as_of_date=dt.date(2026, 1, 15)
        )
        for name, frame in (("spot", _canonical_spot_frame()), ("node", _node_frame()))
    }
    batches["sentinel_row"] = _with_sentinels_in_row_0(batches["node"])
    return batches


def _with_sentinels_in_row_0(batch: Any) -> Any:
    """
    The node batch with every sentinel field at its sentinel in row 0.

    A valid trade always has an underlying, so ``und_ids`` is never null on
    an ingested book. This SoA-level batch puts each sentinel in row 0 and
    keeps the ingested values in row 1.
    """
    changes: dict[str, np.ndarray] = {}
    for field, _dtype, sentinel in _TO_DF_CONTRACT.values():
        if sentinel is None:
            continue
        arr = np.array(getattr(batch, field), copy=True)
        arr[0] = sentinel
        changes[field] = arr
    return dataclasses.replace(batch, **changes)


def _expected_value(column: str, raw: Any) -> Any:
    if _TO_DF_CONTRACT[column][1] == pl.Date():
        return _EPOCH + dt.timedelta(days=int(raw))
    return raw.item()


@pytest.mark.parametrize("accessor", ["to_df", "df"])
@pytest.mark.parametrize("book", ["spot", "node", "sentinel_row"])
def test_s2_t2_to_df_columns_dtypes_and_sentinel_nulls(
    debug_batches: dict[str, Any], book: str, accessor: str
) -> None:
    batch = debug_batches[book]
    frame = batch.to_df() if accessor == "to_df" else batch.df

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == list(_TO_DF_CONTRACT)
    assert frame.height == batch.size
    for column, (field, dtype, sentinel) in _TO_DF_CONTRACT.items():
        series = frame.get_column(column)
        assert series.dtype == dtype, (column, series.dtype)
        soa = getattr(batch, field)
        is_sentinel = (
            np.zeros(batch.size, dtype=bool) if sentinel is None else soa == sentinel
        )
        assert series.is_null().to_list() == is_sentinel.tolist(), column
        expected = [
            None if hit else _expected_value(column, raw)
            for raw, hit in zip(soa, is_sentinel, strict=True)
        ]
        assert series.to_list() == expected, column


def test_s2_t2_books_cover_null_and_value_in_every_sentinel_column(
    debug_batches: dict[str, Any],
) -> None:
    frames = [batch.to_df() for batch in debug_batches.values()]
    for column in _SENTINEL_COLUMNS:
        nulls = sum(f.get_column(column).null_count() for f in frames)
        rows = sum(f.height for f in frames)
        assert 0 < nulls < rows, (column, nulls, rows)


def test_s2_t2_to_df_copy_is_independent_of_the_soa(
    debug_batches: dict[str, Any],
) -> None:
    batch = debug_batches["node"]
    frame = batch.to_df()
    original = int(batch.uids[0])
    batch.uids[0] = original + 1_000
    try:
        assert frame.get_column("uid")[0] == original
    finally:
        batch.uids[0] = original
