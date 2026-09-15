# src/riskquant/config/static_config/loader.py
"""
Controlled-schema xlsx loader for the static-config workbook.

Pipeline position:

    XLSX
      -> controlled-schema read (explicit dtypes, no dtype inference)
      -> normalize (integrality gate + compact Int32 cast for roll fields)
      -> dict[str, pl.DataFrame]
      -> validate_static_config_dfs
      -> compile_...

The loader does not perform semantic validation; that lives in
`validate_static_config_dfs`. It DOES enforce the ingest dtype contract so
Excel dtype inference is never a hidden semantic layer:

  * The nullable roll columns (anchor_offset, roll_month, roll_day) are read
    as Float64 rather than left to inference. A mostly-null column otherwise
    makes calamine emit "could not determine dtype ... falling back to
    string". Reading them as Float64 also preserves a non-integral value
    (e.g. 1.5) so the integrality gate can reject it -- an Int cast at read
    time, or a strict Polars cast, both TRUNCATE 1.5 to 1 silently.
  * fh_bh_mmdd is read as String.
  * `normalize_static_config_dfs` then verifies every numeric roll value is
    integral and casts the columns to Int32. A non-integral value fails
    fast with a table/column/value message; it is never truncated.

The in-memory fixture path (which already carries integer roll columns) and
this XLSX path both converge to Int32 roll columns via
`normalize_static_config_dfs`, so the two ingestion paths are semantically
equivalent.
"""

from __future__ import annotations

from pathlib import Path

import fastexcel
import polars as pl

# NOTE: pl.read_excel()'s default "calamine" engine requires the
# `fastexcel` package (>= 0.7.0) as an explicit runtime dependency --
# it is NOT bundled with a bare `pip install polars`. This module imports
# fastexcel directly (to enumerate sheet names without triggering dtype
# inference) so the dependency is explicit rather than implicit.
from riskquant.config.static_config.validate import REQUIRED_SHEETS
from riskquant.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Ingest dtype contract
# ---------------------------------------------------------------------------

# Numeric roll fields: read as Float64 (see module docstring), then
# integrality-checked and cast to Int32 by normalize.
_ROLL_NUMERIC_COLUMNS: tuple[str, ...] = ("anchor_offset", "roll_month", "roll_day")

# Explicit read-time dtype overrides for sheets that carry roll columns.
_ROLL_READ_OVERRIDES: dict[str, pl.DataType | type[pl.DataType]] = {
    "anchor_offset": pl.Float64,
    "roll_month": pl.Float64,
    "roll_day": pl.Float64,
    "fh_bh_mmdd": pl.String,
}

_SHEETS_WITH_ROLL: frozenset[str] = frozenset({"curve_meta", "surface_meta"})

# Compact target type for compiled/normalised roll counts. Discrete period
# counts and calendar month/day fit int32 with room to spare.
_ROLL_INT_DTYPE: type[pl.DataType] = pl.Int32


def normalize_static_config_dfs(
    dict_of_dfs: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """
    Enforce the roll-column ingest dtype contract.

    For curve_meta and surface_meta: verify every non-null numeric roll value
    is integral, then cast anchor_offset / roll_month / roll_day to Int32 and
    fh_bh_mmdd to String. A non-integral value (e.g. 1.5) is rejected with a
    table/column/value message -- never truncated.

    This is a pure, idempotent normalisation: integer roll columns pass
    straight through as Int32; Float64 columns from Excel are gated and
    narrowed. Semantic validation (ranges, method-specific presence/absence)
    remains in `validate_static_config_dfs`.
    """
    out = dict(dict_of_dfs)
    for sheet in _SHEETS_WITH_ROLL:
        df = out.get(sheet)
        if df is None:
            continue

        casts: list[pl.Expr] = []
        for col in _ROLL_NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            series = df[col]
            if series.dtype.is_float():
                # Reject non-integral values before the narrowing cast.
                offending = df.filter(
                    series.is_not_null() & (series.floor() != series)
                ).select([col])
                if not offending.is_empty():
                    bad = offending[col].to_list()
                    raise ConfigurationError(
                        f"{sheet}.{col}: non-integral value(s) {bad}. "
                        f"{col} is a discrete integer count of periods; "
                        f"non-integral values are rejected, not truncated."
                    )
            casts.append(pl.col(col).cast(_ROLL_INT_DTYPE))
        if "fh_bh_mmdd" in df.columns:
            casts.append(pl.col("fh_bh_mmdd").cast(pl.String))
        if casts:
            out[sheet] = df.with_columns(casts)
    return out


def load_risk_static_config_xlsx_into_dict(
    *,
    path: str | Path,
) -> dict[str, pl.DataFrame]:
    """
    Read every required sheet from an .xlsx workbook into a dict of DataFrames.

    Each sheet is read with an explicit dtype contract (no reliance on Excel
    dtype inference for the roll columns), then normalised so the roll columns
    arrive as Int32 / String. Non-integral roll values fail fast.

    Parameters
    ----------
    path
        Path to the Excel workbook.

    Returns
    -------
    dict[str, pl.DataFrame]
        Sheet-name -> DataFrame for every name in `REQUIRED_SHEETS`.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist.
    ValueError
        If a required sheet is missing from the workbook.
    ConfigurationError
        If a numeric roll cell is non-integral.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Workbook not found: {p}")

    # Enumerate sheet names without triggering dtype inference (warning-free).
    available: set[str] = set(fastexcel.read_excel(str(p)).sheet_names)
    missing = REQUIRED_SHEETS - available
    if missing:
        raise ValueError(
            f"Workbook {p.name} is missing required sheet(s): {sorted(missing)}.\n"
            f"Sheets present: {sorted(available)}."
        )

    dfs: dict[str, pl.DataFrame] = {}
    for name in REQUIRED_SHEETS:
        overrides = _ROLL_READ_OVERRIDES if name in _SHEETS_WITH_ROLL else None
        dfs[name] = pl.read_excel(str(p), sheet_name=name, schema_overrides=overrides)

    return normalize_static_config_dfs(dfs)
