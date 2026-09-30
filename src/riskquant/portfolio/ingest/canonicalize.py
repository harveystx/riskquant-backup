# src/riskquant/portfolio/ingest/canonicalize.py
"""
Canonicalization layer for ingest dataframes.

Purpose
-------
Normalize raw human-facing ingest data into RiskQuant canonical form.

This layer performs ONLY deterministic normalization and returns a new
``pl.DataFrame`` (functional, not in-place):

- canonical column names (strip + lowercase)
- structural schema validation
- string trimming, whitespace collapse, empty-string nulling, uppercasing
- canonical column ordering

This module intentionally does NOT perform:

- semantic validation
- dtype coercion
- business logic
- instrument validation
- RF mapping
- parsing

Those belong downstream in:
- materialize.py
- dispatcher.py
- compile.py

Design Philosophy
-----------------
Human-readable at the edge.
Canonicalized deterministic frames internally.
Functional pipeline: input is never mutated.
"""

from __future__ import annotations

import polars as pl

from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.ingest.schema import (
    ALL_ALLOWED_COLS_SET,
    CANONICAL_COLUMN_ORDER,
    NON_NULLABLE_COLS,
    STRING_COLS,
    UNIVERSAL_REQUIRED_COLS,
)

_CANONICAL_COLUMN_ORDER_SET: frozenset[str] = frozenset(CANONICAL_COLUMN_ORDER)


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _validate_schema(df: pl.DataFrame) -> None:
    """
    Validate ingest dataframe structural schema.

    Validates ONLY:
    - dataframe emptiness
    - unique columns
    - required universal columns
    - allowed column universe

    Does NOT validate:
    - semantics
    - values
    - instruments
    - quantities
    - option logic
    """
    if df.is_empty():
        raise ValueError("Cannot ingest empty dataframe.")

    # Duplicate columns: Polars rejects duplicates at construction, but
    # check explicitly so the error is PrettyKeyError, not Polars internal.
    cols = df.columns
    seen: set[str] = set()
    dupes: list[str] = []
    for c in cols:
        if c in seen:
            dupes.append(c)
        seen.add(c)
    if dupes:
        raise PrettyKeyError(f"Duplicate ingest columns detected:\n{dupes}")

    missing = [col for col in UNIVERSAL_REQUIRED_COLS if col not in cols]
    if missing:
        raise PrettyKeyError(f"Missing required ingest columns:\n{missing}")

    unknown = sorted(set(cols) - ALL_ALLOWED_COLS_SET)
    if unknown:
        raise PrettyKeyError(f"Unknown ingest columns detected:\n{unknown}")


def _canonicalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize dataframe column names.

    Operations:
    - strip whitespace
    - lowercase
    """
    mapping = {c: c.strip().lower() for c in df.columns}
    # Detect collisions BEFORE rename: Polars ``rename`` raises its own
    # DuplicateError, which would replace the PrettyKeyError contract.
    seen: set[str] = set()
    dupes: list[str] = []
    for canonical in mapping.values():
        if canonical in seen:
            dupes.append(canonical)
        seen.add(canonical)
    if dupes:
        raise PrettyKeyError(f"Duplicate ingest columns detected:\n{dupes}")
    return df.rename(mapping)


def _canonicalize_string_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Canonicalize all configured string columns.

    Per-column pipeline:
    1. Validate dtype is Utf8.
    2. Strip leading/trailing whitespace.
    3. Collapse internal whitespace runs to a single space.
    4. Null the result if it is empty or whitespace-only.
    5. Uppercase.
    """
    exprs: list[pl.Expr] = []
    for col in STRING_COLS:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if dtype == pl.Null or isinstance(dtype, (pl.Categorical, pl.Enum)):
            # All-null columns (pl.from_dict gives pl.Null) and dictionary-
            # encoded text are strings. pandas astype("string") accepted them.
            df = df.with_columns(pl.col(col).cast(pl.Utf8))
        elif dtype != pl.Utf8:
            raise TypeError(f"String column '{col}' has invalid dtype={dtype}.")
        norm = (
            pl.col(col).str.strip_chars().str.replace_all(r"\s+", " ").str.strip_chars()
        )
        # Null empty-string results, then uppercase surviving values.
        # pl.when/then/otherwise is the vectorised branchless path.
        exprs.append(
            pl
            .when(norm.is_null() | (norm == ""))
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(norm.str.to_uppercase())
            .alias(col)
        )
    if not exprs:
        return df
    return df.with_columns(exprs)


def _reorder_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reorder dataframe columns into canonical RiskQuant order.

    Known columns appear first in the canonical order; unknown columns
    (not in CANONICAL_COLUMN_ORDER) are appended in their existing order.
    """
    ordered = [col for col in CANONICAL_COLUMN_ORDER if col in df.columns]
    remainder = [col for col in df.columns if col not in _CANONICAL_COLUMN_ORDER_SET]
    return df.select(ordered + remainder)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def canonicalize_raw_ingest_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Canonicalize ingest dataframe into deterministic RiskQuant form.

    Returns a new ``pl.DataFrame``; the caller's frame is never mutated.

    Pipeline
    --------
    1. Normalize column names (strip + lowercase).
    2. Validate structural schema.
    3. Canonicalize string columns (trim, collapse, null-empty, uppercase).
    4. Validate non-nullable columns.
    5. Reorder columns.

    Parameters
    ----------
    df
        Raw human-facing positions frame as a Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        Canonicalized frame. Row order is preserved; no index is carried.
    """
    df = _canonicalize_column_names(df)
    _validate_schema(df)
    df = _canonicalize_string_columns(df)

    for col in NON_NULLABLE_COLS:
        null_count = df[col].null_count()
        if null_count > 0:
            bad_idx = df[col].is_null().to_numpy().nonzero()[0][:10].tolist()
            raise ValueError(
                f"Non-nullable column '{col}' "
                f"contains missing values at rows: {bad_idx}"
            )

    df = _reorder_columns(df)
    return df
