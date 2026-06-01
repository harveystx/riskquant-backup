# src/riskquant/portfolio/ingest/canonicalize.py

"""
Canonicalization layer for ingest dataframes.

Purpose
-------
Normalize raw human-facing ingest data into RiskQuant canonical form.

This layer performs ONLY deterministic normalization in-place:

- canonical column names
- canonical column ordering
- string trimming
- string uppercasing
- empty-string normalization

This module intentionally does NOT perform:

- semantic validation
- dtype coercion
- business logic
- instrument validation
- RF mapping
- parsing

Those belong downstream in:
- materialize.py
- validator.py
- batch_builder.py

Design Philosophy
-----------------
Human-readable at the edge.
Canonicalized deterministic frames internally.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)

from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.ingest.schema import (
    ALL_ALLOWED_COLS_SET,
    CANONICAL_COLUMN_ORDER,
    NON_NULLABLE_COLS,
    STRING_COLS,
    UNIVERSAL_REQUIRED_COLS,
)

# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _validate_schema(df: pd.DataFrame) -> None:
    """
    Validate ingest dataframe structural schema.

    Validates ONLY:
    - dataframe type
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

    # --------------------------------------------------------------------------
    # EMPTY DATAFRAME CHECK
    # --------------------------------------------------------------------------
    if df.empty:
        raise ValueError("Cannot ingest empty dataframe.")

    # --------------------------------------------------------------------------
    # DUPLICATE COLUMNS
    # --------------------------------------------------------------------------

    if not df.columns.is_unique:
        duplicated_ix = df.columns.duplicated()
        duplicated_cols = [
            str(df.columns[i]) for i, is_dup in enumerate(duplicated_ix) if is_dup
        ]

        raise PrettyKeyError(f"Duplicate ingest columns detected:\n{duplicated_cols}")

    # --------------------------------------------------------------------------
    # REQUIRED UNIVERSAL COLS
    # --------------------------------------------------------------------------

    missing = [col for col in UNIVERSAL_REQUIRED_COLS if col not in df.columns]

    if missing:
        raise PrettyKeyError(f"Missing required ingest columns:\n{missing}")

    # --------------------------------------------------------------------------
    # REQUIRED DOMAIN COLS
    # --------------------------------------------------------------------------

    # We do NOT enforce domain-specific columns at the schema level, as not all
    # instruments will require them. Instead, we allow them to be optional at
    # the schema level and defer domain-specific presence/absence logic to the
    # semantic validation layer (validator.py). This keeps the structural schema
    # flexible while still enforcing the core universal contract.

    # --------------------------------------------------------------------------
    # UNKNOWN / ILLEGAL COLS
    # --------------------------------------------------------------------------

    unknown = sorted(set(df.columns) - ALL_ALLOWED_COLS_SET)

    if unknown:
        raise PrettyKeyError(f"Unknown ingest columns detected:\n{unknown}")


def _canonicalize_column_names(df: pd.DataFrame) -> None:
    """
    Normalize dataframe column names in-place.

    Operations:
    - strip whitespace
    - lowercase
    """
    df.columns = df.columns.str.strip().str.lower()


def _canonicalize_string_series(ser: pd.Series) -> pd.Series:
    """
    Helper to process a single string series.
    """
    if not (is_string_dtype(ser.dtype) or is_object_dtype(ser.dtype)):
        raise TypeError(f"String column '{ser.name}' has invalid dtype={ser.dtype}.")

    non_na = ser.dropna()
    vals = non_na.to_numpy(copy=False)

    if not all(isinstance(x, str) for x in vals):
        raise TypeError(f"String column '{ser.name}' contains non-string values.")

    out = ser.astype("string")
    out = out.str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    out = out.mask(out == "", pd.NA)
    out = out.str.upper()

    return out


def _canonicalize_string_columns(df: pd.DataFrame) -> None:
    """
    Canonicalize all configured string columns in-place.
    """
    for col in STRING_COLS:
        if col not in df.columns:
            continue
        # Reassigning the series modifies the DataFrame in-place
        df[col] = _canonicalize_string_series(df[col])


def _reorder_columns(df: pd.DataFrame) -> None:
    """
    Reorder dataframe columns into canonical RiskQuant order in-place.

    Uses pop and insert to modify the internal column order without
    allocating a new DataFrame object.
    """
    ordered_cols = [col for col in CANONICAL_COLUMN_ORDER if col in df.columns]

    for i, col in enumerate(ordered_cols):
        col_data = df.pop(col)
        df.insert(i, col, col_data)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def canonicalize_raw_ingest_df(df: pd.DataFrame) -> None:
    """
    Canonicalize ingest dataframe into deterministic RiskQuant form in-place.

    Pipeline:
    1. canonicalize column names
    2. validate structural schema
    3. canonicalize string columns
    4. reorder columns
    5. validate non-nullable columns

    Returns
    -------
    None
        The dataframe is modified in-place.
    """

    # --------------------------------------------------------------------------
    # COLUMN NAME NORMALIZATION
    # --------------------------------------------------------------------------
    _canonicalize_column_names(df)

    # --------------------------------------------------------------------------
    # STRUCTURAL VALIDATION
    # --------------------------------------------------------------------------
    _validate_schema(df)

    # --------------------------------------------------------------------------
    # STRING NORMALIZATION
    # --------------------------------------------------------------------------
    _canonicalize_string_columns(df)

    # --------------------------------------------------------------------------
    # NON-NULLABLE VALIDATION
    # --------------------------------------------------------------------------
    for col in NON_NULLABLE_COLS:
        if df[col].isna().any():
            bad_idx = df.index[df[col].isna()].tolist()[:10]
            raise ValueError(
                f"Non-nullable column '{col}' "
                f"contains missing values at rows: {bad_idx}"
            )

    # --------------------------------------------------------------------------
    # COLUMN ORDERING
    # --------------------------------------------------------------------------
    _reorder_columns(df)

    # --------------------------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)
