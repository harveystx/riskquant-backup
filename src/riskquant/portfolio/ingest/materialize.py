# src/riskquant/portfolio/ingest/materialize.py

"""
Materialization layer for ingest dataframes.

Purpose
-------
Convert canonicalized ingest dataframes into deterministic,
machine-typed dataframes suitable for downstream validation
and SoA extraction.

This layer performs ONLY deterministic dtype normalization in-place:

- numeric coercion
- integer coercion
- flag coercion
- string dtype normalization
- object dtype elimination

This module intentionally does NOT perform:

- semantic validation
- business logic
- RF mapping
- parsing into engine arrays
- instrument validation

Those belong downstream in:
- validator.py
- batch_builder.py

Design Philosophy
-----------------
Canonicalized human-readable dataframe
    ↓
Deterministic machine-typed dataframe (mutated in-place)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.portfolio.ingest.schema import (
    DATE_COLS,
    DATETIME_COLS,
    FLAG_COLS,
    FLOAT_COLS,
    INT_COLS,
    NULLABLE_INT_COLS,
    STRING_COLS,
)

# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _materialize_string_col(
    df: pd.DataFrame,
    col: str,
) -> None:
    """
    Materialize string columns into deterministic pandas StringDtype in-place.

    Notes
    -----
    String columns intentionally remain human-readable at this stage.

    Strings are eliminated later during:
    - RF mapping
    - batch building
    """
    df[col] = df[col].astype("string")


def _materialize_float_col(
    df: pd.DataFrame,
    col: str,
) -> None:
    """
    Materialize float columns into deterministic float64 dtype in-place.
    """
    try:
        numeric_ser = pd.to_numeric(
            df[col],
            errors="raise",
        )
    except Exception as e:
        raise TypeError(f"Failed to materialize float column '{col}': {e}") from e

    df[col] = numeric_ser.astype(np.float64)


def _materialize_int_col(
    df: pd.DataFrame,
    col: str,
) -> None:
    """
    Materialize integer columns into deterministic int64 dtype in-place.

    Integer columns are:
    - non-nullable
    - exact integers only
    """
    ser = df[col]

    # --------------------------------------------------------------------------
    # NULL VALIDATION
    # --------------------------------------------------------------------------
    if ser.isna().any():
        bad_idx = ser.index[ser.isna()].tolist()[:10]
        raise ValueError(
            f"Integer column '{col}' contains null values at rows: {bad_idx}"
        )

    # --------------------------------------------------------------------------
    # NUMERIC COERCION
    # --------------------------------------------------------------------------
    try:
        numeric_ser = pd.to_numeric(
            ser,
            errors="raise",
        )
    except Exception as e:
        raise TypeError(f"Failed to materialize integer column '{col}': {e}") from e

    # --------------------------------------------------------------------------
    # INTEGER EXACTNESS VALIDATION
    # --------------------------------------------------------------------------
    frac, _ = np.modf(numeric_ser.to_numpy(copy=False))
    valid = frac == 0

    if not np.all(valid):
        bad = ser.iloc[np.flatnonzero(~valid)[:10]].tolist()
        raise TypeError(f"Integer column '{col}' contains non-integer values:\n{bad}")

    df[col] = numeric_ser.astype(np.int64)


def _materialize_nullable_int_col(
    df: pd.DataFrame,
    col: str,
) -> None:
    """
    Materialize a nullable integer column into pandas Int64 in-place.

    Sibling of `_materialize_int_col` for columns where NA is a legitimate
    "absent linkage" rather than an error. NA is preserved; the downstream
    compiler substitutes the typed sentinel.
    """
    ser = df[col]

    # Boolean leakage rejection (mirrors the flag-column contract).
    non_na = ser.dropna()
    is_bool = non_na.map(lambda x: isinstance(x, bool))
    if is_bool.any():
        bad = non_na[is_bool].iloc[:10].tolist()
        raise TypeError(f"Nullable-int column '{col}' contains boolean values:\n{bad}")

    try:
        numeric_ser = pd.to_numeric(ser, errors="raise")
    except Exception as e:
        raise TypeError(
            f"Failed to materialize nullable-int column '{col}': {e}"
        ) from e

    # Integer exactness on populated rows; NA rows pass through.
    frac, _ = np.modf(numeric_ser.to_numpy(dtype=np.float64, copy=False))
    valid = numeric_ser.isna().to_numpy(copy=False) | (frac == 0)
    if not np.all(valid):
        bad = ser.iloc[np.flatnonzero(~valid)[:10]].tolist()
        raise TypeError(
            f"Nullable-int column '{col}' contains non-integer values:\n{bad}"
        )

    df[col] = numeric_ser.astype(np.int64)


def _materialize_flag_col(
    df: pd.DataFrame,
    col: str,
) -> None:
    """
    Materialize nullable flag columns in-place.

    Allowed:
        0
        1
        <NA>

    Rejected:
        True
        False
        arbitrary numerics

    Stored as:
        float64

    Rationale
    ---------
    Downstream ingest logic uses NaN semantics.
    """
    ser = df[col]

    # --------------------------------------------------------------------------
    # BOOLEAN LEAKAGE REJECTION
    # --------------------------------------------------------------------------
    non_na = ser.dropna()
    is_bool = non_na.map(lambda x: isinstance(x, bool))

    if is_bool.any():
        bad = non_na[is_bool].iloc[:10].tolist()
        raise TypeError(
            f"Flag column '{col}' contains {is_bool.sum()} boolean values:\n{bad}"
        )

    # --------------------------------------------------------------------------
    # NUMERIC COERCION
    # --------------------------------------------------------------------------
    try:
        numeric_ser = pd.to_numeric(
            ser,
            errors="raise",
        )
    except Exception as e:
        raise TypeError(f"Failed to materialize flag column '{col}': {e}") from e

    # --------------------------------------------------------------------------
    # DOMAIN VALIDATION
    # --------------------------------------------------------------------------
    valid = numeric_ser.isna() | numeric_ser.isin((0, 1))

    if not valid.all():
        bad = ser[~valid].iloc[:10].tolist()
        raise ValueError(f"Flag column '{col}' contains invalid values:\n{bad}")

    df[col] = numeric_ser.astype(np.float64)


def _materialize_date_col(
    df: pd.DataFrame,
    col: str,
) -> None:
    """
    Materialize date columns into deterministic datetime64[ns] in-place.

    Dates are strictly validated against the ISO 8601 'YYYY-MM-DD' format.
    Providing an explicit format string ensures O(N) C-level parsing speed
    and avoids the heavy overhead of generic date inference.
    """
    try:
        df[col] = pd.to_datetime(
            df[col],
            format="%Y-%m-%d",
            exact=True,  # Strict matching, no extra characters allowed
            errors="raise",
        )
    except Exception as e:
        raise ValueError(
            f"Failed to materialize date column '{col}'. "
            f"Dates must be exactly 'YYYY-MM-DD'. "
            f"Original error: {e}"
        ) from e


def _materialize_datetime_col(
    df: pd.DataFrame,
    col: str,
) -> None:
    """
    Materialize datetime columns into deterministic datetime64[ns] in-place.

    Datetimes are strictly validated against the format 'YYYY-MM-DD HH:MM:SS.FFFFFF'.
    Providing an explicit format string ensures O(N) C-level parsing speed
    and avoids the heavy overhead of generic datetime inference.
    """
    try:
        df[col] = pd.to_datetime(
            df[col],
            format="%Y-%m-%d %H:%M:%S.%f",
            exact=True,  # Strict matching, no extra characters allowed
            errors="raise",
        )
    except Exception as e:
        raise ValueError(
            f"Failed to materialize datetime column '{col}'. "
            f"Datetimes must be exactly 'YYYY-MM-DD HH:MM:SS.FFFFFF'. "
            f"Original error: {e}"
        ) from e


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def materialize_canonicalized_ingest_df(
    df: pd.DataFrame,
) -> None:
    """
    Materialize canonicalized ingest dataframe into deterministic dtypes in-place.

    Pipeline
    --------
    - string normalization
    - float normalization
    - integer normalization
    - nullable integer normalization
    - flag normalization
    - date normalization
    - object dtype elimination

    Returns
    -------
    None
        The dataframe is mutated in-place.
    """

    # --------------------------------------------------------------------------
    # STRING COLS
    # --------------------------------------------------------------------------
    for col in STRING_COLS:
        if col in df.columns:
            _materialize_string_col(df, col)

    # --------------------------------------------------------------------------
    # FLOAT COLS
    # --------------------------------------------------------------------------
    for col in FLOAT_COLS:
        if col in df.columns:
            _materialize_float_col(df, col)

    # --------------------------------------------------------------------------
    # INT COLS
    # --------------------------------------------------------------------------
    for col in INT_COLS:
        if col in df.columns:
            _materialize_int_col(df, col)

    # --------------------------------------------------------------------------
    # NULLABLE INT COLS
    # --------------------------------------------------------------------------
    for col in NULLABLE_INT_COLS:
        if col in df.columns:
            _materialize_nullable_int_col(df, col)

    # --------------------------------------------------------------------------
    # FLAG COLS
    # --------------------------------------------------------------------------
    for col in FLAG_COLS:
        if col in df.columns:
            _materialize_flag_col(df, col)

    # --------------------------------------------------------------------------
    # DATE COLS
    # --------------------------------------------------------------------------
    for col in DATE_COLS:
        if col in df.columns:
            _materialize_date_col(df, col)

    # --------------------------------------------------------------------------
    # DATETIME COLS
    # --------------------------------------------------------------------------
    for col in DATETIME_COLS:
        if col in df.columns:
            _materialize_datetime_col(df, col)

    # --------------------------------------------------------------------------
    # OBJECT DTYPE ELIMINATION
    # --------------------------------------------------------------------------
    obj_cols = df.select_dtypes(include=["object"]).columns

    if len(obj_cols):
        raise RuntimeError(
            f"Object dtype columns remain after materialization:\n{obj_cols.tolist()}"
        )
