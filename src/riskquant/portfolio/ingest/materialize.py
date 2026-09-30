# src/riskquant/portfolio/ingest/materialize.py
"""
Materialization layer for ingest dataframes.

Purpose
-------
Convert canonicalized ingest dataframes into deterministic, machine-typed
dataframes suitable for downstream validation and SoA extraction.

This layer performs ONLY deterministic dtype normalization and returns a new
``pl.DataFrame`` (functional, not in-place):

- numeric coercion  (float64, int64, nullable-int64)
- flag coercion     (float64, null for absent)
- date parsing      (pl.Date from Utf8 'YYYY-MM-DD')
- object dtype elimination (Polars has no object dtype by default)

This module intentionally does NOT perform:

- semantic validation
- business logic
- RF mapping
- parsing into engine arrays
- instrument validation

Design Philosophy
-----------------
Canonicalized human-readable Polars frame
    ↓
Deterministic machine-typed Polars frame (functional return)

Pre-condition
-------------
The caller must pass a frame that has already been processed by
``canonicalize_raw_ingest_df``.  String columns must be ``pl.Utf8``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

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


def _materialize_string_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Cast column to pl.Utf8 if not already."""
    if df[col].dtype == pl.Utf8:
        return df
    try:
        return df.with_columns(pl.col(col).cast(pl.Utf8))
    except Exception as e:
        raise TypeError(f"Failed to materialize string column '{col}': {e}") from e


def _materialize_float_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Materialize float column into deterministic Float64 dtype.

    NaN becomes null. Polars keeps NaN distinct from null, but every
    downstream check (``is_null`` in validate, ``fill_null`` in compile)
    reads null only. The pandas pipeline treated NaN as missing (``isna``);
    without this step a NaN quantity or strike passed validation into the
    SoA.
    """
    try:
        return df.with_columns(pl.col(col).cast(pl.Float64).fill_nan(None))
    except Exception as e:
        raise TypeError(f"Failed to materialize float column '{col}': {e}") from e


def _materialize_int_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Materialize integer column into deterministic Int64 dtype.

    Integer columns are:
    - non-nullable
    - exact integers only (no fractional part)
    """
    ser = df[col]

    # NULL VALIDATION
    if ser.null_count() > 0:
        bad_idx = ser.is_null().to_numpy().nonzero()[0][:10].tolist()
        raise ValueError(
            f"Integer column '{col}' contains null values at rows: {bad_idx}"
        )

    # NUMERIC COERCION (via Float64 to support string integer inputs)
    try:
        as_float = ser.cast(pl.Float64)
    except Exception as e:
        raise TypeError(f"Failed to materialize integer column '{col}': {e}") from e

    # INTEGER EXACTNESS VALIDATION
    vals = as_float.to_numpy()
    frac, _ = np.modf(vals)
    invalid = frac != 0.0
    if invalid.any():
        bad_positions = np.flatnonzero(invalid)[:10]
        bad = ser.to_numpy()[bad_positions].tolist()
        raise TypeError(f"Integer column '{col}' contains non-integer values:\n{bad}")

    try:
        return df.with_columns(pl.col(col).cast(pl.Int64))
    except Exception as e:
        raise TypeError(f"Failed to cast integer column '{col}' to Int64: {e}") from e


def _materialize_nullable_int_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Materialize a nullable integer column into pl.Int64.

    Sibling of ``_materialize_int_col`` for columns where null is a legitimate
    "absent linkage" rather than an error.  Nulls are preserved; the downstream
    compiler substitutes the typed sentinel.

    Why pl.Int64 with native nulls (not np.float64 with NaN):
    Polars Int64 is a proper nullable integer type.  NaN coercion artifacts
    (the pandas 3.0 IntCastingNaNError, or the silent platform-min-int wrapping
    on older NumPy) do not apply here.
    """
    ser = df[col]

    # Boolean leakage rejection: a pl.Boolean column is never a valid integer.
    if ser.dtype == pl.Boolean:
        bad = ser.to_list()[:10]
        raise TypeError(f"Nullable-int column '{col}' contains boolean values:\n{bad}")

    # For Float64 columns arriving with NaN (e.g. from pl.from_pandas), treat
    # NaN as null before coercion so the null semantics are consistent.
    if ser.dtype == pl.Float64:
        ser = ser.fill_nan(None)
        df = df.with_columns(ser.alias(col))

    # NUMERIC COERCION (Float64 intermediate to support string integer inputs).
    try:
        as_float = ser.cast(pl.Float64)
    except Exception as e:
        raise TypeError(
            f"Failed to materialize nullable-int column '{col}': {e}"
        ) from e

    # INTEGER EXACTNESS on non-null rows; null rows pass through.
    null_mask = as_float.is_null().to_numpy()
    vals = as_float.fill_null(0.0).to_numpy()
    frac, _ = np.modf(vals)
    valid = null_mask | (frac == 0.0)
    if not valid.all():
        bad_positions = np.flatnonzero(~valid)[:10]
        bad = ser.to_numpy()[bad_positions].tolist()
        raise TypeError(
            f"Nullable-int column '{col}' contains non-integer values:\n{bad}"
        )

    # Cast to Int64; nulls are preserved natively.
    try:
        return df.with_columns(pl.col(col).cast(pl.Int64))
    except Exception as e:
        raise TypeError(
            f"Failed to cast nullable-int column '{col}' to Int64: {e}"
        ) from e


def _materialize_flag_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Materialize nullable flag columns.

    Allowed:
        0  (integer or numeric)
        1  (integer or numeric)
        null / absent

    Rejected:
        True / False  (booleans)
        arbitrary numerics

    Stored as:
        Float64 (null for absent rows)

    Rationale
    ---------
    Downstream ingest logic uses NaN semantics via ``np.isnan`` on the
    extracted numpy array.  A null Float64 maps to NaN in numpy, preserving
    the same branchless flow as the pandas predecessor.
    """
    ser = df[col]

    # BOOLEAN LEAKAGE REJECTION
    if ser.dtype == pl.Boolean:
        bad = ser.to_list()[:10]
        raise TypeError(
            f"Flag column '{col}' contains {len(bad)} boolean values:\n{bad}"
        )
    # Object-typed booleans (from pl.from_pandas round-trip) would appear as
    # pl.Object dtype; reject anything that isn't numeric.
    if ser.dtype == pl.Object:
        raise TypeError(
            f"Flag column '{col}' has object dtype; expected numeric (0/1/null)."
        )

    # NUMERIC COERCION
    try:
        as_float = ser.cast(pl.Float64).fill_nan(None)  # NaN is absent, as in pandas
    except Exception as e:
        raise TypeError(f"Failed to materialize flag column '{col}': {e}") from e

    # DOMAIN VALIDATION: only 0.0, 1.0, or null
    null_mask = as_float.is_null().to_numpy()
    vals = as_float.fill_null(-1.0).to_numpy()  # -1 sentinel distinguishes null
    valid = null_mask | (vals == 0.0) | (vals == 1.0)
    if not valid.all():
        bad_positions = np.flatnonzero(~valid)[:10]
        bad = ser.to_numpy()[bad_positions].tolist()
        raise ValueError(f"Flag column '{col}' contains invalid values:\n{bad}")

    return df.with_columns(as_float.alias(col))


def _materialize_date_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Materialize date column into pl.Date.

    Accepted input dtypes:
    - pl.Date        : no-op
    - pl.Datetime    : cast to pl.Date (time component dropped)
    - pl.Utf8        : parsed via strict 'YYYY-MM-DD' format

    Timezone-aware pl.Datetime raises, consistent with the prior pandas
    ``DatetimeTZDtype`` rejection.
    """
    dtype = df[col].dtype

    if dtype == pl.Date:
        return df

    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            raise TypeError(
                f"Date column '{col}': timezone-aware datetimes are not supported."
            )
        return df.with_columns(pl.col(col).cast(pl.Date))

    if dtype == pl.Utf8:
        try:
            return df.with_columns(
                pl.col(col).str.to_date(format="%Y-%m-%d", strict=True)
            )
        except Exception as e:
            raise ValueError(
                f"Failed to materialize date column '{col}'. "
                f"Dates must be exactly 'YYYY-MM-DD'. "
                f"Original error: {e}"
            ) from e

    raise TypeError(
        f"Date column '{col}' has unsupported dtype {dtype}. "
        "Expected pl.Utf8 ('YYYY-MM-DD'), pl.Date, or pl.Datetime."
    )


def _materialize_datetime_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Materialize datetime column into pl.Datetime.

    Datetimes are strictly validated against 'YYYY-MM-DD HH:MM:SS.FFFFFF'.
    Currently DATETIME_COLS is empty; this function exists for schema
    completeness.
    """
    dtype = df[col].dtype

    if isinstance(dtype, pl.Datetime):
        return df

    if dtype == pl.Utf8:
        try:
            return df.with_columns(
                pl.col(col).str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=True)
            )
        except Exception as e:
            raise ValueError(
                f"Failed to materialize datetime column '{col}'. "
                f"Datetimes must be exactly 'YYYY-MM-DD HH:MM:SS.FFFFFF'. "
                f"Original error: {e}"
            ) from e

    raise TypeError(f"Datetime column '{col}' has unsupported dtype {dtype}.")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def materialize_canonicalized_ingest_df(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Materialize canonicalized ingest dataframe into deterministic dtypes.

    Returns a new ``pl.DataFrame``; the caller's frame is never mutated.

    Pipeline
    --------
    - string normalization  (Utf8)
    - float normalization   (Float64)
    - integer normalization (Int64, non-nullable)
    - nullable integer      (Int64, nullable)
    - flag normalization    (Float64, null for absent)
    - date normalization    (pl.Date)
    - datetime normalization (pl.Datetime, currently empty set)
    - object dtype check    (Polars has no object dtype outside pl.Object)

    Parameters
    ----------
    df
        Canonicalized ingest frame from ``canonicalize_raw_ingest_df``.

    Returns
    -------
    pl.DataFrame
        Materialized frame.
    """
    for col in STRING_COLS:
        if col in df.columns:
            df = _materialize_string_col(df, col)

    for col in FLOAT_COLS:
        if col in df.columns:
            df = _materialize_float_col(df, col)

    for col in INT_COLS:
        if col in df.columns:
            df = _materialize_int_col(df, col)

    for col in NULLABLE_INT_COLS:
        if col in df.columns:
            df = _materialize_nullable_int_col(df, col)

    for col in FLAG_COLS:
        if col in df.columns:
            df = _materialize_flag_col(df, col)

    for col in DATE_COLS:
        if col in df.columns:
            df = _materialize_date_col(df, col)

    for col in DATETIME_COLS:
        if col in df.columns:
            df = _materialize_datetime_col(df, col)

    # OBJECT DTYPE CHECK: legitimate Polars frames should not have pl.Object
    # columns after materialization.  Any that remain indicate a dtype that
    # the schema registries do not cover.
    obj_cols = [c for c in df.columns if df[c].dtype == pl.Object]
    if obj_cols:
        raise RuntimeError(
            f"Object dtype columns remain after materialization:\n{obj_cols}"
        )

    return df
