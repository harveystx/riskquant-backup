# src/riskquant/portfolio/ingest/validator.py

from typing import Iterable
import numpy as np
import pandas as pd

from riskquant.constants import POSITION_DUST_TOLERANCE
from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.instruments import Instrument
from riskquant.utils.printer import as_list_str

# Grab valid names straight from your enum
VALID_INSTRUMENT_NAMES = {member.name for member in Instrument}


def require_cols_in_df(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    ctx: str,
) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{ctx}: missing required columns: \n{as_list_str(missing)}"
        )


def require_non_nan_in_series(
    ser: pd.Series,
    *,
    ctx: str,
) -> None:
    bad_mask = ser.isna()
    if pd.api.types.is_numeric_dtype(ser.dtype):
        arr = ser.to_numpy(dtype="float64", na_value=np.nan)
        bad_mask = bad_mask | pd.Series(np.isinf(arr), index=ser.index)

    if bad_mask.any():
        bad_idx = ser.index[bad_mask].tolist()
        raise PrettyKeyError(
            f"{ctx}: found {len(bad_idx)} invalid values at rows: {bad_idx[:10]}"
        )


def require_positive_values(
    ser: pd.Series,
    *,
    ctx: str,
) -> None:
    valid_mask = ser.notna()
    violators_mask = ser[valid_mask].to_numpy(dtype=np.float64) <= 0
    if violators_mask.any():
        bad_idx = ser[valid_mask].index[violators_mask].tolist()
        raise ValueError(
            f"{ctx}: found {len(bad_idx)} non-positive values at rows: {bad_idx[:10]}"
        )


def validate_instruments(
    ser: pd.Series,
    *,
    ctx: str,
) -> None:
    """Vectorized check to ensure all instruments match the defined Enum."""
    invalid_mask = ~ser.isin(VALID_INSTRUMENT_NAMES)

    if invalid_mask.any():
        bad_idx = ser.index[invalid_mask].tolist()
        bad_vals = ser[invalid_mask].unique()
        raise ValueError(
            f"{ctx}: Invalid instruments detected {bad_vals.tolist()} "
            f"at rows: {bad_idx[:10]}. Valid options are: {VALID_INSTRUMENT_NAMES}"
        )


def validate_is_call_flags(
    ser: pd.Series,
    *,
    ctx: str,
) -> None:
    """Ensure is_call is strictly 0 (put), 1 (call), or missing/NaN."""
    # We only care about rows where the user actually provided a value
    valid_mask = ser.notna()

    # Check if any provided values are NOT 0 or 1
    provided_vals = ser[valid_mask]
    violators_mask = ~provided_vals.isin([0, 1, 0.0, 1.0])

    if violators_mask.any():
        bad_idx = provided_vals.index[violators_mask].tolist()
        bad_vals = provided_vals[violators_mask].unique()
        raise ValueError(
            f"{ctx}: 'is_call' must be 0, 1, or blank. Found {bad_vals.tolist()} "
            f"at rows: {bad_idx[:10]}"
        )


def validate_conditional_option_fields(
    df: pd.DataFrame,
    *,
    ctx: str,
) -> None:
    """
    Cross-column validation: If it is an option, it MUST have a strike and expiry.
    """
    # Create a fast mask for any row that is an option
    is_option_mask = df["instrument"].str.contains("OPTION", na=False)

    if not is_option_mask.any():
        return  # No options to check

    # Check strikes
    if (
        "strike_price" not in df.columns
        or df.loc[is_option_mask, "strike_price"].isna().any()
    ):
        bad_idx = df[
            is_option_mask
            & df.get("strike_price", pd.Series(np.nan, index=df.index)).isna()
        ].index.tolist()
        raise ValueError(
            f"{ctx}: Options must have a 'strike_price'. Missing at rows: {bad_idx[:10]}"
        )

    # Check expiries
    if (
        "expiry_date" not in df.columns
        or df.loc[is_option_mask, "expiry_date"].isna().any()
    ):
        bad_idx = df[
            is_option_mask
            & df.get("expiry_date", pd.Series(np.nan, index=df.index)).isna()
        ].index.tolist()
        raise ValueError(
            f"{ctx}: Options must have an 'expiry_date'. Missing at rows: {bad_idx[:10]}"
        )


def validate_uids_unique(ser: pd.Series, *, ctx: str) -> None:
    """
    Ensure all trade UIDs are strictly unique across the ingestion batch.
    """
    if not ser.is_unique:
        # Only pay the cost of finding the exact duplicates if we fail
        duplicates = ser[ser.duplicated()].unique()
        raise ValueError(
            f"{ctx}: Trade UIDs must be strictly unique. "
            f"Found duplicates: {duplicates.tolist()[:10]}"
        )


def require_non_zero_values(
    ser: pd.Series,
    *,
    ctx: str,
    tol: float = POSITION_DUST_TOLERANCE,
) -> None:
    """
    Vectorized check to ensure values are meaningfully non-zero.
    Uses the engine's global dust tolerance to filter out upstream float artifacts.
    """
    valid_mask = ser.notna()
    arr = ser[valid_mask].to_numpy(dtype=np.float64)

    # Vectorized check for values that are effectively zero
    violators_mask = np.abs(arr) < tol

    if violators_mask.any():
        bad_idx = ser[valid_mask].index[violators_mask].tolist()
        preview = bad_idx[:25]
        raise ValueError(
            f"{ctx}: found {len(bad_idx):_.0f} trades with effectively zero quantity (abs(qty) < {tol}). "
            f"Zero-quantity and 'dust' trades carry no risk and must be filtered out upstream. "
            f"Row indices: \n{as_list_str(preview)}"
        )
