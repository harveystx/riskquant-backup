# src\riskquant\portfolio\ingest\validator.py
from typing import Iterable

import numpy as np
import pandas as pd

from riskquant.exceptions import PrettyKeyError
from riskquant.utils.printer import as_list_str, print_as_list_str


def require_cols_in_df(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    ctx: str,
) -> None:
    """Check that the specified columns are present in the DataFrame, and raise an error with a helpful message if any are missing.

    Args:
        df: The DataFrame to check for the presence of the required columns.
        cols: An iterable of column names that are required to be present in the DataFrame.
        ctx: A string providing context for error messages, such as the name of the sheet or the type of data being validated.

    Raises:
        TypeError: If the input df is not a pandas DataFrame.
        ValueError: If any of the required columns are missing from the DataFrame, with a message listing the missing columns.

    Returns:
        None if all required columns are present in the DataFrame.
    """
    # Smoke test that the input is a DataFrame, since the error messages are
    # more helpful than if we just let it fail with an AttributeError later on
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{ctx}: expected a pandas DataFrame, got {type(df)}")

    # Get the list of missing columns, if any
    missing = [c for c in cols if c not in df.columns]

    # If any required columns are missing, raise an error with a helpful message
    if missing:
        raise ValueError(
            f"{ctx}: missing required columns: \n" f"{as_list_str(missing)}"
        )


def require_non_nan_in_series(
    ser: pd.Series,
    *,
    ctx: str,
) -> None:
    """
    Require that ser contains no missing values (None/NaN/pd.NA) and no infinities.
    """
    if not isinstance(ser, pd.Series):
        raise TypeError(f"{ctx}: expected a pandas Series, got {type(ser)}")

    # 1) Missing values (works for all dtypes, including nullable dtypes)
    na_mask = ser.isna()

    # 2) Non-finite values (only meaningful for numeric dtypes)
    # Use vectorized check when possible; otherwise treat as "no inf check".
    inf_mask = pd.Series(False, index=ser.index)

    # Handle numeric dtypes (includes pandas nullable Float64)
    if pd.api.types.is_numeric_dtype(ser.dtype):
        # Convert to numpy array; for nullable floats, pandas produces float array with np.nan for missing.
        arr = ser.to_numpy(dtype="float64", na_value=np.nan)
        inf_mask = pd.Series(np.isinf(arr), index=ser.index)

    bad_mask = na_mask | inf_mask

    if bad_mask.any():
        bad_idx = ser.index[bad_mask].tolist()
        num_bad = len(bad_idx)
        preview = bad_idx[:25]
        raise PrettyKeyError(
            f"{ctx}: found {num_bad:_.0f} invalid values (NA or ±inf) at row indices: \n"
            f"{as_list_str(preview)}"
        )
