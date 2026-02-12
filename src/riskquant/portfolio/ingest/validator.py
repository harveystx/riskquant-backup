# src\riskquant\portfolio\ingest\validator.py
import pandas as pd

from riskquant.exceptions import PrettyKeyError
from riskquant.utils.printer import as_list_str


def require_cols_in_df(df: pd.DataFrame, cols, *, ctx: str) -> None:
    # Smoke test that the input is a DataFrame, since the error messages are more
    # helpful than if we just let it fail with an AttributeError later on
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{ctx}: expected a pandas DataFrame, got {type(df)}")

    # Get the list of missing columns, if any
    missing = [c for c in cols if c not in df.columns]

    # If any required columns are missing, raise an error with a helpful message
    if missing:
        raise ValueError(
            f"{ctx}: missing required columns:\n {as_list_str(missing)}"
        )


def require_non_nan_in_series(ser: pd.Series, *, ctx: str) -> None:
    # Smoke test that the input is a Series, since the error messages are more
    # helpful than if we just let it fail with an AttributeError later on
    if not isinstance(ser, pd.Series):
        raise TypeError(f"{ctx}: expected a pandas Series, got {type(ser)}")

    # Check for any NA values in the series and raise an error if found.
    if ser.isna().any():
        bad = ser.index[ser.isna()].tolist()
        num_bad = len(bad)
        bad = bad[:25]
        raise PrettyKeyError(
            f"{ctx}: found {num_bad:_.0f} missing values at row indices: {bad}"
        )
