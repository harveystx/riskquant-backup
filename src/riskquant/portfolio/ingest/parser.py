# src\riskquant\portfolio\ingest\parser.py


from __future__ import annotations
from datetime import date, datetime

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
)

from riskquant.portfolio.ingest.utils import is_na_helper
from riskquant.utils.dates import (
    datetime64D_to_ordinal32,
    is_date_str_valid,
)


def parse_optional_date_series_to_ordinal32(
    ser: pd.Series,
    *,
    col_name: str,
    sentinel: np.int32,
) -> np.ndarray:
    """
    Convert a date-like column to ordinal32 days since DATE_EPOCH using the canonical
    riskquant date encoding (datetime64[D] -> ordinal32).

    Accepted inputs:
      - NaN / None / ""        -> sentinel
      - datetime64 / Timestamp -> encoded via datetime64D_to_ordinal32
      - str "YYYY-MM-DD"       -> validated via is_date_str_valid then encoded
    """
    # Number of rows in the input series
    n = int(len(ser))

    # Initialize the output array with the sentinel value
    out = np.empty(n, dtype=np.int32)
    out.fill(np.int32(sentinel))

    # Early exit if the series is empty, since the rest of the code assumes at least one row
    if n == 0:
        return out

    # Convert the series to a numpy array for efficient access, without copying if possible
    date_vals = ser.to_numpy(copy=False)

    # Create a boolean mask for non-NA values.
    nn_mask = pd.notna(date_vals)

    # If the dtype is object, we also want to treat empty strings as NA.
    if date_vals.dtype == object:
        date_vals_bind = date_vals  # local binding for efficiency
        nn_mask_bind = nn_mask  # local binding for efficiency
        for i in range(n):
            if nn_mask_bind[i]:
                v = date_vals_bind[i]
                if isinstance(v, str) and (v == "" or v.isspace()):
                    nn_mask_bind[i] = False

    # If there are no non-NA values, we can return early with the all-sentinel output.
    if not np.any(nn_mask):
        return out

    # If the dtype of the series is timezone-aware datetime, we should raise an error since we don't support timezones in our date encoding.
    if isinstance(ser.dtype, pd.DatetimeTZDtype):
        raise TypeError(
            f"Invalid {col_name}: timezone-aware datetimes are not supported."
        )

    # If the dtype of the non-NA values is already datetime64, we can convert them directly to ordinal32 without having to iterate over them.
    if is_datetime64_any_dtype(ser.dtype):
        # Convert the series to datetime64[D] without copying if possible.
        d64 = date_vals.astype("datetime64[D]", copy=False)
        out[nn_mask] = datetime64D_to_ordinal32(d64[nn_mask])
        return out

    # Get the indices of the non-NA values
    idx = np.nonzero(nn_mask)[0]

    # Pre-allocate a datetime64[D] array.
    d64 = np.empty(idx.shape[0], dtype="datetime64[D]")

    # Local bindings for efficiency in the loop
    _is_na = is_na_helper
    _is_date_str_valid = is_date_str_valid

    # Iterate over the non-NA values and parse each one according to its type.
    for k, i in enumerate(idx):
        # Get the original value from the series for this index.
        x = date_vals[i]

        # Smoke test! If the value is NA (None, NaN, or empty string), we should have already marked it as NA in the mask, so we can skip it.
        if _is_na(x):
            raise ValueError(
                f"Internal error: encountered NA value at row {int(i)} that was not marked as NA in the mask. Value: {x}"
            )

        # Now we know x is not NA, so we can parse it according to the accepted input types.
        if isinstance(x, (pd.Timestamp, np.datetime64, datetime, date)):
            d64[k] = np.datetime64(x, "D")
            continue

        # If the dtype of the series is object, we may have string values that need to be validated and parsed.
        if isinstance(x, str):
            xs = x.strip()
            if not _is_date_str_valid(xs):
                raise ValueError(
                    f"Invalid {col_name}='{xs}' at row {int(i)}!"
                    f"Got a string that is not a valid date in YYYY-MM-DD format: '{xs}'. "
                    "Expected strict YYYY-MM-DD."
                )
            d64[k] = np.datetime64(xs, "D")
            continue

        # If we get here, it means x is not NA and not one of the accepted types, so we will raise an error with a helpful message.
        raise TypeError(
            f"Invalid {col_name} type at row {int(i)}: value = {x}, type = {type(x).__name__}. "
            "Expected YYYY-MM-DD string, pandas Timestamp, datetime, or date."
        )

    # At this point, we have successfully parsed all the non-NA values into the d64 array as datetime64[D]. We can now convert them to ordinal32 and store them in the output array at the correct indices.
    out[idx] = datetime64D_to_ordinal32(d64)

    # Return.
    return out
