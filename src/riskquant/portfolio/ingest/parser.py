# src\riskquant\portfolio\ingest\parser.py


from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

from riskquant.utils.dates import datetime64D_to_ordinal32, is_date_str_valid


def _is_na_helper(x: object) -> bool:
    return x is None or pd.isna(x)


def parse_optional_date_series_to_ordinal32(
    ser: pd.Series,
    *,
    sentinel: np.int32,
    col_name: str,
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
    sv = ser.to_numpy(copy=False)

    # Create a boolean mask for non-NA values. Start with all True, then set to False where we find NA.
    # Single allocation of the mask (bool[n])
    # nn_mask = np.ones(n, dtype=bool)
    # nn_mask &= ~pd.isna(sv)
    nn_mask = pd.notna(sv)

    # If the dtype is object, we also want to treat empty strings as NA. We can check this without copying the data by iterating over the original series and checking for empty strings where nn_mask is still True.
    if sv.dtype == object:
        vals = sv  # local binding for efficiency
        mask = nn_mask  # local binding for efficiency
        for i in range(n):
            if mask[i]:
                v = vals[i]
                if isinstance(v, str) and (v == "" or v.isspace()):
                    mask[i] = False

    # If there are no non-NA values, we can return early with the all-sentinel output.
    if not np.any(nn_mask):
        return out

    # For the non-NA values, we need to parse them according to the accepted input types. We can do this efficiently by first checking the dtype of the non-NA values. If they are already datetime64, we can convert them directly. Otherwise, we need to iterate over them and parse each one according to its type.

    # # Get the non-NA values as a numpy array without copying if possible
    # # v = ser.loc[nn_mask]     # pandas indexing (slower, may copy)
    # v = ser._values[nn_mask]     # direct access to the underlying array (faster, no copy if already numpy array)

    # If the dtype of the series is timezone-aware datetime64, we should raise an error since we don't support timezones in our date encoding. This is a common source of confusion since pandas will allow timezone-aware datetimes in a datetime64 column, but our encoding only supports timezone-naive datetimes. We can check for this using pandas.api.types.is_datetime64tz_dtype, which will return True for any timezone-aware datetime64 dtype. If we find this, we will raise an error with a helpful message indicating the column name and the issue.
    if is_datetime64tz_dtype(ser.dtype):
        raise TypeError(
            f"Invalid {col_name}: timezone-aware datetimes are not supported."
        )

    # If the dtype of the non-NA values is already datetime64, we can convert them directly to ordinal32 without having to iterate over them. This is a fast path that avoids the overhead of Python-level iteration and type checking for each value. We can check this using pandas.api.types.is_datetime64_any_dtype, which will return True for any datetime64 dtype (including with time units or timezone). If this is the case, we can convert the entire array of non-NA values to datetime64[D] in one step, and then apply the datetime64D_to_ordinal32 conversion to the non-NA positions in the output array.
    if is_datetime64_any_dtype(ser.dtype):
        # Convert the series to datetime64[D] without copying if possible. This will work if the original dtype is already some kind of datetime64, since numpy can view it as datetime64[D] without copying. If the original dtype is object or string, this will create a new array of datetime64[D], but that's unavoidable since we need to parse the strings anyway. We will only convert the non-NA values, since the NA values will be set to the sentinel in the output array.
        d64 = sv.astype("datetime64[D]", copy=False)
        out[nn_mask] = datetime64D_to_ordinal32(d64[nn_mask])
        return out

    # Otherwise, we need to iterate over the non-NA values and parse each one according to its type. We can do this in a single pass over the non-NA values, which should be efficient enough since we expect most values to be NA in many cases. We will validate and convert each value according to the accepted input types, and if any value is invalid we will raise an error with a helpful message indicating the row index and the offending value.

    # Get the indices of the non-NA values
    idx = np.nonzero(nn_mask)[0]

    # Pre-allocate a datetime64[D] array to hold the converted dates for the non-NA values. We will fill this array as we parse the values, and then convert it to ordinal32 at the end. This avoids having to create intermediate lists or arrays of Python objects, which would be less efficient.
    d64 = np.empty(idx.shape[0], dtype="datetime64[D]")

    # Local bindings for efficiency in the loop
    _is_na = _is_na_helper
    _is_date_str_valid = is_date_str_valid

    # Iterate over the non-NA values and parse each one according to its type. We will use the original series for indexing to avoid any issues with the numpy array view, since we may need to access the original Python objects if the dtype is object. We will also keep track of the current position in the d64 array using a separate index variable k, since idx may not be contiguous.
    for k, i in enumerate(idx):
        # Get the original value from the series for this index. We get the value from the array 'sv'. We need to use the original series for indexing to ensure we get the correct Python object, especially if the dtype is object. The numpy array view may not preserve the original Python objects if it had to be converted to a different dtype, so we should avoid using it for indexing in this loop.
        x = sv[i]

        # If the value is NA (None, NaN, or empty string), we should have already marked it as NA in the mask, so we can skip it. This is just a sanity check to ensure our mask logic is correct. If we do encounter an NA value here, it means there is a bug in our mask logic, so we will raise an error to alert us to this issue.
        if _is_na(x):
            raise ValueError(
                f"Internal error: encountered NA value at row {int(i)} that was not marked as NA in the mask. Value: {x}"
            )

        # Now we know x is not NA, so we can parse it according to the accepted input types. We will check the type of x and handle each case accordingly. If the type of x is not one of the accepted types, we will raise an error with a helpful message indicating the row index, the offending value, and the expected types.
        if isinstance(x, (pd.Timestamp, np.datetime64)):
            d64[k] = np.datetime64(x, "D")
            continue

        # If the dtype of the series is object, we may have string values that need to be validated and parsed. We will check if x is a string, and if so we will validate it using is_date_str_valid and then convert it to datetime64[D]. If the string is not a valid date in the expected format, we will raise an error with a helpful message indicating the row index, the offending value, and the expected format.
        if isinstance(x, str):
            xs = x.strip()
            if not _is_date_str_valid(xs):
                raise ValueError(
                    f"Invalid {col_name}='{xs}' at row {int(i)} "
                    "(expected strict YYYY-MM-DD)."
                )
            d64[k] = np.datetime64(xs, "D")
            continue

        # If we get here, it means x is not NA and not one of the accepted types, so we will raise an error with a helpful message.
        raise TypeError(
            f"Invalid {col_name} type at row {int(i)}: {type(x).__name__}. "
            "Expected YYYY-MM-DD string, pandas Timestamp, or datetime64."
        )

    # After parsing all the non-NA values into the d64 array, we can convert it to ordinal32 and fill the corresponding positions in the output array.
    out[idx] = datetime64D_to_ordinal32(d64)

    # Finally, we can return the output array, which contains the parsed ordinal32 values for the non-NA inputs and the sentinel value for the NA inputs.
    return out
