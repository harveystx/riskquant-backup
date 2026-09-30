# src/riskquant/portfolio/ingest/parser.py
"""
Date-column parser for the ingest edge.

Converts ``pl.Series`` date columns to ordinal int32 arrays using the
canonical riskquant date encoding (datetime64[D] -> ordinal32).

Accepted input dtypes
---------------------
- pl.Date        : extracted as datetime64[D], NaT for null -> sentinel
- pl.Datetime    : cast to datetime64[D]
- pl.Utf8        : "YYYY-MM-DD" strings or empty/null -> sentinel
- pl.Null        : all-null series -> sentinel-filled

Rejected:
- timezone-aware pl.Datetime
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

from riskquant.portfolio.ingest.utils import is_na_helper
from riskquant.utils.dates import (
    datetime64D_to_ord32,
    is_date_str_valid,
)


def parse_optional_date_series_to_ord32(
    ser: pl.Series,
    *,
    col_name: str,
    sentinel: np.int32,
) -> np.ndarray:
    """
    Convert a date-like Polars series to ordinal32 days since DATE_EPOCH.

    Null and empty values map to ``sentinel``.

    Parameters
    ----------
    ser
        Polars series with dtype pl.Date, pl.Datetime, or pl.Utf8.
    col_name
        Column label for diagnostic messages.
    sentinel
        Sentinel int32 (from riskquant.constants) for absent/null dates.

    Returns
    -------
    np.ndarray of np.int32
        Length-n, C-contiguous. Absent rows hold ``sentinel``.
    """
    n = int(len(ser))

    out = np.empty(n, dtype=np.int32)
    out.fill(np.int32(sentinel))

    if n == 0:
        return out

    # ---- TIMEZONE CHECK (reject before any extraction) ----
    if isinstance(ser.dtype, pl.Datetime) and ser.dtype.time_zone is not None:
        raise TypeError(
            f"Invalid {col_name}: timezone-aware datetimes are not supported."
        )

    # ---- FAST PATH: pl.Date or tz-naive pl.Datetime ----
    # Polars extracts these as numpy datetime64; NaT marks nulls.
    if isinstance(ser.dtype, (pl.Date, pl.Datetime)):
        d64 = ser.to_numpy(allow_copy=True)
        if isinstance(ser.dtype, pl.Datetime):
            # datetime64[us/ns/ms] -> datetime64[D]; truncates time component.
            d64 = d64.astype("datetime64[D]", copy=False)
        # ~isnat detects both actual NaT and nulls mapped to NaT.
        nn_mask = ~np.isnat(d64)
        if nn_mask.any():
            out[nn_mask] = datetime64D_to_ord32(d64[nn_mask])
        return out

    # ---- ALL-NULL / NULL-TYPED SERIES ----
    # Polars pl.Null dtype or a Utf8 series that is entirely null -> sentinel.
    if ser.dtype == pl.Null or ser.null_count() == n:
        return out

    # ---- STRING / OBJECT FALLBACK ----
    # After materialize, string date columns are pl.Date. This branch handles
    # the blank-sentinel series that compile.py supplies for absent columns.
    null_mask = ser.is_null().to_numpy()
    nn_mask = ~null_mask

    # For Utf8 columns, also treat empty or whitespace-only strings as absent.
    if ser.dtype == pl.Utf8:
        date_vals = ser.to_numpy(allow_copy=True)  # object dtype
        for i in range(n):
            if nn_mask[i]:
                v = date_vals[i]
                if isinstance(v, str) and (v == "" or v.isspace()):
                    nn_mask[i] = False
    else:
        date_vals = ser.to_numpy(allow_copy=True)

    if not np.any(nn_mask):
        return out

    idx = np.nonzero(nn_mask)[0]
    d64 = np.empty(idx.shape[0], dtype="datetime64[D]")

    _is_na = is_na_helper
    _is_date_str_valid = is_date_str_valid

    for k, i in enumerate(idx):
        x = date_vals[i]

        if _is_na(x):
            raise ValueError(
                f"Internal error: NA value at row {int(i)} not marked in mask. "
                f"Value: {x}"
            )

        # numpy.datetime64, datetime, date — direct conversion.
        if isinstance(x, (np.datetime64, datetime, date)):
            d64[k] = np.datetime64(x, "D")
            continue

        if isinstance(x, str):
            xs = x.strip()
            if not _is_date_str_valid(xs):
                raise ValueError(
                    f"Invalid {col_name}='{xs}' at row {int(i)}! "
                    f"Expected strict YYYY-MM-DD format."
                )
            d64[k] = np.datetime64(xs, "D")
            continue

        raise TypeError(
            f"Invalid {col_name} type at row {int(i)}: "
            f"value = {x}, type = {type(x).__name__}. "
            "Expected YYYY-MM-DD string, datetime, or date."
        )

    out[idx] = datetime64D_to_ord32(d64)
    return out
