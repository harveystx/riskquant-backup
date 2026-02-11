# riskquant/utils/dates.py
from __future__ import annotations

import datetime as dt
from typing import Iterable, Final

import numpy as np
import pandas as pd

DATE_EPOCH: Final[dt.date] = dt.date(1970, 1, 1)
DATE_EPOCH_AS_DATETIME64D: Final[np.datetime64] = np.datetime64(
    DATE_EPOCH.isoformat(),
    "D",
)


def coerce_as_of_date_to_pd_timestamp(
    as_of_date: pd.Timestamp | dt.date | pd.Timestamp | str,
) -> pd.Timestamp:
    """
    Accepts: pd.Timestamp | datetime.datetime | datetime.date | str
    Returns: pd.Timestamp (tz-naive)
    """
    # If str, validate format.
    if isinstance(as_of_date, str):
        assert_date_str_is_iso_format(as_of_date)

    # As pd.Timestamp
    ts = pd.Timestamp(as_of_date)

    # If caller passed a date (no time), treat it as end-of-day inclusive
    # so 2026-01-02 includes all times on 2026-01-02.
    if isinstance(as_of_date, dt.date) and not isinstance(as_of_date, dt.datetime):
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    # Normalize tz handling (if any). Adjust policy as you like.
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    return ts


def is_date_str_valid(date_str: str) -> bool:
    """Checks if date_str which must have form %Y-%m-%d is a valid calendar date."""
    if not isinstance(date_str, str):
        return False
    if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
        return False
    if not date_str.replace("-", "").isdigit():
        return False

    try:
        pd.to_datetime(date_str, format="%Y-%m-%d", errors="raise")
    except ValueError:
        return False

    return True


def assert_date_str_is_iso_format(date_str: str) -> None:
    """Assert that date_str is in ISO format 'YYYY-MM-DD'."""
    if not is_date_str_valid(date_str):
        raise ValueError(f"Date string '{date_str}' is not in 'YYYY-MM-DD' format.")


def date_str_to_date(date_str: str) -> dt.date:
    """Convert a date string in 'YYYY-MM-DD' format to a datetime.date object."""
    assert_date_str_is_iso_format(date_str)
    return pd.to_datetime(date_str, format="%Y-%m-%d", errors="raise").date()


def date_to_ordinal32(d: dt.date) -> np.int32:
    return np.int32((d - DATE_EPOCH).days)


def ordinal32_to_date(days: int) -> dt.date:
    return DATE_EPOCH + pd.Timedelta(days=int(days))  # or datetime.timedelta


def datetime64D_to_ordinal32(dts: np.ndarray) -> np.ndarray:
    """
    Convert numpy datetime64[D] array -> int32 days since DATE_EPOCH.
    Returns a view/copy depending on input; no pandas involvement.
    """
    dts = np.asarray(dts, dtype="datetime64[D]")
    return (dts - DATE_EPOCH_AS_DATETIME64D).astype(np.int32, copy=False)


def ordinal32_to_datetime64D(days: np.ndarray) -> np.ndarray:
    """
    Convert int32 days since DATE_EPOCH -> numpy datetime64[D] array.
    """
    days = np.asarray(days, dtype=np.int32)
    # Cast to timedelta64[D] then add epoch
    return DATE_EPOCH_AS_DATETIME64D + days.astype("timedelta64[D]", copy=False)


def ym_str_to_int_tuple(ym: str) -> tuple[int, int]:
    """Convert 'YYYYMM' -> (YYYY, MM)."""
    if len(ym) != 6 or not ym.isdigit():
        raise ValueError(f"Invalid YM string: {ym}")
    y, m = int(ym[:4]), int(ym[4:6])
    if not 1 <= m <= 12:
        raise ValueError(f"Invalid YM string: {ym}")
    return y, m


def ymd_str_to_date(ymd: str) -> dt.date:
    """Convert 'YYYYMMDD' -> datetime.date."""
    if len(ymd) != 8 or not ymd.isdigit():
        raise ValueError(f"Invalid YMD string: {ymd}")
    return dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))


def month_diff(
    *,
    anchor_ym: tuple[int, int],
    target_ym: tuple[int, int],
) -> int:
    """Difference in months: target - anchor."""
    ay, am = anchor_ym
    ty, tm = target_ym
    return (ty - ay) * 12 + (tm - am)


def next_half_year_boundary(
    t0: dt.date,
    ym_boundaries: Iterable[str] = ("0630", "1231"),
) -> dt.date:
    """
    Return the first boundary date >= t0 among {YYYY0630, YYYY1231} (or configured boundaries).
    If none remain in the year, return the first boundary in the next year.
    """
    ym_bounds = list(ym_boundaries)
    if not ym_bounds:
        raise ValueError("No half-year boundaries provided.")

    year = t0.year
    candidates = []
    for b in ym_bounds:
        if len(b) != 4 or not b.isdigit():
            raise ValueError(f"Invalid year-month boundary 'MMDD': {b}")
        m = int(b[:2])
        d = int(b[2:])
        candidates.append(dt.date(year, m, d))

    for d in sorted(candidates):
        if d >= t0:
            return d

    # none left this year -> first boundary next year (sorted)
    year += 1
    b0 = sorted(ym_bounds)[0]
    return dt.date(year, int(b0[:2]), int(b0[2:]))


def half_year_index(d: dt.date) -> tuple[int, int]:
    """
    Map boundary date to a half-year index:
      YYYY-06-30 -> (YYYY, 0)
      YYYY-12-31 -> (YYYY, 1)
    """
    if (d.month, d.day) == (6, 30):
        return d.year, 0
    if (d.month, d.day) == (12, 31):
        return d.year, 1
    raise ValueError(f"Not a recognized half-year boundary: {d.isoformat()}")


def half_year_diff(
    *,
    anchor: dt.date,
    target: dt.date,
) -> int:
    """Difference in half-years: target - anchor, where each boundary step counts as 1."""
    ay, ah = half_year_index(anchor)
    ty, th = half_year_index(target)
    return (ty - ay) * 2 + (th - ah)
