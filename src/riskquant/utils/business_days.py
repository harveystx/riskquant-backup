# src/riskquant/utils/business_days.py
"""
Pandas-free business-day arithmetic on int32 date ordinals (D080).

The compiled static config stores holidays as `HolidayCalendar.holiday_dates_ord`:
a SORTED `int32` array of days since `DATE_EPOCH`. Every function here takes that
array directly, so calendar membership is an O(log N) `np.searchsorted` bisection
and no date object is ever constructed in the loop.

Why a new module rather than `riskquant.utils.dates`: that module imports pandas,
and the curve-construction path must stay pandas-free (the compiled-config
boundary test pins this). Nothing here imports pandas, polars or any config type;
it is scalar integer arithmetic over an ordinal axis.

Weekend rule
------------
Saturday and Sunday. `DATE_EPOCH` (1970-01-01) is a Thursday, so
`(ord + 4) % 7` is 5 for Saturday and 6 for Sunday.

All functions are ordinal-in / ordinal-out. Labels, conventions and calendar
NAMES are resolved by the caller at the edge; this module never sees a string
except the business-day-convention code it dispatches on.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as dt

import numpy as np
import numpy.typing as npt

from riskquant.config.enums.financial_conventions import BusinessDayConvention
from riskquant.constants import DATE_EPOCH

#: Weekday of the ordinal epoch (1970-01-01 is a Thursday, weekday() == 3).
_EPOCH_WEEKDAY: int = DATE_EPOCH.weekday()


def _weekday_ord(date_ord: int) -> int:
    """Monday 0 … Sunday 6 for an ordinal, without constructing a date."""
    return (int(date_ord) + _EPOCH_WEEKDAY) % 7


def is_business_day_ord(*, date_ord: int, holidays_ord: npt.NDArray[np.int32]) -> bool:
    """True when the ordinal is a weekday and not in the calendar."""
    if _weekday_ord(date_ord) >= 5:
        return False
    idx = int(np.searchsorted(holidays_ord, np.int32(date_ord)))
    return not (idx < holidays_ord.shape[0] and int(holidays_ord[idx]) == int(date_ord))


def next_business_day_ord(*, date_ord: int, holidays_ord: npt.NDArray[np.int32]) -> int:
    """The first business day strictly after `date_ord`."""
    cursor = int(date_ord) + 1
    while not is_business_day_ord(date_ord=cursor, holidays_ord=holidays_ord):
        cursor += 1
    return cursor


def business_day_add_ord(
    *, date_ord: int, n_days: int, holidays_ord: npt.NDArray[np.int32]
) -> int:
    """
    Move `n_days` business days from `date_ord`.

    `n_days == 0` rolls FOLLOWING onto the next business day, matching the
    spot-date convention: a spot lag of zero on a holiday still settles on a
    business day.
    """
    if n_days < 0:
        raise ValueError(f"business_day_add_ord: n_days must be >= 0, got {n_days}.")
    cursor = int(date_ord)
    if n_days == 0:
        while not is_business_day_ord(date_ord=cursor, holidays_ord=holidays_ord):
            cursor += 1
        return cursor
    for _ in range(n_days):
        cursor = next_business_day_ord(date_ord=cursor, holidays_ord=holidays_ord)
    return cursor


def adjust_ord(
    *,
    date_ord: int,
    convention: str,
    holidays_ord: npt.NDArray[np.int32],
) -> int:
    """
    Apply a business-day convention to an unadjusted ordinal.

    MOD_FOLLOWING rolls forward unless that crosses into the next calendar
    month, in which case it rolls backward — the only convention D080 uses.
    """
    if is_business_day_ord(date_ord=date_ord, holidays_ord=holidays_ord):
        return int(date_ord)

    if convention == BusinessDayConvention.NONE.value:
        return int(date_ord)

    forward = int(date_ord)
    while not is_business_day_ord(date_ord=forward, holidays_ord=holidays_ord):
        forward += 1
    backward = int(date_ord)
    while not is_business_day_ord(date_ord=backward, holidays_ord=holidays_ord):
        backward -= 1

    if convention == BusinessDayConvention.FOLLOWING.value:
        return forward
    if convention == BusinessDayConvention.PRECEDING.value:
        return backward
    if convention == BusinessDayConvention.MOD_FOLLOWING.value:
        return (
            backward if _month_of_ord(forward) != _month_of_ord(date_ord) else forward
        )
    if convention == BusinessDayConvention.MOD_PRECEDING.value:
        return (
            forward if _month_of_ord(backward) != _month_of_ord(date_ord) else backward
        )
    raise ValueError(
        f"adjust_ord: unsupported business-day convention {convention!r}; "
        f"expected one of {sorted(c.value for c in BusinessDayConvention)}."
    )


def _month_of_ord(date_ord: int) -> tuple[int, int]:
    d = ord_to_date(date_ord)
    return d.year, d.month


def ord_to_date(date_ord: int) -> dt.date:
    """Ordinal -> `datetime.date`. Edge-only helper; never used in a kernel."""
    return DATE_EPOCH + dt.timedelta(days=int(date_ord))


def date_to_ord(value: dt.date) -> int:
    """`datetime.date` -> ordinal. Edge-only helper; never used in a kernel."""
    return (value - DATE_EPOCH).days


def add_months_ord(*, date_ord: int, n_months: int) -> int:
    """
    Calendar month roll, clamped to the target month's last calendar day.

    Unadjusted: business-day adjustment is the caller's separate step, so the
    two conventions stay independently testable.
    """
    d = ord_to_date(date_ord)
    total = d.year * 12 + (d.month - 1) + int(n_months)
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, _calendar.monthrange(year, month)[1])
    return date_to_ord(dt.date(year, month, day))


def add_years_ord(*, date_ord: int, n_years: int) -> int:
    """Calendar year roll; 29 February clamps to 28 February in a common year."""
    return add_months_ord(date_ord=date_ord, n_months=12 * int(n_years))


def last_business_day_of_month_ord(
    *, year: int, month: int, holidays_ord: npt.NDArray[np.int32]
) -> int:
    """Last BUSINESS day of the month, not the last calendar day."""
    cursor = date_to_ord(dt.date(year, month, _calendar.monthrange(year, month)[1]))
    while not is_business_day_ord(date_ord=cursor, holidays_ord=holidays_ord):
        cursor -= 1
    return cursor


def is_last_business_day_of_month_ord(
    *, date_ord: int, holidays_ord: npt.NDArray[np.int32]
) -> bool:
    """True when the ordinal IS its month's last business day."""
    year, month = _month_of_ord(date_ord)
    return int(date_ord) == last_business_day_of_month_ord(
        year=year, month=month, holidays_ord=holidays_ord
    )
