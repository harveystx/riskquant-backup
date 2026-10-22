# src/riskquant/ingest_fixtures/factor_regression/calendars.py
"""
Synthetic exchange calendars for the canonical factor-regression fixture.

Every case simulates its latent paths on one shared weekday grid. Each series
is then observed only on the business days of its own calendar. Two series on
different calendars therefore share only a subset of dates, and
``align_levels`` must record the other dates as ``NOT_COMMON_DATE``.

Why hard-coded holidays
-----------------------
The fixture must not depend on a holiday library or on the static-config
calendar tables. A literal list is auditable by inspection and cannot drift.
Only holidays that fall on a weekday inside the grid are listed. The lists
approximate the 2023 to 2026 calendars of each venue (for example, the SIX
list omits 24 and 31 December). The fixture needs calendars that differ in a
known way, not exact venue calendars.

Grid
----
``GRID_START_DATE`` is the first Monday of 2023. ``GRID_END_DATE`` is the
canonical as-of date (inclusive). The grid holds every weekday in between.
"""

from __future__ import annotations

import enum
from datetime import date, timedelta
from typing import Final

import numpy as np
import numpy.typing as npt

from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.utils.business_days import date_to_ord

__all__ = [
    "GRID_END_DATE",
    "GRID_END_ORD",
    "GRID_SIZE",
    "GRID_START_DATE",
    "GRID_START_ORD",
    "GRID_WINDOW_END_DATE",
    "HOLIDAYS",
    "WEEKDAY_GRID_ORD",
    "Calendar",
    "calendar_mask",
    "grid_index_of",
]


class Calendar(enum.Enum):
    """Observation calendar of one fixture series."""

    FX = "FX"  # Global FX and factor-index fixings: 1 January and 25 December.
    TARGET = "TARGET"  # Euro-area exchanges (Xetra, Euronext, ICE Endex).
    NYSE = "NYSE"  # United States equity exchanges.
    SIX = "SIX"  # Swiss exchange and CHF rates fixings.


GRID_START_DATE: Final[date] = date(2023, 1, 2)
GRID_END_DATE: Final[date] = CANONICAL_AS_OF_DATE

# Exclusive end of the full-history window: the day after the as-of date.
GRID_WINDOW_END_DATE: Final[date] = GRID_END_DATE + timedelta(days=1)

GRID_START_ORD: Final[int] = date_to_ord(GRID_START_DATE)
GRID_END_ORD: Final[int] = date_to_ord(GRID_END_DATE)


def _weekday_grid() -> npt.NDArray[np.int32]:
    # Day 0 (1970-01-01) is a Thursday, so (ord + 3) % 7 is 0 on Monday.
    days = np.arange(GRID_START_ORD, GRID_END_ORD + 1, dtype=np.int32)
    keep = ((days + 3) % 7) < 5
    out = np.ascontiguousarray(days[keep])
    out.setflags(write=False)
    return out


WEEKDAY_GRID_ORD: Final[npt.NDArray[np.int32]] = _weekday_grid()
GRID_SIZE: Final[int] = int(WEEKDAY_GRID_ORD.shape[0])

_GOOD_FRIDAY: Final = (date(2023, 4, 7), date(2024, 3, 29), date(2025, 4, 18))
_EASTER_MONDAY: Final = (date(2023, 4, 10), date(2024, 4, 1), date(2025, 4, 21))

HOLIDAYS: Final[dict[Calendar, tuple[date, ...]]] = {
    Calendar.FX: (
        date(2023, 12, 25),
        date(2024, 1, 1),
        date(2024, 12, 25),
        date(2025, 1, 1),
        date(2025, 12, 25),
        date(2026, 1, 1),
    ),
    Calendar.TARGET: (
        *_GOOD_FRIDAY,
        *_EASTER_MONDAY,
        date(2023, 5, 1),
        date(2023, 12, 25),
        date(2023, 12, 26),
        date(2024, 1, 1),
        date(2024, 5, 1),
        date(2024, 12, 25),
        date(2024, 12, 26),
        date(2025, 1, 1),
        date(2025, 5, 1),
        date(2025, 12, 25),
        date(2025, 12, 26),
        date(2026, 1, 1),
    ),
    Calendar.NYSE: (
        *_GOOD_FRIDAY,
        date(2023, 1, 2),
        date(2023, 1, 16),
        date(2023, 2, 20),
        date(2023, 5, 29),
        date(2023, 6, 19),
        date(2023, 7, 4),
        date(2023, 9, 4),
        date(2023, 11, 23),
        date(2023, 12, 25),
        date(2024, 1, 1),
        date(2024, 1, 15),
        date(2024, 2, 19),
        date(2024, 5, 27),
        date(2024, 6, 19),
        date(2024, 7, 4),
        date(2024, 9, 2),
        date(2024, 11, 28),
        date(2024, 12, 25),
        date(2025, 1, 1),
        date(2025, 1, 9),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        date(2026, 1, 1),
    ),
    Calendar.SIX: (
        *_GOOD_FRIDAY,
        *_EASTER_MONDAY,
        date(2023, 1, 2),
        date(2023, 5, 18),
        date(2023, 5, 29),
        date(2023, 8, 1),
        date(2023, 12, 25),
        date(2023, 12, 26),
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 5, 9),
        date(2024, 5, 20),
        date(2024, 8, 1),
        date(2024, 12, 25),
        date(2024, 12, 26),
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 5, 29),
        date(2025, 6, 9),
        date(2025, 8, 1),
        date(2025, 12, 25),
        date(2025, 12, 26),
        date(2026, 1, 1),
        date(2026, 1, 2),
    ),
}


def grid_index_of(d: date) -> int:
    """
    Return the grid index of weekday ``d``.

    Raises
    ------
    ValueError
        ``d`` is not a weekday inside the grid.
    """
    target = date_to_ord(d)
    i = int(np.searchsorted(WEEKDAY_GRID_ORD, target))
    if i >= GRID_SIZE or int(WEEKDAY_GRID_ORD[i]) != target:
        raise ValueError(f"{d.isoformat()} is not a weekday of the fixture grid")
    return i


def calendar_mask(calendar: Calendar) -> npt.NDArray[np.bool_]:
    """
    Boolean ``(GRID_SIZE,)`` mask of the grid days that ``calendar`` observes.

    Raises
    ------
    ValueError
        A listed holiday is not a weekday of the grid. A stale literal must
        fail loud, not drop out of the mask silently.
    """
    mask = np.ones(GRID_SIZE, dtype=np.bool_)
    holiday_index = np.array(
        [grid_index_of(d) for d in HOLIDAYS[calendar]], dtype=np.int64
    )
    mask[holiday_index] = False
    return mask
