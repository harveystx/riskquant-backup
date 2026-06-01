# src/riskquant/ingest_fixtures/config/risk_static_config/_parts/holiday_calendars.py

from __future__ import annotations

import pandas as pd

from riskquant.ingest_fixtures.schemas import RiskStaticConfigHolidayCalendar
from riskquant.ingest_fixtures.utils import validate_column_lengths

# Five named calendars covering AUD, CHF, EUR, GBP, USD, plus the
# Eurozone TARGET2 settlement calendar (named EUR). Each entry includes
# the year-end pair plus at least one mid-year holiday to exercise the
# business-day engine across the trading year.

_CALENDAR_ENTRIES: list[tuple[str, str]] = [
    # --- USD (Federal Reserve calendar; subset) ---
    ("US", "2026-01-01"),  # New Year's Day
    ("US", "2026-07-03"),  # Independence Day (observed)
    ("US", "2026-11-26"),  # Thanksgiving
    ("US", "2026-12-25"),  # Christmas Day
    ("US", "2027-01-01"),
    ("US", "2027-07-05"),
    ("US", "2027-11-25"),
    ("US", "2027-12-24"),

    # --- GBP (Bank of England subset) ---
    ("GB", "2026-01-01"),
    ("GB", "2026-05-04"),  # Early May bank holiday
    ("GB", "2026-08-31"),  # Summer bank holiday
    ("GB", "2026-12-25"),
    ("GB", "2026-12-28"),  # Boxing Day (observed)
    ("GB", "2027-01-01"),
    ("GB", "2027-05-03"),
    ("GB", "2027-08-30"),
    ("GB", "2027-12-27"),
    ("GB", "2027-12-28"),

    # --- CHF (Zurich SIX) ---
    ("CH", "2026-01-01"),
    ("CH", "2026-01-02"),  # Berchtoldstag
    ("CH", "2026-08-03"),  # Swiss National Day (observed)
    ("CH", "2026-12-25"),
    ("CH", "2026-12-28"),  # Stephen (observed)
    ("CH", "2027-01-01"),
    ("CH", "2027-08-02"),
    ("CH", "2027-12-27"),

    # --- AUD (ASX subset) ---
    ("AU", "2026-01-01"),
    ("AU", "2026-01-26"),  # Australia Day
    ("AU", "2026-04-27"),  # ANZAC Day (observed)
    ("AU", "2026-12-25"),
    ("AU", "2026-12-28"),  # Boxing Day (observed)
    ("AU", "2027-01-01"),
    ("AU", "2027-01-26"),
    ("AU", "2027-04-26"),

    # --- EUR (TARGET2 — settlement calendar for euro-area) ---
    ("EUR", "2026-01-01"),
    ("EUR", "2026-05-01"),  # Labour Day
    ("EUR", "2026-12-25"),
    ("EUR", "2026-12-28"),
    ("EUR", "2027-01-01"),
    ("EUR", "2027-05-03"),
    ("EUR", "2027-12-27"),
    ("EUR", "2027-12-28"),
]


def _get_data() -> RiskStaticConfigHolidayCalendar:
    names, dates = zip(*_CALENDAR_ENTRIES, strict=True)
    return {
        "holiday_calendar_name": list(names),
        "holiday_date":          list(dates),
    }


CANONICAL_HOLIDAY_CALENDARS_DATA: RiskStaticConfigHolidayCalendar = _get_data()

validate_column_lengths(CANONICAL_HOLIDAY_CALENDARS_DATA)

_CANONICAL_HOLIDAY_CALENDARS_DF = pd.DataFrame(CANONICAL_HOLIDAY_CALENDARS_DATA)


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_dict,
        print_success_message,
    )

    print_calling_file()

    print("Canonical Holiday Calendars Data:")
    print_dict(CANONICAL_HOLIDAY_CALENDARS_DATA)

    print("As DataFrame:")
    print_df(_CANONICAL_HOLIDAY_CALENDARS_DF, index=False)

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
