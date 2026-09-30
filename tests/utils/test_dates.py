# tests/utils/test_dates.py
"""
Contract of the ``riskquant.utils.dates`` functions that no longer use pandas.

``riskquant.utils.dates`` is on the ingest import path, so pandas left its
module level (D094). Three functions moved to the standard library:

- ``is_date_str_valid``: the 'YYYY-MM-DD' shape checks are unchanged; the
  calendar check is ``datetime.date.fromisoformat``. The accepted year range
  is Python's ``datetime.date`` range (0001-9999). The pandas check depended
  on the pandas version: 2.2 refused years outside 1677-2262; 3.0 accepted
  year 0000.
- ``date_str_to_date``: ``datetime.date.fromisoformat`` after the same check.
- ``ordinal32_to_date``: ``DATE_EPOCH + datetime.timedelta``.

``coerce_as_of_date_to_pd_timestamp`` and ``get_previous_business_date_as_*``
keep pandas semantics through a function-local import.
"""

from __future__ import annotations

import datetime as dt

import pytest

from riskquant.utils.dates import (
    coerce_as_of_date_to_pd_timestamp,
    date_str_to_date,
    date_to_ord32,
    is_date_str_valid,
    ordinal32_to_date,
)


@pytest.mark.parametrize(
    "text", ["2026-01-15", "2024-02-29", "1970-01-01", "0001-01-01", "9999-12-31"]
)
def test_valid_iso_dates(text: str) -> None:
    assert is_date_str_valid(text)
    assert date_str_to_date(text) == dt.date.fromisoformat(text)


@pytest.mark.parametrize(
    "text",
    [
        "2023-02-29",
        "2026-02-30",
        "2026-13-01",
        "2026-00-10",
        "0000-01-01",
        "2026-2-18",
        " 2026-02-18",
        "20260218",
        "2026/02/18",
    ],
)
def test_invalid_iso_dates(text: str) -> None:
    assert not is_date_str_valid(text)
    with pytest.raises(ValueError, match="is not in 'YYYY-MM-DD' format"):
        date_str_to_date(text)


def test_non_string_is_not_a_date_string() -> None:
    assert not is_date_str_valid(dt.date(2026, 1, 15))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "day", [dt.date(1970, 1, 1), dt.date(2026, 1, 15), dt.date(1900, 3, 1)]
)
def test_ordinal_round_trip_returns_a_plain_date(day: dt.date) -> None:
    back = ordinal32_to_date(int(date_to_ord32(day)))
    assert back == day
    assert type(back) is dt.date


def test_as_of_timestamp_contract_is_unchanged() -> None:
    import pandas as pd

    ts = coerce_as_of_date_to_pd_timestamp(dt.date(2026, 1, 2))
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("2026-01-02 23:59:59.999999999")
    assert coerce_as_of_date_to_pd_timestamp("2026-01-02") == pd.Timestamp("2026-01-02")
