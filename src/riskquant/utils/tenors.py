# src/riskquant/utils/tenors.py
from __future__ import annotations

from datetime import date


def parse_bracket_token(tenor_token: str) -> tuple[str, int | None]:
    """
    "[Y-2]" -> ("Y", -2)
    "[M+0]" -> ("M", 0)
    "[D+24]" -> ("D", 24)
    "[HY+1]" -> ("HY", 1)
    "[ON]" -> ("ON", None)   (LABEL)
    """
    # Error.
    if tenor_token is None:
        raise ValueError("tenor_token cannot be None!")

    # Strip white space.
    s = tenor_token.strip()

    # Must start with '[' and end with ']'.
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    # Find the plus or minus sign.
    plus = s.rfind("+")
    minus = s.rfind("-")
    sign_pos = max(plus, minus)

    # If no sign.
    if sign_pos <= 0:
        return s, None

    # Unit (Y, M, D, HY, etc.)
    unit = s[:sign_pos]

    # Signed offset.
    offset = int(s[sign_pos:])

    # Return.
    return unit, offset


def get_spot_year(
    *,
    as_of_date: date,
    roll_month: int,
    roll_day: int,
) -> int:
    """
    Defines which delivery year corresponds to Y+0 under ROLL_ON_MMDD.

    The roll boundary (roll_month, roll_day) fully determines the spot year:
    on or after the boundary the spot year is the current year; before it,
    the previous year.
    """

    # Get the roll date,
    roll_date = date(as_of_date.year, roll_month, roll_day)

    # As-of-date is on or after the roll date -> current year.
    if as_of_date >= roll_date:
        return as_of_date.year

    # As-of-date is before the roll date -> previous year.
    return as_of_date.year - 1


def add_months_to_date_and_return_yyyymm01(
    *,
    d: date,
    months_to_add: int,
) -> date:
    """Add months to a month-start date (day=1)."""
    y = d.year + (d.month - 1 + months_to_add) // 12
    m = (d.month - 1 + months_to_add) % 12 + 1
    return date(y, m, 1)


def fmt_date_as_str_yyyy(d: date) -> str:
    return f"{d.year:04d}"


def fmt_date_as_str_yyyymm(d: date) -> str:
    return f"{d.year:04d}{d.month:02d}"


def fmt_date_as_str_yyyymmdd(d: date) -> str:
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"
