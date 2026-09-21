# tests/market_data/remediation/d080_oracle.py
"""
Independent D080 oracle: labels -> maturities -> ACT/365F T -> ln DF -> DF.

Imports stdlib + numpy ONLY. It must never import a production maturity,
calendar, conversion, interpolation or curve-construction helper, or the test
suite is checking the implementation against itself. `test_d080_curve_plumbing.py`
machine-checks that with a source scan.

It consumes RAW inputs only: holiday dates as `datetime.date`, raw curve metadata
values, and raw snapshot levels in their declared PERCENT units (so the oracle
applies its own x 0.01 while production applies D069's, independently).
"""

from __future__ import annotations

import calendar as _calendar
import datetime as dt
import math

_EPOCH = dt.date(1970, 1, 1)
_SUPPORTED = ("ON", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y")


def to_ord(value: dt.date) -> int:
    return (value - _EPOCH).days


def is_bd(value: dt.date, holidays: frozenset[dt.date]) -> bool:
    return value.weekday() < 5 and value not in holidays


def next_bd(value: dt.date, holidays: frozenset[dt.date]) -> dt.date:
    cursor = value + dt.timedelta(days=1)
    while not is_bd(cursor, holidays):
        cursor += dt.timedelta(days=1)
    return cursor


def bd_add(value: dt.date, n: int, holidays: frozenset[dt.date]) -> dt.date:
    cursor = value
    if n == 0:
        while not is_bd(cursor, holidays):
            cursor += dt.timedelta(days=1)
        return cursor
    for _ in range(n):
        cursor = next_bd(cursor, holidays)
    return cursor


def mod_following(value: dt.date, holidays: frozenset[dt.date]) -> dt.date:
    forward = value
    while not is_bd(forward, holidays):
        forward += dt.timedelta(days=1)
    if forward.month == value.month:
        return forward
    backward = value
    while not is_bd(backward, holidays):
        backward -= dt.timedelta(days=1)
    return backward


def last_bd_of_month(year: int, month: int, holidays: frozenset[dt.date]) -> dt.date:
    cursor = dt.date(year, month, _calendar.monthrange(year, month)[1])
    while not is_bd(cursor, holidays):
        cursor -= dt.timedelta(days=1)
    return cursor


def add_months(value: dt.date, n: int) -> dt.date:
    total = value.year * 12 + (value.month - 1) + n
    year, month = divmod(total, 12)
    month += 1
    return dt.date(year, month, min(value.day, _calendar.monthrange(year, month)[1]))


def maturity(
    *,
    label: str,
    as_of: dt.date,
    holidays: frozenset[dt.date],
    spot_lag: int = 2,
    end_of_month: bool = True,
) -> dt.date:
    """D080 LABEL -> maturity. ON bypasses spot; nM/nY honour the EOM rule."""
    if label not in _SUPPORTED:
        raise ValueError(f"oracle: unsupported D080 label {label!r}")
    if label == "ON":
        return next_bd(as_of, holidays)
    spot = bd_add(as_of, spot_lag, holidays)
    spot_is_eom = end_of_month and spot == last_bd_of_month(
        spot.year, spot.month, holidays
    )
    unit, count = label[-1], int(label[:-1])
    if unit == "W":
        return mod_following(spot + dt.timedelta(days=7 * count), holidays)
    months = count if unit == "M" else 12 * count
    target = add_months(spot, months)
    if spot_is_eom:
        return last_bd_of_month(target.year, target.month, holidays)
    return mod_following(target, holidays)


def act365f(as_of: dt.date, value: dt.date) -> float:
    return (to_ord(value) - to_ord(as_of)) / 365.0


def pillar_ln_df(*, percent_level: float, t_years: float, shock: float = 0.0) -> float:
    """PERCENT -> decimal (the oracle's own conversion), + shock, then -zT."""
    z = percent_level * 0.01 + shock
    return -z * t_years


def curve(
    *,
    labels: tuple[str, ...],
    percent_levels: dict[str, float],
    as_of: dt.date,
    holidays: frozenset[dt.date],
    shocks: dict[str, float] | None = None,
) -> tuple[list[float], list[float]]:
    """
    Build the canonical pillar arrays, anchor first.

    Returns `(T, lnDF)` sorted by increasing T, with the synthetic `(0.0, 0.0)`
    anchor prepended. Duplicate T is an error here too.
    """
    shocks = shocks or {}
    rows = []
    for label in labels:
        mat = maturity(label=label, as_of=as_of, holidays=holidays)
        t = act365f(as_of, mat)
        rows.append((
            t,
            pillar_ln_df(
                percent_level=percent_levels[label],
                t_years=t,
                shock=shocks.get(label, 0.0),
            ),
        ))
    rows.sort(key=lambda r: r[0])
    coords = [r[0] for r in rows]
    if len(set(coords)) != len(coords):
        raise ValueError(f"oracle: duplicate pillar coordinates {coords}")
    return [0.0, *coords], [0.0, *[r[1] for r in rows]]


def discount_factor(
    *, t_target: float, pillar_t: list[float], pillar_ln: list[float]
) -> float:
    """Linear in ln DF; right tail extends the final slope (flat forward)."""
    if t_target < 0.0:
        raise ValueError("oracle: target before the as-of has no discount factor")
    if t_target <= pillar_t[0]:
        return math.exp(pillar_ln[0])
    if t_target > pillar_t[-1]:
        slope = (pillar_ln[-1] - pillar_ln[-2]) / (pillar_t[-1] - pillar_t[-2])
        return math.exp(pillar_ln[-1] + slope * (t_target - pillar_t[-1]))
    for i in range(1, len(pillar_t)):
        if t_target <= pillar_t[i]:
            w = (t_target - pillar_t[i - 1]) / (pillar_t[i] - pillar_t[i - 1])
            return math.exp(pillar_ln[i - 1] + w * (pillar_ln[i] - pillar_ln[i - 1]))
    raise AssertionError("unreachable")
