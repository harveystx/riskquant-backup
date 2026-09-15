# src/riskquant/config/enums/roll_methods.py
"""
Tenor-resolution taxonomy: the single authoritative roll contract.

A roll method specifies how a relative tenor token (e.g. [M+1], [Y+0],
[HY+0]) resolves to an absolute risk-factor name for a given as-of date.
The convention is not a standalone market entity. Each curve / surface
declares its own ``roll_method`` and roll parameters directly. There is
no separate roll-rule table.

This module owns every rule the validator and the resolver must agree on,
so the two cannot drift:

  RollMethod
      the four resolution methods.
  ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS
      the legal (XAxisKind, RollMethod) combinations.
  ROLL_METHOD_REQUIRED_PARAMS / ROLL_PARAM_FIELDS
      which roll parameters each method requires and which it forbids.
  parse_fh_bh_mmdd
      the one parser for the 'MMDD|MMDD' half-year convention.

Absolute name formats by combination:
  (OFFSET_M,  CALENDAR_PERIOD)     -> "<curve>_YYYYMM"
  (OFFSET_M,  HALF_YEAR_BY_DATE)   -> "<curve>_YYYYMMDD"
  (OFFSET_Y,  CALENDAR_PERIOD)     -> "<curve>_YYYY"
  (OFFSET_Y,  ROLL_ON_MMDD)        -> "<curve>_YYYY"
  (OFFSET_HY, HALF_YEAR_BY_DATE)   -> "<curve>_YYYYMMDD"
  (OFFSET_HY, HALF_YEAR_BY_LABEL)  -> "<curve>_YYYY_FH" or "<curve>_YYYY_BH"

Roll parameters and their semantics:
  anchor_offset  int  CALENDAR_PERIOD only. Offset in tenor units (months
                      for OFFSET_M, years for OFFSET_Y) added to the as-of
                      period to form the spot anchor. The token offset is
                      applied on top of that anchor. It modifies the
                      as-of anchor, never the token.

                      Exact CALENDAR_PERIOD formula:
                          resolved_period = as_of_period
                                            + anchor_offset   (per-curve const)
                                            + token_offset    (per-point [M/Y+N])
                      where the period unit is months (OFFSET_M) or years
                      (OFFSET_Y). anchor_offset and the [M+N]/[Y+N]/[HY+N]
                      token offset are DISTINCT inputs: anchor_offset is a
                      single per-curve constant that positions the spot/front
                      contract; the token offset varies per curve point along
                      the tenor ladder. Both add, but they are not the same
                      quantity.
  roll_month     int  ROLL_ON_MMDD only. Month of the annual roll boundary.
  roll_day       int  ROLL_ON_MMDD only. Day of the annual roll boundary.
  fh_bh_mmdd     str  HALF_YEAR_BY_DATE / HALF_YEAR_BY_LABEL only. The
                      first-half / back-half boundary as 'MMDD|MMDD'.
"""

from __future__ import annotations

from calendar import monthrange
from enum import IntEnum

from riskquant.config.enums.x_axis_kinds import XAxisKind


class RollMethod(IntEnum):
    CALENDAR_PERIOD = 0
    ROLL_ON_MMDD = 1
    HALF_YEAR_BY_DATE = 2
    HALF_YEAR_BY_LABEL = 3


VALID_ROLL_METHODS: frozenset[str] = frozenset(m.name for m in RollMethod)


ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS: frozenset[tuple[XAxisKind, RollMethod]] = (
    frozenset(
        {
            (XAxisKind.OFFSET_M, RollMethod.CALENDAR_PERIOD),
            (XAxisKind.OFFSET_M, RollMethod.HALF_YEAR_BY_DATE),
            (XAxisKind.OFFSET_Y, RollMethod.CALENDAR_PERIOD),
            (XAxisKind.OFFSET_Y, RollMethod.ROLL_ON_MMDD),
            (XAxisKind.OFFSET_HY, RollMethod.HALF_YEAR_BY_DATE),
            (XAxisKind.OFFSET_HY, RollMethod.HALF_YEAR_BY_LABEL),
        }
    )
)


# ------------------------------------------------------------------ #
# Method-specific parameter contract.
# ------------------------------------------------------------------ #
#
# Every roll parameter field. A curve / surface roll config carries
# exactly the fields its method requires; every other field must be null.
# The validator enforces this. The resolver reads only required fields.

ROLL_PARAM_FIELDS: frozenset[str] = frozenset(
    {"anchor_offset", "roll_month", "roll_day", "fh_bh_mmdd"}
)

ROLL_METHOD_REQUIRED_PARAMS: dict[RollMethod, frozenset[str]] = {
    RollMethod.CALENDAR_PERIOD: frozenset({"anchor_offset"}),
    RollMethod.ROLL_ON_MMDD: frozenset({"roll_month", "roll_day"}),
    RollMethod.HALF_YEAR_BY_DATE: frozenset({"fh_bh_mmdd"}),
    RollMethod.HALF_YEAR_BY_LABEL: frozenset({"fh_bh_mmdd"}),
}

# Every method must declare its parameter contract. A new method with no
# entry here is a programming error, caught at import.
assert set(ROLL_METHOD_REQUIRED_PARAMS) == set(RollMethod), (
    "ROLL_METHOD_REQUIRED_PARAMS must cover every RollMethod."
)
assert all(req <= ROLL_PARAM_FIELDS for req in ROLL_METHOD_REQUIRED_PARAMS.values()), (
    "Every required roll parameter must be a member of ROLL_PARAM_FIELDS."
)


def roll_method_forbidden_params(method: RollMethod) -> frozenset[str]:
    """Return the roll-parameter fields that must be null for ``method``."""
    return ROLL_PARAM_FIELDS - ROLL_METHOD_REQUIRED_PARAMS[method]


# ------------------------------------------------------------------ #
# Half-year boundary convention: one parser for validator and resolver.
# ------------------------------------------------------------------ #


def parse_fh_bh_mmdd(fh_bh_mmdd: str) -> tuple[int, int, int, int]:
    """
    Parse and validate a 'MMDD|MMDD' half-year convention.

    Returns (fh_month, fh_day, bh_month, bh_day). Raises ValueError on a
    malformed string, an out-of-range month, or a day invalid for its
    month. This is the single authority; the validator (fail-fast at
    ingest) and the resolver (runtime) both call it.

    Parameters
    ----------
    fh_bh_mmdd
        First-half and back-half boundaries, e.g. "0401|1001".

    Returns
    -------
    tuple[int, int, int, int]
        (fh_month, fh_day, bh_month, bh_day).
    """
    parts = fh_bh_mmdd.split("|")
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 4:
        raise ValueError(f"fh_bh_mmdd must be 'MMDD|MMDD', got {fh_bh_mmdd!r}.")
    if not (parts[0].isdigit() and parts[1].isdigit()):
        raise ValueError(f"fh_bh_mmdd must be all digits, got {fh_bh_mmdd!r}.")

    fh_mm, fh_dd = int(parts[0][:2]), int(parts[0][2:])
    bh_mm, bh_dd = int(parts[1][:2]), int(parts[1][2:])

    for mm, dd, half in ((fh_mm, fh_dd, "first-half"), (bh_mm, bh_dd, "back-half")):
        if not 1 <= mm <= 12:
            raise ValueError(
                f"fh_bh_mmdd {half} month {mm} out of range [1, 12] in {fh_bh_mmdd!r}."
            )
        # 2001 is a non-leap year: rejects 0229, which is not a fixed
        # annual roll boundary.
        max_day = monthrange(2001, mm)[1]
        if not 1 <= dd <= max_day:
            raise ValueError(
                f"fh_bh_mmdd {half} day {dd} out of range [1, {max_day}] for "
                f"month {mm} in {fh_bh_mmdd!r}."
            )

    return fh_mm, fh_dd, bh_mm, bh_dd
