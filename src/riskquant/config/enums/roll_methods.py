# src/riskquant/config/enums/roll_methods.py
"""
Roll method taxonomy for spot roll rules.

Replaces the prior two ad-hoc declarations: the string-keyed
`_VALID_SPOT_ROLL_METHODS` frozenset in validate.py and the
`_SpotRollMethod` constant class in rf_resolution.py.

A roll method specifies how a relative tenor token (e.g. [M+1], [Y+0],
[HY+0]) resolves to an absolute risk-factor name for a given as-of date.
The (XAxisKind, RollMethod) pair determines both the token grammar and
the absolute name format; allowed pairs are enumerated explicitly in
ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS so the validator and the
resolution dispatch cannot drift.

Absolute name formats by combination:
  (OFFSET_M,  CALENDAR_PERIOD)     -> "<curve>_YYYYMM"
  (OFFSET_M,  HALF_YEAR_BY_DATE)   -> "<curve>_YYYYMMDD"
  (OFFSET_Y,  CALENDAR_PERIOD)     -> "<curve>_YYYY"
  (OFFSET_Y,  ROLL_ON_MMDD)        -> "<curve>_YYYY"
  (OFFSET_HY, HALF_YEAR_BY_DATE)   -> "<curve>_YYYYMMDD"
  (OFFSET_HY, HALF_YEAR_BY_LABEL)  -> "<curve>_YYYY_FH" or "<curve>_YYYY_BH"
"""

from __future__ import annotations

from enum import IntEnum

from riskquant.config.enums.x_axis_kinds import XAxisKind


class RollMethod(IntEnum):
    CALENDAR_PERIOD = 0
    ROLL_ON_MMDD = 1
    HALF_YEAR_BY_DATE = 2
    HALF_YEAR_BY_LABEL = 3


VALID_ROLL_METHODS: frozenset[str] = frozenset(m.name for m in RollMethod)


ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS: frozenset[tuple[XAxisKind, RollMethod]] = frozenset(
    {
        (XAxisKind.OFFSET_M, RollMethod.CALENDAR_PERIOD),
        (XAxisKind.OFFSET_M, RollMethod.HALF_YEAR_BY_DATE),
        (XAxisKind.OFFSET_Y, RollMethod.CALENDAR_PERIOD),
        (XAxisKind.OFFSET_Y, RollMethod.ROLL_ON_MMDD),
        (XAxisKind.OFFSET_HY, RollMethod.HALF_YEAR_BY_DATE),
        (XAxisKind.OFFSET_HY, RollMethod.HALF_YEAR_BY_LABEL),
    }
)
