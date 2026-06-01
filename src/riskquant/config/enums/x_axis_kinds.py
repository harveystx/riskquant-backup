# src/riskquant/config/enums/x_axis_kinds.py
"""
X-axis kind taxonomy for curves and surfaces.

Replaces the legacy (TenorType, AliasType, AliasKind) triple. A single
XAxisKind tells the resolution layer exactly how a curve / surface point
addresses its position on the time axis:

  LABEL      - fixed, as-of-independent string labels (e.g. [ON], [1W], [1M]).
               Used by IR conventional curves.
  OFFSET_M   - monthly offsets from a roll-determined spot month (e.g. [M+0],
               [M-1], [M+12]). Used by monthly commodity contracts.
  OFFSET_Y   - annual offsets from a roll-determined spot year (e.g. [Y+0],
               [Y-2]). Used by annual commodity contracts.
  OFFSET_HY  - half-year offsets from a roll-determined spot half (e.g.
               [HY+0], [HY+3]). Used by half-year power contracts.

The X-axis kind defines token grammar only. How an offset resolves to an
absolute date label (calendar period, roll-on-MMDD, half-year boundary)
is independent and lives on the spot roll rule via RollMethod. Allowed
(XAxisKind, RollMethod) pairs are enumerated explicitly in
ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS.
"""

from __future__ import annotations

import re
from enum import IntEnum
from re import Pattern


class XAxisKind(IntEnum):
    LABEL = 0
    OFFSET_M = 1
    OFFSET_Y = 2
    OFFSET_HY = 3


# Token grammars by kind. LABEL matches the standard IR conventions plus the
# generic [\d+(D|W|M|Y)] form. OFFSET kinds match a strictly signed integer
# inside brackets prefixed by the unit (M, Y, HY).
X_AXIS_TOKEN_REGEX: dict[XAxisKind, Pattern[str]] = {
    XAxisKind.LABEL: re.compile(r"^\[(?:ON|SN|TN|\d+D|\d+W|\d+M|\d+Y)\]$"),
    XAxisKind.OFFSET_M: re.compile(r"^\[M[+-]\d+\]$"),
    XAxisKind.OFFSET_Y: re.compile(r"^\[Y[+-]\d+\]$"),
    XAxisKind.OFFSET_HY: re.compile(r"^\[HY[+-]\d+\]$"),
}

X_AXIS_TOKEN_EXAMPLE: dict[XAxisKind, str] = {
    XAxisKind.LABEL: "[3M]",
    XAxisKind.OFFSET_M: "[M+1]",
    XAxisKind.OFFSET_Y: "[Y-2]",
    XAxisKind.OFFSET_HY: "[HY+0]",
}

VALID_X_AXIS_KINDS: frozenset[str] = frozenset(k.name for k in XAxisKind)

# OFFSET kinds require a spot roll rule. LABEL does not (and must not).
X_AXIS_KINDS_REQUIRING_ROLL: frozenset[str] = frozenset(
    k.name for k in (XAxisKind.OFFSET_M, XAxisKind.OFFSET_Y, XAxisKind.OFFSET_HY)
)
