# src/riskquant/config/enums/rate_fixing_convention.py
from __future__ import annotations

from enum import IntEnum


class RateFixingConvention(IntEnum):
    IN_ADVANCE = 0
    IN_ARREARS = 1
