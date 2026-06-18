# src/riskquant/config/enums/fixing_type.py
from __future__ import annotations

from enum import IntEnum


class SwapDirection(IntEnum):
    ADVANCE = 0
    ARREARS = 1
