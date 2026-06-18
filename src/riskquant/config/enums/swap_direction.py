# src/riskquant/config/enums/swap_direction.py
from __future__ import annotations

from enum import IntEnum


class SwapDirection(IntEnum):
    PAY_FIXED = -1
    RECEIVE_FIXED = +1
