# src/riskquant/market_data/risk_factors/shift_types.py
from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum

import numpy as np


class ShiftType(IntEnum):
    NO_SHIFT = -1
    LOG_RETURN = 1
    SIMPLE_RETURN = 2
    DIFFERENCE = 3
    ABSOLUTE_BASE_RETURN = 4


# Kernel-friendly int8 constants (avoid repeated casts in hot code)
SHIFT_NO_SHIFT_ID: np.int8 = np.int8(ShiftType.NO_SHIFT)
SHIFT_LOG_RETURN_ID: np.int8 = np.int8(ShiftType.LOG_RETURN)
SHIFT_SIMPLE_RETURN_ID: np.int8 = np.int8(ShiftType.SIMPLE_RETURN)
SHIFT_DIFFERENCE_ID: np.int8 = np.int8(ShiftType.DIFFERENCE)
SHIFT_ABSOLUTE_BASE_RETURN_ID: np.int8 = np.int8(ShiftType.ABSOLUTE_BASE_RETURN)


# Names tuple
SHIFT_TYPE_NAMES: tuple[str, ...] = tuple(method.name.upper() for method in ShiftType)

# Registry
SHIFT_TYPE_NAME_TO_ID_REGISTRY: dict[str, int] = {
    "NO_SHIFT": int(SHIFT_NO_SHIFT_ID),
    "LOG_RETURN": int(SHIFT_LOG_RETURN_ID),
    "SIMPLE_RETURN": int(SHIFT_SIMPLE_RETURN_ID),
    "DIFFERENCE": int(SHIFT_DIFFERENCE_ID),
    "ABSOLUTE_BASE_RETURN": int(SHIFT_ABSOLUTE_BASE_RETURN_ID),
}

# Canonical set of valid shift-type ids ({-1, 1, 2, 3, 4}). Derived from the
# enum so it stays the single source of truth. `0` is intentionally NOT a
# valid id and must fail fast at the boundary.
VALID_SHIFT_TYPE_IDS: frozenset[int] = frozenset(int(member) for member in ShiftType)


def validate_shift_types_dict(shift_types_dict: Mapping[str, int]) -> None:
    """
    Validate that the Excel-provided mapping exactly matches the code enum.

    Rationale:
      - Code is the contract; Excel is data.
      - Kernels rely on stable integer codes.
    """
    # Expected mapping from enum
    expected: dict[str, int] = {
        name: int(member) for name, member in ShiftType.__members__.items()
    }

    # Normalize "got" mapping to plain ints for stable comparison
    got: dict[str, int] = {str(k): int(v) for k, v in shift_types_dict.items()}

    if got == expected:
        return

    missing = {k: expected[k] for k in expected.keys() - got.keys()}
    extra = {k: got[k] for k in got.keys() - expected.keys()}
    different = {
        k: {"expected": expected[k], "got": got[k]}
        for k in (expected.keys() & got.keys())
        if expected[k] != got[k]
    }

    raise ValueError(
        "shift_types sheet does not match riskquant ShiftType enum.\n"
        f"missing={missing}\n"
        f"extra={extra}\n"
        f"different={different}\n"
        "Fix the Excel shift_types table or update riskquant.market_data.risk_factors.shift_types.ShiftType."
    )
