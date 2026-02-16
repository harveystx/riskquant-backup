# src\riskquant\market_data\curves\interpolation_methods.py

from __future__ import annotations

from enum import IntEnum
from typing import Mapping

import numpy as np


class InterpolationMethod(IntEnum):
    LINEAR = 1
    LOG_LINEAR = 2
    STEP_LEFT = 3
    STEP_RIGHT = 4
    CUBIC_SPLINE = 5


# compiler-friendly int8 constants
INTERP_LINEAR: np.int8 = np.int8(InterpolationMethod.LINEAR)
INTERP_LOG_LINEAR: np.int8 = np.int8(InterpolationMethod.LOG_LINEAR)
INTERP_STEP_LEFT: np.int8 = np.int8(InterpolationMethod.STEP_LEFT)
INTERP_STEP_RIGHT: np.int8 = np.int8(InterpolationMethod.STEP_RIGHT)
INTERP_CUBIC_SPLINE: np.int8 = np.int8(InterpolationMethod.CUBIC_SPLINE)


def validate_interpolation_methods_dict(
    interpolation_methods_dict: Mapping[str, int],
) -> None:
    expected = {
        name: int(member)
        for name, member in InterpolationMethod.__members__.items()
    }
    got = {str(k): int(v) for k, v in interpolation_methods_dict.items()}

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
        "interpolation_methods sheet does not match riskquant InterpolationMethod enum.\n"
        f"missing={missing}\n"
        f"extra={extra}\n"
        f"different={different}\n"
        "Fix the Excel interpolation_methods table or update riskquant.market_data.risk_factors.interpolation_methods.InterpolationMethod."
    )
