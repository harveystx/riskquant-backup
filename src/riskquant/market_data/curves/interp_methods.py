# src/riskquant/market_data/curves/interp_methods.py

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum

import numpy as np


class InterpMethod(IntEnum):
    LINEAR = 1
    LOG_LINEAR = 2
    LOCF = 3
    NOCB = 4
    CUBIC_SPLINE = 5


# compiler-friendly int8 constants
INTERP_LINEAR: np.int8 = np.int8(InterpMethod.LINEAR)
INTERP_LOG_LINEAR: np.int8 = np.int8(InterpMethod.LOG_LINEAR)
INTERP_LOCF: np.int8 = np.int8(InterpMethod.LOCF)
INTERP_NOCB: np.int8 = np.int8(InterpMethod.NOCB)
INTERP_CUBIC_SPLINE: np.int8 = np.int8(InterpMethod.CUBIC_SPLINE)

# Interp methods
INTERP_METHOD_NAMES: tuple[str, ...] = tuple(
    method.name.upper() for method in InterpMethod
)

# Central registry
INTERP_METHOD_REGISTRY: dict[str, InterpMethod] = {
    "LINEAR": InterpMethod.LINEAR,
    "LOG_LINEAR": InterpMethod.LOG_LINEAR,
    "LOCF": InterpMethod.LOCF,
    "NOCB": InterpMethod.NOCB,
    "CUBIC_SPLINE": InterpMethod.CUBIC_SPLINE,
}


def validate_interpolation_methods_dict(
    interpolation_methods_dict: Mapping[str, int],
) -> None:
    expected = {name: int(member) for name, member in InterpMethod.__members__.items()}
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
        "Fix the Excel interpolation_methods table or update riskquant.market_data.curves.interp_methods.InterpMethod."
    )
