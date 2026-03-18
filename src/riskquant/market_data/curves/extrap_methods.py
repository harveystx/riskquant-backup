# src\riskquant\market_data\curves\extrap_methods.py

from __future__ import annotations

from enum import IntEnum

import numpy as np


class ExtrapMethod(IntEnum):
    FLAT = 1
    LINEAR = 2


# compiler-friendly int8 constants
EXTRAP_FLAT: np.int8 = np.int8(ExtrapMethod.FLAT)
EXTRAP_LINEAR: np.int8 = np.int8(ExtrapMethod.LINEAR)


# Extrap methods
EXTRAP_METHOD_NAMES: tuple[str, ...] = tuple(
    method.name.upper() for method in ExtrapMethod
)

# Central registry
EXTRAP_METHOD_REGISTRY: dict[str, ExtrapMethod] = {
    "LINEAR": ExtrapMethod.LINEAR,
    "FLAT": ExtrapMethod.FLAT,
}


def validate_extrapolation_methods_dict(
    extrapolation_methods_dict: dict[str, int],
) -> None:
    expected = {
        name: int(member) for name, member in ExtrapMethod.__members__.items()
    }
    got = {str(k): int(v) for k, v in extrapolation_methods_dict.items()}

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
        "extrapolation_methods sheet does not match riskquant ExtrapMethod enum.\n"
        f"missing={missing}\n"
        f"extra={extra}\n"
        f"different={different}\n"
        "Fix the Excel extrapolation_methods table or update riskquant.market_data.curves.extrap_methods.ExtrapMethod."
    )
