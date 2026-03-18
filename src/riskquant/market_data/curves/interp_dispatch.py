# src/riskquant/market_data/curves/interp_dispatch.py
from __future__ import annotations

import numpy as np
from typing import Callable

# Import your enum constants
from riskquant.market_data.curves.interp_methods import (
    INTERP_LINEAR,
    INTERP_LOG_LINEAR,
    INTERP_STEP_LEFT,
    INTERP_STEP_RIGHT,
)
from riskquant.market_data.curves.extrap_methods import (
    EXTRAP_FLAT,
    EXTRAP_LINEAR,
)

from riskquant.market_data.curves.interp_defs import (
    interp_linear_flat_extrap_3d,
    interp_linear_linear_extrap_3d,
    interp_log_linear_flat_extrap_3d,
    interp_log_linear_linear_extrap_3d,
    interp_step_left_flat_extrap_3d,
    interp_step_left_linear_extrap_3d,
    interp_step_right_flat_extrap_3d,
    interp_step_right_linear_extrap_3d,
)

# ... [All your pure math functions here] ...

# ============================================================
# INTERPOLATION DISPATCH REGISTRY
# ============================================================

# Define the strict signature every curve kernel must follow
InterpKernelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# Map (Interp Method ID, Extrap Method ID) -> Math Kernel
# We cast the np.int8 enums to pure Python ints for safe, fast dictionary hashing.
CURVE_MATH_REGISTRY: dict[tuple[int, int], InterpKernelFn] = {
    (
        int(INTERP_LINEAR),
        int(EXTRAP_FLAT),
    ): interp_linear_flat_extrap_3d,
    (
        int(INTERP_LINEAR),
        int(EXTRAP_LINEAR),
    ): interp_linear_linear_extrap_3d,
    (
        int(INTERP_LOG_LINEAR),
        int(EXTRAP_FLAT),
    ): interp_log_linear_flat_extrap_3d,
    (
        int(INTERP_LOG_LINEAR),
        int(EXTRAP_LINEAR),
    ): interp_log_linear_linear_extrap_3d,
    (
        int(INTERP_STEP_LEFT),
        int(EXTRAP_FLAT),
    ): interp_step_left_flat_extrap_3d,
    (
        int(INTERP_STEP_LEFT),
        int(EXTRAP_LINEAR),
    ): interp_step_left_linear_extrap_3d,
    (
        int(INTERP_STEP_RIGHT),
        int(EXTRAP_FLAT),
    ): interp_step_right_flat_extrap_3d,
    (
        int(INTERP_STEP_RIGHT),
        int(EXTRAP_LINEAR),
    ): interp_step_right_linear_extrap_3d,
}
