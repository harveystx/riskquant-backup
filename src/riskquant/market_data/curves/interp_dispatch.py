# src/riskquant/market_data/curves/interp_dispatch.py
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from riskquant.market_data.curves.extrap_methods import (
    EXTRAP_FLAT,
    EXTRAP_LINEAR,
)

# Import the fused Numba kernels
from riskquant.market_data.curves.interp_defs import (
    interp_linear_flat_extrap_3d_numba_kernel,
    interp_linear_linear_extrap_3d_numba_kernel,
    interp_log_linear_flat_extrap_3d_numba_kernel,
    interp_log_linear_linear_extrap_3d_numba_kernel,
    interp_step_left_flat_extrap_3d_numba_kernel,
    interp_step_left_linear_extrap_3d_numba_kernel,
    interp_step_right_flat_extrap_3d_numba_kernel,
    interp_step_right_linear_extrap_3d_numba_kernel,
)

# Import your enum constants
from riskquant.market_data.curves.interp_methods import (
    INTERP_LINEAR,
    INTERP_LOG_LINEAR,
    INTERP_STEP_LEFT,
    INTERP_STEP_RIGHT,
)

# ============================================================
# INTERPOLATION DISPATCH REGISTRY
# ============================================================

# Define the strict signature every curve kernel must follow.
# Signature: (tenors, values, targets, out) -> None
InterpKernelFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]

# Map (Interp Method ID, Extrap Method ID) -> Math Kernel
# We cast the np.int8 enums to pure Python ints for safe, fast dictionary hashing.
CURVE_MATH_REGISTRY: dict[tuple[int, int], InterpKernelFn] = {
    (
        int(INTERP_LINEAR),
        int(EXTRAP_FLAT),
    ): interp_linear_flat_extrap_3d_numba_kernel,
    (
        int(INTERP_LINEAR),
        int(EXTRAP_LINEAR),
    ): interp_linear_linear_extrap_3d_numba_kernel,
    (
        int(INTERP_LOG_LINEAR),
        int(EXTRAP_FLAT),
    ): interp_log_linear_flat_extrap_3d_numba_kernel,
    (
        int(INTERP_LOG_LINEAR),
        int(EXTRAP_LINEAR),
    ): interp_log_linear_linear_extrap_3d_numba_kernel,
    (
        int(INTERP_STEP_LEFT),
        int(EXTRAP_FLAT),
    ): interp_step_left_flat_extrap_3d_numba_kernel,
    (
        int(INTERP_STEP_LEFT),
        int(EXTRAP_LINEAR),
    ): interp_step_left_linear_extrap_3d_numba_kernel,
    (
        int(INTERP_STEP_RIGHT),
        int(EXTRAP_FLAT),
    ): interp_step_right_flat_extrap_3d_numba_kernel,
    (
        int(INTERP_STEP_RIGHT),
        int(EXTRAP_LINEAR),
    ): interp_step_right_linear_extrap_3d_numba_kernel,
}
