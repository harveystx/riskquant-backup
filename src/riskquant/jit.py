# src/riskquant/numba.py

"""
Centralized Numba decorator definitions used throughout RiskQuant.

Rationale
---------
RiskQuant standardizes Numba compilation settings to ensure consistent
performance characteristics, numerical behavior, and cache utilization
across the codebase.

This module provides three decorators:

    @njit
        Default choice for most compiled functions.

    @parallel_njit
        For kernels that parallelize outer loops using ``numba.prange``.

    @inline_njit
        For tiny helper functions that should be inlined into callers to
        eliminate function-call overhead.

By centralizing these definitions, compilation settings can be adjusted in
one location without touching hundreds of kernels throughout the system.
"""

from __future__ import annotations

from functools import partial

import numba as nb

# Type aliases for Numba-compiled functions — enforce C-contiguous layout for all
# arrays to ensure the compiler emits straight-line SIMD loops with no dynamic
# stride checks. These are used in type signatures for compiled functions below.
_F64_1D = nb.float64[::1]
_F64_2D = nb.float64[:, ::1]
_F64_3D = nb.float64[:, :, ::1]
_F64_4D = nb.float64[:, :, :, ::1]

_F32_1D = nb.float32[::1]
_F32_2D = nb.float32[:, ::1]
_F32_3D = nb.float32[:, :, ::1]
_F32_4D = nb.float32[:, :, :, ::1]

_I64_1D = nb.int64[::1]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]

_BOOL_1D = nb.bool_[::1]

#: Default Numba compiler configuration used throughout RiskQuant.
#:
#: Use for the vast majority of compiled functions.
#:
#: Features:
#:     - Explicit fast-math optimizations
#:     - GIL-free execution
#:     - Bounds-check elimination
#:     - On-disk compilation caching
#:     - NumPy-style floating-point error behavior
#:
#: Example:
#:
#:     @njit
#:     def black_scholes(...):
#:         ...
njit = partial(
    nb.njit,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)


#: Parallel Numba compiler configuration.
#:
#: Use for computational kernels that distribute work across threads using
#: ``numba.prange``.
#:
#: Typical usage:
#:
#:     @parallel_njit
#:     def price_kernel(...):
#:         for i in nb.prange(num_trades):
#:             ...
#:
#: Avoid for:
#:     - Small loops
#:     - Helper functions
#:     - Functions that do not use ``prange``
parallel_njit = partial(
    nb.njit,
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)


#: Inline Numba compiler configuration.
#:
#: Intended for extremely small helper functions that are called repeatedly
#: inside hot loops.
#:
#: Examples:
#:     - Normal CDF approximations
#:     - Small mathematical transforms
#:     - Utility functions used inside pricing kernels
#:
#: The ``inline="always"`` hint requests that Numba inline the function body
#: into callers whenever possible.
#:
#: ``error_model="numpy"`` is set to match ``njit``/``parallel_njit``: division
#: by zero yields inf/nan rather than raising inside a nopython (and possibly
#: parallel) caller. Without it, an unguarded division in an inlined helper can
#: turn into a hard ``ZeroDivisionError`` mid-kernel.
inline_njit = partial(
    nb.njit,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
    inline="always",
)
