# src/riskquant/_njit.py

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
inline_njit = partial(
    nb.njit,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    cache=True,
    inline="always",
)
