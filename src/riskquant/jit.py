# src/riskquant/jit.py

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
from numba.core import types as nb_types

# Type aliases for Numba-compiled functions — enforce C-contiguous layout for all
# arrays to ensure the compiler emits straight-line SIMD loops with no dynamic
# stride checks. These are used in type signatures for compiled functions below.
_F64_1D = nb_types.float64[::1]
_F64_2D = nb_types.float64[:, ::1]
_F64_3D = nb_types.float64[:, :, ::1]
_F64_4D = nb_types.float64[:, :, :, ::1]

_F32_1D = nb_types.float32[::1]
_F32_2D = nb_types.float32[:, ::1]
_F32_3D = nb_types.float32[:, :, ::1]
_F32_4D = nb_types.float32[:, :, :, ::1]

_I64_1D = nb_types.int64[::1]
_I64_2D = nb_types.int64[:, ::1]
_I64_3D = nb_types.int64[:, :, ::1]
_I64_4D = nb_types.int64[:, :, :, ::1]

_I32_1D = nb_types.int32[::1]
_I32_2D = nb_types.int32[:, ::1]
_I32_4D = nb_types.int32[:, :, :, ::1]
_I32_3D = nb_types.int32[:, :, ::1]

_I8_1D = nb_types.int8[::1]
_I8_2D = nb_types.int8[:, ::1]
_I8_3D = nb_types.int8[:, :, ::1]
_I8_4D = nb_types.int8[:, :, :, ::1]

_BOOL_1D = nb_types.bool_[::1]
_BOOL_2D = nb_types.bool_[:, ::1]
_BOOL_3D = nb_types.bool_[:, :, ::1]
_BOOL_4D = nb_types.bool_[:, :, :, ::1]

# Read-only counterparts. A NumPy array with ``writeable=False`` is typed by
# Numba as ``readonly array(...)``, which does NOT unify with the mutable
# aliases above: passing a frozen array to a kernel declared with ``_F64_2D``
# raises a TypeError at dispatch. The compiled sample and rank representation
# own their storage and mark every array non-writeable, so every kernel that
# READS one of them must declare the read-only type. Kernels that WRITE an
# output keep the mutable alias for that argument.
#
# Derived with ``.copy(readonly=True)`` rather than constructed from
# ``nb_types.Array``: it keeps each read-only alias tied to its mutable
# counterpart, so a change to one cannot silently diverge from the other.
_F64_1D_RO = _F64_1D.copy(readonly=True)
_F64_2D_RO = _F64_2D.copy(readonly=True)
_F64_3D_RO = _F64_3D.copy(readonly=True)

_F32_1D_RO = _F32_1D.copy(readonly=True)
_F32_2D_RO = _F32_2D.copy(readonly=True)

_I64_1D_RO = _I64_1D.copy(readonly=True)
_I64_2D_RO = _I64_2D.copy(readonly=True)

_I32_1D_RO = _I32_1D.copy(readonly=True)
_I32_2D_RO = _I32_2D.copy(readonly=True)

_I8_1D_RO = _I8_1D.copy(readonly=True)

_BOOL_1D_RO = _BOOL_1D.copy(readonly=True)
_BOOL_2D_RO = _BOOL_2D.copy(readonly=True)

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
