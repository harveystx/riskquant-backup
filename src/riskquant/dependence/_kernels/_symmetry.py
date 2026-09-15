# src/riskquant/dependence/_kernels/_symmetry.py

"""
In-place symmetry, diagonal and bound enforcement for ``K x K`` outputs.

The NumPy alternatives all allocate. ``(C + C.T) * 0.5`` allocates two full
``K x K`` temporaries; ``C[np.tril_indices(K, -1)] = C.T[np.tril_indices(K, -1)]``
allocates two int64 index vectors of length ``K(K-1)/2`` -- for K = 500 that is
2 MB of indices to move a 2 MB matrix. These kernels touch each cell once, with
no allocation at all.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from riskquant.jit import (
    _F64_2D_C,
    _NB_FLOAT64,
    _NB_INT64,
    _NB_VOID,
    njit,
)


@njit(_NB_VOID(_F64_2D_C, _NB_FLOAT64))
def kernel_mirror_upper_and_set_diagonal(
    matrix: NDArray[np.float64],
    diagonal_value: float,
) -> None:
    """
    Copy the strict upper triangle onto the lower triangle and set the diagonal.

    The strict upper triangle is authoritative. Where a kernel computes both
    triangles -- as a BLAS GEMM does -- the lower half is discarded and
    overwritten from the upper, so exactly one value per unordered pair is
    *retained* even though both were computed.

    This makes ``np.array_equal(M, M.T)`` hold with no tolerance: the two halves
    are the same bit pattern, not merely equal to within rounding. It is applied
    unconditionally, including where the two halves are provably identical, so
    the mechanism is uniform, mechanically auditable, and robust to a future
    reduced-precision execution path.

    Parameters
    ----------
    matrix : Float64Matrix
        ``(K, K)`` C-contiguous. Mutated in place. Only the strict upper
        triangle is read.

    diagonal_value : float
        Written to every diagonal cell.
    """
    size = matrix.shape[0]

    for row in range(size):
        matrix[row, row] = diagonal_value
        for col in range(row + 1, size):
            matrix[col, row] = matrix[row, col]


@njit(_NB_INT64(_F64_2D_C, _NB_FLOAT64, _NB_FLOAT64, _NB_FLOAT64))
def kernel_first_bound_violation(
    matrix: NDArray[np.float64],
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
) -> int:
    """
    Flat index of the first strict-upper-triangle cell outside the tolerance band.

    Returns ``-1`` when every cell lies within ``[lower - tolerance,
    upper + tolerance]``. A cell outside that band is an implementation defect,
    not rounding error, and the caller raises rather than clamping.

    Only the strict upper triangle is inspected, because it is the only part
    populated at this point in the pipeline; the lower triangle is still
    uninitialised.
    """
    size = int(matrix.shape[0])

    for row in range(size):
        for col in range(row + 1, size):
            value = matrix[row, col]
            if value < lower_bound - tolerance or value > upper_bound + tolerance:
                return row * size + col

    return -1


@njit(_NB_VOID(_F64_2D_C, _NB_FLOAT64, _NB_FLOAT64, _NB_FLOAT64))
def kernel_clamp_upper_triangle(
    matrix: NDArray[np.float64],
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
) -> None:
    """
    Clamp strict-upper-triangle excursions that lie inside the tolerance band.

    This is not a statistical adjustment and it is not a general clip. A finite
    declared range says nothing about whether an out-of-range value is rounding
    error; treating it as licence to clamp would silently convert an
    implementation defect into a plausible-looking correlation. That is the
    failure mode the caller's prior bound-violation scan exists to prevent.

    By the time this kernel runs, every cell is known to lie within
    ``[lower - tolerance, upper + tolerance]``. Cauchy-Schwarz places the true
    value inside ``[lower, upper]``, so any remaining excursion is provably pure
    rounding error and is removed at source, which keeps the downstream
    dissimilarity transforms free of NaN.

    Parameters
    ----------
    matrix : Float64Matrix
        ``(K, K)`` C-contiguous. Mutated in place. Only the strict upper
        triangle is read and written, so the still-uninitialised lower triangle
        is never touched.

    lower_bound, upper_bound : float
        Inclusive bounds from the estimator's contract.

    tolerance : float
        Engine-wide numerical excursion tolerance.
    """
    size = matrix.shape[0]

    for row in range(size):
        for col in range(row + 1, size):
            value = matrix[row, col]
            if lower_bound - tolerance <= value < lower_bound:
                matrix[row, col] = lower_bound
            elif upper_bound < value <= upper_bound + tolerance:
                matrix[row, col] = upper_bound
