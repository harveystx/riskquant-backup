# src/riskquant/dependence/_matrix.py

"""
Shared allocation and finalisation for ``K x K`` dependence matrices.

Every matrix estimator follows the same three steps: allocate once, populate the
strict upper triangle, finalise. Nothing here allocates a second ``K x K``
matrix for formatting, and no estimator retains both halves of a symmetric
result.
"""

from __future__ import annotations

import numpy as np

from riskquant.constants import NUMERICAL_EXCURSION_TOLERANCE
from riskquant.dependence._kernels._symmetry import (
    kernel_clamp_upper_triangle,
    kernel_first_bound_violation,
    kernel_mirror_upper_and_set_diagonal,
)
from riskquant.dependence.taxonomy import DependenceContract
from riskquant.observations.arrays import Float64Matrix, Int32Vector


def allocate_dependence_matrix(n_variables: int) -> Float64Matrix:
    """
    Uninitialised ``(K, K)`` float64 C-contiguous output.

    ``np.empty`` rather than ``np.zeros``: every cell is written before the
    matrix is returned -- the strict upper triangle by the estimator, the lower
    triangle and diagonal by :func:`finalise_symmetric`.
    """
    return np.empty((n_variables, n_variables), dtype=np.float64)


def upper_triangle_pairs(n_variables: int) -> tuple[Int32Vector, Int32Vector]:
    """
    Strict-upper-triangle pair indices as int32.

    int32 halves the footprint of ``np.triu_indices``' native int64 result and
    matches the kernel signatures.
    """
    row_index, col_index = np.triu_indices(n_variables, k=1)
    return (
        np.ascontiguousarray(row_index, dtype=np.int32),
        np.ascontiguousarray(col_index, dtype=np.int32),
    )


def finalise_symmetric(
    matrix: Float64Matrix,
    *,
    contract: DependenceContract,
    tolerance: float = NUMERICAL_EXCURSION_TOLERANCE,
) -> Float64Matrix:
    """
    Enforce bounds, symmetry and the analytic diagonal, then freeze.

    The bound rule has three regimes, and the third is the important one:

    * a value inside the declared range is untouched;
    * a value outside the range but within ``tolerance`` of it is clamped to the
      bound, because Cauchy-Schwarz places the true value inside the range and
      the excess is therefore provably rounding error;
    * a value beyond ``tolerance`` **raises**.

    A finite declared range alone never licenses clamping. Under such a rule an
    estimator returning ``1.4`` from an indexing bug would be silently clamped
    to ``1.0`` and would ship as a plausible correlation. Bounds are a
    mathematical statement about the estimand, not permission to overwrite a
    result.

    An infinite endpoint disables the corresponding clamp: for an unbounded
    estimator there is no bound to round to, and a negative value from a
    bias-corrected statistic is a real result rather than an excursion.

    Parameters
    ----------
    matrix : Float64Matrix
        ``(K, K)`` with a populated strict upper triangle. Mutated in place,
        then marked non-writeable.

    contract : DependenceContract
        Supplies the value range and the analytic diagonal.

    tolerance : float
        Engine-wide numerical excursion tolerance.

    Returns
    -------
    matrix : Float64Matrix
        The same object: exactly symmetric, diagonal set, non-writeable.

    Raises
    ------
    ValueError
        If any strict-upper-triangle cell lies beyond the tolerance band.
    """
    lower_bound, upper_bound = contract.value_range

    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        violation = int(
            kernel_first_bound_violation(matrix, lower_bound, upper_bound, tolerance)
        )
        if violation >= 0:
            size = matrix.shape[0]
            row, col = divmod(violation, size)
            raise ValueError(
                f"{contract.estimator.name} produced {matrix[row, col]!r} at "
                f"variable_index pair ({row}, {col}), which lies beyond the "
                f"declared range [{lower_bound}, {upper_bound}] by more than "
                f"the numerical tolerance {tolerance}. This is an "
                "implementation defect, not rounding error, and is not clamped."
            )

        kernel_clamp_upper_triangle(matrix, lower_bound, upper_bound, tolerance)

    kernel_mirror_upper_and_set_diagonal(matrix, contract.diagonal_value)

    matrix.setflags(write=False)

    return matrix
