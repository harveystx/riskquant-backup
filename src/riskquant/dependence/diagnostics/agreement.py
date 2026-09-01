# src/riskquant/dependence/diagnostics/agreement.py

"""
Pairwise agreement between two dependence matrices.

Statistics are taken over the **strict upper triangle**, the set of distinct
unordered variable pairs. The diagonal is excluded by construction and the
mirrored lower triangle is excluded so that each pair is counted once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskquant.dependence.contracts import DependenceResult
from riskquant.observations.arrays import Float64Matrix, Float64Vector


@dataclass(frozen=True, slots=True, kw_only=True)
class AgreementDiagnostics:
    """
    How far apart two estimates of the same sample are.

    Attributes
    ----------
    pair_count : int
        ``K(K-1)/2``.

    maximum_absolute_deviation, mean_absolute_deviation : float
    """

    pair_count: int
    maximum_absolute_deviation: float
    mean_absolute_deviation: float


def _strict_upper_triangle_values(matrix: Float64Matrix) -> Float64Vector:
    """
    Distinct pairwise values of a square matrix, as a flat vector.

    Parameters
    ----------
    matrix : Float64Matrix
        ``(K, K)``, at least 2x2.

    Returns
    -------
    values : Float64Vector
        ``(K(K-1)/2,)`` float64 C-contiguous, in row-major upper-triangle order.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square; got shape={matrix.shape}.")
    if matrix.shape[0] < 2:
        raise ValueError("matrix must be at least 2x2 to contain a pair.")

    return np.ascontiguousarray(
        matrix[np.triu_indices(matrix.shape[0], k=1)], dtype=np.float64
    )


def agreement(
    first: Float64Matrix | DependenceResult,
    second: Float64Matrix | DependenceResult,
) -> AgreementDiagnostics:
    """
    Compare two dependence matrices over their distinct pairs.

    Parameters
    ----------
    first, second : Float64Matrix or DependenceResult
        Same shape.

    Returns
    -------
    diagnostics : AgreementDiagnostics

    Raises
    ------
    ValueError
        On a shape mismatch, or a non-finite off-diagonal cell. Diagnostics
        never repair a matrix.
    """
    left = first.matrix if isinstance(first, DependenceResult) else first
    right = second.matrix if isinstance(second, DependenceResult) else second

    if left.shape != right.shape:
        raise ValueError(
            f"matrices must have the same shape; got {left.shape} and {right.shape}."
        )

    left_values = _strict_upper_triangle_values(left)
    right_values = _strict_upper_triangle_values(right)

    if not (np.isfinite(left_values).all() and np.isfinite(right_values).all()):
        raise ValueError("off-diagonal values must be finite.")

    deviation = np.abs(left_values - right_values)

    return AgreementDiagnostics(
        pair_count=int(left_values.size),
        maximum_absolute_deviation=float(deviation.max()),
        mean_absolute_deviation=float(deviation.mean()),
    )
