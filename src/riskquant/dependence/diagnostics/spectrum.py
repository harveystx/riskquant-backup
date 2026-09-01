# src/riskquant/dependence/diagnostics/spectrum.py

"""
Eigenvalue diagnostics for a dependence matrix.

Consumes a result; estimates nothing. The oracle is LAPACK's symmetric
eigensolver, a code path wholly independent of every estimator in this package,
which is what makes the PSD conformance test non-circular.

PSD, positive definite and Cholesky-eligible are three distinct properties and
this module keeps them distinct. ``psd_status`` in the contract is a *static*
claim about the estimator in exact arithmetic. Whether *this* matrix can be
factorised is a *dynamic* property of the numbers, reported here. A PSD matrix
can be singular, and singularity is guaranteed whenever ``K > S``, since the
Gram matrix then has rank at most ``S < K``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskquant.dependence.contracts import DependenceResult
from riskquant.observations.arrays import Float64Matrix, Float64Vector


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectrumDiagnostics:
    """
    Eigenvalue summary of a symmetric matrix.

    Attributes
    ----------
    eigenvalues : Float64Vector
        ``(K,)`` ascending, from ``numpy.linalg.eigvalsh``.

    minimum_eigenvalue, maximum_eigenvalue : float

    psd_tolerance : float
        ``K * eps * max(0, maximum_eigenvalue)``. The numerical criterion a
        ``PSD_BY_CONSTRUCTION`` estimator's output must satisfy: rounding,
        mirroring and clamping can move the smallest eigenvalue of an exactly
        PSD Gram matrix by about this much.

    satisfies_numerical_psd : bool
        ``minimum_eigenvalue >= -psd_tolerance``.

    condition_number : float
        ``max / min`` of the absolute eigenvalues, or ``inf`` when the smallest
        is exactly zero.
    """

    eigenvalues: Float64Vector
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    psd_tolerance: float
    satisfies_numerical_psd: bool
    condition_number: float


def spectrum(matrix: Float64Matrix | DependenceResult) -> SpectrumDiagnostics:
    """
    Eigenvalue diagnostics of a dependence matrix.

    Parameters
    ----------
    matrix : Float64Matrix or DependenceResult
        Square and exactly symmetric.

    Returns
    -------
    diagnostics : SpectrumDiagnostics

    Raises
    ------
    ValueError
        If the input is not square, not exactly symmetric, or not finite.
    """
    values = matrix.matrix if isinstance(matrix, DependenceResult) else matrix

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"matrix must be square; got shape={values.shape}.")
    if not np.array_equal(values, values.T):
        raise ValueError(
            "matrix must be exactly symmetric. Estimators in this package "
            "mirror their upper triangle, so an asymmetric input did not come "
            "from one of them."
        )
    if not np.isfinite(values).all():
        raise ValueError("matrix must be finite.")

    eigenvalues: Float64Vector = np.asarray(
        np.linalg.eigvalsh(values), dtype=np.float64
    )
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])

    tolerance = values.shape[0] * float(np.finfo(np.float64).eps) * max(0.0, maximum)

    absolute = np.abs(eigenvalues)
    smallest_absolute = float(absolute.min())
    condition = (
        float(absolute.max() / smallest_absolute)
        if smallest_absolute > 0.0
        else float("inf")
    )

    return SpectrumDiagnostics(
        eigenvalues=eigenvalues,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        psd_tolerance=tolerance,
        satisfies_numerical_psd=minimum >= -tolerance,
        condition_number=condition,
    )
