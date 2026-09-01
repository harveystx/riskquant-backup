# src/riskquant/dependence/estimators/pearson.py

"""
Pearson product-moment correlation.

Definition
----------
``rho(x, y) = sum_s (x_s - xbar)(y_s - ybar)
              / sqrt(sum_s (x_s - xbar)^2 * sum_s (y_s - ybar)^2)``

Equivalently the cosine of the angle between the mean-centred observation
vectors. The degrees-of-freedom convention cancels between numerator and
denominator, so no ``ddof`` parameter is exposed: none could change the answer.

What it measures
----------------
Linear association only. ``rho = 0`` does not imply independence -- it is
exactly zero for ``y = x^2`` with ``x`` symmetric about zero. Unbounded
influence function: a single outlier moves it arbitrarily far.

Contracts
---------
Input
    A :class:`~riskquant.observations.compile.CompiledSample`. Constant
    variables are rejected: the denominator is exactly zero.

Output
    ``(K, K)`` float64 C-contiguous, non-writeable, exactly symmetric, diagonal
    exactly 1.0, values in ``[-1, 1]``.

PSD
    Guaranteed by construction: the matrix is a Gram matrix of unit-norm real
    vectors. This does not imply Cholesky eligibility -- see
    :class:`~riskquant.dependence.taxonomy.PsdStatus`.

Complexity
    ``O(K^2 S)`` flops in one BLAS GEMM; ``O(KS + K^2)`` memory. No
    ``K x K x S`` intermediate is formed.
"""

from __future__ import annotations

import numpy as np

from riskquant.dependence._matrix import finalise_symmetric
from riskquant.dependence.contracts import DependenceProvenance, DependenceResult
from riskquant.dependence.taxonomy import (
    ESTIMATOR_CONTRACT,
    DependenceEstimator,
)
from riskquant.observations._kernels import kernel_center_rows, kernel_scale_rows
from riskquant.observations._validate import reject_constant_variables
from riskquant.observations.arrays import Float64Matrix, Float64Vector
from riskquant.observations.compile import CompiledSample


def _gram_correlation_matrix(
    canonical: Float64Matrix,
    sum_of_squares: Float64Vector,
    *,
    estimator: DependenceEstimator,
) -> Float64Matrix:
    """
    Pearson correlation of the rows of a canonical matrix.

    Shared by Pearson (raw observations) and Spearman (average ranks). The two
    estimators differ only in what is fed to it, which is why no ``use_ranks``
    flag exists on either public entry point.

    Parameters
    ----------
    canonical : Float64Matrix
        ``(K, S)`` observations or average ranks. Read only.

    sum_of_squares : Float64Vector
        ``(K,)`` centred sums of squares of ``canonical``, used for the
        degeneracy gate. Supplied by the caller so that the gate never
        allocates a ``(K, S)`` temporary.

    estimator : DependenceEstimator
        Selects the contract used for finalisation.

    Notes
    -----
    Standardising to unit-norm rows turns the correlation into a Gram matrix,
    ``Z @ Z.T``. ``scipy.linalg.blas.dsyrk`` computes only one triangle and was
    benchmarked against the full GEMM at (K, S) of (10, 1000), (50, 2500),
    (100, 5000) and (500, 5000): it ran at 1.0x, 0.94x, 0.74x and 0.90x the GEMM
    speed and returned an F-ordered array needing a further copy. The halved
    flop count does not survive contact with the vendor's better-tuned GEMM
    path, so the GEMM is used and the redundant lower triangle is overwritten by
    the mirror.
    """
    contract = ESTIMATOR_CONTRACT[estimator]

    reject_constant_variables(
        sum_of_squares,
        estimator_name=_ESTIMATOR_DISPLAY_NAME[estimator],
    )

    n_variables, n_observations = canonical.shape

    standardised = np.empty((n_variables, n_observations), dtype=np.float64)
    centred_sum_of_squares = np.empty(n_variables, dtype=np.float64)

    kernel_center_rows(canonical, standardised, centred_sum_of_squares)
    kernel_scale_rows(standardised, 1.0 / np.sqrt(centred_sum_of_squares))

    correlation: Float64Matrix = standardised @ standardised.T

    return finalise_symmetric(correlation, contract=contract)


_ESTIMATOR_DISPLAY_NAME: dict[DependenceEstimator, str] = {
    DependenceEstimator.PEARSON: "Pearson correlation",
    DependenceEstimator.SPEARMAN: "Spearman rank correlation",
    DependenceEstimator.KENDALL_TAU_B: "Kendall tau-b",
}


def pearson(sample: CompiledSample) -> DependenceResult:
    """
    Pearson product-moment correlation matrix.

    Parameters
    ----------
    sample : CompiledSample

    Returns
    -------
    result : DependenceResult

    Raises
    ------
    ValueError
        If any variable has exactly zero dispersion.
    """
    matrix = _gram_correlation_matrix(
        sample.observations,
        sample.sum_of_squares,
        estimator=DependenceEstimator.PEARSON,
    )

    return DependenceResult(
        matrix=matrix,
        contract=ESTIMATOR_CONTRACT[DependenceEstimator.PEARSON],
        provenance=DependenceProvenance(
            observation_count=sample.observation_count,
        ),
    )
