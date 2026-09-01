# src/riskquant/dependence/represent/dissimilarity.py

"""
Similarity-to-dissimilarity transforms. DOWNSTREAM of estimation.

Turning a dependence matrix into something a clustering or embedding routine
will accept is a modelling choice, not part of any estimator's definition.

The word "distance" is avoided unless the metric axioms actually hold. Each
transform states its input domain, output domain, diagonal, and whether the
triangle inequality is guaranteed. Where it is not, that is a hard fact about
the transform, not a caveat.

====================  ===========================  =========================
Transform             Triangle inequality          Identity of indiscernibles
====================  ===========================  =========================
angular               yes, if the matrix is PSD    yes
euclidean             yes, if the matrix is PSD    yes
absolute              **no**                       **no**
linfoot               **no**                       yes
one_minus_dependence  **no**                       yes
====================  ===========================  =========================

"PSD" holds for Pearson and Spearman, which are Gram matrices of centred or
ranked observation vectors. It is **not** guaranteed for Kendall tau-b, so the
angular and euclidean transforms of a tau-b matrix are dissimilarities without a
metric guarantee. Hierarchical clustering tolerates that; anything relying on a
metric embedding does not.
"""

from __future__ import annotations

import numpy as np

from riskquant.constants import CORRELATION_LOWER_BOUND, CORRELATION_UPPER_BOUND
from riskquant.observations.arrays import Float64Matrix


def _validate_square_symmetric(
    matrix: Float64Matrix,
    *,
    allow_infinite_diagonal: bool,
) -> None:
    """Reject non-square, asymmetric, or non-finite input."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square; got shape={matrix.shape}.")

    if not np.array_equal(matrix, matrix.T):
        raise ValueError(
            "matrix must be exactly symmetric. Estimators in this package "
            "mirror their upper triangle, so an asymmetric input did not come "
            "from one of them."
        )

    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    if not np.isfinite(matrix[off_diagonal]).all():
        raise ValueError("off-diagonal cells must be finite.")

    if not allow_infinite_diagonal and not np.isfinite(np.diag(matrix)).all():
        raise ValueError("diagonal cells must be finite for this transform.")


def _validate_correlation_domain(matrix: Float64Matrix) -> None:
    """Reject values outside ``[-1, 1]``."""
    if matrix.min() < CORRELATION_LOWER_BOUND or matrix.max() > CORRELATION_UPPER_BOUND:
        raise ValueError(
            "input must be a correlation-like matrix with all values in "
            f"[{CORRELATION_LOWER_BOUND}, {CORRELATION_UPPER_BOUND}]; got range "
            f"[{matrix.min()!r}, {matrix.max()!r}]."
        )


def angular_dissimilarity(correlation: Float64Matrix) -> Float64Matrix:
    """
    ``d = sqrt((1 - rho) / 2)``.

    Input domain
        ``[-1, 1]``, symmetric, unit diagonal.

    Output domain
        ``[0, 1]``. Zero at ``rho = 1``, one at ``rho = -1``, ``1/sqrt(2)`` at
        ``rho = 0``.

    Diagonal
        Exactly zero, arising from ``rho = 1``. Not forced by a
        ``fill_diagonal`` call: if the input diagonal were not one, the caller
        would want to know.

    Triangle inequality
        Guaranteed when the correlation matrix is positive semi-definite, since
        ``d`` is then half the chord distance between unit vectors on a sphere.
        Not guaranteed otherwise -- a Kendall tau-b matrix need not be PSD.

    Numerical safety
        The square-root argument is non-negative for any input in the validated
        domain, so no clipping is applied and none is needed. The estimators
        bound their off-diagonals into ``[-1, 1]`` precisely so that this holds.
    """
    _validate_square_symmetric(correlation, allow_infinite_diagonal=False)
    _validate_correlation_domain(correlation)

    return np.ascontiguousarray(
        np.sqrt((CORRELATION_UPPER_BOUND - correlation) / 2.0)
    )


def euclidean_dissimilarity(correlation: Float64Matrix) -> Float64Matrix:
    """
    ``d = sqrt(2 (1 - rho))``, the Mantegna distance.

    Input domain ``[-1, 1]``, symmetric, unit diagonal. Output ``[0, 2]``,
    diagonal exactly zero.

    Exactly ``2x`` the angular transform. Any procedure invariant to a positive
    rescaling of the dissimilarity -- single, complete and average linkage --
    cannot distinguish the two. They are the same modelling choice in different
    units, not alternatives.

    Triangle inequality: same condition as the angular transform.
    """
    _validate_square_symmetric(correlation, allow_infinite_diagonal=False)
    _validate_correlation_domain(correlation)

    return np.ascontiguousarray(
        np.sqrt(2.0 * (CORRELATION_UPPER_BOUND - correlation))
    )


def absolute_dissimilarity(correlation: Float64Matrix) -> Float64Matrix:
    """
    ``d = 1 - |rho|``, sign-blind dissimilarity.

    Input domain ``[-1, 1]``, symmetric, unit diagonal. Output ``[0, 1]``,
    diagonal exactly zero.

    **Not a metric.** Two distinct variables with ``rho = -1`` are at
    dissimilarity zero, so identity of indiscernibles fails. Use it only when
    perfect anti-correlation genuinely means "same risk driver", for instance
    when clustering hedge candidates.
    """
    _validate_square_symmetric(correlation, allow_infinite_diagonal=False)
    _validate_correlation_domain(correlation)

    return np.ascontiguousarray(CORRELATION_UPPER_BOUND - np.abs(correlation))


def one_minus_dependence(dependence: Float64Matrix) -> Float64Matrix:
    """
    ``d = 1 - D`` for a dependence measure already bounded in ``[0, 1]``.

    Input domain ``[0, 1]``, symmetric, unit diagonal. Output ``[0, 1]``,
    diagonal exactly zero.

    **Metric status not guaranteed.** An unsigned dependence measure is not the
    cosine of an angle between vectors in an inner-product space, so the PSD
    argument that rescues the angular transform does not apply.
    """
    _validate_square_symmetric(dependence, allow_infinite_diagonal=False)

    if dependence.min() < 0.0 or dependence.max() > CORRELATION_UPPER_BOUND:
        raise ValueError(
            "input must be bounded in [0, 1]; got range "
            f"[{dependence.min()!r}, {dependence.max()!r}]."
        )

    return np.ascontiguousarray(CORRELATION_UPPER_BOUND - dependence)


def linfoot_dissimilarity(mutual_information: Float64Matrix) -> Float64Matrix:
    """
    ``d = 1 - sqrt(1 - exp(-2 I))``, via Linfoot's informational coefficient.

    Mathematical basis
        Linfoot (1957) defines the informational coefficient of correlation
        ``rho_I = sqrt(1 - exp(-2 I))``. For a bivariate Gaussian,
        ``I = -0.5 log(1 - rho^2)``, hence ``rho_I = |rho|`` exactly. The
        transform therefore places an information measure on the same ``[0, 1)``
        scale as an absolute correlation, with a defensible calibration rather
        than an arbitrary rescaling.

    Input domain
        ``[0, inf]`` in **nats**, symmetric. The diagonal may be ``+inf``: the
        self-entropy convention maps through cleanly, since ``exp(-inf) = 0``
        gives ``rho_I = 1`` and ``d = 0`` exactly.

    Output domain
        ``(0, 1]`` off-diagonal, approaching zero as ``I`` grows.

    **Metric status not guaranteed**, inheriting the failure of
    :func:`absolute_dissimilarity`.

    Notes
    -----
    Retained as a representation operation. No estimator in this phase produces
    a mutual-information matrix; the transform is exercised directly on a
    caller-supplied matrix.
    """
    _validate_square_symmetric(mutual_information, allow_infinite_diagonal=True)

    if mutual_information.min() < 0.0:
        raise ValueError(
            "mutual information must be non-negative; got minimum "
            f"{mutual_information.min()!r}."
        )

    informational_coefficient = np.sqrt(
        CORRELATION_UPPER_BOUND - np.exp(-2.0 * mutual_information)
    )

    return np.ascontiguousarray(
        CORRELATION_UPPER_BOUND - informational_coefficient
    )
