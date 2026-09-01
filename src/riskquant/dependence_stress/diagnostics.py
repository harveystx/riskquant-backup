# src/riskquant/dependence_stress/diagnostics.py

"""
Dependence measurement.

This module measures. It never transforms. It has no dependence on the
engine and no knowledge of any downstream risk metric.

Diagnostics are not hot-path risk calculations. Clarity is preferred over
micro-optimization here.
"""

from __future__ import annotations

import numpy as np

from .ranks import _average_ranks_unchecked
from .types import FloatArray
from .validation import (
    validate_observation_matrix,
    validate_probability,
)


def _reject_zero_variance(
    matrix: FloatArray,
    label: str,
) -> None:
    """
    Reject rows with zero variance.

    Correlation is undefined for a constant row. A silent NaN in a
    correlation matrix is a worse failure than an exception.
    """
    variance = np.var(matrix, axis=1)

    if np.any(variance <= 0.0):
        rows = np.flatnonzero(variance <= 0.0)

        raise ValueError(
            f"correlation is undefined for zero-variance {label} rows; "
            f"affected rows={rows.tolist()}."
        )


def pearson_correlation(
    data: FloatArray,
) -> FloatArray:
    """
    Pearson product-moment correlation between rows of a K x S matrix.

    Parameters
    ----------
    data:
        K x S empirical observation matrix.

    Returns
    -------
    FloatArray
        K x K C-contiguous float64 correlation matrix.

    Raises
    ------
    ValueError
        If any row has zero variance.
    """
    matrix = validate_observation_matrix(data)

    _reject_zero_variance(matrix, "observation")

    return np.ascontiguousarray(
        np.corrcoef(matrix),
        dtype=np.float64,
    )


def spearman_correlation(
    data: FloatArray,
) -> FloatArray:
    """
    Spearman rank correlation using average ranks.

    Ranking is delegated to ``ranks.average_ranks`` so that tie semantics
    are defined in exactly one place.

    Parameters
    ----------
    data:
        K x S empirical observation matrix.

    Returns
    -------
    FloatArray
        K x K C-contiguous float64 rank-correlation matrix.

    Raises
    ------
    ValueError
        If any rank row has zero variance, which happens when every
        observation in a row is tied.
    """
    matrix = validate_observation_matrix(data)

    ranks = _average_ranks_unchecked(matrix)

    _reject_zero_variance(ranks, "rank")

    return np.ascontiguousarray(
        np.corrcoef(ranks),
        dtype=np.float64,
    )


def empirical_lower_tail_cooccurrence(
    data: FloatArray,
    threshold: float = 0.05,
) -> FloatArray:
    """
    Finite-sample empirical lower-tail co-occurrence.

    For a sample of size S, the empirical lower tail of a row is its
    lowest ``n_tail = max(1, ceil(threshold * S))`` observations.

    For the pair (i, j):

        lambda_hat(i, j) = #{scenarios in both lower tails} / n_tail

    This is a finite-sample empirical lower-tail co-occurrence statistic.
    It is not the asymptotic copula lower-tail dependence coefficient.
    The two differ at any finite S, and the empirical version depends on
    the chosen threshold.

    A discrete mid-rank comparison such as ``U <= q`` does not in general
    select exactly ``S * q`` observations. The explicit ``n_tail`` count
    is used instead so that the denominator is exact.

    Parameters
    ----------
    data:
        K x S empirical observation matrix.
    threshold:
        Tail fraction in the open interval (0, 1).

    Returns
    -------
    FloatArray
        K x K C-contiguous float64 matrix with exact 1.0 on the diagonal.
    """
    probability = validate_probability(threshold)

    matrix = validate_observation_matrix(data)

    _, s = matrix.shape

    n_tail = max(
        1,
        int(np.ceil(probability * s)),
    )

    sorted_indices = np.argsort(
        matrix,
        axis=1,
        kind="stable",
    )

    # Float indicator rather than bool: the co-occurrence count is a
    # single BLAS GEMM on the K x S indicator, which beats any pairwise
    # Python-level loop.
    tail_indicator = np.zeros(
        matrix.shape,
        dtype=np.float64,
        order="C",
    )

    np.put_along_axis(
        tail_indicator,
        sorted_indices[:, :n_tail],
        1.0,
        axis=1,
    )

    joint_count = tail_indicator @ tail_indicator.T

    return np.ascontiguousarray(
        joint_count / n_tail,
        dtype=np.float64,
    )


def empirical_upper_tail_cooccurrence(
    data: FloatArray,
    threshold: float = 0.05,
) -> FloatArray:
    """
    Finite-sample empirical upper-tail co-occurrence.

    Calculates the proportion of shared scenarios among the highest
    `n_tail = max(1, ceil(threshold * S))` observations for every pair.
    """
    probability = validate_probability(threshold)
    matrix = validate_observation_matrix(data)
    _, s = matrix.shape

    n_tail = max(1, int(np.ceil(probability * s)))

    sorted_indices = np.argsort(matrix, axis=1, kind="stable")

    tail_indicator = np.zeros(matrix.shape, dtype=np.float64, order="C")

    # Place 1.0 at the original scenario indices of the highest n_tail elements
    np.put_along_axis(
        tail_indicator,
        sorted_indices[:, s - n_tail :],
        1.0,
        axis=1,
    )

    joint_count = tail_indicator @ tail_indicator.T

    return np.ascontiguousarray(
        joint_count / n_tail,
        dtype=np.float64,
    )


#: Deprecated alias retained for backward compatibility.
#:
#: The name suggests the asymptotic copula coefficient, which this
#: statistic is not. Prefer ``empirical_lower_tail_cooccurrence``.
lower_tail_dependence = empirical_lower_tail_cooccurrence
