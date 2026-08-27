# src/riskquant/dependence_stress/ranks.py

"""
Empirical rank semantics.

This module is the only place in the package that defines what a rank
is. Diagnostics and transformers import from here. They never
reimplement ranking.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .types import FloatArray
from .validation import validate_observation_matrix


def _average_ranks_unchecked(
    matrix: FloatArray,
) -> FloatArray:
    """
    Row-wise average ranks of an already-validated matrix.

    Internal trusted path. ``scipy.stats.rankdata`` handles the whole
    K x S matrix along axis 1 in one call, so no Python-level row loop
    is required.
    """
    ranks = rankdata(
        matrix,
        method="average",
        axis=1,
    )

    return np.ascontiguousarray(
        ranks,
        dtype=np.float64,
    )


def average_ranks(
    data: FloatArray,
) -> FloatArray:
    """
    Compute row-wise average ranks.

    Tied observations receive the average of the ranks they span. The
    smallest observation has rank 1 and the largest has rank S.

    ``np.argsort(np.argsort(x))`` is deliberately not used. It assigns
    ordinal ranks and is wrong under ties.

    Parameters
    ----------
    data:
        K x S empirical observation matrix.

    Returns
    -------
    FloatArray
        K x S C-contiguous float64 rank matrix.

    Raises
    ------
    ValueError
        If the observation-matrix contract is violated.
    """
    return _average_ranks_unchecked(
        validate_observation_matrix(data),
    )


def _empirical_uniforms_unchecked(
    matrix: FloatArray,
) -> FloatArray:
    """
    Mid-rank uniforms of an already-validated matrix.
    """
    ranks = _average_ranks_unchecked(matrix)

    ranks -= 0.5
    ranks /= matrix.shape[1]

    return ranks


def empirical_uniforms(
    data: FloatArray,
) -> FloatArray:
    """
    Convert empirical observations to mid-rank uniforms.

    For average rank ``r`` among ``S`` observations:

        U = (r - 0.5) / S

    Every uniform therefore lies strictly inside (0, 1), which keeps the
    values safe for an inverse-CDF transform.

    Parameters
    ----------
    data:
        K x S empirical observation matrix.

    Returns
    -------
    FloatArray
        K x S C-contiguous float64 matrix of mid-rank uniforms.
    """
    return _empirical_uniforms_unchecked(
        validate_observation_matrix(data),
    )
