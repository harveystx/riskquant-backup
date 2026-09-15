# tests/dependence_stress/canonical_measures.py

"""
Canonical dependence measurement for dependence_stress transformation tests.

Phase 3 removed the package-local ``diagnostics`` module. Dependence
measurement is owned by ``riskquant.dependence``; ``dependence_stress`` owns
only the transformation. These thin adapters compile a raw ``K x S`` matrix
into the estate-canonical ``CompiledSample`` / ``RankRepresentation`` and call
the canonical estimators, so transformation tests can assert stress EFFECTS
without any package-local statistic.

Tail semantics (canonical, directional)
---------------------------------------
The canonical threshold ``q`` is the CUT QUANTILE: the tail is the extreme
``1 - q`` region, with an exact lower/upper reflection. A tail of size
``tail_fraction`` is therefore ``q = 1 - tail_fraction``. The coefficient is
the directional lower/upper tail-dependence estimate
``lambda_hat[i, j] = P(j in tail | i in tail) = n_ij / m_i``, valid only where
the estimability mask is True. This replaces the old package-local symmetric
``n_ij / n_tail`` co-occurrence, which was a different (non-canonical) quantity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from riskquant.dependence.coefficients.tail_coefficients import tail_coefficients
from riskquant.dependence.contracts import TailSide
from riskquant.dependence.estimators.pearson import pearson
from riskquant.dependence.estimators.spearman import spearman
from riskquant.dependence.estimators.tail_counts import tail_counts
from riskquant.observations.compile import compile_ranks, compile_sample
from riskquant.observations.orientation import ObservationOrientation

_ORIENT = ObservationOrientation.VARIABLES_ON_ROWS

FloatMatrix = NDArray[np.float64]
BoolMatrix = NDArray[np.bool_]


def _sample(data: FloatMatrix):
    return compile_sample(
        np.ascontiguousarray(data, dtype=np.float64), orientation=_ORIENT
    )


def pearson_matrix(data: FloatMatrix) -> FloatMatrix:
    """Canonical Pearson correlation matrix of a raw K x S matrix."""
    return pearson(_sample(data)).matrix


def spearman_matrix(data: FloatMatrix) -> FloatMatrix:
    """Canonical Spearman rank-correlation matrix of a raw K x S matrix."""
    sample = _sample(data)
    return spearman(sample, compile_ranks(sample)).matrix


def _tail_coefficient_result(data: FloatMatrix, tail_fraction: float, tail: TailSide):
    sample = _sample(data)
    ranks = compile_ranks(sample)
    q = 1.0 - float(tail_fraction)  # cut quantile for an extreme tail_fraction tail
    counts = tail_counts(
        sample, ranks, thresholds=np.array([q], dtype=np.float64), tail=tail
    )
    return tail_coefficients(counts)


def lower_tail_lambda(
    data: FloatMatrix, tail_fraction: float
) -> tuple[FloatMatrix, BoolMatrix]:
    """Directional lower-tail dependence lambda_hat (K x K) and its mask."""
    coeff = _tail_coefficient_result(data, tail_fraction, TailSide.LOWER)
    return coeff.lambda_hat[0], coeff.estimable_lambda[0]


def upper_tail_lambda(
    data: FloatMatrix, tail_fraction: float
) -> tuple[FloatMatrix, BoolMatrix]:
    """Directional upper-tail dependence lambda_hat (K x K) and its mask."""
    coeff = _tail_coefficient_result(data, tail_fraction, TailSide.UPPER)
    return coeff.lambda_hat[0], coeff.estimable_lambda[0]


def mean_offdiag(matrix: FloatMatrix, mask: BoolMatrix | None = None) -> float:
    """
    Mean of the off-diagonal entries, restricted to ``mask`` where supplied.

    A supplied mask (from a tail coefficient) selects the estimable cells; the
    unestimable cells hold NaN and are excluded.
    """
    k = matrix.shape[0]
    off = ~np.eye(k, dtype=bool)
    if mask is not None:
        off &= mask
    return float(np.nanmean(matrix[off])) if off.any() else float("nan")


def lower_tail_lambda_mean(data: FloatMatrix, tail_fraction: float) -> float:
    """Mean estimable off-diagonal directional lower-tail lambda_hat."""
    return mean_offdiag(*lower_tail_lambda(data, tail_fraction))


def upper_tail_lambda_mean(data: FloatMatrix, tail_fraction: float) -> float:
    """Mean estimable off-diagonal directional upper-tail lambda_hat."""
    return mean_offdiag(*upper_tail_lambda(data, tail_fraction))
