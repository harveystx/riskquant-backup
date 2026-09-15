# scripts/demos/dependence_stress/_canonical_measure.py

"""
Canonical dependence measurement for the dependence_stress demos.

Measurement is owned by ``riskquant.dependence`` (Phase 3). These adapters
compile a raw ``K x S`` matrix into the estate-canonical ``CompiledSample`` /
``RankRepresentation`` and call the canonical estimators.

Tail dependence here is the canonical DIRECTIONAL coefficient
``lambda_hat[i, j] = P(j in tail | i in tail)``. It is NOT the old symmetric
``n_ij / n_tail`` co-occurrence, which was removed in Phase 3. The threshold
``q`` is the cut quantile, so a tail of fraction ``p`` uses ``q = 1 - p``.
Non-estimable cells (a variable with an empty tail) are reported as 0.0 for
display; under a full tail alignment every marginal has a non-empty tail, so
every off-diagonal is estimable and equals 1.0.
"""

from __future__ import annotations

import numpy as np

from riskquant.dependence.coefficients.tail_coefficients import tail_coefficients
from riskquant.dependence.contracts import TailSide
from riskquant.dependence.estimators.pearson import pearson
from riskquant.dependence.estimators.spearman import spearman
from riskquant.dependence.estimators.tail_counts import tail_counts
from riskquant.observations.compile import compile_ranks, compile_sample
from riskquant.observations.orientation import ObservationOrientation

_ORIENT = ObservationOrientation.VARIABLES_ON_ROWS


def _sample(matrix):
    return compile_sample(
        np.ascontiguousarray(matrix, dtype=np.float64), orientation=_ORIENT
    )


def pearson_correlation(matrix):
    """Canonical Pearson correlation matrix (symmetric)."""
    return pearson(_sample(matrix)).matrix


def spearman_correlation(matrix):
    """Canonical Spearman rank-correlation matrix (symmetric)."""
    sample = _sample(matrix)
    return spearman(sample, compile_ranks(sample)).matrix


def _directional_lambda(matrix, tail_fraction, tail):
    sample = _sample(matrix)
    ranks = compile_ranks(sample)
    q = 1.0 - float(tail_fraction)
    counts = tail_counts(
        sample, ranks, thresholds=np.array([q], dtype=np.float64), tail=tail
    )
    coeff = tail_coefficients(counts)
    lam = coeff.lambda_hat[0].copy()
    lam[~coeff.estimable_lambda[0]] = 0.0
    return lam


def lower_tail_dependence(matrix, tail_fraction):
    """Directional lower-tail dependence lambda_hat (K x K; not symmetric)."""
    return _directional_lambda(matrix, tail_fraction, TailSide.LOWER)


def upper_tail_dependence(matrix, tail_fraction):
    """Directional upper-tail dependence lambda_hat (K x K; not symmetric)."""
    return _directional_lambda(matrix, tail_fraction, TailSide.UPPER)


def average_ranks(matrix):
    """Canonical row-wise average ranks (1..S), from RankRepresentation.ranks."""
    sample = _sample(matrix)
    return compile_ranks(sample).ranks


def empirical_uniforms(matrix):
    """Mid-rank uniforms U = (average_rank - 0.5) / S, strictly inside (0, 1)."""
    ranks = average_ranks(matrix).copy()
    ranks -= 0.5
    ranks /= matrix.shape[1]
    return ranks
