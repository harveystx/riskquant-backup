# src/riskquant/dependence/estimators/spearman.py

"""
Spearman rank correlation.

Definition
----------
Spearman's rho is *defined here* as the Pearson correlation of the average
ranks::

    rho_S(x, y) = rho_P(rank(x), rank(y))

This is the definition that remains correct under ties. The shortcut formula
``1 - 6 * sum d^2 / (S(S^2 - 1))`` is valid only when no ties are present, and a
zero-inflated return series is nothing but ties.

What it measures
----------------
Monotone association. Invariant under any strictly increasing transformation
applied to either variable separately, so log prices and prices give the same
answer -- unlike Pearson.

Tie handling
------------
Average ranks. A constant variable produces a constant rank vector and is
rejected: the rank denominator is exactly zero.

Degeneracy gate
---------------
Gated on the sum of squares of the *ranks*, not of the raw observations. The two
conditions differ only under float64 underflow, where a variable such as
``[1e-300, 0, 0]`` has a well-defined non-degenerate rank vector while its raw
centred sum of squares rounds to exactly zero. Gating on the representation the
estimator actually consumes is the correct rule, and the gate output is ``(K,)``
so it allocates no ``(K, S)`` temporary.

Contracts
---------
Output
    ``(K, K)`` float64 C-contiguous, non-writeable, exactly symmetric, unit
    diagonal, values in ``[-1, 1]``. **PSD guaranteed by construction** -- a
    Gram matrix of rank vectors.

Complexity
    ``O(K^2 S)`` given the shared rank artifact; the ``O(KS log S)`` ranking is
    paid once at compilation and shared with every other rank estimator.
"""

from __future__ import annotations

import numpy as np

from riskquant.dependence.contracts import DependenceProvenance, DependenceResult
from riskquant.dependence.estimators.pearson import _gram_correlation_matrix
from riskquant.dependence.taxonomy import ESTIMATOR_CONTRACT, DependenceEstimator
from riskquant.observations._kernels import kernel_sum_of_squares
from riskquant.observations.compile import CompiledSample, RankRepresentation


def spearman(
    sample: CompiledSample,
    ranks: RankRepresentation,
) -> DependenceResult:
    """
    Spearman rank correlation matrix.

    Parameters
    ----------
    sample : CompiledSample
        Supplies provenance only; the estimate is a function of the ranks.

    ranks : RankRepresentation
        Built from the same sample.

    Returns
    -------
    result : DependenceResult

    Raises
    ------
    ValueError
        If any variable is constant, hence has a degenerate rank vector.
    """
    rank_sum_of_squares = np.empty(ranks.variable_count, dtype=np.float64)
    kernel_sum_of_squares(ranks.ranks, rank_sum_of_squares)

    matrix = _gram_correlation_matrix(
        ranks.ranks,
        rank_sum_of_squares,
        estimator=DependenceEstimator.SPEARMAN,
    )

    return DependenceResult(
        matrix=matrix,
        contract=ESTIMATOR_CONTRACT[DependenceEstimator.SPEARMAN],
        provenance=DependenceProvenance(
            observation_count=sample.observation_count,
        ),
    )
