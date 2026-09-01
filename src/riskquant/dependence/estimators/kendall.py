# src/riskquant/dependence/estimators/kendall.py

"""
Kendall's tau-b.

Variant
-------
**tau-b**, the tie-corrected variant. With ``n0 = S(S-1)/2`` total pairs, ``n1``
pairs tied in x, ``n2`` tied in y, ``n3`` tied in both, and ``D`` discordant
pairs::

    tau_b = (n0 - n1 - n2 + n3 - 2D) / sqrt((n0 - n1)(n0 - n2))

tau-a and tau-c are not supported. tau-b matches
``scipy.stats.kendalltau(variant="b")`` and
``pandas.DataFrame.corr(method="kendall")``, so migrated numbers reconcile.

What it measures
----------------
``P(concordant) - P(discordant)``, the most directly interpretable of the three
association measures. Systematically smaller in magnitude than Spearman on the
same data; the two are not interchangeable.

Why it consumes ranks rather than raw observations
--------------------------------------------------
This is an exact transformation, not an approximation. Average ranking satisfies
two properties by construction:

* ``x_a = x_b  =>  r_a = r_b``  (members of a tie block share the block mean);
* ``x_a < x_b  =>  r_a < r_b``  (distinct values occupy disjoint, order-
  respecting rank intervals, so their means are strictly ordered).

Together these give ``sign(r_a - r_b) = sign(x_a - x_b)`` for every pair.
tau-b is a function of ``n0, n1, n2, n3`` and the discordant count, each defined
purely by the sign pattern of pairwise differences. The sign patterns are
identical, so every count is identical, so the final value is **bitwise**
identical -- the counts are integers and the closing arithmetic is applied to
identical integers.

The one way this could fail is if two distinct tie blocks rounded to the same
average rank. They cannot: consecutive blocks spanning ``[i+1..j]`` and
``[j+1..k]`` have means differing by ``(len_A + len_B)/2 >= 1``, and all such
means are integers or half-integers, exactly representable while ``2S <= 2**53``.
The engine-wide ``S < 2**31`` bound subsumes that.

PSD
---
**Not guaranteed.** tau-b is assembled pair by pair with no Gram
representation. This is the most consequential fact about it for a risk engine
and it is carried in the contract, not in prose.

Contracts
---------
Output
    ``(K, K)`` float64 C-contiguous, non-writeable, exactly symmetric, unit
    diagonal, values in ``[-1, 1]``.

Complexity
    ``O(K^2 S log S)`` time, ``O(n_threads * S + K^2)`` memory beyond the shared
    artifacts.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from riskquant.dependence._kernels._kendall_tau_b import kernel_kendall_tau_b_upper
from riskquant.dependence._matrix import (
    allocate_dependence_matrix,
    finalise_symmetric,
    upper_triangle_pairs,
)
from riskquant.dependence.contracts import DependenceProvenance, DependenceResult
from riskquant.dependence.taxonomy import ESTIMATOR_CONTRACT, DependenceEstimator
from riskquant.observations._kernels import kernel_sum_of_squares
from riskquant.observations._validate import reject_constant_variables
from riskquant.observations.compile import CompiledSample, RankRepresentation


def kendall_tau_b(
    sample: CompiledSample,
    ranks: RankRepresentation,
) -> DependenceResult:
    """
    Kendall tau-b matrix.

    Parameters
    ----------
    sample : CompiledSample
        Supplies provenance only.

    ranks : RankRepresentation
        Supplies the average ranks, the shared sort order and the tied-pair
        counts. The sort order of the average ranks equals the sort order of the
        raw observations under a stable sort, because average ranking is
        strictly increasing on distinct values and constant within tie blocks,
        and both sorts break ties by original position.

    Returns
    -------
    result : DependenceResult

    Raises
    ------
    ValueError
        If any variable is constant, which makes ``n0 - n1 = 0`` and the
        denominator vanish.
    """
    rank_sum_of_squares = np.empty(ranks.variable_count, dtype=np.float64)
    kernel_sum_of_squares(ranks.ranks, rank_sum_of_squares)
    reject_constant_variables(rank_sum_of_squares, estimator_name="Kendall tau-b")

    n_variables = ranks.variable_count
    n_observations = ranks.observation_count

    pair_row, pair_col = upper_triangle_pairs(n_variables)

    thread_count = nb.get_num_threads()
    x_scratch = np.empty((thread_count, n_observations), dtype=np.float64)
    y_scratch = np.empty((thread_count, n_observations), dtype=np.float64)
    merge_scratch = np.empty((thread_count, n_observations), dtype=np.float64)

    tau_b = allocate_dependence_matrix(n_variables)

    kernel_kendall_tau_b_upper(
        ranks.ranks,
        ranks.sort_order,
        ranks.tie_pairs,
        pair_row,
        pair_col,
        x_scratch,
        y_scratch,
        merge_scratch,
        tau_b,
    )

    matrix = finalise_symmetric(
        tau_b,
        contract=ESTIMATOR_CONTRACT[DependenceEstimator.KENDALL_TAU_B],
    )

    return DependenceResult(
        matrix=matrix,
        contract=ESTIMATOR_CONTRACT[DependenceEstimator.KENDALL_TAU_B],
        provenance=DependenceProvenance(
            observation_count=sample.observation_count,
        ),
    )
