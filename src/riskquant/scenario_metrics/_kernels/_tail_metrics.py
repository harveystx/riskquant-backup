# src/riskquant/scenario_metrics/kernels/tail_metrics.py

from __future__ import annotations

import numba as nb
import numpy as np

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    parallel_njit,
)


@parallel_njit(
    nb.void(
        _F64_2D,  # group_value_matrix
        _F64_2D,  # scratchpad
        nb.int64,  # kth_index
        nb.int64,  # tail_count
        _F64_1D,  # var_out
        _F64_1D,  # es_out
    )
)
def _kernel_lower_tail_var_es(
    group_value_matrix: np.ndarray,
    scratchpad: np.ndarray,
    kth_index: int,
    tail_count: int,
    var_out: np.ndarray,
    es_out: np.ndarray,
) -> None:
    """
    Per-group lower-tail VaR and ES.

    Parameters
    ----------
    group_value_matrix : Float64Matrix
        Shape (n_groups, n_scenarios).

    scratchpad : Float64Matrix
        Same shape as group_value_matrix.

    kth_index
        Lower-tail VaR order statistic.

    tail_count
        Number of worst observations included in ES.
    """

    n_groups, n_scenarios = group_value_matrix.shape

    for g in nb.prange(n_groups):
        row = scratchpad[g]

        for s in range(n_scenarios):
            row[s] = group_value_matrix[g, s]

        row.partition(kth_index)

        var_out[g] = row[kth_index]

        tail_sum = 0.0

        for s in range(tail_count):
            tail_sum += row[s]

        es_out[g] = tail_sum / tail_count
