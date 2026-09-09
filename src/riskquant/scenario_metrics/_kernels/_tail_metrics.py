# src/riskquant/scenario_metrics/_kernels/_tail_metrics.py

from __future__ import annotations

import numba as nb
import numpy as np
from numba.core import types as nb_types

from riskquant.jit import (
    _F64_1D_C,
    _F64_1D_C_RO,
    _F64_2D_C,
    _F64_2D_C_RO,
    _I64_1D_C,
    stable_math_parallel_njit,
)


@stable_math_parallel_njit(
    nb_types.void(
        _F64_2D_C_RO,  # group_value_matrix (read only)
        _F64_2D_C,  # scratchpad
        nb_types.int64,  # var_index
        nb_types.int64,  # full_tail_count
        nb_types.float64,  # fractional_weight
        nb_types.float64,  # tail_mass
        _F64_1D_C,  # var_out
        _F64_1D_C,  # es_out
    )
)
def _kernel_lower_tail_var_es(
    group_value_matrix: np.ndarray,
    scratchpad: np.ndarray,
    var_index: int,
    full_tail_count: int,
    fractional_weight: float,
    tail_mass: float,
    var_out: np.ndarray,
    es_out: np.ndarray,
) -> None:
    """
    Compute lower-tail empirical VaR and ES.

    The scratchpad is partitioned only far enough to identify the required
    lower-tail order statistics. A complete sort of each scenario row is
    deliberately avoided.

    Contract
    --------
    Input rows are finite float64 values in C-contiguous storage.

    ``var_index`` is the zero-based empirical VaR order-statistic index.

    ``full_tail_count`` observations carry unit ES weight.

    ``fractional_weight`` is the residual weight on the next observation,
    with ``0 <= fractional_weight < 1``.

    ``tail_mass`` equals:

        full_tail_count + fractional_weight
    """

    n_groups, n_scenarios = group_value_matrix.shape

    for g in nb.prange(n_groups):
        row = scratchpad[g]

        # Copy caller-owned data into private mutable workspace.
        for s in range(n_scenarios):
            row[s] = group_value_matrix[g, s]

        # ---------------------------------------------------------------
        # Quickselect:
        #
        # Move the value at var_index into its final order-statistic
        # position without sorting the remainder of the row.
        # ---------------------------------------------------------------

        left = 0
        right = n_scenarios - 1

        while left < right:
            i = left
            j = right
            pivot = row[(left + right) >> 1]

            while i <= j:
                while row[i] < pivot:
                    i += 1

                while row[j] > pivot:
                    j -= 1

                if i <= j:
                    tmp = row[i]
                    row[i] = row[j]
                    row[j] = tmp
                    i += 1
                    j -= 1

            if var_index <= j:
                right = j
            elif var_index >= i:
                left = i
            else:
                break

        var_value = row[var_index]
        var_out[g] = var_value

        # ---------------------------------------------------------------
        # Exact empirical ES.
        #
        # The first ``full_tail_count`` entries are the complete empirical
        # tail observations. If fractional_weight > 0, var_value is the
        # next order statistic completing the tail mass.
        # ---------------------------------------------------------------

        tail_sum = 0.0

        for s in range(full_tail_count):
            tail_sum += row[s]

        if fractional_weight > 0.0:
            tail_sum += fractional_weight * var_value

        es_out[g] = tail_sum / tail_mass


@stable_math_parallel_njit(
    nb_types.void(
        _F64_2D_C_RO,  # component_matrix (K, S) read only
        _F64_1D_C_RO,  # total_row       (S,)   read only
        _I64_1D_C,  # index_scratch    (S,)   mutable workspace
        nb_types.int64,  # var_index
        nb_types.int64,  # full_tail_count
        nb_types.float64,  # fractional_weight
        nb_types.float64,  # tail_mass
        _F64_1D_C,  # var_contribution_out (K,)
        _F64_1D_C,  # es_contribution_out  (K,)
    )
)
def _kernel_lower_tail_contribution(
    component_matrix: np.ndarray,
    total_row: np.ndarray,
    index_scratch: np.ndarray,
    var_index: int,
    full_tail_count: int,
    fractional_weight: float,
    tail_mass: float,
    var_contribution_out: np.ndarray,
    es_contribution_out: np.ndarray,
) -> None:
    """
    Euler lower-tail VaR and ES contributions for additive components.

    Both contributions are computed over the SAME tail geometry the headline
    VaR/ES kernel uses (identical ``var_index``, ``full_tail_count``,
    ``fractional_weight`` and ``tail_mass``), so the columns foot exactly:

        sum_k es_contribution_out[k]  == ES(total_row)
        sum_k var_contribution_out[k] == VaR(total_row)

    to within floating-point summation error.

    Method
    ------
    The tail SCENARIO SET is defined by the total: the empirical lower tail of
    ``total_row``. An index-quickselect partitions ``index_scratch`` so that the
    first ``full_tail_count`` entries hold the indices of the smallest
    ``full_tail_count`` total values, and ``index_scratch[var_index]`` holds the
    VaR boundary scenario index. Each component is then gathered at exactly
    those scenario indices — never re-ranked independently — which is what makes
    the allocation additive.

    Contract
    --------
    * ``component_matrix`` is ``(K, S)`` finite float64, C-contiguous.
    * ``total_row`` is ``(S,)`` and MUST equal the column sum of
      ``component_matrix`` for the footing identity to hold; the caller owns
      that invariant.
    * ``index_scratch`` is an ``(S,)`` int64 workspace; its contents on entry
      are irrelevant and are overwritten.
    * Tail-geometry scalars are produced by the shared
      ``_compute_lower_tail_parameters`` policy resolver.
    """

    n_components, n_scenarios = component_matrix.shape

    # ------------------------------------------------------------------
    # Index-quickselect on the TOTAL row.
    #
    # Partition index_scratch around var_index by total_row value, so that
    # index_scratch[0:var_index] are indices of the smallest var_index totals
    # and index_scratch[var_index] is the VaR order-statistic scenario. This is
    # the argselect analogue of the value quickselect above; it never sorts the
    # remainder. Serial: the partition is O(S) average and shared by every
    # component, so it is not worth parallelising.
    # ------------------------------------------------------------------

    for s in range(n_scenarios):
        index_scratch[s] = s

    left = 0
    right = n_scenarios - 1

    while left < right:
        i = left
        j = right
        pivot = total_row[index_scratch[(left + right) >> 1]]

        while i <= j:
            while total_row[index_scratch[i]] < pivot:
                i += 1

            while total_row[index_scratch[j]] > pivot:
                j -= 1

            if i <= j:
                tmp = index_scratch[i]
                index_scratch[i] = index_scratch[j]
                index_scratch[j] = tmp
                i += 1
                j -= 1

        if var_index <= j:
            right = j
        elif var_index >= i:
            left = i
        else:
            break

    boundary_idx = index_scratch[var_index]

    # ------------------------------------------------------------------
    # Per-component gather over the shared tail scenario set.
    # ------------------------------------------------------------------

    for k in nb.prange(n_components):
        tail_sum = 0.0

        for j in range(full_tail_count):
            tail_sum += component_matrix[k, index_scratch[j]]

        if fractional_weight > 0.0:
            tail_sum += fractional_weight * component_matrix[k, boundary_idx]

        es_contribution_out[k] = tail_sum / tail_mass

        # Empirical Euler VaR contribution: the component's value in the single
        # VaR scenario. Sums across components to total_row[boundary_idx] = VaR.
        var_contribution_out[k] = component_matrix[k, boundary_idx]
