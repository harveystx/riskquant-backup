# src/riskquant/dependence/_kernels/_kendall_tau_b.py

"""
Knight's O(S log S) Kendall tau-b, evaluated over the strict upper triangle.

Why a kernel exists here at all
-------------------------------
The library rule is: no Numba without a measured win. Two candidate kernels
were benchmarked against ``scipy.stats.kendalltau`` called pair by pair
(single core, S observations, K variables, argsort preparation included in the
kernel timing):

===========  =========  ==============  ============  ==========
K            S          naive O(S^2)    this kernel   SciPy loop
===========  =========  ==============  ============  ==========
10           250        5.0 ms          1.3 ms        13.8 ms
50           250        139 ms          22 ms         341 ms
100          250        562 ms          85 ms         1428 ms
10           1000       81 ms           6.1 ms        21.5 ms
50           1000       2221 ms         110 ms        489 ms
20           5000       2128 ms         117 ms        194 ms
100          5000       n/a             2505 ms       5125 ms
===========  =========  ==============  ============  ==========

The naive O(S^2) kernel loses badly past S = 1000, which is squarely inside the
production range, so it was rejected. This kernel keeps SciPy's asymptotic
complexity, drops the per-pair Python and validation overhead, and parallelises
over pairs, giving 1.7x to 17x single-threaded. Agreement with SciPy is
2.8e-17 maximum absolute deviation across continuous, integer-tied and
zero-inflated inputs.

Definition
----------
With ``n0 = S(S-1)/2`` total pairs, ``n1`` pairs tied in x, ``n2`` pairs tied
in y and ``n3`` pairs tied in both::

    tau_b = (n0 - n1 - n2 + n3 - 2 * discordant) / sqrt((n0 - n1) * (n0 - n2))

This is the tie-corrected variant, identical to ``scipy.stats.kendalltau(...,
variant="b")`` and to ``pandas.DataFrame.corr(method="kendall")``. The
denominator vanishes when a variable is constant, which the estimator rejects
before reaching this kernel.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from riskquant.jit import (
    _F64_1D_C,
    _F64_1D_C_RO,
    _F64_2D_C,
    _F64_2D_C_RO,
    _I32_1D_C_RO,
    _I32_2D_C_RO,
    _NB_FLOAT64,
    _NB_VOID,
    njit,
    parallel_njit,
)


@njit(_NB_FLOAT64(_F64_1D_C, _F64_1D_C))
def _count_inversions(
    values: NDArray[np.float64], buffer: NDArray[np.float64]
) -> float:
    """
    Count inversions by bottom-up merge sort. Destroys ``values``.

    An inversion is a pair ``a < b`` with ``values[a] > values[b]``. When
    ``values`` holds y ordered lexicographically by (x, y), the inversion count
    is exactly the number of discordant pairs.

    The count accumulates in float64 rather than int64 because ``S(S-1)/2``
    exceeds 2^53 only beyond S = 1.3e8, far outside any feasible sample, and
    the tau-b arithmetic is float64 throughout.

    Parameters
    ----------
    values : Float64Vector
        ``(S,)`` C-contiguous scratch. Overwritten.

    buffer : Float64Vector
        ``(S,)`` C-contiguous scratch. Overwritten.

    Returns
    -------
    discordant_count : float
    """
    count = values.size
    inversions = 0.0

    source = values
    target = buffer
    width = 1

    while width < count:
        block_start = 0
        while block_start < count:
            left = block_start
            middle = min(block_start + width, count)
            right = min(block_start + 2 * width, count)

            left_cursor = left
            right_cursor = middle
            write_cursor = left

            while left_cursor < middle and right_cursor < right:
                if source[left_cursor] <= source[right_cursor]:
                    target[write_cursor] = source[left_cursor]
                    left_cursor += 1
                else:
                    # Every unconsumed left element outranks this right element.
                    inversions += middle - left_cursor
                    target[write_cursor] = source[right_cursor]
                    right_cursor += 1
                write_cursor += 1

            while left_cursor < middle:
                target[write_cursor] = source[left_cursor]
                left_cursor += 1
                write_cursor += 1

            while right_cursor < right:
                target[write_cursor] = source[right_cursor]
                right_cursor += 1
                write_cursor += 1

            block_start += 2 * width

        source, target = target, source
        width *= 2

    return inversions


@parallel_njit(
    _NB_VOID(
        _F64_2D_C_RO,  # variables (K, S)
        _I32_2D_C_RO,  # sort_order (K, S)
        _F64_1D_C_RO,  # tied_pairs (K,)
        _I32_1D_C_RO,  # pair_row (P,)
        _I32_1D_C_RO,  # pair_col (P,)
        _F64_2D_C,  # x_scratch (T, S)
        _F64_2D_C,  # y_scratch (T, S)
        _F64_2D_C,  # merge_scratch (T, S)
        _F64_2D_C,  # out (K, K)
    )
)
def kernel_kendall_tau_b_upper(
    variables: NDArray[np.float64],
    sort_order: NDArray[np.int32],
    tied_pairs: NDArray[np.float64],
    pair_row: NDArray[np.int32],
    pair_col: NDArray[np.int32],
    x_scratch: NDArray[np.float64],
    y_scratch: NDArray[np.float64],
    merge_scratch: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Fill the strict upper triangle of ``out`` with Kendall's tau-b.

    Scratch is thread-local, indexed by ``numba.get_thread_id()``, and sized
    ``(n_threads, S)``. Nothing is allocated inside the pair loop, and the peak
    working set stays ``O(K * S + n_threads * S)`` — never ``O(K^2 * S)``.

    Parameters
    ----------
    variables : Float64Matrix
        ``(K, S)`` canonical observations.

    sort_order : Int32Matrix
        ``(K, S)`` stable ascending argsort of each row.

    tied_pairs : Float64Vector
        ``(K,)`` from :func:`kernel_tied_pair_counts`.

    pair_row, pair_col : Int32Vector
        ``(P,)`` strict-upper-triangle pair indices, ``P = K(K-1)/2``.

    x_scratch, y_scratch, merge_scratch : Float64Matrix
        ``(n_threads, S)`` thread-local scratch. Overwritten.

    out : Float64Matrix
        ``(K, K)``. Only the strict upper triangle is written; the caller
        mirrors and sets the diagonal.
    """
    n_observations = variables.shape[1]
    n_pairs = pair_row.size
    total_pairs = n_observations * (n_observations - 1) / 2.0

    for pair in nb.prange(n_pairs):
        thread = nb.get_thread_id()  # type: ignore

        row = pair_row[pair]
        col = pair_col[pair]

        x_values = x_scratch[thread]
        y_values = y_scratch[thread]
        merge_buffer = merge_scratch[thread]

        order_row = sort_order[row]
        for position in range(n_observations):
            source = order_row[position]
            x_values[position] = variables[row, source]
            y_values[position] = variables[col, source]

        # Complete the lexicographic (x, y) order by sorting y inside each
        # x-tie block. Without this the joint-tie count and the inversion count
        # would both be wrong on tie-heavy data such as zero-inflated returns.
        block_start = 0
        while block_start < n_observations:
            block_end = block_start + 1
            while (
                block_end < n_observations
                and x_values[block_end] == x_values[block_start]
            ):
                block_end += 1
            if block_end - block_start > 1:
                y_values[block_start:block_end].sort()
            block_start = block_end

        joint_ties = 0.0
        block_start = 0
        while block_start < n_observations:
            block_end = block_start + 1
            while (
                block_end < n_observations
                and x_values[block_end] == x_values[block_start]
                and y_values[block_end] == y_values[block_start]
            ):
                block_end += 1
            run_length = block_end - block_start
            joint_ties += run_length * (run_length - 1) / 2.0
            block_start = block_end

        discordant = _count_inversions(y_values, merge_buffer)

        row_ties = tied_pairs[row]
        col_ties = tied_pairs[col]

        concordant_minus_discordant = (
            total_pairs - row_ties - col_ties + joint_ties - 2.0 * discordant
        )
        denominator = np.sqrt(total_pairs - row_ties) * np.sqrt(total_pairs - col_ties)

        out[row, col] = concordant_minus_discordant / denominator
