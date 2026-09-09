# src/riskquant/observations/_kernels.py

"""
Zero-allocation scan kernels for the observation contract and compilation.

PRIVATE. Not part of the public surface of ``riskquant.observations``.
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
    _I32_2D_C_RO,
    _NB_INT64,
    _NB_VOID,
    njit,
    parallel_njit,
)


@njit(_NB_INT64(_F64_2D_C_RO))
def kernel_first_non_finite_flat_index(matrix: NDArray[np.float64]) -> int:
    """
    Flat index of the first non-finite cell, or ``-1`` when all cells are finite.

    Single-threaded by construction: the early exit on the first offender is
    worth more than thread scaling, and a ``prange`` cannot short-circuit.
    Returning the index lets the caller report a cell address rather than a bare
    "contains NaN".
    """
    n_rows = int(matrix.shape[0])
    n_cols = int(matrix.shape[1])

    for row in range(n_rows):
        for col in range(n_cols):
            if not np.isfinite(matrix[row, col]):
                return row * n_cols + col

    return -1


@parallel_njit(_NB_VOID(_F64_2D_C_RO, _F64_1D_C))
def kernel_sum_of_squares(
    matrix: NDArray[np.float64],
    sum_of_squares_out: NDArray[np.float64],
) -> None:
    """
    Per-row centred sum of squares, ``(K,)`` output, no ``(K, S)`` scratch.

    The arithmetic sequence is identical to :func:`kernel_center_rows`: the same
    accumulation order for the row sum, the same reciprocal multiply for the
    mean, the same accumulation order for the squared deviations. The two
    therefore produce bitwise-identical sums, which is asserted in the test
    suite and is what allows the degeneracy gate to move out of the centring
    pass without changing any estimator's numbers.

    The sum of squares is returned rather than a standard deviation so that
    callers can apply an exact ``== 0.0`` degeneracy test before any division.

    Parameters
    ----------
    matrix : Float64Matrix
        ``(K, S)`` C-contiguous. Never mutated.

    sum_of_squares_out : Float64Vector
        ``(K,)``. Overwritten with ``sum((x - mean(x))^2)`` per row.
    """
    n_rows, n_cols = matrix.shape
    inverse_count = 1.0 / n_cols

    for row in nb.prange(n_rows):
        row_sum = 0.0
        for col in range(n_cols):
            row_sum += matrix[row, col]

        row_mean = row_sum * inverse_count

        accumulated = 0.0
        for col in range(n_cols):
            deviation = matrix[row, col] - row_mean
            accumulated += deviation * deviation

        sum_of_squares_out[row] = accumulated


@parallel_njit(_NB_VOID(_F64_2D_C_RO, _F64_2D_C, _F64_1D_C))
def kernel_center_rows(
    matrix: NDArray[np.float64],
    centered_out: NDArray[np.float64],
    sum_of_squares_out: NDArray[np.float64],
) -> None:
    """
    Row-wise mean removal plus the centred sum of squares, in one pass.

    Writing the centred values and accumulating ``sum((x - mean)^2)`` in the same
    traversal removes three NumPy temporaries and measured 4-6x faster than the
    equivalent NumPy expression on inputs from 10x1000 to 500x5000.

    Used by the estimator, never by a degeneracy gate: it allocates a ``(K, S)``
    output, which a gate must not.
    """
    n_rows, n_cols = matrix.shape
    inverse_count = 1.0 / n_cols

    for row in nb.prange(n_rows):
        row_sum = 0.0
        for col in range(n_cols):
            row_sum += matrix[row, col]

        row_mean = row_sum * inverse_count

        accumulated = 0.0
        for col in range(n_cols):
            deviation = matrix[row, col] - row_mean
            centered_out[row, col] = deviation
            accumulated += deviation * deviation

        sum_of_squares_out[row] = accumulated


@parallel_njit(_NB_VOID(_F64_2D_C, _F64_1D_C_RO))
def kernel_scale_rows(
    matrix: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> None:
    """In-place row scaling, ``matrix[k, :] *= scale[k]``."""
    n_rows, n_cols = matrix.shape

    for row in nb.prange(n_rows):
        factor = scale[row]
        for col in range(n_cols):
            matrix[row, col] *= factor


@parallel_njit(_NB_VOID(_F64_2D_C_RO, _I32_2D_C_RO, _F64_1D_C))
def kernel_tied_pair_counts(
    variables: NDArray[np.float64],
    sort_order: NDArray[np.int32],
    tied_pairs_out: NDArray[np.float64],
) -> None:
    """
    Per-variable count of tied pairs, ``sum over tie blocks of t(t-1)/2``.

    Depends on one variable only, so it is computed once per variable at
    compilation rather than once per pair inside Kendall, a saving of a factor
    ``(K-1)/2``.
    """
    n_variables, n_observations = variables.shape

    for variable in nb.prange(n_variables):
        order_row = sort_order[variable]
        total = 0.0
        run_length = 1

        previous = variables[variable, order_row[0]]
        for position in range(1, n_observations):
            current = variables[variable, order_row[position]]
            if current == previous:
                run_length += 1
            else:
                total += run_length * (run_length - 1) / 2.0
                run_length = 1
            previous = current

        total += run_length * (run_length - 1) / 2.0
        tied_pairs_out[variable] = total


@parallel_njit(_NB_VOID(_F64_2D_C_RO, _I32_2D_C_RO, _F64_1D_C))
def kernel_tie_rate(
    variables: NDArray[np.float64],
    sort_order: NDArray[np.int32],
    tie_rate_out: NDArray[np.float64],
) -> None:
    """
    Fraction of observations belonging to a tie block of size two or more.

    Chosen for interpretability at both ends: all values distinct gives exactly
    ``0.0``, all values equal gives exactly ``1.0``, and a series with 40 per
    cent exact zeros and otherwise distinct values gives ``0.4``. The tied-pair
    count of :func:`kernel_tied_pair_counts` is quadratic in the block size and
    does not have that reading.

    Reported per variable rather than as a sample-wide maximum: identification
    of a copula functional is a pairwise property, and a scalar cannot localise
    which pairs are affected.
    """
    n_variables, n_observations = variables.shape

    for variable in nb.prange(n_variables):
        order_row = sort_order[variable]
        tied_observations = 0

        block_start = 0
        while block_start < n_observations:
            block_end = block_start + 1
            anchor = variables[variable, order_row[block_start]]
            while (
                block_end < n_observations
                and variables[variable, order_row[block_end]] == anchor
            ):
                block_end += 1
            if block_end - block_start > 1:
                tied_observations += block_end - block_start
            block_start = block_end

        tie_rate_out[variable] = tied_observations / n_observations
