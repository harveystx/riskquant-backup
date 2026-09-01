# tests/observations/test_compile.py

"""Compilation contract: validation, ownership, and the shared rank artifacts."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy.stats import rankdata

from riskquant.observations._kernels import kernel_center_rows, kernel_sum_of_squares
from riskquant.observations._validate import reject_constant_variables
from riskquant.observations.compile import compile_ranks, compile_sample
from riskquant.observations.orientation import ObservationOrientation

VARIABLES_ROWS = ObservationOrientation.VARIABLES_ON_ROWS
OBSERVATIONS_ROWS = ObservationOrientation.OBSERVATIONS_ON_ROWS


def _valid() -> np.ndarray:
    return np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 3.0, 2.0]])


# ------------------------------------------------------------------ validation


def test_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        compile_sample(np.arange(6.0), orientation=VARIABLES_ROWS)


def test_three_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        compile_sample(np.zeros((2, 3, 4)), orientation=VARIABLES_ROWS)


def test_too_few_observations_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        compile_sample(np.zeros((3, 1)), orientation=VARIABLES_ROWS)


def test_too_few_variables_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 variables"):
        compile_sample(np.zeros((1, 5)), orientation=VARIABLES_ROWS)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_is_rejected_with_a_cell_address(bad_value: float) -> None:
    matrix = _valid()
    matrix[1, 2] = bad_value
    with pytest.raises(ValueError) as excinfo:
        compile_sample(matrix, orientation=VARIABLES_ROWS)
    message = str(excinfo.value)
    assert "variable_index=1" in message
    assert "observation_index=2" in message


def test_first_non_finite_cell_is_reported_in_row_major_order() -> None:
    matrix = _valid()
    matrix[0, 3] = np.nan
    matrix[1, 1] = np.nan
    with pytest.raises(ValueError, match=r"variable_index=0, observation_index=3"):
        compile_sample(matrix, orientation=VARIABLES_ROWS)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([["a", "b"], ["c", "d"]]),
        np.array([[1 + 1j, 2 + 0j], [3 + 0j, 4 + 0j]]),
        np.array([[True, False], [False, True]]),
        np.array([[None, 1], [2, 3]], dtype=object),
    ],
)
def test_unsupported_dtype_is_rejected(bad: np.ndarray) -> None:
    with pytest.raises(TypeError, match="real numeric dtype"):
        compile_sample(bad, orientation=VARIABLES_ROWS)


def test_integer_input_is_promoted_to_float64() -> None:
    sample = compile_sample(
        np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int32), orientation=VARIABLES_ROWS
    )
    assert sample.observations.dtype == np.float64


def test_orientation_must_be_an_enum_member() -> None:
    """
    The string is cast rather than ignored: the point is the *runtime* rejection
    of a value that a caller without type checking could easily supply.
    """
    not_an_orientation = cast(ObservationOrientation, "variables_on_rows")
    with pytest.raises(TypeError, match="ObservationOrientation"):
        compile_sample(_valid(), orientation=not_an_orientation)


def test_observations_on_rows_is_transposed_exactly_once() -> None:
    sample = compile_sample(_valid().T.copy(), orientation=OBSERVATIONS_ROWS)
    assert sample.observations.shape == (2, 4)
    assert np.array_equal(sample.observations, _valid())


def test_excessive_observation_count_is_rejected_loudly() -> None:
    """S >= 2**31 would overflow the int32 sort_order and corrupt tie detection."""

    class _HugeShape(np.ndarray):
        pass

    fake = np.zeros((2, 2)).view(_HugeShape)
    # Exercise the rule directly: allocating a 2**31-column array is infeasible.
    from riskquant.observations._validate import reject_excessive_observation_count

    reject_excessive_observation_count(2**31 - 1)
    with pytest.raises(ValueError, match="sort_order is int32"):
        reject_excessive_observation_count(2**31)
    assert fake.shape == (2, 2)


# ------------------------------------------------------------------- ownership


def test_compiled_sample_owns_a_copy_and_is_read_only() -> None:
    """
    Replaces test_already_canonical_input_is_not_copied.

    Zero-copy canonicalisation was deliberately removed: an artifact retained
    across many estimator calls must not alias a live caller buffer.
    """
    source = np.ascontiguousarray(_valid())
    expected = source.copy()

    sample = compile_sample(source, orientation=VARIABLES_ROWS)

    assert sample.observations is not source
    assert not sample.observations.flags.writeable
    assert not sample.sum_of_squares.flags.writeable

    source[0, 0] = 999.0

    assert np.array_equal(sample.observations, expected)


def test_estimates_are_unchanged_after_the_caller_mutates_its_array() -> None:
    from riskquant.dependence.estimators.pearson import pearson

    rng = np.random.default_rng(0)
    source = np.ascontiguousarray(rng.standard_normal((4, 60)))

    sample = compile_sample(source, orientation=VARIABLES_ROWS)
    before = pearson(sample).matrix.copy()

    source *= 3.0
    source[0, 0] = 1e6

    assert np.array_equal(pearson(sample).matrix, before)


def test_compiled_matrix_cannot_be_written_through() -> None:
    sample = compile_sample(_valid(), orientation=VARIABLES_ROWS)
    with pytest.raises(ValueError):
        sample.observations[0, 0] = 1.0


# ------------------------------------------------------ sum of squares and gate


def test_sum_of_squares_is_exactly_zero_for_a_constant_variable() -> None:
    sample = compile_sample(
        np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]]), orientation=VARIABLES_ROWS
    )
    assert sample.sum_of_squares[0] == 0.0
    assert sample.sum_of_squares[1] == pytest.approx(2.0)


def test_sum_of_squares_kernel_is_bitwise_identical_to_the_centring_kernel() -> None:
    """
    The gate may only move out of the centring pass if the arithmetic matches.

    Both kernels accumulate the row sum in the same order, form the mean by the
    same reciprocal multiply, and accumulate squared deviations in the same
    order, so the sums are the same bit patterns.
    """
    rng = np.random.default_rng(3)
    for shape in [(4, 50), (13, 977), (2, 2)]:
        matrix = np.ascontiguousarray(rng.standard_normal(shape))

        gate = np.empty(shape[0], dtype=np.float64)
        kernel_sum_of_squares(matrix, gate)

        centred = np.empty(shape, dtype=np.float64)
        from_centring = np.empty(shape[0], dtype=np.float64)
        kernel_center_rows(matrix, centred, from_centring)

        assert np.array_equal(gate, from_centring)


def test_degeneracy_gate_allocates_no_matrix_sized_scratch() -> None:
    """The gate output is (K,); a (K, S) temporary would be a defect."""
    import tracemalloc

    rng = np.random.default_rng(4)
    n_variables, n_observations = 20, 4000
    matrix = np.ascontiguousarray(rng.standard_normal((n_variables, n_observations)))

    out = np.empty(n_variables, dtype=np.float64)
    kernel_sum_of_squares(matrix, out)  # warm the JIT

    tracemalloc.start()
    gate = np.empty(n_variables, dtype=np.float64)
    kernel_sum_of_squares(matrix, gate)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    matrix_bytes = n_variables * n_observations * 8
    assert peak < matrix_bytes // 10


def test_constant_variables_are_rejected_with_their_indices() -> None:
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        reject_constant_variables(
            np.array([1.0, 0.0, 3.0, 0.0]), estimator_name="Test estimator"
        )


def test_tiny_but_non_zero_dispersion_is_accepted() -> None:
    """
    Rejection is exact-zero, not a hidden tolerance.

    Bounded below by float64 underflow: a dispersion of 1e-300 squares to zero
    and is genuinely rejected, so the magnitude here stays above that.
    """
    sample = compile_sample(
        np.array([[0.0, 1e-100, 0.0], [1.0, 2.0, 3.0]]), orientation=VARIABLES_ROWS
    )
    reject_constant_variables(sample.sum_of_squares, estimator_name="Test estimator")


# ------------------------------------------------------------------- rank artifacts


def test_ranks_are_average_ranks_along_the_observation_axis() -> None:
    sample = compile_sample(
        np.array([[1.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0]]),
        orientation=VARIABLES_ROWS,
    )
    ranks = compile_ranks(sample)
    assert np.array_equal(ranks.ranks[0], [1.5, 1.5, 3.0, 4.0])
    assert np.array_equal(ranks.ranks[1], [4.0, 3.0, 2.0, 1.0])


def test_ranks_match_scipy_average_ranks_on_tie_heavy_data() -> None:
    rng = np.random.default_rng(1)
    matrix = np.where(rng.random((4, 60)) < 0.5, 0.0, rng.standard_normal((4, 60)))
    sample = compile_sample(matrix, orientation=VARIABLES_ROWS)
    expected = rankdata(sample.observations, method="average", axis=1)
    assert np.array_equal(compile_ranks(sample).ranks, expected)


def test_average_ranks_differ_from_double_argsort_under_ties() -> None:
    """np.argsort(np.argsort(x)) breaks ties by position; average ranks do not."""
    sample = compile_sample(
        np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0]]),
        orientation=VARIABLES_ROWS,
    )
    average = compile_ranks(sample).ranks[0]
    ordinal = np.argsort(np.argsort(sample.observations[0])) + 1.0
    assert np.array_equal(average, [2.0, 2.0, 2.0, 4.0])
    assert not np.array_equal(average, ordinal)


def test_rank_artifacts_have_the_declared_layout_and_are_frozen() -> None:
    rng = np.random.default_rng(2)
    sample = compile_sample(rng.standard_normal((5, 40)), orientation=VARIABLES_ROWS)
    ranks = compile_ranks(sample)

    assert ranks.ranks.dtype == np.float64 and ranks.ranks.flags.c_contiguous
    assert ranks.sort_order.dtype == np.int32 and ranks.sort_order.flags.c_contiguous
    assert ranks.tie_pairs.shape == (5,)
    assert not ranks.ranks.flags.writeable
    assert not ranks.sort_order.flags.writeable
    assert not ranks.tie_pairs.flags.writeable


def test_sort_order_is_a_stable_ascending_argsort() -> None:
    sample = compile_sample(
        np.array([[3.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0]]),
        orientation=VARIABLES_ROWS,
    )
    ranks = compile_ranks(sample)
    assert np.array_equal(ranks.sort_order[0], [1, 2, 3, 0])


def test_tie_pairs_counts_tied_pairs_per_variable() -> None:
    sample = compile_sample(
        np.array([[1.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0]]),
        orientation=VARIABLES_ROWS,
    )
    ranks = compile_ranks(sample)
    assert ranks.tie_pairs[0] == pytest.approx(3.0)
    assert ranks.tie_pairs[1] == 0.0


def test_ranks_are_invariant_to_strictly_increasing_transformations() -> None:
    rng = np.random.default_rng(5)
    matrix = np.ascontiguousarray(rng.standard_normal((3, 50)))
    plain = compile_ranks(compile_sample(matrix, orientation=VARIABLES_ROWS))
    mapped = compile_ranks(compile_sample(np.exp(matrix), orientation=VARIABLES_ROWS))
    assert np.array_equal(plain.ranks, mapped.ranks)
