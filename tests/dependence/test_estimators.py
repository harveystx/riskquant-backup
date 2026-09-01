# tests/dependence/test_estimators.py

"""Hand-verifiable values and independent-reference agreement, Class 1."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import kendalltau, rankdata, spearmanr

from riskquant.dependence.estimators.kendall import kendall_tau_b
from riskquant.dependence.estimators.pearson import pearson
from riskquant.dependence.estimators.spearman import spearman
from riskquant.observations.arrays import Float64Matrix
from tests.dependence.conftest import (
    compile_both,
    continuous_sample,
    zero_inflated_sample,
)


def _scipy_kendall(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    reference = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            reference[i, j] = kendalltau(matrix[i], matrix[j], variant="b").statistic
    return reference


# ------------------------------------------------------------------- Pearson


def test_pearson_perfect_positive_linear_relation_is_exactly_one() -> None:
    sample, _ = compile_both(np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0]]))
    assert pearson(sample).matrix[0, 1] == 1.0


def test_pearson_perfect_negative_linear_relation_is_exactly_minus_one() -> None:
    sample, _ = compile_both(np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]))
    assert pearson(sample).matrix[0, 1] == -1.0


def test_pearson_hand_calculated_example() -> None:
    """
    x = [1, 2, 3, 4],  y = [2, 4, 5, 4]
    xbar = 2.5, ybar = 3.75
    dx = [-1.5, -0.5, 0.5, 1.5], dy = [-1.75, 0.25, 1.25, 0.25]
    sum(dx*dy) = 2.625 - 0.125 + 0.625 + 0.375 = 3.5
    sum(dx^2) = 5.0,  sum(dy^2) = 4.75
    rho = 3.5 / sqrt(5.0 * 4.75)
    """
    sample, _ = compile_both(np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 4.0]]))
    assert pearson(sample).matrix[0, 1] == pytest.approx(3.5 / np.sqrt(5.0 * 4.75))


def test_pearson_uncorrelated_symmetric_example_is_zero() -> None:
    sample, _ = compile_both(np.array([[-2.0, -1.0, 1.0, 2.0], [1.0, 0.0, 0.0, 1.0]]))
    assert pearson(sample).matrix[0, 1] == pytest.approx(0.0, abs=1e-15)


def test_pearson_agrees_with_numpy_corrcoef() -> None:
    matrix = continuous_sample(2, 8, 200)
    sample, _ = compile_both(matrix)
    assert pearson(sample).matrix == pytest.approx(np.corrcoef(matrix), abs=1e-12)


def test_pearson_rejects_a_zero_variance_variable() -> None:
    sample, _ = compile_both(np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]))
    with pytest.raises(ValueError, match=r"Pearson correlation is undefined"):
        pearson(sample)


# ------------------------------------------------------------------ Spearman


def test_spearman_equals_pearson_of_average_ranks_exactly() -> None:
    matrix = continuous_sample(0, 5, 120)
    sample, ranks = compile_both(matrix)
    rank_sample, _ = compile_both(np.ascontiguousarray(ranks.ranks))
    assert np.array_equal(spearman(sample, ranks).matrix, pearson(rank_sample).matrix)


def test_spearman_agrees_with_scipy() -> None:
    matrix = continuous_sample(1, 6, 200)
    sample, ranks = compile_both(matrix)
    assert spearman(sample, ranks).matrix == pytest.approx(
        spearmanr(matrix.T).statistic, abs=1e-12
    )


def test_spearman_agrees_with_scipy_on_tie_heavy_data() -> None:
    matrix = zero_inflated_sample(2, 5, 150)
    sample, ranks = compile_both(matrix)
    assert spearman(sample, ranks).matrix == pytest.approx(
        spearmanr(matrix.T).statistic, abs=1e-12
    )


def test_spearman_average_rank_semantics_under_ties() -> None:
    matrix = np.array([[1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    sample, ranks = compile_both(matrix)
    expected = np.corrcoef(rankdata(matrix, method="average", axis=1))
    assert spearman(sample, ranks).matrix == pytest.approx(expected, abs=1e-12)


def test_spearman_monotone_increasing_relation_is_exactly_one() -> None:
    sample, ranks = compile_both(
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 8.0, 27.0, 64.0, 125.0]])
    )
    assert spearman(sample, ranks).matrix[0, 1] == 1.0


def test_spearman_rejects_a_degenerate_rank_vector() -> None:
    sample, ranks = compile_both(
        np.array([[2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
    )
    with pytest.raises(ValueError, match=r"Spearman rank correlation is undefined"):
        spearman(sample, ranks)


def test_spearman_accepts_a_variable_whose_raw_dispersion_underflows() -> None:
    """
    The gate is on the ranks, the representation Spearman actually consumes.

    The raw centred sum of squares of this variable rounds to exactly zero under
    float64 underflow, but its rank vector is perfectly non-degenerate.
    """
    matrix = np.array([[0.0, 1e-320, 2e-320], [1.0, 2.0, 3.0]])
    sample, ranks = compile_both(matrix)

    assert sample.sum_of_squares[0] == 0.0, "raw dispersion must underflow here"
    with pytest.raises(ValueError, match="Pearson correlation is undefined"):
        pearson(sample)

    # Spearman gates on the ranks and therefore succeeds. Off-diagonal
    # maximality holds to within rounding; only the diagonal is exactly 1.0.
    assert spearman(sample, ranks).matrix[0, 1] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------- Kendall


def test_kendall_hand_counted_concordance() -> None:
    """
    x = [1, 2, 3, 4], y = [1, 3, 2, 4]. Six pairs, no ties.
    Concordant: (1,2) (1,3) (1,4) (2,4) (3,4) = 5. Discordant: (2,3) = 1.
    tau = (5 - 1) / 6 = 2/3.
    """
    sample, ranks = compile_both(
        np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]])
    )
    assert kendall_tau_b(sample, ranks).matrix[0, 1] == pytest.approx(2.0 / 3.0)


def test_kendall_hand_counted_tie_corrected_example() -> None:
    """
    x = [1, 1, 2, 3], y = [1, 2, 2, 3]. n0 = 6, n1 = 1, n2 = 1, n3 = 0, D = 0.
    tau_b = (6 - 1 - 1 + 0 - 0) / sqrt(5 * 5) = 0.8
    """
    sample, ranks = compile_both(
        np.array([[1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]])
    )
    assert kendall_tau_b(sample, ranks).matrix[0, 1] == pytest.approx(0.8)


def test_kendall_all_concordant_is_exactly_one() -> None:
    sample, ranks = compile_both(
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    )
    assert kendall_tau_b(sample, ranks).matrix[0, 1] == 1.0


def test_kendall_all_discordant_is_exactly_minus_one() -> None:
    sample, ranks = compile_both(
        np.array([[1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]])
    )
    assert kendall_tau_b(sample, ranks).matrix[0, 1] == -1.0


@pytest.mark.parametrize("builder", [continuous_sample, zero_inflated_sample])
def test_kendall_agrees_with_scipy(
    builder: Callable[[int, int, int], Float64Matrix],
) -> None:
    matrix = builder(3, 6, 200)
    sample, ranks = compile_both(matrix)
    assert kendall_tau_b(sample, ranks).matrix == pytest.approx(
        _scipy_kendall(matrix), abs=1e-14
    )


def test_kendall_agrees_with_scipy_on_integer_data() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.integers(-2, 3, (6, 150)).astype(np.float64)
    sample, ranks = compile_both(matrix)
    assert kendall_tau_b(sample, ranks).matrix == pytest.approx(
        _scipy_kendall(matrix), abs=1e-14
    )


def test_kendall_two_observation_edge_case() -> None:
    sample, ranks = compile_both(np.array([[1.0, 2.0], [5.0, 3.0]]))
    assert kendall_tau_b(sample, ranks).matrix[0, 1] == -1.0


def test_kendall_is_smaller_in_magnitude_than_spearman() -> None:
    """Documented: the two are not interchangeable."""
    matrix = continuous_sample(6, 4, 500)
    sample, ranks = compile_both(matrix)
    tau = kendall_tau_b(sample, ranks).matrix
    rho = spearman(sample, ranks).matrix
    off = ~np.eye(4, dtype=bool)
    assert np.all(np.abs(tau[off]) < np.abs(rho[off]))


def test_kendall_rejects_a_constant_variable() -> None:
    sample, ranks = compile_both(np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match=r"Kendall tau-b is undefined"):
        kendall_tau_b(sample, ranks)
