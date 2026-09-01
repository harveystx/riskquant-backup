# tests/observations/test_preprocess.py

"""Tests for the explicitly-named, sample-altering preprocessing policy."""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.observations.orientation import ObservationOrientation
from riskquant.observations.preprocess import (
    ImputationStrategy,
    ZeroFilterMode,
    apply_zero_dead_zone,
    filter_incomplete_observations,
    filter_zero_observations,
    impute_observations,
)

ROWS = ObservationOrientation.VARIABLES_ON_ROWS


def test_complete_case_selection_returns_the_retained_index() -> None:
    matrix = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    retained, index = filter_incomplete_observations(matrix, orientation=ROWS)
    assert np.array_equal(index, [0, 2])
    assert np.array_equal(retained, [[1.0, 3.0], [4.0, 6.0]])


def test_zero_filter_any_versus_all_differ() -> None:
    matrix = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 3.0]])

    _, any_index = filter_zero_observations(
        matrix, orientation=ROWS, mode=ZeroFilterMode.DROP_IF_ANY_ZERO, zero_tolerance=0.0
    )
    _, all_index = filter_zero_observations(
        matrix, orientation=ROWS, mode=ZeroFilterMode.DROP_IF_ALL_ZERO, zero_tolerance=0.0
    )

    assert np.array_equal(any_index, [1])
    assert np.array_equal(all_index, [1, 2])


def test_zero_tolerance_of_exactly_zero_is_not_coerced() -> None:
    """The prototype turned a falsy 0.0 into 1e-12 and matched near-zeros."""
    matrix = np.array([[1e-13, 1.0], [2.0, 3.0]])
    _, index = filter_zero_observations(
        matrix, orientation=ROWS, mode=ZeroFilterMode.DROP_IF_ANY_ZERO, zero_tolerance=0.0
    )
    assert np.array_equal(index, [0, 1])


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_zero_tolerance_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        filter_zero_observations(
            np.ones((2, 3)),
            orientation=ROWS,
            mode=ZeroFilterMode.DROP_IF_ALL_ZERO,
            zero_tolerance=bad,
        )


def test_dead_zone_snaps_only_small_magnitudes() -> None:
    matrix = np.array([[1e-9, 0.5], [-1e-9, -0.5]])
    snapped = apply_zero_dead_zone(matrix, orientation=ROWS, zero_tolerance=1e-6)
    assert np.array_equal(snapped, [[0.0, 0.5], [0.0, -0.5]])


def test_constant_imputation_fills_only_missing_cells() -> None:
    matrix = np.array([[1.0, np.nan], [np.inf, 4.0]])
    filled = impute_observations(
        matrix, orientation=ROWS, strategy=ImputationStrategy.CONSTANT, constant_value=-1.0
    )
    assert np.array_equal(filled, [[1.0, -1.0], [-1.0, 4.0]])


def test_mean_imputation_uses_the_observed_marginal() -> None:
    matrix = np.array([[1.0, 3.0, np.nan], [1.0, 2.0, 3.0]])
    filled = impute_observations(
        matrix, orientation=ROWS, strategy=ImputationStrategy.VARIABLE_MEAN
    )
    assert filled[0, 2] == pytest.approx(2.0)


def test_median_imputation_uses_the_observed_marginal() -> None:
    matrix = np.array([[1.0, 2.0, 100.0, np.nan], [1.0, 2.0, 3.0, 4.0]])
    filled = impute_observations(
        matrix, orientation=ROWS, strategy=ImputationStrategy.VARIABLE_MEDIAN
    )
    assert filled[0, 3] == pytest.approx(2.0)


def test_fully_missing_variable_cannot_be_imputed_marginally() -> None:
    matrix = np.array([[np.nan, np.nan], [1.0, 2.0]])
    with pytest.raises(ValueError, match=r"fully missing"):
        impute_observations(
            matrix, orientation=ROWS, strategy=ImputationStrategy.VARIABLE_MEAN
        )


def test_knn_imputation_produces_a_finite_matrix() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((4, 40))
    matrix[1, 5] = np.nan
    filled = impute_observations(
        matrix,
        orientation=ROWS,
        strategy=ImputationStrategy.K_NEAREST_NEIGHBOURS,
        neighbour_count=3,
    )
    assert np.isfinite(filled).all()
    assert filled.shape == (4, 40)


def test_mean_imputation_shrinks_estimated_correlation() -> None:
    """Documented consequence: synthetic observations at the marginal centre."""
    from riskquant.dependence.estimators.pearson import pearson
    from riskquant.observations.compile import compile_sample

    rng = np.random.default_rng(3)
    base = rng.standard_normal(200)
    complete = np.vstack([base, 0.9 * base + 0.436 * rng.standard_normal(200)])

    holed = complete.copy()
    holed[0, :80] = np.nan
    imputed = impute_observations(
        holed, orientation=ROWS, strategy=ImputationStrategy.VARIABLE_MEAN
    )

    def correlation(matrix: np.ndarray) -> float:
        sample = compile_sample(matrix, orientation=ROWS)
        return float(pearson(sample).matrix[0, 1])

    assert abs(correlation(imputed)) < abs(correlation(complete))
