# tests/market_state/test_indices.py

"""Market-state indices: definitions, scale handling, no silent NaN skipping."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from riskquant.market_state.indices import (
    robust_rms_z_index,
    robust_scale_mad,
    volatility_normalised_directional_index,
    volatility_normalised_dispersion_index,
    volatility_normalised_shock_index,
)
from riskquant.observations.arrays import Float64Matrix, Float64Vector

ALL_INDICES = (
    robust_rms_z_index,
    volatility_normalised_shock_index,
    volatility_normalised_directional_index,
    volatility_normalised_dispersion_index,
)


def _sample() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.ascontiguousarray(rng.standard_normal((5, 200)))


def test_mad_scale_matches_its_definition() -> None:
    series = np.array([[1.0, 2.0, 3.0, 4.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    # median = 3, absolute deviations = [2, 1, 0, 1, 97], median = 1
    assert robust_scale_mad(series)[0] == pytest.approx(1.4826)


def test_mad_scale_resists_an_outlier_that_moves_the_standard_deviation() -> None:
    series = np.array([[1.0, 2.0, 3.0, 4.0, 1000.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    assert robust_scale_mad(series)[0] < np.std(series[0], ddof=1) / 100.0


@pytest.mark.parametrize("index", ALL_INDICES)
def test_output_shape_dtype_and_layout(index: Callable[[Float64Matrix], Float64Vector]) -> None:
    result = index(_sample())
    assert result.shape == (200,)
    assert result.dtype == np.float64
    assert result.flags.c_contiguous


@pytest.mark.parametrize("index", ALL_INDICES)
def test_indices_are_non_negative_and_finite(index: Callable[[Float64Matrix], Float64Vector]) -> None:
    result = index(_sample())
    assert (result >= 0.0).all()
    assert np.isfinite(result).all()


@pytest.mark.parametrize("index", ALL_INDICES)
def test_indices_are_invariant_to_per_variable_rescaling(index: Callable[[Float64Matrix], Float64Vector]) -> None:
    matrix = _sample()
    rescaled = matrix * np.array([1.0, 1e4, 1e-4, 3.0, 0.2])[:, None]
    assert index(matrix) == pytest.approx(index(rescaled), abs=1e-9)


@pytest.mark.parametrize("index", ALL_INDICES)
def test_zero_dispersion_variable_is_rejected(index: Callable[[Float64Matrix], Float64Vector]) -> None:
    matrix = np.vstack([np.ones(50), np.arange(50.0)])
    with pytest.raises(ValueError, match="is zero for variable_index"):
        index(matrix)


def test_shock_index_hand_calculated() -> None:
    """Two variables with unit scale; index = sqrt(mean of squared z)."""
    matrix = np.array([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]])
    sigma = np.std(matrix, axis=1, ddof=1)
    expected = np.sqrt(((matrix / sigma[:, None]) ** 2).mean(axis=0))
    assert volatility_normalised_shock_index(matrix) == pytest.approx(expected)


def test_directional_index_is_zero_when_the_cross_section_offsets() -> None:
    matrix = np.array([[1.0, -1.0, 2.0, -2.0], [-1.0, 1.0, -2.0, 2.0]])
    assert volatility_normalised_directional_index(matrix) == pytest.approx(
        np.zeros(4), abs=1e-14
    )


def test_dispersion_index_is_zero_when_the_cross_section_moves_together() -> None:
    base = np.array([1.0, -1.0, 2.0, -2.0])
    matrix = np.vstack([base, base])
    assert volatility_normalised_dispersion_index(matrix) == pytest.approx(
        np.zeros(4), abs=1e-14
    )


def test_directional_and_dispersion_separate_two_market_regimes() -> None:
    """A common move scores on direction; an offsetting move scores on dispersion."""
    together = np.vstack([np.array([3.0, -1.0, 1.0, -3.0])] * 2)
    apart = np.vstack([np.array([3.0, -1.0, 1.0, -3.0]), np.array([-3.0, 1.0, -1.0, 3.0])])

    assert volatility_normalised_directional_index(together).max() > 1.0
    assert volatility_normalised_dispersion_index(together).max() < 1e-12
    assert volatility_normalised_directional_index(apart).max() < 1e-12
    assert volatility_normalised_dispersion_index(apart).max() > 1.0


def test_ddof_choice_changes_the_scale_but_not_the_shape() -> None:
    matrix = _sample()
    population = volatility_normalised_shock_index(matrix, ddof=0)
    sample = volatility_normalised_shock_index(matrix, ddof=1)
    assert not np.allclose(population, sample)
    assert np.corrcoef(population, sample)[0, 1] == pytest.approx(1.0, abs=1e-12)


def test_robust_and_non_robust_indices_disagree_on_a_fat_tailed_sample() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_t(df=3, size=(4, 400))
    assert not np.allclose(
        robust_rms_z_index(matrix), volatility_normalised_shock_index(matrix)
    )


@pytest.mark.parametrize("index", ALL_INDICES)
def test_input_is_not_mutated(index: Callable[[Float64Matrix], Float64Vector]) -> None:
    matrix = _sample()
    before = matrix.copy()
    index(matrix)
    assert np.array_equal(matrix, before)
