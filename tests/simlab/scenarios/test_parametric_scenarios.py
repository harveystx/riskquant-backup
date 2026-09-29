# tests/simlab/scenarios/test_parametric_scenarios.py
from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric import ParametricGenerator

# =========================================================================
# MOCK PORTFOLIO CONSTANTS
# =========================================================================
# These define the 'Universe A' requirements that the tests will pass to iter_batches
MOCK_REQ_FACTOR_IDS = np.array([101, 202], dtype=np.int32)
MOCK_REQ_SHIFT_TYPES = np.full(
    2, SHIFT_LOG_RETURN_ID, dtype=np.int8
)  # LOG_RETURN (id 1)
MOCK_REQ_SHOCK_SCALES = np.ones(2, dtype=np.float32)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def _iter_batches(generator: ParametricGenerator, batch_size: int):
    return list(
        generator.iter_batches(
            chart_ids=MOCK_REQ_SHIFT_TYPES,
            driver_rf_ids=MOCK_REQ_FACTOR_IDS,
            batch_size=batch_size,
        )
    )


def _collect_shocks(generator: ParametricGenerator, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [
            batch.shocks.copy()
            for batch in _iter_batches(generator, batch_size=batch_size)
        ],
        axis=0,
    )


def _expected_covariance(
    volatilities: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    return correlation * np.outer(volatilities, volatilities)


# =========================================================================
# TESTS
# =========================================================================


def test_standard_normal_replays_seeded_stream() -> None:
    correlation = np.array([[1.0, 0.35], [0.35, 1.0]], dtype=np.float64)
    generator = ParametricGenerator.standard_normal(
        available_factor_ids=MOCK_REQ_FACTOR_IDS,
        cholesky_matrix=np.linalg.cholesky(correlation),
        volatilities=np.array([0.2, 0.4], dtype=np.float64),
        total_scenarios=32,
        seed=17,
    )

    first_batches = _iter_batches(generator, batch_size=7)

    # Re-instantiate exactly the same generator to test stream replay
    generator2 = ParametricGenerator.standard_normal(
        available_factor_ids=MOCK_REQ_FACTOR_IDS,
        cholesky_matrix=np.linalg.cholesky(correlation),
        volatilities=np.array([0.2, 0.4], dtype=np.float64),
        total_scenarios=32,
        seed=17,
    )
    second_batches = _iter_batches(generator2, batch_size=7)

    assert len(first_batches) == len(second_batches)
    for first_batch, second_batch in zip(first_batches, second_batches, strict=True):
        assert first_batch.shocks.flags["F_CONTIGUOUS"]
        assert first_batch.shocks.dtype == np.float32
        np.testing.assert_array_equal(first_batch.shocks, second_batch.shocks)


def test_student_t_rejects_invalid_df() -> None:
    with pytest.raises(ValueError, match="greater than 2"):
        ParametricGenerator.student_t(
            available_factor_ids=MOCK_REQ_FACTOR_IDS,
            student_t_df=2.0,
            cholesky_matrix=np.eye(2, dtype=np.float64),
            volatilities=np.ones(2, dtype=np.float64),
            total_scenarios=8,
        )


def test_iter_batches_rejects_non_positive_batch_size() -> None:
    generator = ParametricGenerator.standard_normal(
        available_factor_ids=MOCK_REQ_FACTOR_IDS,
        cholesky_matrix=np.eye(2, dtype=np.float64),
        volatilities=np.ones(2, dtype=np.float64),
        total_scenarios=8,
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        next(
            generator.iter_batches(
                chart_ids=MOCK_REQ_SHIFT_TYPES,
                driver_rf_ids=MOCK_REQ_FACTOR_IDS,
                batch_size=0,
            )
        )


def test_standard_normal_covariance_matches_calibration() -> None:
    correlation = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)
    volatilities = np.array([0.2, 0.4], dtype=np.float64)
    generator = ParametricGenerator.standard_normal(
        available_factor_ids=MOCK_REQ_FACTOR_IDS,
        cholesky_matrix=np.linalg.cholesky(correlation),
        volatilities=volatilities,
        total_scenarios=40000,
        seed=123,
    )

    shocks = _collect_shocks(generator, batch_size=4096)
    empirical_covariance = np.cov(shocks, rowvar=False, ddof=0)
    target_covariance = _expected_covariance(volatilities, correlation)

    np.testing.assert_allclose(empirical_covariance, target_covariance, atol=0.01)


def test_student_t_covariance_matches_calibration() -> None:
    correlation = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float64)
    volatilities = np.array([0.3, 0.5], dtype=np.float64)
    generator = ParametricGenerator.student_t(
        student_t_df=7.0,
        available_factor_ids=MOCK_REQ_FACTOR_IDS,
        cholesky_matrix=np.linalg.cholesky(correlation),
        volatilities=volatilities,
        total_scenarios=60000,
        seed=321,
    )

    shocks = _collect_shocks(generator, batch_size=4096)
    empirical_covariance = np.cov(shocks, rowvar=False, ddof=0)
    target_covariance = _expected_covariance(volatilities, correlation)

    np.testing.assert_allclose(empirical_covariance, target_covariance, atol=0.02)


def test_regime_switching_covariance_matches_mixture_target() -> None:
    correlation = np.array([[1.0, 0.45], [0.45, 1.0]], dtype=np.float64)
    volatilities = np.array([0.25, 0.4], dtype=np.float64)
    crisis_multiplier = np.array([1.8, 2.5], dtype=np.float64)
    prob_crisis = 0.35
    generator = ParametricGenerator.regime_switching_gmm(
        prob_crisis=prob_crisis,
        crisis_vol_multipliers=crisis_multiplier,
        available_factor_ids=MOCK_REQ_FACTOR_IDS,
        cholesky_matrix=np.linalg.cholesky(correlation),
        volatilities=volatilities,
        total_scenarios=60000,
        seed=999,
    )

    shocks = _collect_shocks(generator, batch_size=4096)
    empirical_covariance = np.cov(shocks, rowvar=False, ddof=0)
    base_covariance = _expected_covariance(volatilities, correlation)
    mixture_multiplier = (1.0 - prob_crisis) + prob_crisis * np.outer(
        crisis_multiplier,
        crisis_multiplier,
    )
    target_covariance = base_covariance * mixture_multiplier

    np.testing.assert_allclose(empirical_covariance, target_covariance, atol=0.03)
