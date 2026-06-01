# tests/simlab/scenarios/test_historical_math.py
import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.utils.printer import print_df


@pytest.fixture
def math_audit_data():
    """
    Fixture providing a fully deterministic matrix of levels built exactly
    from known 1-day returns, alongside the expected arrays.
    """
    # 1. Define the EXACT 1-day shocks we want to hide in the prices
    expected_log_1d = np.array([0.01, -0.02, 0.03, -0.01, 0.05], dtype=np.float64)
    expected_sim_1d = np.array([0.01, -0.02, 0.03, -0.01, 0.05], dtype=np.float64)
    expected_diff_1d = np.array([1.0, -2.0, 3.0, -1.0, 5.0], dtype=np.float64)

    T = 6  # 6 days of prices = 5 days of 1-day returns
    K = 3
    base_price = 100.0

    # 2. Mathematically build the price levels forward through time
    levels = np.zeros((T, K), dtype=np.float64)
    levels[0, :] = base_price

    for t in range(1, T):
        # Inverse of Log Return
        levels[t, 0] = levels[t - 1, 0] * np.exp(expected_log_1d[t - 1])
        # Inverse of Simple Return
        levels[t, 1] = levels[t - 1, 1] * (1.0 + expected_sim_1d[t - 1])
        # Inverse of Difference
        levels[t, 2] = levels[t - 1, 2] + expected_diff_1d[t - 1]

    factor_ids = np.array([101, 102, 103], dtype=np.int32)

    matrix = RiskFactorMatrix(
        factor_ids=factor_ids,
        dates=np.arange(T, dtype=np.int32),
        values=np.asfortranarray(levels),
    )

    print("Constructed Price Levels:\n")
    print_df(matrix.df)

    req_shifts = np.array(
        [SHIFT_LOG_RETURN_ID, SHIFT_SIMPLE_RETURN_ID, SHIFT_DIFFERENCE_ID],
        dtype=np.int8,
    )
    req_scales = np.ones(K, dtype=np.float32)

    return (
        matrix,
        factor_ids,
        req_shifts,
        req_scales,
        expected_log_1d,
        expected_sim_1d,
        expected_diff_1d,
    )


def test_historical_math_h1(math_audit_data):
    """
    Validates that the Numba kernel perfectly recovers the 1-day historical
    returns from the constructed price levels.
    """
    matrix, factor_ids, req_shifts, req_scales, exp_log, exp_sim, exp_diff = (
        math_audit_data
    )

    generator = HistoricalGenerator(
        matrix=matrix,
        horizon=1,
        shock_dtype=np.float64,  # Keep precision high for the assert
    )

    batch = next(
        generator.iter_batches(
            required_shift_type_id=req_shifts,
            source_factor_ids=factor_ids,
            shock_scale=req_scales,
            batch_size=10,
        )
    )

    shocks = batch.shocks

    # S_total = T - 1 = 5 scenarios
    assert shocks.shape == (5, 3)

    # Assert recovery to near-machine precision
    np.testing.assert_allclose(shocks[:, 0], exp_log, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shocks[:, 1], exp_sim, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shocks[:, 2], exp_diff, rtol=1e-12, atol=1e-12)


def test_historical_math_h2_path_aggregation(math_audit_data):
    """
    Validates that h=2 path aggregation correctly bridges the start and
    end prices of a 2-day window, exactly matching algebraic expectations.
    """
    (
        matrix,
        factor_ids,
        req_shifts,
        req_scales,
        exp_log_1d,
        exp_sim_1d,
        exp_diff_1d,
    ) = math_audit_data

    generator = HistoricalGenerator(matrix=matrix, horizon=2, shock_dtype=np.float64)

    batch = next(
        generator.iter_batches(
            required_shift_type_id=req_shifts,
            source_factor_ids=factor_ids,
            shock_scale=req_scales,
            batch_size=10,
        )
    )

    shocks = batch.shocks

    # S_total = T - 2 = 4 scenarios
    assert shocks.shape == (4, 3)

    # --- Algebraically Derive Expected 2-Day Paths ---
    # Log Returns sum additively over time: ln(V2/V0) = ln(V2/V1) + ln(V1/V0)
    exp_log_2d = exp_log_1d[1:] + exp_log_1d[:-1]

    # Simple Returns compound multiplicatively: (1+r2)(1+r1) - 1
    exp_sim_2d = (1.0 + exp_sim_1d[1:]) * (1.0 + exp_sim_1d[:-1]) - 1.0

    # Differences sum additively
    exp_diff_2d = exp_diff_1d[1:] + exp_diff_1d[:-1]

    # Assert 2-day recovery
    np.testing.assert_allclose(shocks[:, 0], exp_log_2d, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shocks[:, 1], exp_sim_2d, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shocks[:, 2], exp_diff_2d, rtol=1e-12, atol=1e-12)
