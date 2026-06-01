# tests/simlab/scenarios/core_math/test_ar1_filter.py
import numpy as np
import pytest

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def _run_kernel(values, cols, shift_types):
    T = values.shape[0]
    K_req = len(cols)
    out = np.zeros((T, K_req), dtype=np.float64, order="F")

    kernel_ar1_residuals(values, cols, shift_types, out)
    return out


def _compute_returns(values, shift_type):
    T = len(values)
    r = np.zeros(T)

    for t in range(1, T):
        if shift_type == SHIFT_LOG_RETURN_ID:
            r[t] = np.log(values[t] / values[t - 1])
        elif shift_type == SHIFT_SIMPLE_RETURN_ID:
            r[t] = values[t] / values[t - 1] - 1.0
        elif shift_type == SHIFT_DIFFERENCE_ID:
            r[t] = values[t] - values[t - 1]
    return r


# ---------------------------------------------------------
# 1. Basic correctness
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "shift_type",
    [
        SHIFT_LOG_RETURN_ID,
        SHIFT_SIMPLE_RETURN_ID,
        SHIFT_DIFFERENCE_ID,
    ],
)
def test_ar1_residuals_mean_zero(shift_type):
    """
    Residuals from AR(1) should be approximately mean-zero.
    """
    np.random.seed(42)

    T = 500
    values = np.cumprod(1 + 0.01 * np.random.randn(T)).reshape(-1, 1)
    values = np.asfortranarray(values)

    out = _run_kernel(values, np.array([0]), np.array([shift_type]))

    residuals = out[2:, 0]  # exclude unstable initial points

    assert abs(np.mean(residuals)) < 1e-2


# ---------------------------------------------------------
# 2. AR(1) structure removal
# ---------------------------------------------------------


def test_ar1_residuals_remove_autocorrelation():
    """
    Residuals should have significantly reduced lag-1 autocorrelation.
    """
    np.random.seed(0)

    T = 1000
    phi = 0.8

    r = np.zeros(T)
    noise = np.random.randn(T)

    for t in range(1, T):
        r[t] = phi * r[t - 1] + noise[t]

    values = np.exp(np.cumsum(r)).reshape(-1, 1)
    values = np.asfortranarray(values)

    out = _run_kernel(values, np.array([0]), np.array([SHIFT_LOG_RETURN_ID]))
    residuals = out[:, 0]

    # autocorrelation at lag 1
    def autocorr(x):
        return np.corrcoef(x[1:], x[:-1])[0, 1]

    raw_returns = _compute_returns(values[:, 0], SHIFT_LOG_RETURN_ID)

    acf_raw = autocorr(raw_returns[2:])
    acf_res = autocorr(residuals[2:])

    assert abs(acf_res) < abs(acf_raw)
    assert abs(acf_res) < 0.2  # strong reduction


# ---------------------------------------------------------
# 3. Constant series
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "shift_type",
    [
        SHIFT_LOG_RETURN_ID,
        SHIFT_SIMPLE_RETURN_ID,
        SHIFT_DIFFERENCE_ID,
    ],
)
def test_constant_series(shift_type):
    """
    Constant input should produce zero residuals.
    """
    T = 100
    values = np.ones((T, 1), order="F")

    out = _run_kernel(values, np.array([0]), np.array([shift_type]))

    assert np.allclose(out, 0.0)


# ---------------------------------------------------------
# 4. Small T edge cases
# ---------------------------------------------------------


def test_small_T_cases():
    """
    Ensure no crashes and correct handling for small T.
    """
    for T in [1, 2]:
        values = np.ones((T, 1), order="F")
        out = _run_kernel(values, np.array([0]), np.array([SHIFT_DIFFERENCE_ID]))

        assert out.shape == (T, 1)


# ---------------------------------------------------------
# 5. Multi-column correctness
# ---------------------------------------------------------


def test_multiple_columns_independence():
    """
    Ensure columns are processed independently.
    """
    np.random.seed(123)

    T = 300
    K = 3

    values = np.cumprod(1 + 0.01 * np.random.randn(T, K), axis=0)
    values = np.asfortranarray(values)

    cols = np.array([0, 1, 2])
    shift_types = np.array(
        [
            SHIFT_LOG_RETURN_ID,
            SHIFT_SIMPLE_RETURN_ID,
            SHIFT_DIFFERENCE_ID,
        ]
    )

    out = _run_kernel(values, cols, shift_types)

    assert out.shape == (T, K)

    # Check columns differ (not accidentally shared memory / overwritten)
    assert not np.allclose(out[:, 0], out[:, 1])
    assert not np.allclose(out[:, 1], out[:, 2])


# ---------------------------------------------------------
# 6. Numerical stability (low variance)
# ---------------------------------------------------------


def test_near_zero_variance():
    """
    Test stability when returns have near-zero variance.
    """
    T = 200
    values = np.linspace(1.0, 1.000001, T).reshape(-1, 1)
    values = np.asfortranarray(values)

    out = _run_kernel(values, np.array([0]), np.array([SHIFT_LOG_RETURN_ID]))

    assert np.all(np.isfinite(out))


# ---------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------


def test_deterministic_output():
    """
    Kernel should produce deterministic results.
    """
    np.random.seed(999)

    T = 400
    values = np.cumprod(1 + 0.01 * np.random.randn(T)).reshape(-1, 1)
    values = np.asfortranarray(values)

    out1 = _run_kernel(values, np.array([0]), np.array([SHIFT_LOG_RETURN_ID]))
    out2 = _run_kernel(values, np.array([0]), np.array([SHIFT_LOG_RETURN_ID]))

    assert np.allclose(out1, out2)


# ---------------------------------------------------------
# 8. Memory layout safety
# ---------------------------------------------------------


def test_fortran_layout_preserved():
    """
    Ensure output remains Fortran contiguous (important for downstream performance).
    """
    T = 100
    values = np.ones((T, 1), order="F")

    out = _run_kernel(values, np.array([0]), np.array([SHIFT_DIFFERENCE_ID]))

    assert out.flags["F_CONTIGUOUS"]


# ---------------------------------------------------------
# 9. No NaN propagation (except pathological log cases)
# ---------------------------------------------------------


def test_no_nan_output():
    """
    Ensure no NaNs are produced under normal conditions.
    """
    np.random.seed(7)

    T = 300
    values = np.cumprod(1 + 0.01 * np.random.randn(T)).reshape(-1, 1)
    values = np.asfortranarray(values)

    out = _run_kernel(values, np.array([0]), np.array([SHIFT_LOG_RETURN_ID]))

    assert not np.isnan(out).any()


# ---------------------------------------------------------
# 10. De-meaning only (Efficient Market / White Noise)
# ---------------------------------------------------------


def test_ar1_filter_demean_only_for_white_noise():
    """
    When returns are pure white noise (no autocorrelation), the t-statistic
    for beta should fail the significance test (< 1.96). The kernel should
    set beta=0 and only de-mean the series.
    """
    np.random.seed(42)
    T = 1000
    mu_true = 0.005  # Intentional drift
    sigma_true = 0.015

    # 1. Simulate white noise returns (log returns)
    raw_returns = np.random.normal(loc=mu_true, scale=sigma_true, size=T)
    prices = np.exp(np.cumsum(raw_returns))
    values = np.asfortranarray(prices.reshape(-1, 1))

    # 2. Run kernel
    out = _run_kernel(values, np.array([0]), np.array([SHIFT_LOG_RETURN_ID]))
    residuals = out[:, 0]

    # 3. Compute raw returns exactly as the kernel sees them
    computed_returns = _compute_returns(values[:, 0], SHIFT_LOG_RETURN_ID)

    # 4. Diagnostics
    # The kernel calculates the mean from t=2 to T-1 (n_obs = T - 2)
    mean_raw = np.mean(computed_returns[2:])
    mean_res = np.mean(residuals[2:])

    # The residuals must be perfectly mean-centered
    assert abs(mean_res) < 1e-12

    # Because beta was deemed statistically insignificant, the kernel
    # should have ONLY subtracted the mean (alpha = mean_y, beta = 0).
    # Therefore, residual[t] == raw_return[t] - mean_raw
    expected_residuals = computed_returns[2:] - mean_raw

    np.testing.assert_allclose(
        residuals[2:],
        expected_residuals,
        atol=1e-10,
        err_msg="Kernel applied beta scaling instead of strictly de-meaning.",
    )
