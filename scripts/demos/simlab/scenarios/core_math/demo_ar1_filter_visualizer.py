# scripts/demos/simlab/scenarios/core_math/audit_ar1_filter.py
from __future__ import annotations

import numpy as np
from scipy.stats import linregress

# Assuming these are available in your environment; mocked here for clarity if needed
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals


def generate_test_data(n_days: int = 500, seed: int = 42) -> np.ndarray:
    """
    Generates a (T, 3) matrix of asset levels.
    Col 0: Strong positive autocorrelation (Beta should be kept)
    Col 1: Pure white noise (Beta should fail t-test and drop to 0)
    Col 2: Strong negative autocorrelation (Beta should be kept)
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros((n_days, 3))

    returns[0, :] = rng.standard_normal(3) * 0.01

    for t in range(1, n_days):
        noise = rng.standard_normal(3) * 0.01
        returns[t, 0] = 0.8 * returns[t - 1, 0] + noise[0]  # Strong AR(1)
        returns[t, 1] = noise[1]  # White noise
        returns[t, 2] = -0.6 * returns[t - 1, 2] + noise[2]  # Negative AR(1)

    # Convert returns to levels (since the kernel expects values and calculates returns)
    levels = 100.0 * np.exp(np.cumsum(returns, axis=0))
    # Prepend an initial level so we have n_days of returns
    initial = np.array([[100.0, 100.0, 100.0]])
    return np.vstack([initial, levels])


def audit_ar1_kernel() -> None:
    levels = generate_test_data()
    t_max, k_total = levels.shape

    # ------------------------------------------------------------------
    # 1. Run the Numba Kernel
    # ------------------------------------------------------------------
    # The kernel expects F-contiguous arrays for 2D inputs
    values_f = np.asfortranarray(levels)
    cols = np.arange(k_total, dtype=np.int32)
    shift_types = np.full(k_total, SHIFT_LOG_RETURN_ID, dtype=np.int8)
    out_residuals_f = np.asfortranarray(np.empty_like(values_f))

    kernel_ar1_residuals(values_f, cols, shift_types, out_residuals_f)

    # ------------------------------------------------------------------
    # 2. Run the SciPy Reference Implementation
    # ------------------------------------------------------------------
    print(
        f"{'Col':<5} | {'Type':<12} | {'SciPy Beta':<12} | {'SciPy t-stat':<12} | {'Max Abs Diff (Numba vs SciPy)'}"
    )
    print("-" * 75)

    T_STAT_CRIT = 1.96

    for k in range(k_total):
        # Calculate log returns manually
        col_levels = levels[:, k]
        returns = np.log(col_levels[1:] / col_levels[:-1])

        # OLS on r_t ~ r_{t-1}
        x = returns[:-1]
        y = returns[1:]

        # SciPy regression
        res = linregress(x, y)
        beta = res.slope
        alpha = res.intercept
        t_stat = abs(beta / res.stderr) if res.stderr > 0 else 0.0

        # Replicate the branching logic of the kernel
        if t_stat < T_STAT_CRIT:
            beta = 0.0
            # If beta is rejected, alpha becomes the simple mean of y
            alpha = np.mean(y)
            label = "White Noise"
        else:
            label = "Strong AR(1)"

        # Calculate expected residuals
        expected_residuals = np.zeros(t_max)
        expected_residuals[1] = returns[0] - alpha
        for t in range(2, t_max):
            expected_residuals[t] = returns[t - 1] - (alpha + beta * returns[t - 2])

        # Compare
        numba_residuals = out_residuals_f[:, k]
        max_diff = np.max(np.abs(expected_residuals - numba_residuals))

        print(
            f"{k:<5} | {label:<12} | {res.slope:>10.5f} | {t_stat:>10.3f} | {max_diff:>10.2e}"
        )

        # Strict assertion to ensure mathematical equivalence
        np.testing.assert_allclose(
            expected_residuals,
            numba_residuals,
            atol=1e-10,
            err_msg=f"Mismatch on column {k}",
        )


if __name__ == "__main__":
    audit_ar1_kernel()
