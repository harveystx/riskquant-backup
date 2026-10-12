# scripts/demos/simlab/scenarios/core_math/demo_ar1_filter_visualizer.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals


def compute_acf(series: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Computes empirical autocorrelation up to max_lag."""
    acf = np.ones(max_lag + 1)
    mean = np.mean(series)
    var = np.var(series)

    if var == 0:
        return np.zeros(max_lag + 1)

    for lag in range(1, max_lag + 1):
        cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        acf[lag] = cov / var
    return acf


def demo_ar1_filter(n_days: int = 500, phi: float = 0.85, seed: int = 101) -> None:
    # 1. Generate highly autocorrelated levels
    rng = np.random.default_rng(seed)
    returns = np.empty(n_days)
    returns[0] = rng.standard_normal() * 0.01

    for t in range(1, n_days):
        returns[t] = phi * returns[t - 1] + rng.standard_normal() * 0.01

    levels = 100.0 * np.exp(np.cumsum(returns))
    levels = np.concatenate(([100.0], levels))

    # 2. Filter via Numba Kernel
    # Numba types (N, 1) arrays as C-contiguous by default because the stride is trivial.
    # To force strict F-contiguous type inference, we create a (N, 2) array in Fortran order
    # and simply tell the kernel to only process column 0.
    values_f = np.empty((len(levels), 2), order="F")
    values_f[:, 0] = levels

    cols = np.array([0], dtype=np.int32)
    shift_types = np.array([SHIFT_LOG_RETURN_ID], dtype=np.int8)

    out_residuals_f = np.empty_like(values_f, order="F")

    kernel_ar1_residuals(values_f, cols, shift_types, out_residuals_f)

    kernel_ar1_residuals(values_f, cols, shift_types, out_residuals_f)

    # Extract data for plotting
    original_returns = returns
    filtered_residuals = out_residuals_f[1:, 0]  # Drop t=0 which is always 0.0

    # 3. Calculate ACFs
    max_lag = 15
    acf_orig = compute_acf(original_returns, max_lag)
    acf_filt = compute_acf(filtered_residuals, max_lag)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top Left: Original Returns
    axes[0, 0].plot(original_returns, color="tab:red", lw=1)
    axes[0, 0].set_title(f"Original Returns (AR(1) with $\\phi={phi}$)")
    axes[0, 0].grid(True, alpha=0.3)

    # Top Right: Filtered Residuals
    axes[0, 1].plot(filtered_residuals, color="tab:blue", lw=1)
    axes[0, 1].set_title("Filtered Residuals (Numba Kernel Output)")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom Left: ACF of Original
    lags = np.arange(max_lag + 1)
    axes[1, 0].bar(lags, acf_orig, color="tab:red", alpha=0.7)
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].axhline(1.96 / np.sqrt(n_days), color="grey", ls="--")
    axes[1, 0].axhline(-1.96 / np.sqrt(n_days), color="grey", ls="--")
    axes[1, 0].set_title("Autocorrelation of Original Returns")
    axes[1, 0].set_ylim(-0.2, 1.0)

    # Bottom Right: ACF of Residuals
    axes[1, 1].bar(lags, acf_filt, color="tab:blue", alpha=0.7)
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].axhline(1.96 / np.sqrt(n_days), color="grey", ls="--")
    axes[1, 1].axhline(-1.96 / np.sqrt(n_days), color="grey", ls="--")
    axes[1, 1].set_title("Autocorrelation of Residuals (Should look like White Noise)")
    axes[1, 1].set_ylim(-0.2, 1.0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_ar1_filter()
