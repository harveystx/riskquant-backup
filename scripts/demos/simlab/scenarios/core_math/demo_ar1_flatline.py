# scripts/demos/simlab/scenarios/core_math/demo_ar1_flatline.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals


def demo_ar1_zero_variance() -> None:
    n_days = 250

    # Create two edge-case time series
    # 1. A perfectly flat asset (e.g., a suspended stock)
    flat_levels = np.full(n_days, 100.0)

    # 2. A micro-variance asset (moves, but extremely tightly, testing the 1e-12 floor)
    # e.g., a strictly pegged stablecoin
    rng = np.random.default_rng(42)
    micro_returns = rng.standard_normal(n_days) * 1e-8
    micro_levels = 100.0 * np.exp(np.cumsum(micro_returns))

    # Combine into an F-contiguous matrix
    levels = np.column_stack([flat_levels, micro_levels])
    values_f = np.asfortranarray(levels)

    cols = np.array([0, 1], dtype=np.int32)
    shift_types = np.array([SHIFT_LOG_RETURN_ID, SHIFT_LOG_RETURN_ID], dtype=np.int8)
    out_residuals_f = np.empty_like(values_f, order="F")

    # If the _VAR_X_FLOOR check fails, this will throw a RuntimeWarning: divide by zero
    # or populate out_residuals with NaNs.
    kernel_ar1_residuals(values_f, cols, shift_types, out_residuals_f)

    flat_res = out_residuals_f[:, 0]
    micro_res = out_residuals_f[:, 1]

    print("--- Flatline Audit Results ---")
    print(
        f"Flat Asset Max Residual:  {np.max(np.abs(flat_res)):.2e} (Expected: 0.00e+00)"
    )
    print(f"Flat Asset NaN Count:     {np.isnan(flat_res).sum()} (Expected: 0)")
    print(f"Micro Asset Max Residual: {np.max(np.abs(micro_res)):.2e}")
    print(f"Micro Asset NaN Count:    {np.isnan(micro_res).sum()} (Expected: 0)")

    # Visual Proof
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(flat_levels, label="Levels", color="k")
    axes[0].plot(flat_res, label="Residuals", color="tab:red", linestyle="--")
    axes[0].set_title("Perfectly Stale Asset (Zero Variance)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(micro_res, label="Residuals", color="tab:blue")
    axes[1].set_title("Micro-Variance Asset (Testing Floor Logic)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("AR(1) Filter Numerical Safety Validator", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_ar1_zero_variance()
