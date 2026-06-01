# scripts/demos/simlab/scenarios/core_math/demo_ar1_illiquid.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals


def create_illiquid_series(
    n_days: int = 1000, updates_per_year: int = 35, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a hidden continuous random walk and an observed "step function"
    price that only updates sporadically, snapping to the hidden true value.
    """
    rng = np.random.default_rng(seed)

    # 1. The hidden "true" economic price accumulating information
    true_returns = rng.standard_normal(n_days) * 0.012 + 0.0005
    true_log_price = np.cumsum(true_returns)

    # 2. The observed price (updates ~35 times per 252-day year)
    prob_update = updates_per_year / 252.0
    is_observed = rng.random(n_days) < prob_update
    is_observed[0] = True  # Always observe the first day

    obs_log_price = np.zeros(n_days)
    last_known_price = true_log_price[0]

    for t in range(n_days):
        if is_observed[t]:
            last_known_price = true_log_price[t]
        obs_log_price[t] = last_known_price

    true_levels = 100.0 * np.exp(true_log_price)
    obs_levels = 100.0 * np.exp(obs_log_price)

    return true_levels, obs_levels


def demo_ar1_illiquid_jumps() -> None:
    n_days = 1260  # Exactly 5 trading years
    true_levels, obs_levels = create_illiquid_series(
        n_days=n_days, updates_per_year=35, seed=105
    )

    # Calculate observed returns for the console printout
    obs_returns = np.log(obs_levels[1:] / obs_levels[:-1])
    non_zero_days = np.sum(np.abs(obs_returns) > 1e-8)

    print("--- Illiquid Asset Audit ---")
    print(f"Total Trading Days:   {n_days}")
    print(
        f"Non-Zero Price Moves: {non_zero_days} (~{non_zero_days / (n_days / 252):.1f} per year)"
    )
    print("Expected Behavior:    The AR(1) t-test should fail, rejecting beta.")
    print(
        "                      The filter MUST NOT smear jumps into the adjacent stale days.\n"
    )

    # ------------------------------------------------------------------
    # Run the Numba Kernel
    # ------------------------------------------------------------------
    # Force strict F-contiguous array to satisfy the kernel signature
    values_f = np.empty((len(obs_levels), 2), order="F")
    values_f[:, 0] = obs_levels

    cols = np.array([0], dtype=np.int32)
    shift_types = np.array([SHIFT_LOG_RETURN_ID], dtype=np.int8)
    out_residuals_f = np.empty_like(values_f, order="F")

    kernel_ar1_residuals(values_f, cols, shift_types, out_residuals_f)

    filtered_residuals = out_residuals_f[1:, 0]  # Drop t=0

    # ------------------------------------------------------------------
    # Check for "Phantom Smearing"
    # ------------------------------------------------------------------
    # If the kernel erroneously applied a beta, the days immediately following
    # a jump would have a non-zero residual. We verify that stale days remain stale.
    stale_mask = np.abs(obs_returns) < 1e-8
    smear_errors = np.sum(np.abs(filtered_residuals[stale_mask]) > 1e-6)

    print(f"Phantom Smear Errors: {smear_errors} (Expected: 0)")
    if smear_errors == 0:
        print(
            "Conclusion:           SUCCESS. The kernel safely bypassed the stale jumps."
        )
    else:
        print(
            "Conclusion:           FAIL. The kernel injected fake returns on stale days."
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    # Top Plot: The Price Step Function
    axes[0].plot(
        true_levels, color="lightgrey", lw=1.5, label="Hidden True Economic Value"
    )
    axes[0].plot(obs_levels, color="tab:red", lw=2, label="Observed Illiquid Price")
    axes[0].set_title("Illiquid Asset Profile (Information Buildup -> Sudden Pricing)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Middle Plot: Original Observed Returns
    axes[1].plot(obs_returns, color="tab:red", lw=1)
    axes[1].set_title("Original Observed Returns (Sparse Jumps)")
    axes[1].grid(True, alpha=0.3)

    # Bottom Plot: Filtered Residuals
    axes[2].plot(filtered_residuals, color="tab:blue", lw=1)
    axes[2].set_title(
        "Kernel Residuals (Notice: No smearing, jumps are perfectly preserved)"
    )
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_ar1_illiquid_jumps()
