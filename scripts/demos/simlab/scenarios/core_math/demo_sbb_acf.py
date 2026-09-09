# scripts/demos/simlab/scenarios/core_math/demo_sbb_acf.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


def create_ar1_process(
    n_days: int = 1000, phi: float = 0.8, seed: int = 42
) -> np.ndarray:
    """Generates an AR(1) process with strong autocorrelation."""
    rng = np.random.default_rng(seed)
    returns = np.empty(n_days)
    returns[0] = rng.standard_normal() * 0.01

    # AR(1) generation: r_t = phi * r_{t-1} + noise
    for t in range(1, n_days):
        returns[t] = phi * returns[t - 1] + rng.standard_normal() * 0.01

    return returns


def compute_acf(series: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Computes the autocorrelation function up to max_lag."""
    acf = np.ones(max_lag + 1)
    mean = np.mean(series)
    var = np.var(series)

    if var == 0:
        return np.zeros(max_lag + 1)

    for lag in range(1, max_lag + 1):
        cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        acf[lag] = cov / var
    return acf


def plot_acf_preservation(
    n_days: int = 1000,
    horizon: int = 500,
    total_scenarios: int = 100,
    max_lag: int = 15,
    seed: int = 101,
) -> None:
    # 1. Generate historical AR(1) returns
    hist_returns = create_ar1_process(n_days=n_days, phi=0.85, seed=seed)
    hist_acf = compute_acf(hist_returns, max_lag=max_lag)
    t_max = len(hist_returns)

    # Test different block sizes
    # 1.0 = standard bootstrap (destroys ACF)
    # 5.0 = weak preservation
    # 25.0 = strong preservation
    block_sizes = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 25.0]
    colors = [
        "tab:red",
        "tab:orange",
        "yellow",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:brown",
    ]

    rng = np.random.default_rng(seed)

    plt.figure(figsize=(10, 6))

    # Plot historical ACF
    lags = np.arange(max_lag + 1)
    plt.plot(lags, hist_acf, "k-o", lw=2, label="Historical AR(1) Data")

    # Generate SBB paths and plot their average ACF
    for b_size, color in zip(block_sizes, colors, strict=True):
        path_indices = generate_sbb_path_indices(
            t_max=t_max,
            h=horizon,
            total_scenarios=total_scenarios,
            expected_block_size=b_size,
            rng=rng,
        )

        # Map indices to returns
        bootstrapped_returns = hist_returns[path_indices - 1]

        # Compute ACF for each scenario, then average them
        scenario_acfs = np.array(
            [
                compute_acf(bootstrapped_returns[b], max_lag=max_lag)
                for b in range(total_scenarios)
            ]
        )
        avg_simulated_acf = np.mean(scenario_acfs, axis=0)

        plt.plot(
            lags,
            avg_simulated_acf,
            "--s",
            color=color,
            alpha=0.8,
            label=f"SBB (Expected Block Size = {b_size})",
        )

    plt.axhline(0, color="grey", linestyle="-", alpha=0.3)
    plt.title(
        f"SBB Autocorrelation Preservation\n(Averaged over {total_scenarios} scenarios of length $h={horizon}$)"
    )
    plt.xlabel("Lag (days)")
    plt.ylabel("Autocorrelation")
    plt.xticks(lags)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_acf_preservation()
