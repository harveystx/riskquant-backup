# scripts/demos/simlab/scenarios/core_math/demo_sbb_fat_tails.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


def create_fat_tailed_returns(n_days: int = 1000, seed: int = 42) -> np.ndarray:
    """Generates a long bull market punctuated by a severe, contiguous crash."""
    rng = np.random.default_rng(seed)

    # Regime 1: Long, slow bull market (800 days)
    # Low volatility, slight positive drift
    bull_days = rng.standard_normal(800) * 0.005 + 0.001

    # Regime 2: The Crash (200 days)
    # High volatility, severe negative drift
    crash_days = rng.standard_normal(200) * 0.03 - 0.01

    # Do NOT shuffle. We need the crash to be contiguous so SBB
    # can sample whole blocks of pure panic.
    returns = np.concatenate([bull_days, crash_days])

    return returns


def calculate_tail_metrics(
    terminal_levels: np.ndarray, initial_level: float, confidence: float = 1.0
) -> tuple[float, float]:
    """Calculates VaR and Expected Shortfall (ES) at the given confidence level (e.g., 1.0 for 99%)."""
    # Convert terminal levels to cumulative percentage returns
    returns = (terminal_levels / initial_level) - 1.0

    # Value at Risk (1st percentile for 99% confidence)
    var = np.percentile(returns, confidence)

    # Expected Shortfall (Average of all returns worse than VaR)
    es = np.mean(returns[returns <= var])

    return var, es


def plot_fat_tails_validation(
    horizon: int = 60,
    total_scenarios: int = 10_000,
    expected_block_size: float = 10.0,
    seed: int = 101,
) -> None:
    historical_returns = create_fat_tailed_returns(n_days=1000, seed=seed)
    t_max = len(historical_returns)

    mu_daily = np.mean(historical_returns)
    sigma_daily = np.std(historical_returns)
    initial_level = 100.0

    rng = np.random.default_rng(seed)

    # Simulate SBB
    path_indices = generate_sbb_path_indices(
        t_max=t_max,
        h=horizon,
        total_scenarios=total_scenarios,
        expected_block_size=expected_block_size,
        rng=rng,
    )
    sbb_cumulative = np.cumsum(historical_returns[path_indices - 1], axis=1)
    sbb_terminal_levels = initial_level * np.exp(sbb_cumulative[:, -1])

    # Simulate GBM
    z = rng.standard_normal(total_scenarios)
    drift = (mu_daily - 0.5 * sigma_daily**2) * horizon
    diffusion = sigma_daily * np.sqrt(horizon) * z
    gbm_terminal_levels = initial_level * np.exp(drift + diffusion)

    # Calculate tail metrics
    gbm_var, gbm_es = calculate_tail_metrics(
        gbm_terminal_levels, initial_level, confidence=1.0
    )
    sbb_var, sbb_es = calculate_tail_metrics(
        sbb_terminal_levels, initial_level, confidence=1.0
    )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    bins = np.linspace(40, 180, 100)

    # Left Plot: Linear Scale
    ax1.hist(
        gbm_terminal_levels,
        bins=bins,
        alpha=0.5,
        density=True,
        color="tab:orange",
        label="GBM (Lognormal)",
    )
    ax1.hist(
        sbb_terminal_levels,
        bins=bins,
        alpha=0.5,
        density=True,
        color="tab:blue",
        label="SBB (Empirical)",
    )
    ax1.set_title("Terminal Price Distribution (Linear Scale)")
    ax1.set_xlabel(f"Asset Level at h={horizon}")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Right Plot: Log Scale for Tails
    ax2.hist(
        gbm_terminal_levels,
        bins=bins,
        alpha=0.5,
        density=True,
        color="tab:orange",
        label="GBM",
    )
    ax2.hist(
        sbb_terminal_levels,
        bins=bins,
        alpha=0.5,
        density=True,
        color="tab:blue",
        label="SBB",
    )
    ax2.set_title("Terminal Price Distribution (Log Scale - Tail Focus)")
    ax2.set_xlabel(f"Asset Level at h={horizon}")
    ax2.set_ylabel("Log Density")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Add Quantitative Metrics Table to the Chart
    stats_text = (
        "Tail Risk Metrics (99% Confidence)\n"
        "--------------------------------------------------\n"
        f"GBM 99% VaR:  {gbm_var * 100:6.2f}%\n"
        f"GBM 99% ES:   {gbm_es * 100:6.2f}%\n\n"
        f"SBB 99% VaR:  {sbb_var * 100:6.2f}%\n"
        f"SBB 99% ES:   {sbb_es * 100:6.2f}%\n"
        "--------------------------------------------------\n"
        "Notice how GBM systematically underestimates\n"
        "Expected Shortfall by ignoring empirical jumps."
    )

    # Place text box in the first subplot
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    ax1.text(
        0.05,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )

    plt.suptitle(
        f"Validation of Empirical Tail Preservation ({total_scenarios:,} Paths)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


if __name__ == "__main__":
    plot_fat_tails_validation()
