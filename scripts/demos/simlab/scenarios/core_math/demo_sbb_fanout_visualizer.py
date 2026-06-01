# scripts/demos/simlab/scenarios/core_math/demo_sbb_fanout_visualizer.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


def inject_stale_periods(
    returns: np.ndarray,
    n_stale: int,
    n_stale_len: int,
    rng: np.random.Generator,
) -> None:
    """
    Injects non-overlapping contiguous periods of zero returns in-place.
    """
    n = len(returns)
    if n_stale * n_stale_len > n:
        raise ValueError("Requested stale periods exceed total array length.")

    # Maintain a list of valid starting indices that can fit the block length
    valid_starts = list(range(n - n_stale_len + 1))

    for _ in range(n_stale):
        if not valid_starts:
            raise ValueError(
                "Array is too crowded to fit another non-overlapping stale block."
            )

        # Pick a random valid starting index
        start_idx = rng.choice(valid_starts)

        # Zero out the return block
        returns[start_idx : start_idx + n_stale_len] = 0.0

        # Prune the candidate pool to prevent overlapping blocks.
        # If we chose start_idx, any new start index 's' must be far enough
        # away so the blocks don't bleed into each other.
        min_invalid = start_idx - n_stale_len + 1
        max_invalid = start_idx + n_stale_len - 1

        valid_starts = [x for x in valid_starts if x < min_invalid or x > max_invalid]


def create_complex_timeseries(
    seed: int = 101,
    n_days: int = 330,  # Added parameter (defaults to original length)
    n_stale: int = 5,
    n_stale_len: int = 22,
) -> np.ndarray:
    """Generates log-returns featuring regime changes, contiguous stale periods, and jumps."""
    rng = np.random.default_rng(seed)

    # Proportional split: ~33% / 33% / 34%
    len1 = int(n_days * 0.33)
    len2 = int(n_days * 0.33)
    len3 = n_days - len1 - len2  # Absorbs any rounding differences

    # Safety checks: ensure the regimes are long enough to fit the anomalies
    if len1 < n_stale * n_stale_len:
        raise ValueError(
            f"n_days={n_days} makes Regime 1 ({len1} days) too short to "
            f"fit {n_stale} stale periods of {n_stale_len} days."
        )
    if len2 < 4:
        raise ValueError(
            f"n_days={n_days} makes Regime 2 ({len2} days) too short to fit 4 jumps."
        )

    # Regime 1: Low Volatility with CONTIGUOUS Stale periods (Illiquid phase)
    ret1 = rng.standard_normal(len1) * 0.005

    # Regime 2: High Volatility with Jumps (Crisis/Earnings phase)
    ret2 = rng.standard_normal(len2) * 0.030
    jump_indices = rng.choice(len2, size=4, replace=False)
    ret2[jump_indices] = rng.choice([-0.08, 0.07, -0.12, 0.09], size=4)

    # Regime 3: Medium Volatility with Drift (Recovery phase)
    ret3 = rng.standard_normal(len3) * 0.01 + 0.001

    log_returns = np.concatenate([ret1, ret2, ret3])

    inject_stale_periods(log_returns, n_stale=n_stale, n_stale_len=n_stale_len, rng=rng)

    # Construct level prices starting at 100.0
    levels = 100.0 * np.exp(np.cumsum(log_returns))
    return np.concatenate(([100.0], levels))


def plot_multipath_fanout(
    n_days: int = 306,  # Shortened for demo purposes
    n_stale: int = 3,
    n_stale_len: int = 20,
    horizon: int = 50,
    total_scenarios: int = 20,
    expected_block_size: float = 20.0,
    seed: int = 123_234,
) -> None:
    levels = create_complex_timeseries(
        n_days=n_days,
        n_stale=n_stale,
        n_stale_len=n_stale_len,
        seed=seed,
    )
    t_max = len(levels)

    # Extract historical shifts
    shifts = np.log(levels[1:] / levels[:-1])

    rng = np.random.default_rng(seed)

    # Generate the index matrix (B, h)
    path_indices = generate_sbb_path_indices(
        t_max=t_max,
        h=horizon,
        total_scenarios=total_scenarios,
        expected_block_size=expected_block_size,
        rng=rng,
    )

    # Map indices to actual historical returns
    bootstrapped_shifts = shifts[path_indices - 1]

    # Vectorized path reconstruction
    cumulative_shifts = np.cumsum(bootstrapped_shifts, axis=1)

    terminal_level = levels[-1]
    simulated_levels = terminal_level * np.exp(cumulative_shifts)

    # ------------------------------------------------------------------
    # Calculate dynamic breakpoints for plotting
    # ------------------------------------------------------------------
    len1 = int(n_days * 0.33)
    len2 = int(n_days * 0.33)
    break1 = len1
    break2 = len1 + len2

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plt.figure(figsize=(14, 7))

    # Plot historical data
    hist_x = np.arange(t_max)
    plt.plot(hist_x, levels, color="k", lw=2, label="Historical Series")

    # Annotate structural breaks dynamically
    plt.axvline(break1, color="grey", linestyle="--", alpha=0.5)
    plt.text(break1 / 2, np.max(levels), "Stale/Low Vol", ha="center", alpha=0.7)

    plt.axvline(break2, color="grey", linestyle="--", alpha=0.5)
    plt.text(
        break1 + (len2 / 2), np.max(levels), "High Vol + Jumps", ha="center", alpha=0.7
    )

    plt.text(
        break2 + ((n_days - break2) / 2),
        np.max(levels),
        "Medium Vol",
        ha="center",
        alpha=0.7,
    )

    # Plot simulated fan-out
    sim_x = np.arange(t_max - 1, t_max - 1 + horizon)

    plt.plot(
        sim_x,
        simulated_levels[0],
        color="tab:blue",
        alpha=0.15,
        lw=1.5,
        label=f"{total_scenarios} SBB Simulated Paths (Block Size $\\approx$ {expected_block_size})",
    )

    for b in range(1, total_scenarios):
        plt.plot(sim_x, simulated_levels[b], color="tab:blue", alpha=0.15, lw=1.5)

    plt.plot(t_max - 1, terminal_level, marker="o", color="red", markersize=6, zorder=5)

    plt.title(f"SBB Fan-Out Simulation (h={horizon} days)")
    plt.xlabel("Time (days)")
    plt.ylabel("Asset Level")
    plt.grid(True, alpha=0.33)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_multipath_fanout()
