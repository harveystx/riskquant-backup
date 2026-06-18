# scripts/demos/simlab/scenarios/core_math/demo_sbb_fanout_visualizer.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.synthetic_level_generators import create_complex_levels_series
from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


def plot_multipath_fanout(
    num_obs: int = 400,  # Shortened for demo purposes
    n_stale: int = 2,
    n_stale_len: int = 50,
    horizon: int = 100,
    total_scenarios: int = 30,
    expected_block_size: float = 25.0,
    rng_seed: int = 999_999,
    len_01_proportion: float = 0.30,
    len_02_proportion: float = 0.30,
) -> None:
    levels = create_complex_levels_series(
        num_obs=num_obs,
        n_stale=n_stale,
        n_stale_len=n_stale_len,
        rng_seed=rng_seed,
        len_01_proportion=len_01_proportion,
        len_02_proportion=len_02_proportion,
    )
    t_max = len(levels)

    # Extract historical shifts
    shifts = np.log(levels[1:] / levels[:-1])

    rng = np.random.default_rng(rng_seed)

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
    len1 = int(num_obs * len_01_proportion)
    len2 = int(num_obs * len_02_proportion)
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
        break2 + ((num_obs - break2) / 2),
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
