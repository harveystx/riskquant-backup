# scripts/demos/simlab/scenarios/core_math/demo_sbb_fanout_visualizer.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)
from riskquant.synthetic_level_generators import create_complex_levels_series
from riskquant.utils.printer import (
    print_calling_file,
    print_success_message,
    print_title,
)


def plot_multipath_fanout(
    num_obs: int = 400,
    n_stale: int = 2,
    n_stale_len: int = 50,
    horizon: int = 100,
    n_simulations: int = 1_000,  # total number of simulated paths
    n_plot_lines: int = 50,  # number of paths to draw in the fan-out
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

    shifts = np.log(levels[1:] / levels[:-1])
    rng = np.random.default_rng(rng_seed)

    path_indices = generate_sbb_path_indices(
        t_max=t_max,
        h=horizon,
        total_scenarios=n_simulations,
        expected_block_size=expected_block_size,
        rng=rng,
    )

    bootstrapped_shifts = shifts[path_indices - 1]
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
    # Set up two-panel figure (sharey=True ensures identical y-limits)
    # ------------------------------------------------------------------
    _, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=True,
    )

    # ==================================================================
    # LEFT PANEL: Historical data + fan-out simulation
    # ==================================================================
    hist_x = np.arange(t_max)
    ax1.plot(hist_x, levels, color="k", lw=2, label="Historical Series")

    # Structural break annotations
    ax1.axvline(break1, color="grey", linestyle="--", alpha=0.5)
    ax1.text(break1 / 2, np.max(levels), "Stale/Low Vol", ha="center", alpha=0.7)

    ax1.axvline(break2, color="grey", linestyle="--", alpha=0.5)
    ax1.text(
        break1 + (len2 / 2), np.max(levels), "High Vol + Jumps", ha="center", alpha=0.7
    )
    ax1.text(
        break2 + ((num_obs - break2) / 2),
        np.max(levels),
        "Medium Vol",
        ha="center",
        alpha=0.7,
    )

    # Simulated fan-out paths
    sim_x = np.arange(t_max - 1, t_max - 1 + horizon)

    lines_to_plot = min(n_plot_lines, n_simulations)

    # First path (with label)
    ax1.plot(
        sim_x,
        simulated_levels[0],
        color="tab:blue",
        alpha=0.15,
        lw=1.5,
        label=(
            f"{lines_to_plot} plotted paths "
            f"(out of {n_simulations} SBB simulations, "
            f"Block Size $\\approx$ {expected_block_size})"
        ),
    )
    # Remaining plotted paths
    for b in range(1, lines_to_plot):
        ax1.plot(sim_x, simulated_levels[b], color="tab:blue", alpha=0.15, lw=1.5)

    # Mark the terminal point of the historical series
    ax1.plot(t_max - 1, terminal_level, marker="o", color="red", markersize=6, zorder=5)

    ax1.set_title(f"SBB Fan-Out Simulation (h={horizon} days)")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Asset Level")
    ax1.grid(True, alpha=0.33)
    ax1.legend(loc="upper left")

    # ==================================================================
    # RIGHT PANEL: Distribution of terminal simulated values (all paths)
    # ==================================================================
    terminal_values = simulated_levels[:, -1]
    ax2.hist(
        terminal_values,
        bins=30,
        orientation="horizontal",
        color="tab:blue",
        alpha=0.6,
        edgecolor="white",
    )
    ax2.axhline(
        terminal_level, color="red", linestyle="--", lw=1.5, label="Historical Terminal"
    )
    ax2.set_title("Terminal Value Distribution")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Terminal Asset Level")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.33)

    # Hide y tick labels on the right panel (labels remain on the left)
    ax2.tick_params(axis="y", labelleft=False)

    # ------------------------------------------------------------------
    # Synchronize y-limits based ONLY on the data shown in ax1
    # (historical series + the subset of simulated paths that are drawn)
    # ------------------------------------------------------------------
    plotted_levels = np.concatenate(
        [levels, simulated_levels[:lines_to_plot].flatten()]
    )
    y_min, y_max = np.min(plotted_levels), np.max(plotted_levels)
    y_min = min(y_min, min(levels)) * 0.90
    y_max = max(y_max, max(levels)) * 1.10
    padding = 0.05 * (y_max - y_min)
    ax1.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    plt.show()


def main():
    print_calling_file()
    print_title("SBB Fan-Out Visualizer Demo")
    plot_multipath_fanout()
    print_success_message("SBB Fan-Out Visualizer Demo Completed Successfully!")


if __name__ == "__main__":
    main()
