# scripts/demos/simlab/scenarios/core_math/demo_sbb_wrap.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


def plot_circular_wrap_edge_case(
    n_days: int = 150,
    horizon: int = 250,
    expected_block_size: float = 35.0,
    seed: int = 102,
) -> None:
    # Generate a distinct historical pattern: upward drift + sine wave
    # This makes the "wrap" from the end back to the beginning extremely obvious
    t = np.arange(n_days)
    historical_returns = 0.005 + 0.02 * np.sin(t / 10.0)

    t_max = len(historical_returns)
    max_valid_length = t_max - 1

    rng = np.random.default_rng(seed)

    # Generate a single long path
    path_indices = generate_sbb_path_indices(
        t_max=t_max,
        h=horizon,
        total_scenarios=1,
        expected_block_size=expected_block_size,
        rng=rng,
    )[0]

    # Map to returns and compute a cumulative stitched path
    bootstrapped_returns = historical_returns[path_indices - 1]
    stitched_path = np.cumsum(bootstrapped_returns)

    # Identify where the index wraps from the end of history back to the start
    # A wrap occurs if the current index doesn't logically follow the previous
    wrap_points_x = []
    wrap_points_y = []

    for i in range(1, horizon):
        prev_idx = path_indices[i - 1]
        curr_idx = path_indices[i]
        expected_next = 1 + ((prev_idx - 1 + 1) % max_valid_length)

        # We specifically want to highlight the circular wrap back to the start,
        # not just standard block transitions.
        if curr_idx != expected_next and curr_idx < prev_idx:
            wrap_points_x.append(i)
            wrap_points_y.append(stitched_path[i])

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plt.figure(figsize=(14, 8))

    # Top Plot: The Historical Series
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(np.cumsum(historical_returns), color="k", lw=2)
    ax1.set_title(f"Historical Trend (Length: {n_days} days)")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.2)

    # Bottom Plot: The Stitched Path with Wraps Highlighted
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(stitched_path, color="tab:blue", lw=2, label="Stitched SBB Path")

    if wrap_points_x:
        ax2.scatter(
            wrap_points_x,
            wrap_points_y,
            color="red",
            s=100,
            zorder=5,
            label="Circular Wrap Discontinuity",
        )

        # Add vertical dashed lines for clarity
        for wx in wrap_points_x:
            ax2.axvline(wx, color="red", linestyle="--", alpha=0.3)

    ax2.set_title(
        f"Simulated Path (Horizon: {horizon} days | Expected Block: {expected_block_size})\n"
        "Red dots indicate where the block stitched the end of history to the beginning."
    )
    ax2.set_xlabel("Simulation Day")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_circular_wrap_edge_case()
