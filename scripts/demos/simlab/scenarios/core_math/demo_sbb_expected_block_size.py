# scripts/demos/simlab/scenarios/core_math/demo_sbb_expected_block_size.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import geom

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


def identify_blocks_from_matrix(
    path_indices: np.ndarray,
    max_valid_index: int,
) -> np.ndarray:
    """Extracts all block lengths across a (B, h) matrix of SBB indices."""
    b_blocks, h = path_indices.shape
    all_lengths = []

    for b in range(b_blocks):
        block_start = 0
        for i in range(1, h):
            prev_idx = path_indices[b, i - 1]
            curr_idx = path_indices[b, i]

            # If the current index doesn't logically follow the previous one, the block ended
            expected_next = 1 + ((prev_idx - 1 + 1) % max_valid_index)
            if curr_idx != expected_next:
                all_lengths.append(i - block_start)
                block_start = i

        # Append the final block for this scenario
        all_lengths.append(h - block_start)

    return np.array(all_lengths)


def validate_block_sizes(
    expected_block_size: float = 25.0,
    total_scenarios: int = 50,
    horizon: int = 10_000,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    t_max = 500  # Arbitrary historical length
    max_valid_index = t_max - 1

    # Generate massive pool of indices to test the law of large numbers
    path_indices = generate_sbb_path_indices(
        t_max=t_max,
        h=horizon,
        total_scenarios=total_scenarios,
        expected_block_size=expected_block_size,
        rng=rng,
    )

    empirical_lengths = identify_blocks_from_matrix(path_indices, max_valid_index)

    # Calculate stats
    empirical_mean = np.mean(empirical_lengths)
    print(f"Target Expected Block Size: {expected_block_size:.2f}")
    print(f"Empirical Mean Block Size:  {empirical_mean:.2f}")
    print(f"Total Blocks Sampled:       {len(empirical_lengths)}")

    # Plotting
    plt.figure(figsize=(12, 6))
    max_len = min(int(expected_block_size * 5), np.max(empirical_lengths))

    # Histogram of Empirical data
    bins = list(np.arange(1, max_len + 2) - 0.5)
    plt.hist(
        empirical_lengths,
        bins=bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        label="Empirical (SBB Kernel Output)",
    )

    # Line for Theoretical Geometric PMF
    p = 1.0 / expected_block_size
    x = np.arange(1, max_len + 1)
    theoretical_pmf = geom.pmf(x, p)

    plt.plot(
        x,
        theoretical_pmf,
        "ro-",
        lw=2,
        markersize=4,
        label=f"Theoretical Geometric (p={p:.4f})",
    )

    plt.title("SBB Block Size Distribution Validation")
    plt.xlabel("Block Length (days)")
    plt.ylabel("Probability")
    plt.xlim(0.5, max_len + 0.5)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate_block_sizes(expected_block_size=25.0)
