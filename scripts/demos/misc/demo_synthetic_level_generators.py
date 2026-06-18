# scripts/demos/misc/demo_synthetic_level_generators.py
"""
Demo script for visualising the synthetic time series generators.
"""

import matplotlib.pyplot as plt

from riskquant.synthetic_level_generators import (
    create_ar1_process_levels_series,
    create_ar2_process_levels_series,
    create_complex_levels_series,
    create_illiquid_levels_series,
    create_mean_reverting_levels_series,
    create_random_walk_levels_series,
    create_seasonal_levels_series,
    create_structural_break_levels_series,
    create_volatility_clustering_levels_series,
)

NUM_OBS: int = 1_250  # ~5 years
RNG_SEED: int = 999_999


def main() -> None:

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 14),
        constrained_layout=True,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------
    # Random Walk
    # ------------------------------------------------------------------
    axes[0].plot(
        create_random_walk_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[0].set_title("Random Walk")
    axes[0].grid(True)

    # ------------------------------------------------------------------
    # AR(1)
    # ------------------------------------------------------------------
    axes[1].plot(
        create_ar1_process_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[1].set_title("AR(1)")
    axes[1].grid(True)

    # ------------------------------------------------------------------
    # AR(2)
    # ------------------------------------------------------------------
    axes[2].plot(
        create_ar2_process_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[2].set_title("AR(2)")
    axes[2].grid(True)

    # ------------------------------------------------------------------
    # Illiquid
    # ------------------------------------------------------------------
    true_levels, observed_levels = create_illiquid_levels_series(
        num_obs=NUM_OBS,
        rng_seed=RNG_SEED,
    )

    axes[3].plot(
        true_levels,
        label="True",
        alpha=0.5,
    )
    axes[3].plot(
        observed_levels,
        label="Observed",
        color="tab:red",
        alpha=0.8,
    )

    axes[3].set_title("Illiquid")
    axes[3].legend()
    axes[3].grid(True)

    # ------------------------------------------------------------------
    # Complex
    # ------------------------------------------------------------------
    axes[4].plot(
        create_complex_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[4].set_title("Complex")
    axes[4].grid(True)

    # ------------------------------------------------------------------
    # Volatility Clustering
    # ------------------------------------------------------------------
    axes[5].plot(
        create_volatility_clustering_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[5].set_title("Volatility Clustering")
    axes[5].grid(True)

    # ------------------------------------------------------------------
    # Mean Reverting
    # ------------------------------------------------------------------
    axes[6].plot(
        create_mean_reverting_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[6].set_title("Mean Reverting")
    axes[6].grid(True)

    # ------------------------------------------------------------------
    # Seasonal
    # ------------------------------------------------------------------
    axes[7].plot(
        create_seasonal_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[7].set_title("Seasonal")
    axes[7].grid(True)

    # ------------------------------------------------------------------
    # Structural Break
    # ------------------------------------------------------------------
    axes[8].plot(
        create_structural_break_levels_series(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )
    )
    axes[8].set_title("Structural Break")
    axes[8].grid(True)

    fig.suptitle(
        "Synthetic Time Series Generators",
        fontsize=18,
        fontweight="bold",
    )

    plt.show()


if __name__ == "__main__":
    main()
