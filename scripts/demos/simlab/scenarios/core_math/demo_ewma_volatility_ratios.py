# scripts/demos/simlab/scenarios/core_math/demo_ewma_volatility_ratios.py
"""
Visualise EWMA volatility-ratio behaviour across synthetic time series.

For each synthetic series we plot:

    Left : Price levels
    Right: EWMA volatility ratio

where

    Z_t = sigma_today / sigma_t

Interpretation:

    Z_t > 1
        Historical volatility was lower than today's volatility.

    Z_t < 1
        Historical volatility was higher than today's volatility.

    Z_t ~= 1
        Historical volatility was similar to today's volatility.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios,
)
from riskquant.synthetic_level_generators import (
    _DEFAULT_RNG_SEED,
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

NUM_OBS: int = 1_250
RNG_SEED: int = _DEFAULT_RNG_SEED

DECAY_FACTOR: float = 0.98
MAX_VOL_RATIO: float = 10.0


def compute_ewma_ratio(
    levels: np.ndarray,
    *,
    decay_factor: float = DECAY_FACTOR,
    max_vol_ratio: float = MAX_VOL_RATIO,
) -> np.ndarray:

    values = np.empty(
        (levels.size, 2),
        dtype=np.float64,
        order="F",
    )

    values[:, 0] = levels
    values[:, 1] = levels  # dummy column

    out = np.empty(
        values.shape,
        dtype=np.float64,
        order="F",
    )

    kernel_ewma_vol_ratios(
        values=values,
        cols=np.array([0], dtype=np.int32),
        shift_types=np.array([SHIFT_LOG_RETURN_ID], dtype=np.int8),
        decay_factor=decay_factor,
        out_ratios=out,
        max_vol_ratio=max_vol_ratio,
    )

    return out[:, 0]


def main() -> None:

    series = []

    # --------------------------------------------------------------
    # Random Walk
    # --------------------------------------------------------------
    series.append(
        (
            "Random Walk",
            create_random_walk_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    # --------------------------------------------------------------
    # AR(1)
    # --------------------------------------------------------------
    series.append(
        (
            "AR(1)",
            create_ar1_process_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    # --------------------------------------------------------------
    # AR(2)
    # --------------------------------------------------------------
    series.append(
        (
            "AR(2)",
            create_ar2_process_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    # --------------------------------------------------------------
    # Illiquid
    # --------------------------------------------------------------
    _, observed_levels = create_illiquid_levels_series(
        num_obs=NUM_OBS,
        rng_seed=RNG_SEED,
    )

    series.append(
        (
            "Illiquid",
            observed_levels,
        )
    )

    # --------------------------------------------------------------
    # Complex
    # --------------------------------------------------------------
    series.append(
        (
            "Complex",
            create_complex_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    # --------------------------------------------------------------
    # Volatility Clustering
    # --------------------------------------------------------------
    series.append(
        (
            "Volatility Clustering",
            create_volatility_clustering_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    # --------------------------------------------------------------
    # Mean Reverting
    # --------------------------------------------------------------
    series.append(
        (
            "Mean Reverting",
            create_mean_reverting_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    # --------------------------------------------------------------
    # Seasonal
    # --------------------------------------------------------------
    series.append(
        (
            "Seasonal",
            create_seasonal_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    # --------------------------------------------------------------
    # Structural Break
    # --------------------------------------------------------------
    series.append(
        (
            "Structural Break",
            create_structural_break_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    n_series = len(series)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(20, 12),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, (name, levels) in zip(axes, series):
        ratios = compute_ewma_ratio(levels)

        # ----------------------------------------------------------
        # Levels
        # ----------------------------------------------------------
        ax.plot(
            levels,
            color="tab:blue",
            linewidth=1.5,
            label="Levels",
        )

        ax.set_title(name)

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.set_ylabel("Level")

        # ----------------------------------------------------------
        # EWMA Ratio
        # ----------------------------------------------------------
        ax2 = ax.twinx()

        ax2.plot(
            ratios,
            color="tab:red",
            linewidth=1.0,
            alpha=0.8,
            label="EWMA Ratio",
        )

        ax2.axhline(
            1.0,
            linestyle="--",
            color="black",
            linewidth=1.0,
            alpha=0.5,
        )

        ax2.set_ylabel("EWMA Ratio")

        # Keep scales somewhat comparable across panels
        ax2.set_ylim(
            bottom=0.0,
            top=max(2.0, np.nanmax(ratios) * 1.05),
        )

    fig.suptitle(
        (
            "EWMA Volatility Ratios Across Synthetic Time Series\n"
            f"(Decay={DECAY_FACTOR:.2f})"
        ),
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
