# scripts/demos/simlab/scenarios/core_math/demo_ar1_filter_synthetics.py
"""
Visualise AR(1) filter behaviour across synthetic time series.

Blue axis:
    Price levels

Red axis:
    Raw log returns

Black axis:
    AR(1) residuals

The AR(1) filter should materially alter AR(1)-driven series while
having little impact on IID random walks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import (
    kernel_ar1_residuals,
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


def log_returns(levels: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(levels)
    ret[1:] = np.diff(np.log(levels))
    return ret


def compute_ar1_residuals(levels: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper around kernel_ar1_residuals.
    """

    values = np.empty(
        (levels.size, 2),
        dtype=np.float64,
        order="F",
    )

    values[:, 0] = levels
    values[:, 1] = levels  # dummy column

    residuals = np.empty(
        values.shape,
        dtype=np.float64,
        order="F",
    )

    kernel_ar1_residuals(
        values=values,
        cols=np.array([0], dtype=np.int32),
        shift_types=np.array([SHIFT_LOG_RETURN_ID], dtype=np.int8),
        out_residuals=residuals,
    )

    return residuals[:, 0]


def main() -> None:

    series = []

    series.append(
        (
            "Random Walk",
            create_random_walk_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    series.append(
        (
            "AR(1)",
            create_ar1_process_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    series.append(
        (
            "AR(2)",
            create_ar2_process_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    _, illiquid = create_illiquid_levels_series(
        num_obs=NUM_OBS,
        rng_seed=RNG_SEED,
    )

    series.append(("Illiquid", illiquid))

    series.append(
        (
            "Complex",
            create_complex_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    series.append(
        (
            "Volatility Clustering",
            create_volatility_clustering_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    series.append(
        (
            "Mean Reverting",
            create_mean_reverting_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    series.append(
        (
            "Seasonal",
            create_seasonal_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    series.append(
        (
            "Structural Break",
            create_structural_break_levels_series(
                num_obs=NUM_OBS,
                rng_seed=RNG_SEED,
            ),
        )
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(20, 12),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, (name, levels) in zip(axes, series):
        residuals = compute_ar1_residuals(levels)
        returns = log_returns(levels)

        # ----------------------------------------------------------
        # Levels
        # ----------------------------------------------------------

        ax.plot(
            levels,
            color="tab:blue",
            linewidth=1.5,
            alpha=0.8,
        )

        ax.set_title(name)

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.set_ylabel("Level")

        # ----------------------------------------------------------
        # Returns vs Residuals
        # ----------------------------------------------------------

        ax2 = ax.twinx()

        ax2.plot(
            returns,
            color="tab:red",
            alpha=0.5,
            linewidth=0.8,
            label="Returns",
        )

        ax2.plot(
            residuals,
            color="black",
            alpha=0.8,
            linewidth=0.8,
            label="AR(1) Residuals",
        )

        if name == "Random Walk":
            ax2.legend(
                loc="upper right",
                fontsize=8,
            )

        ax2.set_ylabel("Return / Residual")

    fig.suptitle(
        "AR(1) Filter Behaviour Across Synthetic Time Series",
        fontsize=18,
        fontweight="bold",
    )

    plt.show()


if __name__ == "__main__":
    main()
