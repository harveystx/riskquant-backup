# scripts/synthetic_level_generators/demo_all_generators.py
"""
Demonstration of ``SyntheticLevelGenerator``.

The demo:

1. Generates every supported synthetic series type.
2. Repeats the experiment for several innovation-dependence strengths.
3. Plots normalised levels and log returns for every dependence strength.
4. Computes and prints the realised Pearson correlation matrix of returns
   for every dependence strength.
5. Produces a pairwise return-scatter matrix for one selected dependence
   strength.

Plot conventions
----------------
Levels are rebased to 100 at the first observation. This makes the different
process shapes directly comparable even when their absolute levels diverge.

Returns are plotted on a symmetric logarithmic y-axis (``symlog``), which
keeps the central return distribution visible while allowing large jumps,
including those from illiquid and complex series, to remain visible.

Run from the RiskQuant repository root:

    python scripts/synthetic_level_generators/demo_all_generators.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from riskquant.synthetic_level_generators import (
    SyntheticLevelGenerator,
    SyntheticSeriesKind,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_OBS: Final[int] = 1_301
RNG_SEED: Final[int] = 123_456

CORRELATIONS: Final[tuple[float, ...]] = (
    0.00,
    0.25,
    0.50,
    0.75,
)

# Full pairwise scatter matrix is generated for this dependence scenario.
SCATTER_CORRELATION: Final[float] = 0.75

# Set to True to write figures instead of only displaying them.
SAVE_FIGURES: Final[bool] = False

FIGURE_DIR: Final[Path] = Path("demo_output") / "synthetic_level_generators"

# Symmetric-log transition scale for returns.
#
# Values inside roughly +/- LINTHRESH remain visually linear. Outside that
# region, symlog compresses the tails while retaining their sign.
RETURN_LINTHRESH: Final[float] = 0.1

# Base size of each cell in the scatter matrix.
SCATTER_CELL_SIZE: Final[float] = 2.6


# ============================================================================
# SERIES DEFINITIONS
# ============================================================================

SERIES_SPECS: Final[dict[str, SyntheticSeriesKind]] = {
    "random_walk": "random_walk",
    "ar1": "ar1",
    "ar2": "ar2",
    "garch": "garch",
    "mean_reverting": "mean_reverting",
    "seasonal": "seasonal",
    "structural_break": "structural_break",
    "complex": "complex",
    "illiquid_observed": "illiquid_observed",
    "illiquid_latent": "illiquid_latent",
}


# ============================================================================
# DATA HELPERS
# ============================================================================


def _returns_from_levels(
    levels: np.ndarray,
) -> np.ndarray:
    """
    Compute log returns from strictly positive levels.
    """
    if levels.ndim != 1:
        raise ValueError("Levels must be one-dimensional.")

    if len(levels) < 2:
        raise ValueError("At least two levels are required.")

    if not np.all(np.isfinite(levels)):
        raise ValueError("Levels contain non-finite values.")

    if np.any(levels <= 0.0):
        raise ValueError("Levels must be strictly positive.")

    return np.diff(np.log(levels))


def _normalise_levels(
    levels: np.ndarray,
) -> np.ndarray:
    """
    Rebase a level series to 100 at its first observation.
    """
    if levels.ndim != 1:
        raise ValueError("Levels must be one-dimensional.")

    if len(levels) == 0:
        raise ValueError("Levels must not be empty.")

    initial_level = levels[0]

    if not np.isfinite(initial_level) or initial_level <= 0.0:
        raise ValueError(
            f"Initial level must be finite and strictly positive; "
            f"got {initial_level!r}."
        )

    return 100.0 * levels / initial_level


def _compute_return_matrix(
    series: dict[str, np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """
    Convert named level series into a common return matrix.

    Returns
    -------
    names:
        Series names in matrix column order.

    returns:
        Shape ``(n_observations - 1, n_series)``.
    """
    names = tuple(series)

    if not names:
        raise ValueError("No series supplied.")

    returns = np.column_stack([_returns_from_levels(series[name]) for name in names])

    return names, np.asarray(
        returns,
        dtype=np.float64,
    )


def _compute_correlation_matrix(
    series: dict[str, np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """
    Compute the Pearson correlation matrix of log returns.
    """
    names, returns = _compute_return_matrix(series)

    correlation_matrix = np.corrcoef(
        returns,
        rowvar=False,
    )

    return names, np.asarray(
        correlation_matrix,
        dtype=np.float64,
    )


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================


def _print_correlation_matrix(
    *,
    correlation: float,
    names: tuple[str, ...],
    matrix: np.ndarray,
) -> None:
    """
    Print a readable realised return-correlation matrix.
    """
    print()
    print("=" * 125)
    print(
        "REALIZED RETURN CORRELATION MATRIX "
        f"| TARGET INNOVATION CORRELATION = {correlation:.2f}"
    )
    print("=" * 125)

    label_width = max(
        18,
        max(len(name) for name in names) + 2,
    )

    header = " " * label_width
    header += "".join(f"{name:>16s}" for name in names)

    print(header)

    for i, name in enumerate(names):
        row = f"{name:<{label_width}s}"

        row += "".join(f"{matrix[i, j]:16.3f}" for j in range(len(names)))

        print(row)

    print()


def _print_return_summary(
    *,
    correlation: float,
    names: tuple[str, ...],
    returns: np.ndarray,
) -> None:
    """
    Print useful return-scale diagnostics.

    This is especially useful for the demo because the synthetic series are
    intentionally heterogeneous.
    """
    print()
    print("=" * 125)
    print(f"RETURN SUMMARY | TARGET INNOVATION CORRELATION = {correlation:.2f}")
    print("=" * 125)

    header = (
        f"{'Series':<22s}"
        f"{'Mean':>14s}"
        f"{'Std':>14s}"
        f"{'Min':>16s}"
        f"{'Max':>16s}"
        f"{'Zero %':>12s}"
    )

    print(header)
    print("-" * len(header))

    for i, name in enumerate(names):
        x = returns[:, i]

        zero_pct = 100.0 * np.count_nonzero(x == 0.0) / len(x)

        print(
            f"{name:<22s}"
            f"{np.mean(x):14.6g}"
            f"{np.std(x, ddof=1):14.6g}"
            f"{np.min(x):16.6g}"
            f"{np.max(x):16.6g}"
            f"{zero_pct:12.2f}"
        )

    print()


# ============================================================================
# PLOTTING
# ============================================================================


def _plot_levels_and_returns(
    *,
    correlation: float,
    series: dict[str, np.ndarray],
) -> None:
    """
    Plot rebased levels and log returns.
    """
    names, returns = _compute_return_matrix(series)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 11),
        constrained_layout=True,
    )

    levels_axis = axes[0]
    returns_axis = axes[1]

    # ------------------------------------------------------------------------
    # Normalised levels
    # ------------------------------------------------------------------------

    for name in names:
        levels = series[name]
        normalised_levels = _normalise_levels(levels)

        levels_axis.plot(
            normalised_levels,
            label=name,
            linewidth=1.0,
        )

    levels_axis.axhline(
        100.0,
        linewidth=0.8,
    )

    levels_axis.set_title(
        "Synthetic Risk-Factor Levels "
        f"(rebased to 100) | "
        f"Innovation Correlation = {correlation:.2f}"
    )
    levels_axis.set_xlabel("Observation")
    levels_axis.set_ylabel("Normalised level")
    levels_axis.grid(
        alpha=0.25,
    )
    levels_axis.legend(
        loc="best",
        ncol=2,
        fontsize=8,
    )

    # ------------------------------------------------------------------------
    # Log returns
    # ------------------------------------------------------------------------

    for i, name in enumerate(names):
        returns_axis.plot(
            returns[:, i],
            label=name,
            linewidth=0.75,
        )

    returns_axis.axhline(
        0.0,
        linewidth=0.8,
    )

    returns_axis.set_yscale(
        "symlog",
        linthresh=RETURN_LINTHRESH,
    )

    returns_axis.set_title(
        "Synthetic Log Returns "
        f"(symlog scale) | "
        f"Innovation Correlation = {correlation:.2f}"
    )
    returns_axis.set_xlabel("Observation")
    returns_axis.set_ylabel("Log return")
    returns_axis.grid(
        alpha=0.25,
    )

    # Ten series makes the return-panel legend too dense.
    # The level panel already identifies all series.
    if len(names) <= 8:
        returns_axis.legend(
            loc="best",
            ncol=2,
            fontsize=8,
        )

    _save_or_show(
        fig,
        FIGURE_DIR / f"levels_returns_corr_{correlation:.2f}.png",
    )


def _plot_return_scatter_matrix(
    *,
    correlation: float,
    series: dict[str, np.ndarray],
) -> None:
    """
    Plot a full pairwise scatter matrix of log returns.

    Diagonal:
        Return histogram.

    Off-diagonal:
        Pairwise return scatter with realised Pearson correlation.
    """
    names, returns = _compute_return_matrix(series)

    n_series = len(names)

    if n_series == 0:
        raise ValueError("No series supplied.")

    fig_size = SCATTER_CELL_SIZE * n_series

    # ``figsize`` must be a (width, height) tuple.
    fig, axes = plt.subplots(
        n_series,
        n_series,
        figsize=(fig_size, fig_size),
        squeeze=False,
    )

    correlation_matrix = np.corrcoef(
        returns,
        rowvar=False,
    )

    for row in range(n_series):
        for col in range(n_series):
            axis = axes[row][col]

            x = returns[:, col]
            y = returns[:, row]

            if row == col:
                axis.hist(
                    x,
                    bins=35,
                )

            else:
                axis.scatter(
                    x,
                    y,
                    s=5,
                    alpha=0.30,
                )

                realised_corr = correlation_matrix[row, col]

                axis.text(
                    0.05,
                    0.92,
                    f"rho = {realised_corr:.3f}",
                    transform=axis.transAxes,
                    fontsize=8,
                    va="top",
                )

            # Use symlog for all scatter axes as well. This is important for
            # the complex and illiquid factors where a few observations can
            # otherwise dominate the entire panel.
            axis.set_xscale(
                "symlog",
                linthresh=RETURN_LINTHRESH,
            )

            if row != col:
                axis.set_yscale(
                    "symlog",
                    linthresh=RETURN_LINTHRESH,
                )

            if row == n_series - 1:
                axis.set_xlabel(
                    names[col],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
            else:
                axis.set_xticklabels([])

            if col == 0:
                axis.set_ylabel(
                    names[row],
                    fontsize=8,
                )
            else:
                axis.set_yticklabels([])

            axis.grid(
                alpha=0.15,
            )

    fig.suptitle(
        "Pairwise Log-Return Scatter Matrix "
        f"| Innovation Correlation = {correlation:.2f}",
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.97),
    )

    _save_or_show(
        fig,
        FIGURE_DIR / f"return_scatter_matrix_corr_{correlation:.2f}.png",
    )


def _plot_correlation_matrix(
    *,
    correlation: float,
    names: tuple[str, ...],
    matrix: np.ndarray,
) -> None:
    """
    Plot the realised return correlation matrix.

    This is intentionally separate from the scatter matrix so the numerical
    dependence structure can be inspected at a glance.
    """
    fig_size = max(
        7.0,
        0.85 * len(names),
    )

    fig, axis = plt.subplots(
        figsize=(fig_size, fig_size),
        constrained_layout=True,
    )

    image = axis.imshow(
        matrix,
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
    )

    axis.set_xticks(
        np.arange(len(names)),
        labels=names,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        np.arange(len(names)),
        labels=names,
    )

    axis.set_title(
        "Realised Return Correlation Matrix "
        f"| Innovation Correlation = {correlation:.2f}"
    )

    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    fig.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    _save_or_show(
        fig,
        FIGURE_DIR / f"return_correlation_matrix_corr_{correlation:.2f}.png",
    )


def _save_or_show(
    fig: Figure,
    output_path: Path,
) -> None:
    """
    Save a figure when requested; otherwise display it.
    """
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        print(f"Saved: {output_path}")

    plt.show()

    # Explicitly close after display/save so running all scenarios does not
    # leave a large collection of live Matplotlib figures.
    plt.close(fig)


# ============================================================================
# VALIDATION
# ============================================================================


def _validate_configuration() -> None:
    """
    Fail fast on demo configuration errors.
    """
    if NUM_OBS < 2:
        raise ValueError(f"{NUM_OBS=} must be at least 2.")

    if not CORRELATIONS:
        raise ValueError("CORRELATIONS must contain at least one value.")

    for correlation in CORRELATIONS:
        if not 0.0 <= correlation <= 1.0:
            raise ValueError(
                f"Invalid correlation {correlation}; expected 0 <= correlation <= 1."
            )

    if SCATTER_CORRELATION not in CORRELATIONS:
        raise ValueError(f"{SCATTER_CORRELATION=} must be one of {CORRELATIONS=}.")

    if RETURN_LINTHRESH <= 0.0:
        raise ValueError(f"{RETURN_LINTHRESH=} must be strictly positive.")

    if SCATTER_CELL_SIZE <= 0.0:
        raise ValueError(f"{SCATTER_CELL_SIZE=} must be strictly positive.")


# ============================================================================
# MAIN DEMO
# ============================================================================


def main() -> None:
    """
    Execute the complete synthetic-series demonstration.
    """
    _validate_configuration()

    print()
    print("=" * 125)
    print("RISKQUANT SYNTHETIC LEVEL GENERATOR DEMO")
    print("=" * 125)
    print(f"{NUM_OBS=}")
    print(f"{RNG_SEED=}")
    print(f"{CORRELATIONS=}")
    print(f"{SCATTER_CORRELATION=}")
    print(f"{RETURN_LINTHRESH=}")
    print()

    all_results: dict[
        float,
        dict[str, np.ndarray],
    ] = {}

    for correlation in CORRELATIONS:
        print()
        print("#" * 125)
        print(f"GENERATING SERIES | TARGET INNOVATION CORRELATION = {correlation:.2f}")
        print("#" * 125)

        # Use a fresh generator with the same seed for each dependence
        # scenario. This isolates the effect of dependence from Monte Carlo
        # path differences.
        generator = SyntheticLevelGenerator(
            num_obs=NUM_OBS,
            rng_seed=RNG_SEED,
        )

        series = generator.generate_many(
            SERIES_SPECS,
            correlation=correlation,
        )

        all_results[correlation] = series

        names, returns = _compute_return_matrix(series)

        _, correlation_matrix = _compute_correlation_matrix(
            series,
        )

        _print_return_summary(
            correlation=correlation,
            names=names,
            returns=returns,
        )

        _print_correlation_matrix(
            correlation=correlation,
            names=names,
            matrix=correlation_matrix,
        )

        _plot_levels_and_returns(
            correlation=correlation,
            series=series,
        )

        _plot_correlation_matrix(
            correlation=correlation,
            names=names,
            matrix=correlation_matrix,
        )

    # ------------------------------------------------------------------------
    # Full pairwise return scatter matrix for the selected dependence case.
    # ------------------------------------------------------------------------

    _plot_return_scatter_matrix(
        correlation=SCATTER_CORRELATION,
        series=all_results[SCATTER_CORRELATION],
    )


if __name__ == "__main__":
    main()
