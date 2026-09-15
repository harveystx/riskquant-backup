# scripts/demos/simlab/scenarios/core_math/demo_sbb_fanout_visualizer_mf.py
"""
Two-factor correlated SBB fan-out simulation demo.

The demo generates two synthetic risk factors with a configurable dependence
structure, applies the same stationary block-bootstrap (SBB) path indices to
both factors, and visualises the resulting paired simulations.

The two synthetic factor types are controlled by the module-level constants:

    SERIES_A_KIND
    SERIES_B_KIND

The historical factors are generated jointly using
``SyntheticLevelGenerator.generate_many`` so that their innovations have the
requested common-factor dependence.

The same SBB path-index matrix is then applied to both historical return
series. Each simulated path therefore represents one paired scenario across
the two risk factors.

The figure contains:

    LEFT:
        Historical factor A + SBB fan-out and its terminal distribution.

        Historical factor B + SBB fan-out and its terminal distribution.

    RIGHT:
        Scatter plot of simulated terminal percentage changes for A versus B.

The terminal scatter reports the realised correlation between the paired
terminal percentage changes.

Run from the RiskQuant repository root, for example:

    python scripts/demos/simlab/scenarios/core_math/demo_sbb_fanout_visualizer_mf.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)
from riskquant.synthetic_level_generators import (
    SyntheticLevelGenerator,
    SyntheticSeriesKind,
)
from riskquant.utils.printer import (
    print_calling_file,
    print_success_message,
    print_title,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_OBS: int = 400

SERIES_A_KIND: SyntheticSeriesKind = "complex"
SERIES_B_KIND: SyntheticSeriesKind = "garch"

TARGET_CORRELATION: float = 0.90

N_STALE: int = 2
N_STALE_LEN: int = 25

HORIZON: int = 100

N_SIMULATIONS: int = 2_000
N_PLOT_LINES: int = 50

EXPECTED_BLOCK_SIZE: float = 25.0

RNG_SEED: int = 123_456_789

LEN_01_PROPORTION: float = 0.30
LEN_02_PROPORTION: float = 0.30

# Scatter transparency.
SCATTER_ALPHA: float = 0.25

# Scatter marker size.
SCATTER_SIZE: float = 12.0


# ============================================================================
# VALIDATION
# ============================================================================


def _validate_configuration() -> None:
    """Fail fast on invalid demo configuration."""
    if NUM_OBS < 2:
        raise ValueError(f"{NUM_OBS=} must be at least 2.")

    if N_SIMULATIONS < 1:
        raise ValueError(f"{N_SIMULATIONS=} must be strictly positive.")

    if N_PLOT_LINES < 1:
        raise ValueError(f"{N_PLOT_LINES=} must be strictly positive.")

    if N_PLOT_LINES > N_SIMULATIONS:
        raise ValueError(f"{N_PLOT_LINES=} cannot exceed {N_SIMULATIONS=}.")

    if HORIZON < 1:
        raise ValueError(f"{HORIZON=} must be strictly positive.")

    if not 0.0 <= TARGET_CORRELATION <= 1.0:
        raise ValueError(f"{TARGET_CORRELATION=} must satisfy 0 <= correlation <= 1.")

    if N_STALE < 0:
        raise ValueError(f"{N_STALE=} must be non-negative.")

    if N_STALE_LEN < 1:
        raise ValueError(f"{N_STALE_LEN=} must be strictly positive.")

    if EXPECTED_BLOCK_SIZE <= 0.0:
        raise ValueError(f"{EXPECTED_BLOCK_SIZE=} must be strictly positive.")

    if not 0.0 < LEN_01_PROPORTION < 1.0:
        raise ValueError(f"{LEN_01_PROPORTION=} must be strictly between 0 and 1.")

    if not 0.0 < LEN_02_PROPORTION < 1.0:
        raise ValueError(f"{LEN_02_PROPORTION=} must be strictly between 0 and 1.")

    if LEN_01_PROPORTION + LEN_02_PROPORTION >= 1.0:
        raise ValueError(
            "LEN_01_PROPORTION + LEN_02_PROPORTION must be strictly less than 1."
        )


# ============================================================================
# HELPERS
# ============================================================================


def _log_returns(
    levels: np.ndarray,
) -> np.ndarray:
    """Calculate log returns from strictly positive levels."""
    if levels.ndim != 1:
        raise ValueError("Levels must be one-dimensional.")

    if len(levels) < 2:
        raise ValueError("At least two levels are required.")

    if not np.all(np.isfinite(levels)):
        raise ValueError("Levels contain non-finite values.")

    if np.any(levels <= 0.0):
        raise ValueError("Levels must be strictly positive.")

    return np.diff(np.log(levels))


def _build_sbb_fanout(
    *,
    levels: np.ndarray,
    path_indices: np.ndarray,
) -> np.ndarray:
    """
    Apply an SBB path-index matrix to one historical level series.

    Parameters
    ----------
    levels:
        Historical levels.

    path_indices:
        Integer matrix of shape ``(n_simulations, horizon)``.

        The SBB generator returns indices in the range ``1 .. t_max - 1``.
        Subtracting one converts these to the zero-based return array.

    Returns
    -------
    np.ndarray
        Simulated levels with shape
        ``(n_simulations, horizon)``.
    """
    shifts = _log_returns(levels)

    if path_indices.ndim != 2:
        raise ValueError("path_indices must be two-dimensional.")

    if np.min(path_indices) < 1:
        raise ValueError("SBB path indices must be one-based positive indices.")

    max_valid_index = len(levels) - 1

    if np.max(path_indices) > max_valid_index:
        raise ValueError(f"SBB path indices contain values above {max_valid_index}.")

    bootstrapped_shifts = shifts[path_indices - 1]

    cumulative_shifts = np.cumsum(
        bootstrapped_shifts,
        axis=1,
    )

    terminal_level = levels[-1]

    return terminal_level * np.exp(
        cumulative_shifts,
    )


def _terminal_percentage_changes(
    *,
    terminal_values: np.ndarray,
    starting_level: float,
) -> np.ndarray:
    """
    Calculate simulated terminal percentage changes from the simulation start.

    The simulation starts at the historical terminal level immediately before
    the SBB horizon. Therefore:

        percentage_change = terminal_value / starting_level - 1
    """
    if not np.isfinite(starting_level):
        raise ValueError("starting_level must be finite.")

    if starting_level <= 0.0:
        raise ValueError("starting_level must be strictly positive.")

    if terminal_values.ndim != 1:
        raise ValueError("terminal_values must be one-dimensional.")

    if not np.all(np.isfinite(terminal_values)):
        raise ValueError("terminal_values contain non-finite values.")

    return terminal_values / starting_level - 1.0


def _terminal_correlation(
    terminal_a: np.ndarray,
    terminal_b: np.ndarray,
) -> float:
    """Calculate the Pearson correlation of paired terminal values."""
    if terminal_a.shape != terminal_b.shape:
        raise ValueError("Terminal arrays must have identical shapes.")

    if len(terminal_a) < 2:
        raise ValueError("At least two simulations are required.")

    return float(
        np.corrcoef(
            terminal_a,
            terminal_b,
        )[0, 1]
    )


# ============================================================================
# MAIN DEMO
# ============================================================================


def plot_two_factor_fanout(
    *,
    num_obs: int = NUM_OBS,
    series_a_kind: SyntheticSeriesKind = SERIES_A_KIND,
    series_b_kind: SyntheticSeriesKind = SERIES_B_KIND,
    target_correlation: float = TARGET_CORRELATION,
    n_stale: int = N_STALE,
    n_stale_len: int = N_STALE_LEN,
    horizon: int = HORIZON,
    n_simulations: int = N_SIMULATIONS,
    n_plot_lines: int = N_PLOT_LINES,
    expected_block_size: float = EXPECTED_BLOCK_SIZE,
    rng_seed: int = RNG_SEED,
    len_01_proportion: float = LEN_01_PROPORTION,
    len_02_proportion: float = LEN_02_PROPORTION,
) -> None:
    """
    Generate and plot two correlated SBB fan-outs.

    Figure layout
    -------------
    [0, 0]
        Historical series A + SBB fan-out.

    [0, 1]
        Horizontal distribution of series A terminal values.

    [0, 2]
        Scatter plot of paired terminal percentage changes.
        This axis spans both rows.

    [1, 0]
        Historical series B + SBB fan-out.

    [1, 1]
        Horizontal distribution of series B terminal values.

    The fan-out and corresponding terminal distribution share the same x-axis.
    The scatter plot has independent x/y limits.
    """

    if not 0.0 <= target_correlation <= 1.0:
        raise ValueError(f"{target_correlation=} must satisfy 0 <= correlation <= 1.")

    # ------------------------------------------------------------------------
    # Generate the historical factors jointly.
    # ------------------------------------------------------------------------

    generator = SyntheticLevelGenerator(
        num_obs=num_obs,
        rng_seed=rng_seed,
    )

    historical = generator.generate_many(
        {
            "series_a": {
                "kind": series_a_kind,
                **(
                    {
                        "n_stale": n_stale,
                        "n_stale_len": n_stale_len,
                        "len_01_proportion": len_01_proportion,
                        "len_02_proportion": len_02_proportion,
                    }
                    if series_a_kind == "complex"
                    else {}
                ),
            },
            "series_b": {
                "kind": series_b_kind,
                **(
                    {
                        "n_stale": n_stale,
                        "n_stale_len": n_stale_len,
                        "len_01_proportion": len_01_proportion,
                        "len_02_proportion": len_02_proportion,
                    }
                    if series_b_kind == "complex"
                    else {}
                ),
            },
        },
        correlation=target_correlation,
    )

    levels_a = historical["series_a"]
    levels_b = historical["series_b"]

    returns_a = _log_returns(levels_a)
    returns_b = _log_returns(levels_b)

    historical_return_correlation = float(
        np.corrcoef(
            returns_a,
            returns_b,
        )[0, 1]
    )

    # ------------------------------------------------------------------------
    # Generate ONE SBB index matrix and apply the same paired historical
    # scenario to both factors.
    # ------------------------------------------------------------------------

    t_max = len(levels_a)

    rng = np.random.default_rng(rng_seed)

    path_indices = generate_sbb_path_indices(
        t_max=t_max,
        h=horizon,
        total_scenarios=n_simulations,
        expected_block_size=expected_block_size,
        rng=rng,
    )

    simulated_levels_a = _build_sbb_fanout(
        levels=levels_a,
        path_indices=path_indices,
    )

    simulated_levels_b = _build_sbb_fanout(
        levels=levels_b,
        path_indices=path_indices,
    )

    # ------------------------------------------------------------------------
    # Terminal values and percentage changes.
    #
    # Each SBB simulation starts from the historical terminal level.
    # ------------------------------------------------------------------------

    terminal_values_a = np.asarray(
        simulated_levels_a[:, -1],
        dtype=np.float64,
    )

    terminal_values_b = np.asarray(
        simulated_levels_b[:, -1],
        dtype=np.float64,
    )

    starting_level_a = float(levels_a[-1])
    starting_level_b = float(levels_b[-1])

    terminal_changes_a = _terminal_percentage_changes(
        terminal_values=terminal_values_a,
        starting_level=starting_level_a,
    )

    terminal_changes_b = _terminal_percentage_changes(
        terminal_values=terminal_values_b,
        starting_level=starting_level_b,
    )

    terminal_corr = _terminal_correlation(
        terminal_changes_a,
        terminal_changes_b,
    )

    # ------------------------------------------------------------------------
    # Figure layout:
    #
    #   [fan-out A] [distribution A] [               ]
    #   [fan-out B] [distribution B] [    scatter    ]
    #                                    spans rows
    # ------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(22, 11),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[3.6, 1.35, 3.2],
        wspace=0.10,
        hspace=0.12,
    )

    ax_a_fanout = fig.add_subplot(grid[0, 0])
    ax_a_dist = fig.add_subplot(
        grid[0, 1],
        sharex=ax_a_fanout,
        sharey=ax_a_fanout,
    )

    ax_b_fanout = fig.add_subplot(grid[1, 0])
    ax_b_dist = fig.add_subplot(
        grid[1, 1],
        sharex=ax_b_fanout,
        sharey=ax_b_fanout,
    )

    ax_scatter = fig.add_subplot(grid[:, 2])

    # ========================================================================
    # Common plotting configuration
    # ========================================================================

    hist_x = np.arange(t_max)

    sim_x = np.arange(
        t_max - 1,
        t_max - 1 + horizon,
    )

    lines_to_plot = min(
        n_plot_lines,
        n_simulations,
    )

    # ========================================================================
    # SERIES A FAN-OUT
    # ========================================================================

    ax_a_fanout.plot(
        hist_x,
        levels_a,
        color="tab:blue",
        lw=2.0,
        label=f"Historical A: {series_a_kind}",
    )

    for simulation_idx in range(lines_to_plot):
        ax_a_fanout.plot(
            sim_x,
            simulated_levels_a[simulation_idx],
            color="tab:blue",
            alpha=0.12,
            lw=1.2,
            label=(
                f"SBB fan-out ({lines_to_plot}/{n_simulations} paths)"
                if simulation_idx == 0
                else None
            ),
        )

    ax_a_fanout.axvline(
        t_max - 1,
        color="grey",
        linestyle="--",
        alpha=0.6,
        lw=1.0,
    )

    ax_a_fanout.plot(
        t_max - 1,
        starting_level_a,
        marker="o",
        color="tab:red",
        markersize=7,
        zorder=5,
        label="Simulation starting level",
    )

    ax_a_fanout.set_title(f"Series A Fan-Out ({series_a_kind})")

    ax_a_fanout.set_ylabel("Level")
    ax_a_fanout.grid(True, alpha=0.25)
    ax_a_fanout.legend(
        loc="best",
        fontsize=8,
    )

    # The A fan-out shares its x-axis with the A terminal distribution.
    ax_a_fanout.tick_params(
        axis="x",
        labelbottom=False,
    )

    # ========================================================================
    # SERIES A TERMINAL DISTRIBUTION
    # ========================================================================

    ax_a_dist.hist(
        terminal_values_a,
        bins=30,
        orientation="horizontal",
        color="tab:blue",
        alpha=0.65,
    )

    # ax_a_dist.axhline(
    #     starting_level_a,
    #     color="tab:red",
    #     linestyle="--",
    #     lw=1.5,
    #     label="Simulation starting level",
    # )

    ax_a_dist.set_title("Terminal Dist.")

    ax_a_dist.grid(
        True,
        alpha=0.25,
    )

    ax_a_dist.tick_params(
        axis="y",
        labelleft=False,
    )

    ax_a_dist.legend(
        loc="best",
        fontsize=8,
    )

    ax_a_dist.set_xticklabels([])

    # ========================================================================
    # SERIES B FAN-OUT
    # ========================================================================

    ax_b_fanout.plot(
        hist_x,
        levels_b,
        color="tab:green",
        lw=2.0,
        label=f"Historical B: {series_b_kind}",
    )

    for simulation_idx in range(lines_to_plot):
        ax_b_fanout.plot(
            sim_x,
            simulated_levels_b[simulation_idx],
            color="tab:green",
            alpha=0.12,
            lw=1.2,
            label=(
                f"SBB fan-out ({lines_to_plot}/{n_simulations} paths)"
                if simulation_idx == 0
                else None
            ),
        )

    ax_b_fanout.axvline(
        t_max - 1,
        color="grey",
        linestyle="--",
        alpha=0.6,
        lw=1.0,
    )

    ax_b_fanout.plot(
        t_max - 1,
        starting_level_b,
        marker="o",
        color="tab:red",
        markersize=7,
        zorder=5,
        label="Simulation starting level",
    )

    ax_b_fanout.set_title(f"Series B Fan-Out ({series_b_kind})")

    ax_b_fanout.set_xlabel("Time (days)")
    ax_b_fanout.set_ylabel("Level")
    ax_b_fanout.grid(True, alpha=0.25)
    ax_b_fanout.legend(
        loc="best",
        fontsize=8,
    )

    # ========================================================================
    # SERIES B TERMINAL DISTRIBUTION
    # ========================================================================

    ax_b_dist.hist(
        terminal_values_b,
        bins=30,
        orientation="horizontal",
        color="tab:green",
        alpha=0.65,
    )

    # ax_b_dist.axhline(
    #     starting_level_b,
    #     color="tab:red",
    #     linestyle="--",
    #     lw=1.5,
    #     label="Simulation starting level",
    # )

    ax_b_dist.set_title("Terminal Dist.")

    ax_b_dist.set_xlabel("Frequency")
    ax_b_dist.grid(
        True,
        alpha=0.25,
    )

    ax_b_dist.tick_params(
        axis="y",
        labelleft=False,
    )

    ax_b_dist.legend(
        loc="best",
        fontsize=8,
    )

    # ========================================================================
    # TERMINAL PERCENTAGE-CHANGE SCATTER
    # ========================================================================

    ax_scatter.scatter(
        terminal_changes_a,
        terminal_changes_b,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        c="k",
    )

    ax_scatter.scatter(
        0.0,
        0.0,
        marker="x",
        s=90,
        linewidths=2.0,
        color="tab:red",
        zorder=5,
        label="Simulation starting point",
    )

    ax_scatter.set_title(f"Paired SBB Terminal Changes (rho = {terminal_corr:.3f})")

    ax_scatter.set_xlabel(f"Series A terminal change ({series_a_kind})")

    ax_scatter.set_ylabel(f"Series B terminal change ({series_b_kind})")

    ax_scatter.grid(
        True,
        alpha=0.25,
    )

    ax_scatter.legend(
        loc="best",
        fontsize=8,
    )

    # ------------------------------------------------------------------------
    # Independent scatter limits.
    #
    # The scatter is in percentage-change space, so its axes should be driven
    # independently by the realised A and B terminal changes.
    # ------------------------------------------------------------------------

    scatter_x_min = float(np.min(terminal_changes_a))
    scatter_x_max = float(np.max(terminal_changes_a))

    scatter_y_min = float(np.min(terminal_changes_b))
    scatter_y_max = float(np.max(terminal_changes_b))

    scatter_x_range = max(
        scatter_x_max - scatter_x_min,
        1e-12,
    )

    scatter_y_range = max(
        scatter_y_max - scatter_y_min,
        1e-12,
    )

    scatter_x_padding = 0.05 * scatter_x_range
    scatter_y_padding = 0.05 * scatter_y_range

    ax_scatter.set_xlim(
        scatter_x_min - scatter_x_padding,
        scatter_x_max + scatter_x_padding,
    )

    ax_scatter.set_ylim(
        scatter_y_min - scatter_y_padding,
        scatter_y_max + scatter_y_padding,
    )

    ax_scatter.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    # ========================================================================
    # Figure-level information
    # ========================================================================

    fig.suptitle(
        "Two-Factor Correlated Stationary Block Bootstrap Fan-Out",
        fontsize=15,
    )

    fig.text(
        0.5,
        0.015,
        (
            f"Target innovation correlation = {target_correlation:.3f}   |   "
            f"Historical return correlation = "
            f"{historical_return_correlation:.3f}   |   "
            f"Terminal-change correlation = {terminal_corr:.3f}   |   "
            f"{n_simulations:,} simulations   |   "
            f"{horizon} day horizon   |   "
            f"SBB block size ≈ {expected_block_size:.1f}"
        ),
        ha="center",
        fontsize=7,
    )

    plt.tight_layout(
        rect=(0.0, 0.035, 1.0, 0.95),
    )

    plt.show()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Execute the demo."""
    _validate_configuration()

    print_calling_file()
    print_title("Correlated Two-Factor SBB Fan-Out Visualizer Demo")

    print()
    print(f"Series A: {SERIES_A_KIND}")
    print(f"Series B: {SERIES_B_KIND}")
    print(f"Target innovation correlation: {TARGET_CORRELATION:.3f}")
    print(f"Historical observations: {NUM_OBS:,}")
    print(f"SBB simulations: {N_SIMULATIONS:,}")
    print(f"SBB horizon: {HORIZON}")
    print(f"Expected block size: {EXPECTED_BLOCK_SIZE:.1f}")
    print()

    plot_two_factor_fanout()

    print_success_message(
        "Correlated Two-Factor SBB Fan-Out Visualizer Demo Completed Successfully!"
    )


if __name__ == "__main__":
    main()
