# scripts/demos/market_data/curves/demo_interp_strike_linear_flat_extrap_visualizer.py

"""
DEMO: Strike-Surface Interpolation (Linear + Flat Extrapolation)

This demo visualizes exactly how the
`interp_strike_linear_flat_extrap_numba_kernel`
behaves.

Concept
-------
For each trade:

    pillars = known strikes
    values  = volatility smile values at those strikes
    target  = simulated strike (or underlying level)

The kernel:

1. Finds the two surrounding strike pillars.
2. Interpolates linearly between their volatility values.
3. Applies flat extrapolation outside the pillar range.

Flat extrapolation means:

    target < min(strikes)
        -> use first smile value

    target > max(strikes)
        -> use last smile value

Visual Guide
------------
■ = Input strike pillars

— = Continuous interpolated smile produced by the kernel

X = Example Monte-Carlo scenario targets

The plots intentionally include targets both inside and outside
the pillar range so the extrapolation behaviour is obvious.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from riskquant.market_data.curves.interp_defs import (
    interp_strike_linear_flat_extrap_numba_kernel,
)


def _build_dense_surface(
    pillars: np.ndarray,
    base_smiles: list[list[float]],
    dense_targets: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the kernel on a dense grid so the interpolated
    smile can be drawn as a continuous line.
    """
    n_trades = pillars.shape[0]
    k_strikes = pillars.shape[1]
    s_dense = dense_targets.shape[1]

    dense_values = np.empty((n_trades, k_strikes, s_dense))

    for s in range(s_dense):
        for n in range(n_trades):
            dense_values[n, :, s] = base_smiles[n]

    dense_out = np.empty((n_trades, s_dense))

    interp_strike_linear_flat_extrap_numba_kernel(
        pillars,
        dense_values,
        dense_targets,
        dense_out,
    )

    return dense_out


def _print_results(
    pillars: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """Print scenario results to the console."""
    print()
    print("=" * 90)
    print("STRIKE INTERPOLATION RESULTS")
    print("=" * 90)

    for trade_idx in range(targets.shape[0]):
        strike_min = pillars[trade_idx, 0]
        strike_max = pillars[trade_idx, -1]

        print()
        print(
            f"Trade {trade_idx} | Strike Range = [{strike_min:.1f}, {strike_max:.1f}]"
        )

        for scenario_idx in range(targets.shape[1]):
            target = targets[trade_idx, scenario_idx]
            vol = out[trade_idx, scenario_idx]

            if target < strike_min:
                location = "LEFT FLAT EXTRAP"
            elif target > strike_max:
                location = "RIGHT FLAT EXTRAP"
            else:
                location = "INTERPOLATION"

            print(
                f"  Scenario {scenario_idx:<2d} | "
                f"Target = {target:7.2f} | "
                f"Vol = {vol:.6f} | "
                f"{location}"
            )


def run_strike_surface_demo() -> None:
    # ==================================================================
    # DEMO INPUTS
    # ==================================================================

    n_trades = 2
    k_strikes = 5
    s_scenarios = 5

    # Strike grids
    pillars = np.array(
        [
            [80.0, 90.0, 100.0, 110.0, 120.0],
            [50.0, 75.0, 100.0, 125.0, 150.0],
        ],
        dtype=np.float64,
    )

    # Volatility smiles
    base_smile_0 = [0.25, 0.22, 0.20, 0.23, 0.28]
    base_smile_1 = [0.40, 0.35, 0.30, 0.32, 0.38]

    base_smiles = [base_smile_0, base_smile_1]

    # Build values tensor expected by the kernel
    values = np.empty((n_trades, k_strikes, s_scenarios))

    for s in range(s_scenarios):
        values[0, :, s] = base_smile_0
        values[1, :, s] = base_smile_1

    # Example scenario targets
    #
    # Includes:
    #   - left extrapolation
    #   - interpolation
    #   - exact pillar hits
    #   - right extrapolation
    targets = np.array(
        [
            [70.0, 85.0, 100.0, 115.0, 130.0],
            [40.0, 62.5, 100.0, 112.5, 160.0],
        ],
        dtype=np.float64,
    )

    # ==================================================================
    # RUN KERNEL
    # ==================================================================

    out = np.empty((n_trades, s_scenarios))

    interp_strike_linear_flat_extrap_numba_kernel(
        pillars,
        values,
        targets,
        out,
    )

    _print_results(
        pillars=pillars,
        targets=targets,
        out=out,
    )

    # ==================================================================
    # DENSE EVALUATION FOR PLOTTING
    # ==================================================================

    s_dense = 500

    dense_targets_1d = np.linspace(40.0, 170.0, s_dense)

    dense_targets = np.vstack([dense_targets_1d for _ in range(n_trades)])

    dense_out = _build_dense_surface(
        pillars=pillars,
        base_smiles=base_smiles,
        dense_targets=dense_targets,
    )

    # ==================================================================
    # PLOTTING
    # ==================================================================

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 7),
    )

    fig.suptitle(
        "Strike Surface Interpolation\nLinear Interpolation + Flat Extrapolation",
        fontsize=16,
    )

    trade_titles = [
        "Trade 0 — Narrow Strike Grid",
        "Trade 1 — Wide Strike Grid",
    ]

    for trade_idx in range(n_trades):
        ax = axs[trade_idx]

        strike_min = pillars[trade_idx, 0]
        strike_max = pillars[trade_idx, -1]

        # --------------------------------------------------------------
        # Continuous smile
        # --------------------------------------------------------------

        ax.plot(
            dense_targets[trade_idx],
            dense_out[trade_idx],
            linewidth=2,
            label="Kernel Output",
        )

        # --------------------------------------------------------------
        # Input pillars
        # --------------------------------------------------------------

        ax.scatter(
            pillars[trade_idx],
            values[trade_idx, :, 0],
            marker="s",
            s=100,
            zorder=5,
            label="Input Pillars",
        )

        # --------------------------------------------------------------
        # Extrapolation boundaries
        # --------------------------------------------------------------

        ax.axvline(
            strike_min,
            linestyle="--",
            alpha=0.5,
            label="Pillar Boundary",
        )

        ax.axvline(
            strike_max,
            linestyle="--",
            alpha=0.5,
        )

        # --------------------------------------------------------------
        # Scenario locations
        # --------------------------------------------------------------

        for scenario_idx in range(s_scenarios):
            ax.scatter(
                targets[trade_idx, scenario_idx],
                out[trade_idx, scenario_idx],
                marker="X",
                s=140,
                zorder=10,
            )

            ax.annotate(
                f"S{scenario_idx}",
                (
                    targets[trade_idx, scenario_idx],
                    out[trade_idx, scenario_idx],
                ),
                xytext=(5, 5),
                textcoords="offset points",
            )

        # --------------------------------------------------------------
        # Region labels
        # --------------------------------------------------------------

        y_mid = (np.min(dense_out[trade_idx]) + np.max(dense_out[trade_idx])) / 2.0

        ax.text(
            strike_min - 15,
            y_mid,
            "Flat\nExtrap",
            ha="center",
            va="center",
        )

        ax.text(
            (strike_min + strike_max) / 2,
            y_mid,
            "Linear\nInterpolation",
            ha="center",
            va="center",
        )

        ax.text(
            strike_max + 15,
            y_mid,
            "Flat\nExtrap",
            ha="center",
            va="center",
        )

        # --------------------------------------------------------------
        # Cosmetics
        # --------------------------------------------------------------

        ax.set_title(trade_titles[trade_idx])
        ax.set_xlabel("Target Strike / Simulated Underlying")
        ax.set_ylabel("Implied Volatility")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_strike_surface_demo()
