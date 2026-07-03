# src/riskquant_lab/plotting/credit_var.py
"""Plotting utilities for RiskQuant CreditVar solutions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from riskquant_lab.plotting.plot_utils import (
    fit_kernel_distribution,
    format_tick_labels,
)


def plot_credit_loss_distribution(
    *,
    # Histogram data
    hist_bin_edges: np.ndarray | list[float],
    hist_counts: np.ndarray | list[int],
    # Statistics
    expected_loss: float,
    var_levels: dict[str, float],
    # Losses for KDE plotting (optional)
    losses: np.ndarray | None = None,
    # Metadata (optional)
    model_name: str | None = None,
    n_obligors: int | None = None,
    n_scenarios: int | None = None,
    # Plotting options
    figsize: tuple[float, float] = (12, 6),
    kde_display_percentile: float = 99.90,
) -> tuple[Figure, Axes]:
    """
    Plot the simulated credit loss distribution.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    hist_bin_edges = np.asarray(hist_bin_edges, dtype=np.float64)
    hist_counts = np.asarray(hist_counts, dtype=np.int64)
    bin_widths = np.diff(hist_bin_edges) * 0.975
    bin_centers = (hist_bin_edges[:-1] + hist_bin_edges[1:]) / 2

    ax.bar(
        bin_centers,  # Use centers instead of left edges
        hist_counts,
        width=bin_widths,
        align="center",  # Center bars on bin_centers
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.75,
        label="Simulated Losses",
    )

    # ------------------------------------------------------------------
    # KDE (scaled into histogram count units)
    # ------------------------------------------------------------------

    kde_x_values, kde_density = None, None

    if losses is not None:
        kde_result = fit_kernel_distribution(
            losses,
            display_percentile=kde_display_percentile,
        )

        if kde_result is not None:
            kde_x_values, kde_density = kde_result

            n = n_scenarios if n_scenarios is not None else hist_counts.sum()
            bin_width = hist_bin_edges[1] - hist_bin_edges[0]

            kde_counts = np.asarray(kde_density) * n * bin_width

            ax.plot(
                kde_x_values,
                kde_counts,
                color="#DD8452",
                linewidth=2.5,
                label="Kernel Density Estimate",
            )

    # ------------------------------------------------------------------
    # Expected Loss
    # ------------------------------------------------------------------

    ax.axvline(
        expected_loss,
        color="black",
        linewidth=2,
        label=f"Expected Loss ({expected_loss:,.0f})",
    )

    # ------------------------------------------------------------------
    # VaR levels
    # ------------------------------------------------------------------

    var_colors = (
        "#83EA9B",
        "#7A9BD1",
        "#EA4D52",
        "#9889CA",
    )

    for (confidence, value), color in zip(var_levels.items(), var_colors, strict=True):
        ax.axvline(
            value,
            color=color,
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label=f"VaR {confidence} ({value:,.0f})",
        )

    max_var = max(var_levels.values())

    # ------------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------------

    xmin = hist_bin_edges[0]
    xmax = max(
        hist_bin_edges[-1],
        expected_loss,
        *var_levels.values(),
    )

    if kde_x_values is not None:
        xmin = min(xmin, float(np.min(kde_x_values)))
        xmax = max(xmax, float(np.max(kde_x_values)))

    xmax = min(max_var * 1.10, xmax)

    padding = 0.01 * (xmax - xmin)

    ax.set_xlim(
        max(xmin, 0.0) - padding,
        xmax + padding,
    )

    ax = format_tick_labels(ax=ax, axis="x", style="millions")
    ax = format_tick_labels(ax=ax, axis="y", style="auto")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    title_parts = []
    title_parts.append("Credit Loss Distribution")
    if model_name is not None:
        title_parts[-1] += f" (Model: {model_name.replace('_', ' ').upper()})"

    subtitle_parts = []
    if n_obligors is not None:
        subtitle_parts.append(f"{n_obligors:,} obligors")
    if n_scenarios is not None:
        subtitle_parts.append(f"{n_scenarios:,} scenarios")

    if subtitle_parts:
        title_parts.append(" • ".join(subtitle_parts))

    ax.set_title(
        "\n".join(title_parts),
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Credit Loss")
    ax.set_ylabel("Scenario Count")

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.30,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()
    plt.show()

    return fig, ax


# ======================================================================
# Multi-model comparison overlays
# ======================================================================


def _model_colors(n: int) -> list:
    """A distinct colour per model (tab20 supports up to 20 series)."""
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def plot_loss_distribution_comparison(
    model_losses: dict[str, np.ndarray],
    *,
    display_percentile: float = 99.9,
    log_y: bool = False,
    linthresh_y: float = 1e-7,
    figsize: tuple[float, float] = (13, 7),
    title: str | None = None,
    ax: Axes | None = None,
    smooth_kde: bool = True,
) -> tuple[Figure, Axes]:
    """Overlay reflection-corrected loss-distribution KDEs for many models.

    ``model_losses`` maps a model label to its per-scenario portfolio-loss vector
    (e.g. from ``simulate_portfolio_losses``). Each curve is a probability density,
    so the models are directly comparable on one axis. Degenerate distributions
    (constant losses, e.g. ``forced_default``) are skipped -- they have no smooth
    density and would otherwise raise.
    """

    # 1. Determine if the user supplied the axes
    ax_provided = ax is not None

    if ax_provided:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    assert isinstance(fig, Figure), "fig from Axes must be a matplotlib Figure"

    colors = _model_colors(len(model_losses))
    for color, (label, losses) in zip(colors, model_losses.items(), strict=True):
        fitted = fit_kernel_distribution(
            np.asarray(losses, dtype=np.float64),
            display_percentile=display_percentile,
            smooth_kde=smooth_kde,
        )
        if fitted is None:
            continue
        x_values, density = fitted
        ax.plot(x_values, density, label=label, color=color, linewidth=1.8)

    if log_y:
        ax.set_yscale("symlog", linthresh=linthresh_y)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Portfolio Credit Loss")
    ax.set_ylabel("Density")
    ax.set_title(title or "Simulated Credit Loss Distribution by Model (KDE)")
    format_tick_labels(ax, axis="x", style="millions")
    ax.legend(fontsize=8, ncol=1)

    # 2. Only apply layout adjustments if we created the figure ourselves
    if not ax_provided:
        fig.tight_layout()

    # plt.show() has been removed. The caller handles rendering.

    return fig, ax


def plot_ecdf_comparison(
    model_losses: dict[str, np.ndarray],
    *,
    display_percentile: float = 99.9,
    n_points: int = 512,
    figsize: tuple[float, float] = (13, 7),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Overlay empirical CDFs of portfolio loss for many models.

    Drawn as the inverse CDF on a quantile grid (light and smooth regardless of
    scenario count). The x-axis is clipped to ``display_percentile`` across models
    so the tail, where the models separate, stays legible.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = _model_colors(len(model_losses))
    quantiles = np.linspace(0.0, 1.0, n_points)

    x_hi = 1.0
    for color, (label, losses) in zip(colors, model_losses.items(), strict=True):
        arr = np.asarray(losses, dtype=np.float64)
        x_values = np.quantile(arr, quantiles)
        ax.plot(x_values, quantiles, label=label, color=color, linewidth=1.6)
        x_hi = max(x_hi, float(np.percentile(arr, display_percentile)))

    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Portfolio Credit Loss")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title or "Empirical CDF of Portfolio Credit Loss by Model")
    format_tick_labels(ax, axis="x", style="auto")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plt.show()

    return fig, ax


def plot_var_es_comparison(
    labels: list[str],
    var_values: list[float],
    es_values: list[float],
    *,
    alpha_label: str,
    figsize: tuple[float, float] = (13, 7),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Grouped bar chart of VaR and ES by model at one confidence level."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(
        x - width / 2, var_values, width, label=f"VaR {alpha_label}", color="#4C72B0"
    )
    ax.bar(x + width / 2, es_values, width, label=f"ES {alpha_label}", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Credit Loss")
    ax.set_title(title or f"VaR & ES by Model at {alpha_label}")
    format_tick_labels(ax, axis="y", style="auto")
    ax.legend()
    fig.tight_layout()
    plt.show()

    return fig, ax
