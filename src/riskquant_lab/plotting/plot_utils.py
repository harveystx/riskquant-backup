# src/riskquant_lab/plotting/plot_utils.py
"""Plotting utilities for RiskQuant solutions."""

from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, ScalarFormatter

TICK_LABEL_STYLES = [
    "comma",
    "scientific",
    "millions",
    "billions",
    "percent",
    "currency_usd",
    "auto",
]

TICK_LABEL_STYLE = Literal[
    "comma",
    "scientific",
    "millions",
    "billions",
    "percent",
    "currency_usd",
    "auto",
]


# KDE shape and bandwidth are stable well below this; cost is O(n_points * n_data),
# so capping the data is what keeps the fit off the engine's hot path.
_KDE_MIN_GRID = 1.0


def fit_kernel_distribution(
    losses: np.ndarray,
    *,
    n_points: int = 250,
    display_percentile: float = 99.9,
    smooth_kde: bool = True,
    smooth_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Smoothed curve of the portfolio-loss distribution for display, or None.

    Returned density is a probability density (PDF), not histogram counts.

    A reflection-corrected Gaussian KDE over the non-negative losses:

    * **Reflection** about 0 stops kernel mass leaking below the hard zero bound:
      fit on ``[L, -L]`` and take ``2 * f(x)`` for ``x >= 0``.
    * **Robust Silverman bandwidth** from the original sample (not the reflected
      set, whose inflated variance oversmooths), via ``min(std, IQR/1.349)`` so the
      long right tail / zero spike don't blow up the width.
    * **Grid spans ``[0, p_display]``** (matched to ``loss_histogram``) so a single
      tail outlier can't compress the visible mass and the two overlay on one axis.

    Returns ``None`` for a degenerate (constant / zero-variance) distribution --
    e.g. ``forced_default`` or a no-default run -- where a smooth density is
    meaningless and ``gaussian_kde`` would raise; the histogram still shows the
    point mass. The curve is a display *shape*, not a normalised density
    (truncated at ``p_display``; reflection carries a small known boundary bias).
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import gaussian_kde

    losses = np.ascontiguousarray(losses, dtype=np.float64)
    n = losses.size
    sigma = float(np.std(losses))
    if n < 2 or sigma <= 0.0:
        return None

    q75, q25, q_display = np.percentile(losses, [75.0, 25.0, display_percentile])
    iqr = float(q75 - q25)
    scale = min(sigma, iqr / 1.349) if iqr > 0.0 else sigma
    scale = max(scale, np.finfo(float).eps)
    h = 0.9 * scale * losses.size ** (-0.2)  # Silverman rule-of-thumb, 1D

    reflected = np.concatenate([losses, -losses])

    mu = losses.mean()
    reflected_std = np.sqrt(sigma**2 + mu**2)

    # Set scipy's factor so the realised kernel bandwidth equals h in data units
    # (covariance = factor**2 * cov(reflected)).
    # factor = h / float(np.sqrt(np.cov(reflected)))
    factor = h / reflected_std
    kde = gaussian_kde(reflected, bw_method=factor)

    # x_hi = max(float(q_display), _KDE_MIN_GRID)
    # x_grid = np.linspace(0.0, x_hi, n_points)
    # density = 2.0 * kde.evaluate(points=x_grid)

    x_hi = max(float(q_display), _KDE_MIN_GRID)

    # Improvement 1: Use log-spaced grid in the tail for better resolution
    # where density is small and changing rapidly
    if x_hi > 0:
        # Start with linear spacing up to some threshold, then log-space
        threshold = max(np.percentile(losses, 95), _KDE_MIN_GRID)

        if threshold < x_hi:
            n_linear = int(0.6 * n_points)  # 60% linear, 40% log
            n_log = n_points - n_linear

            # Linear grid for the bulk of the distribution
            linear_grid = np.linspace(0.0, threshold, n_linear)

            # Log-spaced grid for the tail
            log_grid = np.geomspace(
                max(threshold, 1e-10),  # Avoid log(0)
                x_hi,
                n_log,
            )

            # Combine and remove duplicates
            x_grid = np.unique(np.concatenate([linear_grid, log_grid]))
        else:
            x_grid = np.linspace(0.0, x_hi, n_points)
    else:
        x_grid = np.linspace(0.0, x_hi, n_points)

    # Improvement 2: Evaluate with higher numerical precision
    density = 2.0 * kde.evaluate(x_grid)

    # Improvement 3: Clip tiny values to avoid log-scale artifacts
    # but preserve the shape
    density = np.maximum(density, 1e-300)  # Avoid exact zeros

    if smooth_kde:
        density = gaussian_filter1d(density, sigma=smooth_sigma)

    return x_grid, density


def format_tick_labels(
    ax: Axes,
    *,
    axis: Literal["x", "y"] = "y",
    style: TICK_LABEL_STYLE = "comma",
    percent_is_fraction: bool = False,
) -> Axes:
    """
    Format x or y tick labels consistently.

    Parameters
    ----------
    ax : Axes
    axis : Literal["x", "y"]
        'x' or 'y'
    style : TICK_LABEL_STYLE
        One of TICK_LABEL_STYLES
    percent_is_fraction : bool
        If True, converts 0.5 -> 50%
    """

    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    if style not in TICK_LABEL_STYLES:
        raise ValueError(f"style must be one of {TICK_LABEL_STYLES}")

    target = ax.xaxis if axis == "x" else ax.yaxis

    def fmt_commas(x, _):
        return f"{x:,.0f}"

    def fmt_millions(x, _):
        return f"{x / 1e6:.2f}M"

    def fmt_billions(x, _):
        return f"{x / 1e9:.2f}B"

    def fmt_currency_usd(x, _):
        return f"${x:,.0f}"

    def fmt_percent(x, _):
        if percent_is_fraction:
            x = x * 100
        return f"{x:.1f}%"

    def fmt_auto(x, _):
        ax_abs = abs(x)
        if ax_abs >= 1e9:
            return f"{x / 1e9:.2f}B"
        if ax_abs >= 1e6:
            return f"{x / 1e6:.2f}M"
        if ax_abs >= 1e3:
            return f"{x:,.0f}"
        return f"{x:.2f}"

    formatters = {
        "comma": fmt_commas,
        "millions": fmt_millions,
        "billions": fmt_billions,
        "percent": fmt_percent,
        "currency_usd": fmt_currency_usd,
        "auto": fmt_auto,
    }

    if style == "scientific":
        sf = ScalarFormatter(useMathText=True)
        sf.set_powerlimits((0, 0))
        target.set_major_formatter(sf)
    else:
        if style not in formatters:
            raise ValueError(f"Unknown style: {style}")
        target.set_major_formatter(FuncFormatter(formatters[style]))

    target.axes.grid(True, axis=axis, alpha=0.3, linestyle="--")

    return ax
