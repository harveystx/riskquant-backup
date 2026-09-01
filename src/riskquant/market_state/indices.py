# src/riskquant/market_state/indices.py

"""
Cross-sectional market-state indices.

These are not dependence estimators and do not belong in one. They collapse a
``(K, S)`` observation matrix into a ``(S,)`` series describing how unusual each
observation date was across the variable cross-section.

Naming correction
-----------------
The prototype's ``robust_market_turbulence_index`` is **not** the Kritzman-Li
turbulence index. Kritzman-Li turbulence is the Mahalanobis distance
``sqrt((x - mu)' Sigma^-1 (x - mu))``, whose entire content is the inverse
covariance: it flags dates on which variables moved in historically unusual
*combinations*, even at ordinary magnitudes. The prototype computes a
root-mean-square robust z-score, which uses only the marginal scales and is
blind to the correlation structure. Two dates with identical marginal
magnitudes score identically whether the moves were aligned or opposed. The
function is renamed :func:`robust_rms_z_index` to say what it computes. The
Mahalanobis index is not implemented here; adding it is a separate, deliberate
piece of work because it needs a covariance-estimation policy of its own.

Missing-data correction
-----------------------
The prototype used pandas reductions that skip NaN, so each date's index was an
average over however many variables happened to have data — a date with two
observed variables and a date with fifty produced comparable-looking numbers
from incomparable samples. These functions require a fully finite canonical
matrix and reject anything else. Use
:mod:`riskquant.observations.preprocess` to decide, explicitly, what to do
about gaps.

Common contract
---------------
Input
    ``(K, S)`` canonical observation matrix, finite, at least two variables.
    Variables with zero dispersion are rejected rather than silently converted
    to NaN.

Output
    ``(S,)`` float64 C-contiguous, finite, non-negative.

Normalisation
    Every index divides by a per-variable scale and does **not** mean-centre
    the numerator, matching the prototype: for a return series the meaningful
    notion of a shock is displacement from zero.
"""

from __future__ import annotations

import numpy as np

from riskquant.constants import (
    MAD_TO_SIGMA_NORMAL,
    MARKET_STATE_DEFAULT_DDOF,
)
from riskquant.observations.arrays import Float64Matrix, Float64Vector


def _reject_zero_scale(scale: Float64Vector, *, scale_name: str) -> None:
    """Reject variables whose dispersion estimate is exactly zero."""
    degenerate = np.flatnonzero(scale == 0.0)
    if degenerate.size:
        raise ValueError(
            f"{scale_name} is zero for variable_index values "
            f"{degenerate.tolist()}; the index is undefined. The prototype "
            "replaced these with NaN and propagated them silently."
        )


def robust_scale_mad(canonical: Float64Matrix) -> Float64Vector:
    """
    Per-variable robust scale: ``1.4826 * median(|x - median(x)|)``.

    The constant makes the estimator consistent for the standard deviation
    under normality (it is ``1 / Phi^-1(0.75)``). Under fat tails it is
    deliberately smaller than the sample standard deviation, which is the point:
    a few large moves do not inflate the yardstick used to judge them.

    Breakdown point 50 per cent. Returns exactly zero when more than half the
    observations of a variable are identical — common for a stale or heavily
    zero-inflated series, and rejected downstream rather than hidden.

    Returns
    -------
    scale : Float64Vector
        ``(K,)`` float64.
    """
    median = np.median(canonical, axis=1, keepdims=True)
    scale: Float64Vector = MAD_TO_SIGMA_NORMAL * np.median(
        np.abs(canonical - median), axis=1
    )
    return scale


def _standardised(
    canonical: Float64Matrix,
    scale: Float64Vector,
) -> Float64Matrix:
    """Scale-normalised matrix, without mean centring."""
    return canonical / scale[:, None]


def robust_rms_z_index(canonical: Float64Matrix) -> Float64Vector:
    """
    Root-mean-square robust z-score across variables.

    ``index_s = sqrt( mean_k ( (x_ks / mad_scale_k)^2 ) )``

    Uses the MAD scale, so a single extreme date does not raise the yardstick
    that judges it. Ignores the correlation structure entirely — see the module
    docstring.

    Parameters
    ----------
    canonical : Float64Matrix
        ``(K, S)``.

    Returns
    -------
    index : Float64Vector
        ``(S,)``, non-negative.
    """
    scale = robust_scale_mad(canonical)
    _reject_zero_scale(scale, scale_name="MAD robust scale")

    standardised = _standardised(canonical, scale)
    index: Float64Vector = np.ascontiguousarray(
        np.sqrt((standardised * standardised).mean(axis=0))
    )
    return index


def volatility_normalised_shock_index(
    canonical: Float64Matrix,
    *,
    ddof: int = MARKET_STATE_DEFAULT_DDOF,
) -> Float64Vector:
    """
    Root-mean-square z-score using the sample standard deviation.

    ``index_s = sqrt( mean_k ( (x_ks / sigma_k)^2 ) )``

    Identical in form to :func:`robust_rms_z_index` but with a non-robust
    scale, so it is the more conservative reading during a volatility spike:
    ``sigma`` already contains the spike, which damps the index.

    Parameters
    ----------
    canonical : Float64Matrix
        ``(K, S)``.

    ddof : int, default 1
        Degrees-of-freedom correction. Exposed because the prototype inherited
        the pandas default of one for these three functions while the MAD
        variant used a median-based scale, an inconsistency that was invisible
        at the call site.

    Returns
    -------
    index : Float64Vector
        ``(S,)``, non-negative.
    """
    scale = np.std(canonical, axis=1, ddof=ddof)
    _reject_zero_scale(scale, scale_name="Sample standard deviation")

    standardised = _standardised(canonical, scale)
    index: Float64Vector = np.ascontiguousarray(
        np.sqrt((standardised * standardised).mean(axis=0))
    )
    return index


def volatility_normalised_directional_index(
    canonical: Float64Matrix,
    *,
    ddof: int = MARKET_STATE_DEFAULT_DDOF,
) -> Float64Vector:
    """
    Absolute mean z-score across variables — a common-direction move.

    ``index_s = | mean_k ( x_ks / sigma_k ) |``

    Large only when the cross-section moved together. A date on which half the
    variables rose and half fell by the same normalised amount scores zero here
    while scoring high on :func:`volatility_normalised_shock_index`; comparing
    the two separates a directional market move from a dispersion event.

    Parameters
    ----------
    canonical : Float64Matrix
        ``(K, S)``.

    ddof : int, default 1
        Degrees-of-freedom correction for the per-variable scale.

    Returns
    -------
    index : Float64Vector
        ``(S,)``, non-negative.
    """
    scale = np.std(canonical, axis=1, ddof=ddof)
    _reject_zero_scale(scale, scale_name="Sample standard deviation")

    standardised = _standardised(canonical, scale)
    index: Float64Vector = np.ascontiguousarray(np.abs(standardised.mean(axis=0)))
    return index


def volatility_normalised_dispersion_index(
    canonical: Float64Matrix,
    *,
    ddof: int = MARKET_STATE_DEFAULT_DDOF,
) -> Float64Vector:
    """
    Cross-sectional dispersion after removing the common direction.

    ``index_s = sqrt( mean_k ( (z_ks - mean_k z_ks)^2 ) )``

    The residual scatter once the average normalised move is stripped out.
    Divides by K, not ``K - 1``: the mean removed is the cross-sectional mean of
    the same K values, so this is the population second moment of that
    cross-section, not an estimate of a wider population's variance.

    Parameters
    ----------
    canonical : Float64Matrix
        ``(K, S)``.

    ddof : int, default 1
        Degrees-of-freedom correction for the per-variable time-series scale
        only. It does not affect the cross-sectional divisor.

    Returns
    -------
    index : Float64Vector
        ``(S,)``, non-negative.
    """
    scale = np.std(canonical, axis=1, ddof=ddof)
    _reject_zero_scale(scale, scale_name="Sample standard deviation")

    standardised = _standardised(canonical, scale)
    residual = standardised - standardised.mean(axis=0, keepdims=True)
    index: Float64Vector = np.ascontiguousarray(
        np.sqrt((residual * residual).mean(axis=0))
    )
    return index
