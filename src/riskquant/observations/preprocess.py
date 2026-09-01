# src/riskquant/observations/preprocess.py

"""
Sample-altering preprocessing policy. EDGE TIER.

Every function here changes the empirical sample and therefore changes every
statistic subsequently estimated from it. That is why these operations were
removed from the dependence estimators: an estimator that silently drops
observations does not estimate the dependence of the data it was given.

Concretely, dropping the rows where any variable printed a zero return removes
exactly the illiquid, stale-quote days. Those days carry the low-volatility,
low-covariance part of the sample, so removing them biases every pairwise
correlation upward. Imputing a missing return with the column mean inserts a
synthetic observation at the centre of the marginal distribution, which pulls
the estimate toward zero. Neither is wrong as a desk policy; both are wrong as
a hidden default.

Contract
--------
Inputs may contain NaN — that is the point of this module. Outputs are
``(K, S')`` float64 C-contiguous, and every function returns the retained
observation index so the caller can realign dates, weights, or a market-state
series. Nothing here is applied automatically anywhere in the library.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from riskquant.observations.orientation import ObservationOrientation


class ZeroFilterMode(Enum):
    """
    Which observations a zero-return filter removes.

    Attributes
    ----------
    DROP_IF_ANY_ZERO
        Drop the observation when at least one variable is zero. Aggressive:
        with K variables each independently zero with probability p, the
        expected retention is ``(1 - p)^K``, so at K = 50 and p = 0.05 only
        eight per cent of the sample survives.

    DROP_IF_ALL_ZERO
        Drop the observation only when every variable is zero — a genuinely
        empty market day (holiday, no fixings). Far less biasing.
    """

    DROP_IF_ANY_ZERO = 0
    DROP_IF_ALL_ZERO = 1


class ImputationStrategy(Enum):
    """
    How missing observations are replaced.

    Attributes
    ----------
    CONSTANT
        Replace with a caller-supplied constant.

    VARIABLE_MEAN
        Replace with the variable's mean over its observed values.

    VARIABLE_MEDIAN
        Replace with the variable's median over its observed values.

    K_NEAREST_NEIGHBOURS
        ``sklearn.impute.KNNImputer`` over the observation cross-section.
        Creates synthetic observations that borrow the dependence structure of
        the complete rows, so it can materially inflate estimated dependence.
    """

    CONSTANT = 0
    VARIABLE_MEAN = 1
    VARIABLE_MEDIAN = 2
    K_NEAREST_NEIGHBOURS = 3


def _to_variables_by_observations(
    observations: NDArray[np.generic],
    *,
    orientation: ObservationOrientation,
) -> NDArray[np.float64]:
    """Orient to ``(K, S)`` float64 C-contiguous without a finiteness check."""
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2-dimensional; got ndim={observations.ndim}."
        )

    oriented = (
        observations
        if orientation is ObservationOrientation.VARIABLES_ON_ROWS
        else observations.T
    )

    return np.ascontiguousarray(oriented, dtype=np.float64)


def filter_incomplete_observations(
    observations: NDArray[np.generic],
    *,
    orientation: ObservationOrientation,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Complete-case selection: keep observations where every variable is finite.

    This is the only sample repair that leaves the marginal distributions of
    the retained observations untouched. It is still not free: it is unbiased
    only under missing-completely-at-random.

    Parameters
    ----------
    observations : NDArray
        2-D, may contain NaN or Inf.

    orientation : ObservationOrientation
        Layout of ``observations``.

    Returns
    -------
    retained : Float64Matrix
        ``(K, S')`` C-contiguous.

    retained_index : NDArray[int64]
        ``(S',)`` positions of the kept observations in the input.
    """
    matrix = _to_variables_by_observations(observations, orientation=orientation)

    keep_mask = np.isfinite(matrix).all(axis=0)
    retained_index = np.flatnonzero(keep_mask)

    return np.ascontiguousarray(matrix[:, retained_index]), retained_index


def filter_zero_observations(
    observations: NDArray[np.generic],
    *,
    orientation: ObservationOrientation,
    mode: ZeroFilterMode,
    zero_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Drop observations according to an explicit zero-return rule.

    Parameters
    ----------
    observations : NDArray
        2-D. Non-finite cells are treated as non-zero.

    orientation : ObservationOrientation
        Layout of ``observations``.

    mode : ZeroFilterMode
        Selection rule.

    zero_tolerance : float
        Absolute dead zone: ``|x| <= zero_tolerance`` counts as zero. Must be
        finite and non-negative. Pass ``0.0`` for exact zeros — the value is
        used as given, never coerced.

    Returns
    -------
    retained : Float64Matrix
        ``(K, S')`` C-contiguous.

    retained_index : NDArray[int64]
        ``(S',)`` positions of the kept observations in the input.

    Notes
    -----
    The prototype coerced a caller-supplied ``zero_epsilon`` of ``0.0`` to
    ``1e-12`` through a falsiness test, so asking for exact-zero matching
    silently produced a dead zone instead. No coercion happens here.
    """
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError(
            f"zero_tolerance must be finite and non-negative; got {zero_tolerance!r}."
        )

    matrix = _to_variables_by_observations(observations, orientation=orientation)

    is_zero = np.abs(matrix) <= zero_tolerance

    if mode is ZeroFilterMode.DROP_IF_ANY_ZERO:
        keep_mask = ~is_zero.any(axis=0)
    elif mode is ZeroFilterMode.DROP_IF_ALL_ZERO:
        keep_mask = ~is_zero.all(axis=0)
    else:
        raise ValueError(f"Unsupported ZeroFilterMode: {mode!r}.")

    retained_index = np.flatnonzero(keep_mask)

    return np.ascontiguousarray(matrix[:, retained_index]), retained_index


def apply_zero_dead_zone(
    observations: NDArray[np.generic],
    *,
    orientation: ObservationOrientation,
    zero_tolerance: float,
) -> NDArray[np.float64]:
    """
    Snap small magnitudes to exact zero.

    Creates ties. Rank-based estimators handle ties through average ranks and
    tau-b tie corrections, but the KSG mutual-information estimator degrades
    badly on tie-heavy data, so applying a dead zone before mutual information
    is a methodology decision, not a cleanup step.

    Parameters
    ----------
    observations : NDArray
        2-D.

    orientation : ObservationOrientation
        Layout of ``observations``.

    zero_tolerance : float
        Absolute dead zone, finite and non-negative.

    Returns
    -------
    snapped : Float64Matrix
        ``(K, S)`` C-contiguous copy.
    """
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError(
            f"zero_tolerance must be finite and non-negative; got {zero_tolerance!r}."
        )

    matrix = _to_variables_by_observations(observations, orientation=orientation)

    np.copyto(matrix, 0.0, where=np.abs(matrix) <= zero_tolerance)

    return matrix


def impute_observations(
    observations: NDArray[np.generic],
    *,
    orientation: ObservationOrientation,
    strategy: ImputationStrategy,
    constant_value: float = 0.0,
    neighbour_count: int = 5,
) -> NDArray[np.float64]:
    """
    Replace non-finite cells with synthetic values.

    Parameters
    ----------
    observations : NDArray
        2-D, may contain NaN.

    orientation : ObservationOrientation
        Layout of ``observations``.

    strategy : ImputationStrategy
        Replacement rule.

    constant_value : float, default 0.0
        Used by ``CONSTANT`` only. Must be finite.

    neighbour_count : int, default 5
        Used by ``K_NEAREST_NEIGHBOURS`` only. Must be at least one.

    Returns
    -------
    imputed : Float64Matrix
        ``(K, S)`` C-contiguous, finite.

    Raises
    ------
    ValueError
        If a variable is entirely missing under a marginal strategy, or if the
        result still contains a non-finite cell.

    Notes
    -----
    ``VARIABLE_MEAN`` inserts the marginal centre, which mechanically shrinks
    the estimated covariance of that variable with every other. Infinities are
    treated as missing.
    """
    matrix = _to_variables_by_observations(observations, orientation=orientation)
    missing = ~np.isfinite(matrix)

    if not missing.any():
        return matrix

    if strategy is ImputationStrategy.CONSTANT:
        if not np.isfinite(constant_value):
            raise ValueError(
                f"constant_value must be finite; got {constant_value!r}."
            )
        matrix[missing] = constant_value
        return matrix

    if strategy in (
        ImputationStrategy.VARIABLE_MEAN,
        ImputationStrategy.VARIABLE_MEDIAN,
    ):
        fully_missing = np.flatnonzero(missing.all(axis=1))
        if fully_missing.size:
            raise ValueError(
                "Cannot impute a marginal statistic for a fully missing "
                f"variable. Offending variable_index values: {fully_missing.tolist()}."
            )

        observed = np.where(missing, np.nan, matrix)
        replacement = (
            np.nanmean(observed, axis=1)
            if strategy is ImputationStrategy.VARIABLE_MEAN
            else np.nanmedian(observed, axis=1)
        )
        row_index = np.nonzero(missing)[0]
        matrix[missing] = replacement[row_index]
        return matrix

    if strategy is ImputationStrategy.K_NEAREST_NEIGHBOURS:
        if neighbour_count < 1:
            raise ValueError(
                f"neighbour_count must be >= 1; got {neighbour_count}."
            )

        from sklearn.impute import KNNImputer

        imputer = KNNImputer(
            missing_values=np.nan,
            n_neighbors=neighbour_count,
            weights="uniform",
        )
        # KNNImputer expects (n_samples, n_features) = observations x variables.
        filled = imputer.fit_transform(np.where(missing, np.nan, matrix).T)
        result = np.ascontiguousarray(filled.T, dtype=np.float64)

        if not np.isfinite(result).all():
            raise ValueError(
                "K-nearest-neighbour imputation left non-finite cells; a "
                "variable or observation is too sparse to impute."
            )
        return result

    raise ValueError(f"Unsupported ImputationStrategy: {strategy!r}.")
