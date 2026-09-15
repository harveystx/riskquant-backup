"""
Synthetic level generators for testing, demonstrations, and diagnostics.

The module provides two APIs:

1. Backwards-compatible standalone generators.

   Existing callers can continue using functions such as
   ``create_random_walk_levels_series`` directly.

2. ``SyntheticLevelGenerator``.

   A stateful façade for generating multiple heterogeneous series with a
   consistent length and an optional common-factor dependence structure.

Design principles
-----------------
* Existing public standalone functions remain backwards compatible.
* The class API always returns exactly ``num_obs`` levels.
* The class API uses an initial level at observation zero, so all generated
  series share the same time grid and innovation alignment.
* Dependence is introduced at the innovation level before process dynamics
  are applied.
* Process-specific characteristics such as AR dynamics, GARCH clustering,
  mean reversion, seasonality, jumps, structural breaks, and stale periods
  are therefore preserved.
* Fail fast on invalid configuration rather than silently producing dubious
  synthetic data.

These generators are intended for controlled testing and demonstration
purposes. They are not intended to reproduce every empirical feature of
financial markets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, cast

import numpy as np

# ============================================================================
# DEFAULTS
# ============================================================================

_DEFAULT_RNG_SEED: Final[int] = 999_999

_DEFAULT_NUM_OBS: Final[int] = (
    22 * 12 * 5
) + 1  # Five years of daily data + initial level.

_DEFAULT_VOL_PA: Final[float] = 0.25
_DEFAULT_VOL_PA_LOW: Final[float] = 0.05
_DEFAULT_VOL_PA_HIGH: Final[float] = 0.50
_DEFAULT_DRIFT_PA: Final[float] = 0.010

_DEFAULT_BDAYS_PER_YEAR: Final[int] = 22 * 12

_DEFAULT_UPDATES_PER_YEAR_IN_ILLQUID_SERIES: Final[int] = 15

_DEFAULT_INIT_LEVEL: Final[float] = 100.0

_DEFAULT_PHI_AR1: Final[float] = 0.60
_DEFAULT_PHI_1_AR2: Final[float] = 0.0
_DEFAULT_PHI_2_AR2: Final[float] = 0.70

_DEFAULT_COMPLEX_SR_LEN_01_PROPORTION: Final[float] = 0.20
_DEFAULT_COMPLEX_SR_LEN_02_PROPORTION: Final[float] = 0.50
_DEFAULT_COMPLEX_SR_INJECTED_STALE_PERIODS: Final[int] = 10
_DEFAULT_COMPLEX_SR_INJECTED_STALE_LENGTH: Final[int] = 22

_DEFAULT_COMPLEX_SR_JUMP_SIZES: Final[tuple[float, ...]] = (
    -0.45,
    +0.35,
    -0.28,
    +0.39,
    -0.42,
    +0.40,
    -0.48,
    -0.50,
    +0.45,
)

_DEFAULT_COMPLEX_SR_DRIFT_PA_03: Final[float] = 0.10

_DEFAULT_GARCH_OMEGA: Final[float] = 1e-6
_DEFAULT_GARCH_ALPHA: Final[float] = 0.08
_DEFAULT_GARCH_BETA: Final[float] = 0.90

_DEFAULT_MR_SPEED: Final[float] = 3.0
_DEFAULT_MR_LONG_RUN_LEVEL: Final[float] = _DEFAULT_INIT_LEVEL

_DEFAULT_SEASONAL_AMPLITUDE_PA: Final[float] = 0.15
_DEFAULT_SEASONAL_PERIOD_DAYS: Final[int] = _DEFAULT_BDAYS_PER_YEAR

_DEFAULT_BREAK_SIZE: Final[float] = -0.35
_DEFAULT_BREAK_LOCATION_PROPORTION: Final[float] = 0.60

_DEFAULT_PRE_BREAK_VOL_PA: Final[float] = 0.10
_DEFAULT_POST_BREAK_VOL_PA: Final[float] = 0.50

_DEFAULT_PRE_BREAK_DRIFT_PA: Final[float] = 0.02
_DEFAULT_POST_BREAK_DRIFT_PA: Final[float] = -0.10


# ============================================================================
# PUBLIC TYPE DEFINITIONS
# ============================================================================

SyntheticSeriesKind = Literal[
    "illiquid_observed",
    "illiquid_latent",
    "random_walk",
    "ar1",
    "ar2",
    "complex",
    "garch",
    "mean_reverting",
    "seasonal",
    "structural_break",
]

SyntheticSeriesSpec = str | Mapping[str, object]

_SYNTHETIC_SERIES_KINDS: Final[tuple[SyntheticSeriesKind, ...]] = (
    "illiquid_observed",
    "illiquid_latent",
    "random_walk",
    "ar1",
    "ar2",
    "complex",
    "garch",
    "mean_reverting",
    "seasonal",
    "structural_break",
)

_SYNTHETIC_SERIES_KIND_SET: Final[frozenset[str]] = frozenset(_SYNTHETIC_SERIES_KINDS)


# ============================================================================
# GENERAL VALIDATION / HELPERS
# ============================================================================


def _validate_num_obs(num_obs: int) -> None:
    if isinstance(num_obs, bool) or not isinstance(num_obs, (int, np.integer)):
        raise TypeError(f"{num_obs=} must be an integer.")

    if num_obs < 1:
        raise ValueError(f"{num_obs=} must be at least 1.")


def _validate_num_obs_for_class(num_obs: int) -> None:
    _validate_num_obs(num_obs)

    if num_obs < 2:
        raise ValueError(f"{num_obs=} must be at least 2 for SyntheticLevelGenerator.")


def _validate_bdays_per_year(bdays_per_year: int) -> None:
    if isinstance(bdays_per_year, bool) or not isinstance(
        bdays_per_year, (int, np.integer)
    ):
        raise TypeError(f"{bdays_per_year=} must be an integer.")

    if bdays_per_year <= 0:
        raise ValueError(f"{bdays_per_year=} must be strictly positive.")


def _validate_positive_finite(value: float, *, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}={value!r} must be finite and strictly positive.")


def _validate_non_negative_finite(value: float, *, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name}={value!r} must be finite and non-negative.")


def _validate_finite(value: float, *, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name}={value!r} must be finite.")


def _validate_log_return_shocks(
    shocks: np.ndarray,
    *,
    expected_length: int,
) -> None:
    if shocks.ndim != 1:
        raise ValueError("Innovation shocks must be one-dimensional.")

    if len(shocks) != expected_length:
        raise ValueError(
            f"Innovation shocks have length {len(shocks)}; expected {expected_length}."
        )

    if not np.all(np.isfinite(shocks)):
        raise ValueError("Innovation shocks contain non-finite values.")


def _returns_to_levels(
    returns: np.ndarray,
    *,
    init_level: float,
    include_initial_level: bool,
) -> np.ndarray:
    """
    Convert log returns into strictly positive levels.

    Parameters
    ----------
    returns:
        Log returns.
    init_level:
        Level from which the process starts.
    include_initial_level:
        If true, the returned array starts with ``init_level``.
    """
    _validate_positive_finite(init_level, name="init_level")

    if returns.ndim != 1:
        raise ValueError("Returns must be one-dimensional.")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns contain non-finite values.")

    levels = init_level * np.exp(np.cumsum(returns))

    if not include_initial_level:
        return levels

    result = np.empty(len(levels) + 1, dtype=np.float64)
    result[0] = init_level
    result[1:] = levels
    return result


def _draw_standard_normal_shocks(
    rng: np.random.Generator,
    num_shocks: int,
) -> np.ndarray:
    if num_shocks < 0:
        raise ValueError(f"{num_shocks=} must be non-negative.")

    return np.asarray(
        rng.standard_normal(num_shocks),
        dtype=np.float64,
    )


# ============================================================================
# ILLIQUID SERIES
# ============================================================================


def _create_illiquid_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    updates_per_year: int,
    vol_pa: float,
    drift_pa: float,
    bdays_per_year: int,
    rng: np.random.Generator,
    include_initial_level: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create hidden and observed illiquid series from supplied innovations.
    """
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)

    if not isinstance(updates_per_year, (int, np.integer)):
        raise TypeError(f"{updates_per_year=} must be an integer.")

    if updates_per_year < 0 or updates_per_year > bdays_per_year:
        raise ValueError(
            f"{updates_per_year=} must satisfy "
            f"0 <= updates_per_year <= {bdays_per_year}."
        )

    _validate_non_negative_finite(vol_pa, name="vol_pa")
    _validate_finite(drift_pa, name="drift_pa")
    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    dt = 1.0 / bdays_per_year

    true_returns = shocks * vol_pa * np.sqrt(dt) + drift_pa * dt

    true_levels = _returns_to_levels(
        true_returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )

    if include_initial_level:
        is_observed = rng.random(expected_length) < (updates_per_year / bdays_per_year)
        is_observed[0] = True

        true_log_price = np.empty(expected_length + 1, dtype=np.float64)
        true_log_price[0] = 0.0
        true_log_price[1:] = np.cumsum(true_returns)

        observed_log_price = np.empty_like(true_log_price)
        last_known_price = true_log_price[0]
        observed_log_price[0] = last_known_price

        for t in range(1, expected_length + 1):
            if is_observed[t - 1]:
                last_known_price = true_log_price[t]
            observed_log_price[t] = last_known_price

        observed_levels = init_level * np.exp(observed_log_price)

        return true_levels, observed_levels

    is_observed = rng.random(expected_length) < (updates_per_year / bdays_per_year)
    is_observed[0] = True

    true_log_price = np.cumsum(true_returns)

    observed_log_price = np.empty_like(true_log_price)
    last_known_price = true_log_price[0]

    for t in range(expected_length):
        if is_observed[t]:
            last_known_price = true_log_price[t]
        observed_log_price[t] = last_known_price

    observed_levels = init_level * np.exp(observed_log_price)

    return true_levels, observed_levels


# ============================================================================
# AR(1)
# ============================================================================


def _create_ar1_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    phi: float,
    vol_pa: float,
    drift_pa: float,
    bdays_per_year: int,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)
    _validate_finite(phi, name="phi")
    _validate_non_negative_finite(vol_pa, name="vol_pa")
    _validate_finite(drift_pa, name="drift_pa")

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    if expected_length == 0:
        return np.array([init_level], dtype=np.float64)

    dt = 1.0 / bdays_per_year

    returns = np.empty(expected_length, dtype=np.float64)

    returns[0] = shocks[0] * vol_pa * np.sqrt(dt) + drift_pa * dt

    for t in range(1, expected_length):
        returns[t] = (
            phi * returns[t - 1] + shocks[t] * vol_pa * np.sqrt(dt) + drift_pa * dt
        )

    return _returns_to_levels(
        returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )


# ============================================================================
# AR(2)
# ============================================================================


def _create_ar2_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    phi_1: float,
    phi_2: float,
    vol_pa: float,
    drift_pa: float,
    bdays_per_year: int,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)
    _validate_finite(phi_1, name="phi_1")
    _validate_finite(phi_2, name="phi_2")
    _validate_non_negative_finite(vol_pa, name="vol_pa")
    _validate_finite(drift_pa, name="drift_pa")

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    if expected_length < 2:
        raise ValueError("AR(2) generation requires at least two innovations.")

    dt = 1.0 / bdays_per_year

    returns = np.empty(expected_length, dtype=np.float64)

    returns[0] = shocks[0] * vol_pa * np.sqrt(dt) + drift_pa * dt

    returns[1] = shocks[1] * vol_pa * np.sqrt(dt) + drift_pa * dt

    for t in range(2, expected_length):
        returns[t] = (
            phi_1 * returns[t - 1]
            + phi_2 * returns[t - 2]
            + shocks[t] * vol_pa * np.sqrt(dt)
            + drift_pa * dt
        )

    return _returns_to_levels(
        returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )


# ============================================================================
# RANDOM WALK
# ============================================================================


def _create_random_walk_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    vol_pa: float,
    drift_pa: float,
    bdays_per_year: int,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)
    _validate_non_negative_finite(vol_pa, name="vol_pa")
    _validate_finite(drift_pa, name="drift_pa")

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    dt = 1.0 / bdays_per_year

    returns = shocks * vol_pa * np.sqrt(dt) + drift_pa * dt

    return _returns_to_levels(
        returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )


# ============================================================================
# STALE PERIOD INJECTION
# ============================================================================


def inject_stale_periods_into_returns(
    returns: np.ndarray,
    *,
    n_stale: int = _DEFAULT_COMPLEX_SR_INJECTED_STALE_PERIODS,
    n_stale_len: int = _DEFAULT_COMPLEX_SR_INJECTED_STALE_LENGTH,
    rng: np.random.Generator,
) -> None:
    """
    Inject non-overlapping contiguous periods of zero returns in-place.

    The blocks are placed uniformly across the available free space subject
    to the non-overlap constraint.

    The implementation guarantees successful placement whenever the requested
    blocks fit in the supplied array:

        n_stale * n_stale_len <= len(returns)

    Parameters
    ----------
    returns:
        One-dimensional return array modified in-place.
    n_stale:
        Number of stale blocks to inject.
    n_stale_len:
        Length of each stale block.
    rng:
        Random-number generator controlling block placement.
    """
    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional.")

    if isinstance(n_stale, bool) or not isinstance(
        n_stale,
        (int, np.integer),
    ):
        raise TypeError(f"{n_stale=} must be an integer.")

    if isinstance(n_stale_len, bool) or not isinstance(
        n_stale_len,
        (int, np.integer),
    ):
        raise TypeError(f"{n_stale_len=} must be an integer.")

    if n_stale < 0:
        raise ValueError(f"{n_stale=} must be non-negative.")

    if n_stale_len <= 0:
        raise ValueError(f"{n_stale_len=} must be strictly positive.")

    if n_stale == 0:
        return

    n = len(returns)

    required_length = n_stale * n_stale_len

    if required_length > n:
        raise ValueError(
            f"Cannot fit {n_stale} stale blocks of length "
            f"{n_stale_len} into an array of length {n}."
        )

    # ``free_space`` is distributed over the n_stale + 1 gaps:
    #
    #     gap | block | gap | block | ... | block | gap
    #
    # Every block therefore has exactly n_stale_len observations and the
    # blocks can never overlap.
    free_space = n - required_length

    gap_sizes = rng.multinomial(
        free_space,
        np.full(
            n_stale + 1,
            1.0 / (n_stale + 1),
            dtype=np.float64,
        ),
    )

    start_idx = int(gap_sizes[0])

    for block_idx in range(n_stale):
        end_idx = start_idx + n_stale_len

        returns[start_idx:end_idx] = 0.0

        start_idx = end_idx + int(gap_sizes[block_idx + 1])


# ============================================================================
# COMPLEX SERIES
# ============================================================================


def _create_complex_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    n_stale: int,
    n_stale_len: int,
    len_01_proportion: float,
    len_02_proportion: float,
    low_vol_pa: float,
    vol_pa: float,
    high_vol_pa: float,
    bdays_per_year: int,
    drift_03_pa: float,
    jump_sizes: tuple[float, ...],
    rng: np.random.Generator,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)

    _validate_non_negative_finite(
        len_01_proportion,
        name="len_01_proportion",
    )
    _validate_non_negative_finite(
        len_02_proportion,
        name="len_02_proportion",
    )

    if len_01_proportion + len_02_proportion >= 1.0:
        raise ValueError(
            "len_01_proportion + len_02_proportion must be strictly less than 1."
        )

    _validate_non_negative_finite(low_vol_pa, name="low_vol_pa")
    _validate_non_negative_finite(vol_pa, name="vol_pa")
    _validate_non_negative_finite(high_vol_pa, name="high_vol_pa")
    _validate_finite(drift_03_pa, name="drift_03_pa")

    if any(not np.isfinite(size) for size in jump_sizes):
        raise ValueError("jump_sizes must contain only finite values.")

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    if include_initial_level:
        num_returns = expected_length
    else:
        num_returns = expected_length

    if num_returns < 1:
        raise ValueError("Complex synthetic series requires at least one return.")

    len1 = int(num_returns * len_01_proportion)
    len2 = int(num_returns * len_02_proportion)
    len3 = num_returns - len1 - len2

    if len1 < n_stale * n_stale_len:
        raise ValueError(
            f"Generated Regime 1 ({len1} returns) is too short to fit "
            f"{n_stale} stale periods of {n_stale_len} returns."
        )

    if len2 < len(jump_sizes):
        raise ValueError(
            f"Generated Regime 2 ({len2} returns) is too short to fit "
            f"{len(jump_sizes)} jumps."
        )

    if len3 < 1:
        raise ValueError("Generated Regime 3 must contain at least one return.")

    dt = 1.0 / bdays_per_year

    ret1 = shocks[:len1] * low_vol_pa * np.sqrt(dt)

    ret2_start = len1
    ret2_end = len1 + len2

    ret2 = shocks[ret2_start:ret2_end] * high_vol_pa * np.sqrt(dt)

    ret3 = shocks[ret2_end:] * vol_pa * np.sqrt(dt) + drift_03_pa * dt

    jump_indices = rng.choice(
        len2,
        size=len(jump_sizes),
        replace=False,
    )

    ret2[jump_indices] = rng.permutation(jump_sizes)

    # The stale periods are explicitly injected into Regime 1.
    # This fixes the semantic mismatch in the previous implementation,
    # where the entire concatenated return array was passed.
    inject_stale_periods_into_returns(
        returns=ret1,
        n_stale=n_stale,
        n_stale_len=n_stale_len,
        rng=rng,
    )

    log_returns = np.concatenate((ret1, ret2, ret3))

    return _returns_to_levels(
        log_returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )


# ============================================================================
# GARCH(1,1)-STYLE VOLATILITY CLUSTERING
# ============================================================================


def _create_garch_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    drift_pa: float,
    bdays_per_year: int,
    omega: float,
    alpha: float,
    beta: float,
    long_run_vol_pa: float,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)
    _validate_finite(drift_pa, name="drift_pa")

    _validate_non_negative_finite(omega, name="omega")
    _validate_non_negative_finite(alpha, name="alpha")
    _validate_non_negative_finite(beta, name="beta")
    _validate_non_negative_finite(
        long_run_vol_pa,
        name="long_run_vol_pa",
    )

    if alpha + beta >= 1.0:
        raise ValueError(
            f"GARCH parameters require alpha + beta < 1; got {alpha + beta}."
        )

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    if expected_length == 0:
        return np.array([init_level], dtype=np.float64)

    dt = 1.0 / bdays_per_year

    # Make the supplied long-run volatility and GARCH parameters coherent.
    #
    # The unconditional variance of:
    #
    #   sigma_t^2 = omega + alpha * r_(t-1)^2 + beta * sigma_(t-1)^2
    #
    # is omega / (1 - alpha - beta).
    #
    # The user-facing ``long_run_vol_pa`` is therefore used to derive the
    # unconditional daily variance, and omega is adjusted consistently.
    unconditional_var = long_run_vol_pa**2 * dt
    effective_omega = unconditional_var * (1.0 - alpha - beta)

    sigma2 = np.empty(expected_length, dtype=np.float64)
    returns = np.empty(expected_length, dtype=np.float64)

    sigma2[0] = unconditional_var

    for t in range(expected_length):
        if t > 0:
            sigma2[t] = (
                effective_omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
            )

        returns[t] = drift_pa * dt + np.sqrt(sigma2[t]) * shocks[t]

    return _returns_to_levels(
        returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )


# ============================================================================
# MEAN REVERSION
# ============================================================================


def _create_mean_reverting_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    bdays_per_year: int,
    mean_reversion_speed: float,
    long_run_level: float,
    vol_pa: float,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_positive_finite(long_run_level, name="long_run_level")
    _validate_bdays_per_year(bdays_per_year)

    _validate_non_negative_finite(
        mean_reversion_speed,
        name="mean_reversion_speed",
    )
    _validate_non_negative_finite(vol_pa, name="vol_pa")

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    if include_initial_level:
        num_levels = expected_length + 1
    else:
        num_levels = expected_length

    if num_levels < 1:
        return np.array([init_level], dtype=np.float64)

    dt = 1.0 / bdays_per_year

    x = np.empty(num_levels, dtype=np.float64)
    x[0] = np.log(init_level)

    theta = np.log(long_run_level)

    for t in range(1, num_levels):
        x[t] = (
            x[t - 1]
            + mean_reversion_speed * (theta - x[t - 1]) * dt
            + vol_pa * np.sqrt(dt) * shocks[t - 1]
        )

    if include_initial_level:
        return np.exp(x)

    return np.exp(x)


# ============================================================================
# SEASONAL
# ============================================================================


def _create_seasonal_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    vol_pa: float,
    drift_pa: float,
    amplitude_pa: float,
    period_days: int,
    bdays_per_year: int,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)
    _validate_non_negative_finite(vol_pa, name="vol_pa")
    _validate_finite(drift_pa, name="drift_pa")
    _validate_finite(amplitude_pa, name="amplitude_pa")

    if isinstance(period_days, bool) or not isinstance(period_days, (int, np.integer)):
        raise TypeError(f"{period_days=} must be an integer.")

    if period_days <= 0:
        raise ValueError(f"{period_days=} must be strictly positive.")

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    dt = 1.0 / bdays_per_year

    if include_initial_level:
        t = np.arange(1, expected_length + 1, dtype=np.float64)
    else:
        t = np.arange(expected_length, dtype=np.float64)

    seasonal_component = amplitude_pa * np.sin(2.0 * np.pi * t / period_days) * dt

    random_component = drift_pa * dt + vol_pa * np.sqrt(dt) * shocks

    returns = seasonal_component + random_component

    return _returns_to_levels(
        returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )


# ============================================================================
# STRUCTURAL BREAK
# ============================================================================


def _create_structural_break_levels_from_shocks(
    shocks: np.ndarray,
    *,
    init_level: float,
    bdays_per_year: int,
    break_location_proportion: float,
    break_size: float,
    pre_break_vol_pa: float,
    post_break_vol_pa: float,
    pre_break_drift_pa: float,
    post_break_drift_pa: float,
    include_initial_level: bool,
) -> np.ndarray:
    expected_length = len(shocks)

    _validate_positive_finite(init_level, name="init_level")
    _validate_bdays_per_year(bdays_per_year)

    _validate_finite(
        break_location_proportion,
        name="break_location_proportion",
    )

    if not 0.0 < break_location_proportion < 1.0:
        raise ValueError(
            f"{break_location_proportion=} must be strictly between 0 and 1."
        )

    _validate_finite(break_size, name="break_size")
    _validate_non_negative_finite(
        pre_break_vol_pa,
        name="pre_break_vol_pa",
    )
    _validate_non_negative_finite(
        post_break_vol_pa,
        name="post_break_vol_pa",
    )
    _validate_finite(
        pre_break_drift_pa,
        name="pre_break_drift_pa",
    )
    _validate_finite(
        post_break_drift_pa,
        name="post_break_drift_pa",
    )

    _validate_log_return_shocks(
        shocks,
        expected_length=expected_length,
    )

    if expected_length < 2:
        raise ValueError("Structural-break generation requires at least two returns.")

    dt = 1.0 / bdays_per_year

    if include_initial_level:
        num_levels = expected_length + 1
        break_idx = int(num_levels * break_location_proportion)

        if break_idx < 1 or break_idx >= num_levels:
            raise ValueError(
                f"{break_location_proportion=} produces an invalid "
                f"{break_idx=} for {num_levels=} levels."
            )

        num_pre_returns = break_idx - 1

    else:
        num_levels = expected_length
        break_idx = int(num_levels * break_location_proportion)

        if break_idx < 1 or break_idx >= num_levels:
            raise ValueError(
                f"{break_location_proportion=} produces an invalid "
                f"{break_idx=} for {num_levels=} observations."
            )

        num_pre_returns = break_idx

    num_post_returns = expected_length - num_pre_returns

    pre_returns = (
        shocks[:num_pre_returns] * pre_break_vol_pa * np.sqrt(dt)
        + pre_break_drift_pa * dt
    )

    post_returns = (
        shocks[num_pre_returns:] * post_break_vol_pa * np.sqrt(dt)
        + post_break_drift_pa * dt
    )

    returns = np.concatenate((pre_returns, post_returns))

    levels = _returns_to_levels(
        returns,
        init_level=init_level,
        include_initial_level=include_initial_level,
    )

    levels[break_idx:] *= np.exp(break_size)

    return levels


# ============================================================================
# BACKWARDS-COMPATIBLE PUBLIC GENERATORS
# ============================================================================


def create_illiquid_levels_series(
    *,
    num_obs: int = _DEFAULT_NUM_OBS,
    updates_per_year: int = _DEFAULT_UPDATES_PER_YEAR_IN_ILLQUID_SERIES,
    rng_seed: int = _DEFAULT_RNG_SEED,
    vol_pa: float = _DEFAULT_VOL_PA,
    drift_pa: float = _DEFAULT_DRIFT_PA,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    init_level: float = _DEFAULT_INIT_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a hidden continuous random walk and an observed step-function
    price that updates sporadically.

    This function retains its historical API and output convention.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_illiquid_levels_from_shocks(
        shocks,
        init_level=init_level,
        updates_per_year=updates_per_year,
        vol_pa=vol_pa,
        drift_pa=drift_pa,
        bdays_per_year=bdays_per_year,
        rng=rng,
        include_initial_level=False,
    )


def create_ar1_process_levels_series(
    num_obs: int = _DEFAULT_NUM_OBS,
    phi: float = _DEFAULT_PHI_AR1,
    rng_seed: int = _DEFAULT_RNG_SEED,
    vol_pa: float = _DEFAULT_VOL_PA,
    drift_pa: float = _DEFAULT_DRIFT_PA,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    init_level: float = _DEFAULT_INIT_LEVEL,
) -> np.ndarray:
    """
    Simulates an AR(1) process with specified parameters.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_ar1_levels_from_shocks(
        shocks,
        init_level=init_level,
        phi=phi,
        vol_pa=vol_pa,
        drift_pa=drift_pa,
        bdays_per_year=bdays_per_year,
        include_initial_level=False,
    )


def create_ar2_process_levels_series(
    num_obs: int = _DEFAULT_NUM_OBS,
    phi_1: float = _DEFAULT_PHI_1_AR2,
    phi_2: float = _DEFAULT_PHI_2_AR2,
    rng_seed: int = _DEFAULT_RNG_SEED,
    vol_pa: float = _DEFAULT_VOL_PA,
    drift_pa: float = _DEFAULT_DRIFT_PA,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    init_level: float = _DEFAULT_INIT_LEVEL,
) -> np.ndarray:
    """
    Simulates an AR(2) process with specified parameters.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_ar2_levels_from_shocks(
        shocks,
        init_level=init_level,
        phi_1=phi_1,
        phi_2=phi_2,
        vol_pa=vol_pa,
        drift_pa=drift_pa,
        bdays_per_year=bdays_per_year,
        include_initial_level=False,
    )


def create_random_walk_levels_series(
    num_obs: int = _DEFAULT_NUM_OBS,
    rng_seed: int = _DEFAULT_RNG_SEED,
    vol_pa: float = _DEFAULT_VOL_PA,
    drift_pa: float = _DEFAULT_DRIFT_PA,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    init_level: float = _DEFAULT_INIT_LEVEL,
) -> np.ndarray:
    """
    Simulates a simple geometric random walk with drift.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_random_walk_levels_from_shocks(
        shocks,
        init_level=init_level,
        vol_pa=vol_pa,
        drift_pa=drift_pa,
        bdays_per_year=bdays_per_year,
        include_initial_level=False,
    )


def create_complex_levels_series(
    *,
    num_obs: int = _DEFAULT_NUM_OBS,
    rng_seed: int = _DEFAULT_RNG_SEED,
    n_stale: int = _DEFAULT_COMPLEX_SR_INJECTED_STALE_PERIODS,
    n_stale_len: int = _DEFAULT_COMPLEX_SR_INJECTED_STALE_LENGTH,
    len_01_proportion: float = _DEFAULT_COMPLEX_SR_LEN_01_PROPORTION,
    len_02_proportion: float = _DEFAULT_COMPLEX_SR_LEN_02_PROPORTION,
    init_level: float = _DEFAULT_INIT_LEVEL,
    low_vol_pa: float = _DEFAULT_VOL_PA_LOW,
    vol_pa: float = _DEFAULT_VOL_PA,
    high_vol_pa: float = _DEFAULT_VOL_PA_HIGH,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    drift_03_pa: float = _DEFAULT_COMPLEX_SR_DRIFT_PA_03,
    jump_sizes: tuple[float, ...] = _DEFAULT_COMPLEX_SR_JUMP_SIZES,
) -> np.ndarray:
    """
    Generates log returns featuring regime changes, contiguous stale periods,
    and jumps.

    Historical output convention is retained: ``num_obs + 1`` levels are
    returned, including the initial level.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_complex_levels_from_shocks(
        shocks,
        init_level=init_level,
        n_stale=n_stale,
        n_stale_len=n_stale_len,
        len_01_proportion=len_01_proportion,
        len_02_proportion=len_02_proportion,
        low_vol_pa=low_vol_pa,
        vol_pa=vol_pa,
        high_vol_pa=high_vol_pa,
        bdays_per_year=bdays_per_year,
        drift_03_pa=drift_03_pa,
        jump_sizes=jump_sizes,
        rng=rng,
        include_initial_level=True,
    )


def create_volatility_clustering_levels_series(
    *,
    num_obs: int = _DEFAULT_NUM_OBS,
    rng_seed: int = _DEFAULT_RNG_SEED,
    drift_pa: float = _DEFAULT_DRIFT_PA,
    init_level: float = _DEFAULT_INIT_LEVEL,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    omega: float = _DEFAULT_GARCH_OMEGA,
    alpha: float = _DEFAULT_GARCH_ALPHA,
    beta: float = _DEFAULT_GARCH_BETA,
    long_run_vol_pa: float = _DEFAULT_VOL_PA,
) -> np.ndarray:
    """
    Simulates returns exhibiting volatility clustering using a GARCH(1,1)-style
    process.

    ``long_run_vol_pa`` is used consistently with ``omega``, ``alpha`` and
    ``beta`` to determine the unconditional variance.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_garch_levels_from_shocks(
        shocks,
        init_level=init_level,
        drift_pa=drift_pa,
        bdays_per_year=bdays_per_year,
        omega=omega,
        alpha=alpha,
        beta=beta,
        long_run_vol_pa=long_run_vol_pa,
        include_initial_level=False,
    )


def create_mean_reverting_levels_series(
    *,
    num_obs: int = _DEFAULT_NUM_OBS,
    rng_seed: int = _DEFAULT_RNG_SEED,
    init_level: float = _DEFAULT_INIT_LEVEL,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    mean_reversion_speed: float = _DEFAULT_MR_SPEED,
    long_run_level: float = _DEFAULT_MR_LONG_RUN_LEVEL,
    vol_pa: float = _DEFAULT_VOL_PA,
) -> np.ndarray:
    """
    Simulates a mean-reverting process in log-price space.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs - 1,
    )

    return _create_mean_reverting_levels_from_shocks(
        shocks,
        init_level=init_level,
        bdays_per_year=bdays_per_year,
        mean_reversion_speed=mean_reversion_speed,
        long_run_level=long_run_level,
        vol_pa=vol_pa,
        include_initial_level=False,
    )


def create_seasonal_levels_series(
    *,
    num_obs: int = _DEFAULT_NUM_OBS,
    rng_seed: int = _DEFAULT_RNG_SEED,
    init_level: float = _DEFAULT_INIT_LEVEL,
    vol_pa: float = _DEFAULT_VOL_PA_LOW,
    drift_pa: float = _DEFAULT_DRIFT_PA,
    amplitude_pa: float = _DEFAULT_SEASONAL_AMPLITUDE_PA,
    period_days: int = _DEFAULT_SEASONAL_PERIOD_DAYS,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
) -> np.ndarray:
    """
    Simulates a seasonal asset with a sinusoidal return component.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_seasonal_levels_from_shocks(
        shocks,
        init_level=init_level,
        vol_pa=vol_pa,
        drift_pa=drift_pa,
        amplitude_pa=amplitude_pa,
        period_days=period_days,
        bdays_per_year=bdays_per_year,
        include_initial_level=False,
    )


def create_structural_break_levels_series(
    *,
    num_obs: int = _DEFAULT_NUM_OBS,
    rng_seed: int = _DEFAULT_RNG_SEED,
    init_level: float = _DEFAULT_INIT_LEVEL,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    break_location_proportion: float = _DEFAULT_BREAK_LOCATION_PROPORTION,
    break_size: float = _DEFAULT_BREAK_SIZE,
    pre_break_vol_pa: float = _DEFAULT_PRE_BREAK_VOL_PA,
    post_break_vol_pa: float = _DEFAULT_POST_BREAK_VOL_PA,
    pre_break_drift_pa: float = _DEFAULT_PRE_BREAK_DRIFT_PA,
    post_break_drift_pa: float = _DEFAULT_POST_BREAK_DRIFT_PA,
) -> np.ndarray:
    """
    Simulates a time series containing a clear structural break.

    The break consists of:

    1. An instantaneous permanent level shift.
    2. A change in volatility regime.
    3. A change in drift regime.
    """
    _validate_num_obs(num_obs)

    rng = np.random.default_rng(rng_seed)

    shocks = _draw_standard_normal_shocks(
        rng,
        num_obs,
    )

    return _create_structural_break_levels_from_shocks(
        shocks,
        init_level=init_level,
        bdays_per_year=bdays_per_year,
        break_location_proportion=break_location_proportion,
        break_size=break_size,
        pre_break_vol_pa=pre_break_vol_pa,
        post_break_vol_pa=post_break_vol_pa,
        pre_break_drift_pa=pre_break_drift_pa,
        post_break_drift_pa=post_break_drift_pa,
        include_initial_level=False,
    )


# ============================================================================
# SYNTHETIC LEVEL GENERATOR FACADE
# ============================================================================


class SyntheticLevelGenerator:
    """
    Stateful generator for heterogeneous synthetic risk-factor level series.

    The class is deliberately a façade rather than a new hierarchy of model
    objects. Process-specific mathematics remains in small private kernels,
    while this class owns orchestration, random-number streams, validation,
    and optional cross-series dependence.

    Class contract
    --------------
    Every series generated through this class:

    * contains exactly ``num_obs`` levels;
    * starts at its configured initial level;
    * uses the same observation grid;
    * can receive the same common innovation factor.

    This differs intentionally from the historical standalone functions,
    some of which do not include their initial level and
    ``create_complex_levels_series`` of which returns ``num_obs + 1`` values.

    Dependence
    ----------
    ``correlation`` is implemented using a one-factor Gaussian innovation
    construction:

        z_i,t = sqrt(rho) * f_t + sqrt(1-rho) * e_i,t

    where ``f`` is common across all series and ``e_i`` is idiosyncratic.

    Therefore the supplied ``correlation`` is the theoretical pairwise
    correlation of the standard-normal innovations.

    Realised correlations of final returns need not equal that value because
    the individual generators may introduce:

    * serial dependence;
    * volatility clustering;
    * regime changes;
    * deterministic seasonality;
    * jumps;
    * stale observations;
    * mean reversion.

    This is intentional. The common factor controls dependence in the
    underlying innovations rather than artificially forcing correlation onto
    already-generated price paths.

    Examples
    --------
    Generate heterogeneous independent series::

        generator = SyntheticLevelGenerator(
            num_obs=1301,
            rng_seed=42,
        )

        data = generator.generate_many(
            {
                "spot": "random_walk",
                "momentum": "ar1",
                "volatility": "garch",
                "reverting": "mean_reverting",
            }
        )

    Generate dependent series::

        data = generator.generate_many(
            {
                "spot": "random_walk",
                "momentum": "ar1",
                "volatility": "garch",
                "reverting": "mean_reverting",
            },
            correlation=0.60,
        )

    Customize individual processes::

        data = generator.generate_many(
            {
                "spot": {
                    "kind": "random_walk",
                    "vol_pa": 0.20,
                    "init_level": 80.0,
                },
                "momentum": {
                    "kind": "ar1",
                    "phi": 0.85,
                    "vol_pa": 0.15,
                },
                "reverting": {
                    "kind": "mean_reverting",
                    "mean_reversion_speed": 5.0,
                    "long_run_level": 120.0,
                },
            },
            correlation=0.50,
        )
    """

    def __init__(
        self,
        *,
        num_obs: int = _DEFAULT_NUM_OBS,
        rng_seed: int = _DEFAULT_RNG_SEED,
        bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
        init_level: float = _DEFAULT_INIT_LEVEL,
    ) -> None:
        _validate_num_obs_for_class(num_obs)
        _validate_bdays_per_year(bdays_per_year)
        _validate_positive_finite(init_level, name="init_level")

        self._num_obs = int(num_obs)
        self._rng = np.random.default_rng(rng_seed)
        self._bdays_per_year = int(bdays_per_year)
        self._init_level = float(init_level)

    @property
    def num_obs(self) -> int:
        """Number of levels returned by every class-generated series."""
        return self._num_obs

    @property
    def bdays_per_year(self) -> int:
        """Business days per year used by the generator."""
        return self._bdays_per_year

    @property
    def init_level(self) -> float:
        """Default initial level used by generated series."""
        return self._init_level

    @property
    def supported_kinds(self) -> tuple[SyntheticSeriesKind, ...]:
        """Supported synthetic generator kinds."""
        return _SYNTHETIC_SERIES_KINDS

    def generate(
        self,
        kind: SyntheticSeriesKind,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate one synthetic level series.

        ``num_obs``, ``rng_seed`` and ``bdays_per_year`` are controlled by
        the generator instance. ``init_level`` may be overridden for an
        individual series.

        The returned array always contains exactly ``self.num_obs`` levels.
        """
        self._validate_kind(kind)
        self._validate_reserved_kwargs(kwargs)

        init_level = float(kwargs.pop("init_level", self._init_level))

        _validate_positive_finite(
            init_level,
            name="init_level",
        )

        shocks = self._draw_correlated_shocks(
            num_series=1,
            correlation=0.0,
        )[0]

        return self._generate_kind_from_shocks(
            kind=kind,
            shocks=shocks,
            init_level=init_level,
            kwargs=kwargs,
            rng=self._rng,
        )

    def generate_many(
        self,
        series: Mapping[str, SyntheticSeriesSpec],
        *,
        correlation: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """
        Generate multiple heterogeneous series on a common time grid.

        Parameters
        ----------
        series:
            Mapping from output name to a series specification.

            Simple form::

                {
                    "spot": "random_walk",
                    "spread": "ar1",
                }

            Expanded form::

                {
                    "spot": {
                        "kind": "random_walk",
                        "vol_pa": 0.20,
                        "init_level": 80.0,
                    },
                    "spread": {
                        "kind": "ar1",
                        "phi": 0.80,
                    },
                }

        correlation:
            Common-factor innovation correlation.

            ``0.0`` gives independent innovations.

            ``1.0`` gives identical innovations across all series.

            Values between zero and one give a one-factor dependence
            structure.

        Returns
        -------
        dict[str, np.ndarray]
            Named arrays, each containing exactly ``self.num_obs`` levels.
        """
        if not series:
            raise ValueError("At least one series specification is required.")

        self._validate_correlation(correlation)

        names = list(series)

        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    f"Series names must be strings; got {type(name).__name__}."
                )

            if not name.strip():
                raise ValueError("Series names must not be empty.")

        if len(names) != len(set(names)):
            raise ValueError("Series names must be unique.")

        normalised_specs: list[tuple[str, SyntheticSeriesKind, dict[str, Any]]] = []

        for name, specification in series.items():
            kind, kwargs = self._normalise_specification(
                name=name,
                specification=specification,
            )

            normalised_specs.append(
                (
                    name,
                    kind,
                    kwargs,
                )
            )

        shocks = self._draw_correlated_shocks(
            num_series=len(normalised_specs),
            correlation=correlation,
        )

        result: dict[str, np.ndarray] = {}

        for i, (name, kind, kwargs) in enumerate(normalised_specs):
            init_level = float(
                kwargs.pop(
                    "init_level",
                    self._init_level,
                )
            )

            _validate_positive_finite(
                init_level,
                name=f"{name}.init_level",
            )

            levels = self._generate_kind_from_shocks(
                kind=kind,
                shocks=shocks[i],
                init_level=init_level,
                kwargs=kwargs,
                rng=self._rng,
            )

            if len(levels) != self._num_obs:
                raise RuntimeError(
                    f"Generator {kind!r} returned {len(levels)} levels; "
                    f"expected {self._num_obs}."
                )

            if levels.ndim != 1:
                raise RuntimeError(f"Generator {kind!r} returned a non-1D array.")

            if not np.all(np.isfinite(levels)):
                raise RuntimeError(f"Generator {kind!r} returned non-finite levels.")

            if np.any(levels <= 0.0):
                raise RuntimeError(f"Generator {kind!r} returned non-positive levels.")

            result[name] = levels

        return result

    def _generate_kind_from_shocks(
        self,
        *,
        kind: SyntheticSeriesKind,
        shocks: np.ndarray,
        init_level: float,
        kwargs: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Dispatch one process using externally supplied innovations.

        This method is the key architectural seam that allows the same
        process mathematics to operate under either independent or correlated
        innovations.
        """
        expected_shocks = self._num_obs - 1

        _validate_log_return_shocks(
            shocks,
            expected_length=expected_shocks,
        )

        if kind == "illiquid_observed":
            _, observed = _create_illiquid_levels_from_shocks(
                shocks,
                init_level=init_level,
                updates_per_year=int(
                    kwargs.get(
                        "updates_per_year",
                        _DEFAULT_UPDATES_PER_YEAR_IN_ILLQUID_SERIES,
                    )
                ),
                vol_pa=float(kwargs.get("vol_pa", _DEFAULT_VOL_PA)),
                drift_pa=float(kwargs.get("drift_pa", _DEFAULT_DRIFT_PA)),
                bdays_per_year=self._bdays_per_year,
                rng=rng,
                include_initial_level=True,
            )
            return observed

        if kind == "illiquid_latent":
            true_levels, _ = _create_illiquid_levels_from_shocks(
                shocks,
                init_level=init_level,
                updates_per_year=int(
                    kwargs.get(
                        "updates_per_year",
                        _DEFAULT_UPDATES_PER_YEAR_IN_ILLQUID_SERIES,
                    )
                ),
                vol_pa=float(kwargs.get("vol_pa", _DEFAULT_VOL_PA)),
                drift_pa=float(kwargs.get("drift_pa", _DEFAULT_DRIFT_PA)),
                bdays_per_year=self._bdays_per_year,
                rng=rng,
                include_initial_level=True,
            )
            return true_levels

        if kind == "random_walk":
            return _create_random_walk_levels_from_shocks(
                shocks,
                init_level=init_level,
                vol_pa=float(kwargs.get("vol_pa", _DEFAULT_VOL_PA)),
                drift_pa=float(kwargs.get("drift_pa", _DEFAULT_DRIFT_PA)),
                bdays_per_year=self._bdays_per_year,
                include_initial_level=True,
            )

        if kind == "ar1":
            return _create_ar1_levels_from_shocks(
                shocks,
                init_level=init_level,
                phi=float(kwargs.get("phi", _DEFAULT_PHI_AR1)),
                vol_pa=float(kwargs.get("vol_pa", _DEFAULT_VOL_PA)),
                drift_pa=float(kwargs.get("drift_pa", _DEFAULT_DRIFT_PA)),
                bdays_per_year=self._bdays_per_year,
                include_initial_level=True,
            )

        if kind == "ar2":
            return _create_ar2_levels_from_shocks(
                shocks,
                init_level=init_level,
                phi_1=float(kwargs.get("phi_1", _DEFAULT_PHI_1_AR2)),
                phi_2=float(kwargs.get("phi_2", _DEFAULT_PHI_2_AR2)),
                vol_pa=float(kwargs.get("vol_pa", _DEFAULT_VOL_PA)),
                drift_pa=float(kwargs.get("drift_pa", _DEFAULT_DRIFT_PA)),
                bdays_per_year=self._bdays_per_year,
                include_initial_level=True,
            )

        if kind == "complex":
            return _create_complex_levels_from_shocks(
                shocks,
                init_level=init_level,
                n_stale=int(
                    kwargs.get(
                        "n_stale",
                        _DEFAULT_COMPLEX_SR_INJECTED_STALE_PERIODS,
                    )
                ),
                n_stale_len=int(
                    kwargs.get(
                        "n_stale_len",
                        _DEFAULT_COMPLEX_SR_INJECTED_STALE_LENGTH,
                    )
                ),
                len_01_proportion=float(
                    kwargs.get(
                        "len_01_proportion",
                        _DEFAULT_COMPLEX_SR_LEN_01_PROPORTION,
                    )
                ),
                len_02_proportion=float(
                    kwargs.get(
                        "len_02_proportion",
                        _DEFAULT_COMPLEX_SR_LEN_02_PROPORTION,
                    )
                ),
                low_vol_pa=float(
                    kwargs.get(
                        "low_vol_pa",
                        _DEFAULT_VOL_PA_LOW,
                    )
                ),
                vol_pa=float(
                    kwargs.get(
                        "vol_pa",
                        _DEFAULT_VOL_PA,
                    )
                ),
                high_vol_pa=float(
                    kwargs.get(
                        "high_vol_pa",
                        _DEFAULT_VOL_PA_HIGH,
                    )
                ),
                bdays_per_year=self._bdays_per_year,
                drift_03_pa=float(
                    kwargs.get(
                        "drift_03_pa",
                        _DEFAULT_COMPLEX_SR_DRIFT_PA_03,
                    )
                ),
                jump_sizes=tuple(
                    kwargs.get(
                        "jump_sizes",
                        _DEFAULT_COMPLEX_SR_JUMP_SIZES,
                    )
                ),
                rng=rng,
                include_initial_level=True,
            )

        if kind == "garch":
            return _create_garch_levels_from_shocks(
                shocks,
                init_level=init_level,
                drift_pa=float(kwargs.get("drift_pa", _DEFAULT_DRIFT_PA)),
                bdays_per_year=self._bdays_per_year,
                omega=float(kwargs.get("omega", _DEFAULT_GARCH_OMEGA)),
                alpha=float(kwargs.get("alpha", _DEFAULT_GARCH_ALPHA)),
                beta=float(kwargs.get("beta", _DEFAULT_GARCH_BETA)),
                long_run_vol_pa=float(
                    kwargs.get(
                        "long_run_vol_pa",
                        _DEFAULT_VOL_PA,
                    )
                ),
                include_initial_level=True,
            )

        if kind == "mean_reverting":
            return _create_mean_reverting_levels_from_shocks(
                shocks,
                init_level=init_level,
                bdays_per_year=self._bdays_per_year,
                mean_reversion_speed=float(
                    kwargs.get(
                        "mean_reversion_speed",
                        _DEFAULT_MR_SPEED,
                    )
                ),
                long_run_level=float(
                    kwargs.get(
                        "long_run_level",
                        _DEFAULT_MR_LONG_RUN_LEVEL,
                    )
                ),
                vol_pa=float(kwargs.get("vol_pa", _DEFAULT_VOL_PA)),
                include_initial_level=True,
            )

        if kind == "seasonal":
            return _create_seasonal_levels_from_shocks(
                shocks,
                init_level=init_level,
                vol_pa=float(
                    kwargs.get(
                        "vol_pa",
                        _DEFAULT_VOL_PA_LOW,
                    )
                ),
                drift_pa=float(kwargs.get("drift_pa", _DEFAULT_DRIFT_PA)),
                amplitude_pa=float(
                    kwargs.get(
                        "amplitude_pa",
                        _DEFAULT_SEASONAL_AMPLITUDE_PA,
                    )
                ),
                period_days=int(
                    kwargs.get(
                        "period_days",
                        _DEFAULT_SEASONAL_PERIOD_DAYS,
                    )
                ),
                bdays_per_year=self._bdays_per_year,
                include_initial_level=True,
            )

        if kind == "structural_break":
            return _create_structural_break_levels_from_shocks(
                shocks,
                init_level=init_level,
                bdays_per_year=self._bdays_per_year,
                break_location_proportion=float(
                    kwargs.get(
                        "break_location_proportion",
                        _DEFAULT_BREAK_LOCATION_PROPORTION,
                    )
                ),
                break_size=float(
                    kwargs.get(
                        "break_size",
                        _DEFAULT_BREAK_SIZE,
                    )
                ),
                pre_break_vol_pa=float(
                    kwargs.get(
                        "pre_break_vol_pa",
                        _DEFAULT_PRE_BREAK_VOL_PA,
                    )
                ),
                post_break_vol_pa=float(
                    kwargs.get(
                        "post_break_vol_pa",
                        _DEFAULT_POST_BREAK_VOL_PA,
                    )
                ),
                pre_break_drift_pa=float(
                    kwargs.get(
                        "pre_break_drift_pa",
                        _DEFAULT_PRE_BREAK_DRIFT_PA,
                    )
                ),
                post_break_drift_pa=float(
                    kwargs.get(
                        "post_break_drift_pa",
                        _DEFAULT_POST_BREAK_DRIFT_PA,
                    )
                ),
                include_initial_level=True,
            )

        raise RuntimeError(f"Unhandled synthetic series kind: {kind!r}")

    def _draw_correlated_shocks(
        self,
        *,
        num_series: int,
        correlation: float,
    ) -> np.ndarray:
        """
        Draw a matrix of common-factor correlated Gaussian innovations.

        Returns
        -------
        np.ndarray
            Shape ``(num_series, num_obs - 1)``.
        """
        if num_series < 1:
            raise ValueError(f"{num_series=} must be strictly positive.")

        n_returns = self._num_obs - 1

        common = np.asarray(
            self._rng.standard_normal(n_returns),
            dtype=np.float64,
        )

        if num_series == 1 or correlation == 0.0:
            return np.asarray(
                self._rng.standard_normal((num_series, n_returns)),
                dtype=np.float64,
            )

        if correlation == 1.0:
            return np.broadcast_to(
                common,
                (num_series, n_returns),
            ).copy()

        idiosyncratic = np.asarray(
            self._rng.standard_normal((num_series, n_returns)),
            dtype=np.float64,
        )

        return (
            np.sqrt(correlation) * common[None, :]
            + np.sqrt(1.0 - correlation) * idiosyncratic
        )

    def _normalise_specification(
        self,
        *,
        name: str,
        specification: SyntheticSeriesSpec,
    ) -> tuple[SyntheticSeriesKind, dict[str, Any]]:
        if isinstance(specification, str):
            kind = self._validate_kind(specification)
            return kind, {}

        if not isinstance(specification, Mapping):
            raise TypeError(
                f"Series {name!r} specification must be a string "
                f"or mapping; got {type(specification).__name__}."
            )

        if "kind" not in specification:
            raise ValueError(
                f"Series {name!r} specification must contain a 'kind' entry."
            )

        raw_kind = specification["kind"]

        if not isinstance(raw_kind, str):
            raise TypeError(f"Series {name!r} has non-string kind {raw_kind!r}.")

        kind = self._validate_kind(raw_kind)

        kwargs = {key: value for key, value in specification.items() if key != "kind"}

        self._validate_reserved_kwargs(kwargs)

        return kind, kwargs

    @staticmethod
    def _validate_kind(
        kind: str,
    ) -> SyntheticSeriesKind:
        if kind not in _SYNTHETIC_SERIES_KIND_SET:
            valid = ", ".join(_SYNTHETIC_SERIES_KINDS)
            raise ValueError(
                f"Unsupported synthetic series kind {kind!r}. Expected one of: {valid}."
            )

        return cast(SyntheticSeriesKind, kind)

    @staticmethod
    def _validate_reserved_kwargs(
        kwargs: Mapping[str, object],
    ) -> None:
        reserved = {
            "num_obs",
            "rng_seed",
            "bdays_per_year",
        }

        conflicting = sorted(reserved.intersection(kwargs))

        if conflicting:
            raise ValueError(
                "The following parameters are controlled by "
                "SyntheticLevelGenerator and cannot be overridden: "
                f"{', '.join(conflicting)}."
            )

    @staticmethod
    def _validate_correlation(
        correlation: float,
    ) -> None:
        if not np.isfinite(correlation):
            raise ValueError(f"{correlation=} must be finite.")

        if not 0.0 <= correlation <= 1.0:
            raise ValueError(f"{correlation=} must satisfy 0 <= correlation <= 1.")
