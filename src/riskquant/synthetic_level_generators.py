# src/riskquant/synthetic_level_generators.py
"""
Synthetic level generators for various types of time series.

Contains functions to generate synthetic time series data for testing and
demonstration purposes, including random walks, AR processes, and other
complex behaviors.

Noe that these generators are designed for illustrative purposes and may not capture
all nuances of real financial data. They are intended to provide
controlled environments for testing filters, diagnostics, and risk models.

"""

from typing import Final

import numpy as np

_DEFAULT_RNG_SEED: Final[int] = 999_999


_DEFAULT_NUM_OBS: Final[int] = (
    22 * 12 * 5
) + 1  # 5 years of daily data + initial level
_DEFAULT_VOL_PA: Final[float] = 0.25
_DEFAULT_VOL_PA_LOW: Final[float] = 0.05
_DEFAULT_VOL_PA_HIGH: Final[float] = 0.50
_DEFAULT_DRIFT_PA: Final[float] = 0.010
_DEFAULT_BDAYS_PER_YEAR: Final[int] = (
    22 * 12
)  # Approximate number of business days in a year

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
    Simulates a hidden continuous random walk and an observed "step function"
    price that only updates sporadically, snapping to the hidden true value.
    """
    rng = np.random.default_rng(rng_seed)

    dt = 1 / bdays_per_year

    # 1. The hidden "true" economic price accumulating information
    true_returns = rng.standard_normal(num_obs) * vol_pa * np.sqrt(dt) + drift_pa * dt
    true_log_price = np.cumsum(true_returns)

    # 2. The observed price (e.g., updates ~35 times per 252-day year)
    prob_update = updates_per_year / bdays_per_year
    is_observed = rng.random(num_obs) < prob_update
    is_observed[0] = True  # Always observe the first day

    obs_log_price = np.zeros(num_obs)
    last_known_price = true_log_price[0]

    for t in range(num_obs):
        if is_observed[t]:
            last_known_price = true_log_price[t]
        obs_log_price[t] = last_known_price

    true_levels = init_level * np.exp(true_log_price)
    obs_levels = init_level * np.exp(obs_log_price)

    return true_levels, obs_levels


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
    Simulates an AR(1) process with specified parameters, which can be used to
    test AR(1) filter acceptance.
    """
    rng = np.random.default_rng(rng_seed)
    returns = np.zeros(num_obs)
    dt = 1 / bdays_per_year
    returns[0] = rng.standard_normal() * vol_pa * np.sqrt(dt) + drift_pa * dt

    for t in range(1, num_obs):
        returns[t] = (
            phi * returns[t - 1]
            + rng.standard_normal() * vol_pa * np.sqrt(dt)
            + drift_pa * dt
        )

    levels = init_level * np.exp(np.cumsum(returns))
    return levels


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
    Simulates an AR(2) process with specified parameters, which can be used to
    test AR(1) filter rejection.
    """
    rng = np.random.default_rng(rng_seed)
    returns = np.zeros(num_obs)
    dt = 1 / bdays_per_year
    returns[0] = rng.standard_normal() * vol_pa * np.sqrt(dt) + drift_pa * dt
    returns[1] = rng.standard_normal() * vol_pa * np.sqrt(dt) + drift_pa * dt

    for t in range(2, num_obs):
        returns[t] = (
            phi_1 * returns[t - 1]
            + phi_2 * returns[t - 2]
            + rng.standard_normal() * vol_pa * np.sqrt(dt)
            + drift_pa * dt
        )

    levels = init_level * np.exp(np.cumsum(returns))
    return levels


def create_random_walk_levels_series(
    num_obs: int = _DEFAULT_NUM_OBS,
    rng_seed: int = _DEFAULT_RNG_SEED,
    vol_pa: float = _DEFAULT_VOL_PA,
    drift_pa: float = _DEFAULT_DRIFT_PA,
    bdays_per_year: int = _DEFAULT_BDAYS_PER_YEAR,
    init_level: float = _DEFAULT_INIT_LEVEL,
) -> np.ndarray:
    """
    Simulates a simple random walk with drift, which can be used to test AR(1)
    filter acceptance.
    """
    rng = np.random.default_rng(rng_seed)
    dt = 1 / bdays_per_year
    returns = rng.standard_normal(num_obs) * vol_pa * np.sqrt(dt) + drift_pa * dt
    levels = init_level * np.exp(np.cumsum(returns))
    return levels


def inject_stale_periods_into_returns(
    returns: np.ndarray,
    *,
    n_stale: int = _DEFAULT_COMPLEX_SR_INJECTED_STALE_PERIODS,
    n_stale_len: int = _DEFAULT_COMPLEX_SR_INJECTED_STALE_LENGTH,
    rng: np.random.Generator,
) -> None:
    """
    Injects non-overlapping contiguous periods of zero returns in-place.
    """
    n = len(returns)
    if n_stale * n_stale_len > n:
        raise ValueError("Requested stale periods exceed total array length.")

    # Maintain a list of valid starting indices that can fit the block length
    valid_starts = list(range(n - n_stale_len + 1))

    for _ in range(n_stale):
        if not valid_starts:
            raise ValueError(
                "Array is too crowded to fit another non-overlapping stale block."
            )

        # Pick a random valid starting index
        start_idx = rng.choice(valid_starts)

        # Zero out the return block
        returns[start_idx : start_idx + n_stale_len] = 0.0

        # Prune the candidate pool to prevent overlapping blocks.
        # If we chose start_idx, any new start index 's' must be far enough
        # away so the blocks don't bleed into each other.
        min_invalid = start_idx - n_stale_len + 1
        max_invalid = start_idx + n_stale_len - 1

        valid_starts = [x for x in valid_starts if x < min_invalid or x > max_invalid]


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
    """Generates log-returns featuring regime changes, contiguous stale periods, and jumps."""
    rng = np.random.default_rng(rng_seed)

    # Proportional split: ~33% / 33% / 34%
    len1 = int(num_obs * len_01_proportion)
    len2 = int(num_obs * len_02_proportion)
    len3 = num_obs - len1 - len2  # Absorbs any rounding differences

    # Safety checks: ensure the regimes are long enough to fit the anomalies
    if len1 < n_stale * n_stale_len:
        raise ValueError(
            f"num_obs={num_obs} makes Regime 1 ({len1} days) too short to "
            f"fit {n_stale} stale periods of {n_stale_len} days."
        )
    if len2 < 4:
        raise ValueError(
            f"num_obs={num_obs} makes Regime 2 ({len2} days) too short to fit 4 jumps."
        )

    dt = 1 / bdays_per_year

    # Regime 1: Low Volatility with CONTIGUOUS Stale periods (Illiquid phase)
    ret1 = rng.standard_normal(len1) * low_vol_pa * np.sqrt(dt)

    # Regime 2: High Volatility with Jumps (Crisis/Earnings phase)
    ret2 = rng.standard_normal(len2) * high_vol_pa * np.sqrt(dt)
    num_jumps = len(jump_sizes)
    if len2 < num_jumps:
        raise ValueError(
            f"num_obs={num_obs} makes Regime 2 ({len2} days) too short "
            f"to fit {num_jumps} jumps."
        )
    jump_indices = rng.choice(len2, size=num_jumps, replace=False)
    ret2[jump_indices] = rng.permutation(jump_sizes)

    # Regime 3: Medium Volatility with Drift (Recovery phase)
    ret3 = rng.standard_normal(len3) * vol_pa * np.sqrt(dt) + drift_03_pa * dt

    log_returns = np.concatenate([ret1, ret2, ret3])

    # Inject contiguous stale periods into the first regime
    inject_stale_periods_into_returns(
        returns=log_returns,
        n_stale=n_stale,
        n_stale_len=n_stale_len,
        rng=rng,
    )

    # Construct level prices starting at init_level
    levels = init_level * np.exp(np.cumsum(log_returns))
    return np.concatenate(([init_level], levels))


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
    Simulates returns exhibiting volatility clustering using a simple
    GARCH(1,1)-style process.
    """
    rng = np.random.default_rng(rng_seed)

    dt = 1.0 / bdays_per_year

    unconditional_var = long_run_vol_pa**2 * dt

    sigma2 = np.empty(num_obs)
    returns = np.empty(num_obs)

    sigma2[0] = unconditional_var

    for t in range(num_obs):
        if t > 0:
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

        returns[t] = drift_pa * dt + np.sqrt(sigma2[t]) * rng.standard_normal()

    levels = init_level * np.exp(np.cumsum(returns))
    return levels


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
    rng = np.random.default_rng(rng_seed)

    dt = 1.0 / bdays_per_year

    x = np.empty(num_obs)

    x[0] = np.log(init_level)

    theta = np.log(long_run_level)

    for t in range(1, num_obs):
        x[t] = (
            x[t - 1]
            + mean_reversion_speed * (theta - x[t - 1]) * dt
            + vol_pa * np.sqrt(dt) * rng.standard_normal()
        )

    return np.exp(x)


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
    rng = np.random.default_rng(rng_seed)

    dt = 1.0 / bdays_per_year

    t = np.arange(num_obs)

    seasonal_component = amplitude_pa * np.sin(2.0 * np.pi * t / period_days) * dt

    random_component = drift_pa * dt + vol_pa * np.sqrt(dt) * rng.standard_normal(
        num_obs
    )

    returns = seasonal_component + random_component

    levels = init_level * np.exp(np.cumsum(returns))
    return levels


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

    This produces a visually obvious change point that is useful for
    testing regime detection, change-point algorithms, rolling statistics,
    VaR models, and risk-factor diagnostics.
    """
    if not 0.0 < break_location_proportion < 1.0:
        raise ValueError(
            f"{break_location_proportion=} must be strictly between 0 and 1."
        )

    rng = np.random.default_rng(rng_seed)

    dt = 1.0 / bdays_per_year

    break_idx = int(num_obs * break_location_proportion)

    if break_idx < 1 or break_idx >= num_obs:
        raise ValueError(
            f"{break_location_proportion=} produces an invalid "
            f"{break_idx=} for {num_obs=}."
        )

    pre_returns = (
        rng.standard_normal(break_idx) * pre_break_vol_pa * np.sqrt(dt)
        + pre_break_drift_pa * dt
    )

    post_returns = (
        rng.standard_normal(num_obs - break_idx) * post_break_vol_pa * np.sqrt(dt)
        + post_break_drift_pa * dt
    )

    returns = np.concatenate((pre_returns, post_returns))

    levels = init_level * np.exp(np.cumsum(returns))

    # Permanent level shift at the break point
    levels[break_idx:] *= np.exp(break_size)

    return levels
