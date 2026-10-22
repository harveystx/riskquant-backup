# src/riskquant/ingest_fixtures/factor_regression/_generators.py
"""
Deterministic path primitives for the canonical factor-regression fixture.

Each case draws from its own ``numpy.random.Generator``. The generator seed is
``SeedSequence(CANONICAL_RNG_SEED, spawn_key=(case_salt,))``. Why not
``get_canonical_rng()``: that function returns the same stream to every
caller, so two cases would draw identical innovations and become correlated
by construction. A spawn key gives independent streams that do not depend on
import order.

Layout
------
Innovation blocks are ``(n_steps, n_factors)`` F-order: every path primitive
reads one factor down the time axis, and F-order makes that axis contiguous.
``np.cumsum(..., out=...)`` and ``scipy.signal.lfilter`` run along it in C.
No Python loop runs over time steps.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from riskquant.ingest_fixtures.constants import (
    BUSINESS_DAYS_PER_YEAR,
    CANONICAL_RNG_SEED,
)

__all__ = [
    "DAILY_DT",
    "ar1_path",
    "case_rng",
    "correlated_normals",
    "log_brownian_path",
]

# One grid step is one weekday.
DAILY_DT: float = 1.0 / BUSINESS_DAYS_PER_YEAR


def case_rng(case_salt: int) -> np.random.Generator:
    """Independent generator for one case; see the module docstring."""
    seq = np.random.SeedSequence(CANONICAL_RNG_SEED, spawn_key=(case_salt,))
    return np.random.default_rng(seq)


def correlated_normals(
    rng: np.random.Generator,
    *,
    n_steps: int,
    correlation: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Draw ``(n_steps, n)`` F-order standard normals with correlation ``correlation``.

    Raises
    ------
    ValueError
        ``correlation`` is not square, not symmetric, or not positive definite.
    """
    n = int(correlation.shape[0])
    if correlation.shape != (n, n) or not np.array_equal(correlation, correlation.T):
        raise ValueError("correlation must be a symmetric square matrix")
    # cholesky raises LinAlgError (a ValueError subclass) if not PD.
    chol = np.linalg.cholesky(correlation)
    z = rng.standard_normal((n_steps, n))
    out = np.empty((n_steps, n), dtype=np.float64, order="F")
    np.matmul(z, chol.T, out=out)
    return out


def log_brownian_path(
    z: npt.NDArray[np.float64],
    *,
    log_s0: float,
    sigma_per_step: npt.NDArray[np.float64] | float,
    drift_per_step: float,
    out: npt.NDArray[np.float64],
) -> None:
    """
    Fill ``out`` with ``log_s0 + cumsum(drift + sigma * z)``.

    The first step already carries one increment, so ``out[0] != log_s0``.
    ``sigma_per_step`` may be a ``(n_steps,)`` array: a volatility regime
    scales the step, not the level.
    """
    np.multiply(z, sigma_per_step, out=out)
    out += drift_per_step
    np.cumsum(out, out=out)
    out += log_s0


def ar1_path(
    z: npt.NDArray[np.float64],
    *,
    phi: float,
    stationary_sd: float,
) -> npt.NDArray[np.float64]:
    """
    Stationary AR(1) path ``u_t = phi * u_{t-1} + e_t`` with ``sd(u) = stationary_sd``.

    The first value is drawn from the stationary law, so the path has no
    burn-in transient: ``u_0 = stationary_sd * z_0`` and
    ``e_t = stationary_sd * sqrt(1 - phi**2) * z_t``.

    Raises
    ------
    ValueError
        ``phi`` is outside ``(-1, 1)`` or ``stationary_sd`` is not above 0.
    """
    if not (-1.0 < phi < 1.0):
        raise ValueError(f"phi must be in (-1, 1), got {phi!r}")
    if not (stationary_sd > 0.0):
        raise ValueError(f"stationary_sd must be > 0, got {stationary_sd!r}")
    e = z * (stationary_sd * np.sqrt(1.0 - phi * phi))
    e[0] = stationary_sd * z[0]
    out = np.asarray(lfilter([1.0], [1.0, -phi], e), dtype=np.float64)
    return np.ascontiguousarray(out)
