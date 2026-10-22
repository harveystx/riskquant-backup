# src/riskquant/ingest_fixtures/factor_regression/credit_factor_loadings.py
"""
Case ``credit_factor_loadings``: estimate obligor factor loadings for credit VaR.

Business question
-----------------
The multi-factor Gaussian copula of the credit VaR engine needs, per obligor,
loadings on the systematic factors and the idiosyncratic weight
``sqrt(1 - R**2)``. With equity log returns as the asset-return proxy (Merton
view), what loadings on ``CREDIT_FACTOR_GLOBAL`` and ``CREDIT_FACTOR_EUROPE``
does each obligor carry, and does a US obligor load on the European factor?

Data-generating process
-----------------------
Factor log returns are independent with annual volatility 18 % (global) and
12 % (Europe). With ``x_k`` the standardized factor returns and ``a_ik`` the
standardized loadings::

    d ln S_i = sigma_i * ( a_ig x_g + a_ie x_e + sqrt(1 - a_ig**2 - a_ie**2) eps_i )

so the copula correlation of obligor ``i`` is ``R_i**2 = a_ig**2 + a_ie**2``
and the OLS slope on factor ``k`` is ``beta_ik = sigma_i * a_ik / sigma_k``.

A stress regime (2024-08-01 to 2024-10-31) multiplies every innovation by
2.5. Standardized loadings and ``R**2`` do not change, but the residual
variance moves with the regressors. IID and HC0 standard errors then differ.

The window starts on 2023-07-03, half a year after the first observation, so
``align_levels`` records ``OUTSIDE_WINDOW`` exclusions. Weekly rows
(``step = stride = 5``, non-overlapping) are the usual calibration sample.

Calendars
---------
Factor indices on FX, European obligors on TARGET, the US obligor on NYSE.
"""

from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np

from riskquant.ingest_fixtures.factor_regression._generators import (
    DAILY_DT,
    case_rng,
    correlated_normals,
    log_brownian_path,
)
from riskquant.ingest_fixtures.factor_regression.calendars import (
    GRID_SIZE,
    GRID_WINDOW_END_DATE,
    Calendar,
    calendar_mask,
    grid_index_of,
)
from riskquant.ingest_fixtures.factor_regression.case import (
    FactorRegressionCase,
    build_series_frame,
    observe,
)

__all__ = ["CREDIT_FACTOR_LOADINGS_CASE"]

CASE_NAME: Final = "credit_factor_loadings"
_CASE_SALT: Final = 3

SIGMA_GLOBAL: Final = 0.18
SIGMA_EUROPE: Final = 0.12
STRESS_START: Final = date(2024, 8, 1)
STRESS_END: Final = date(2024, 10, 31)  # inclusive
STRESS_MULTIPLIER: Final = 2.5
WINDOW_START: Final = date(2023, 7, 3)

# (name, unit, calendar, s0, sigma_i, a_global, a_europe)
_OBLIGORS: Final = (
    ("OBLIGOR_EU_BANK", "EUR", Calendar.TARGET, 11.0, 0.30, 0.50, 0.40),
    ("OBLIGOR_EU_UTILITY", "EUR", Calendar.TARGET, 24.0, 0.22, 0.35, 0.25),
    ("OBLIGOR_US_INDUSTRIAL", "USD", Calendar.NYSE, 140.0, 0.28, 0.55, 0.0),
)


def _build() -> FactorRegressionCase:
    rng = case_rng(_CASE_SALT)
    n = GRID_SIZE
    n_obligors = len(_OBLIGORS)
    z = correlated_normals(
        rng, n_steps=n, correlation=np.eye(2 + n_obligors, dtype=np.float64)
    )
    regime = np.ones(n, dtype=np.float64)
    regime[grid_index_of(STRESS_START) : grid_index_of(STRESS_END) + 1] = (
        STRESS_MULTIPLIER
    )
    z *= regime[:, np.newaxis]
    sqrt_dt = float(np.sqrt(DAILY_DT))

    ln_factors = np.empty((n, 2), dtype=np.float64, order="F")
    for j, (s0, sigma) in enumerate(((1000.0, SIGMA_GLOBAL), (500.0, SIGMA_EUROPE))):
        log_brownian_path(
            z[:, j],
            log_s0=float(np.log(s0)),
            sigma_per_step=sigma * sqrt_dt,
            drift_per_step=-0.5 * sigma * sigma * DAILY_DT,
            out=ln_factors[:, j],
        )

    fx = calendar_mask(Calendar.FX)
    observations = [
        observe(
            name="CREDIT_FACTOR_GLOBAL",
            unit="index points",
            grid_values=np.exp(ln_factors[:, 0]),
            mask=fx,
        ),
        observe(
            name="CREDIT_FACTOR_EUROPE",
            unit="index points",
            grid_values=np.exp(ln_factors[:, 1]),
            mask=fx,
        ),
    ]
    truth: dict[str, float] = {
        "factor_sigma_CREDIT_FACTOR_GLOBAL": SIGMA_GLOBAL,
        "factor_sigma_CREDIT_FACTOR_EUROPE": SIGMA_EUROPE,
    }
    ln_obligor = np.empty(n, dtype=np.float64)
    for i, (name, unit, calendar, s0, sigma_i, a_g, a_e) in enumerate(_OBLIGORS):
        r_squared = a_g * a_g + a_e * a_e
        shock = a_g * z[:, 0] + a_e * z[:, 1] + np.sqrt(1.0 - r_squared) * z[:, 2 + i]
        log_brownian_path(
            shock,
            log_s0=float(np.log(s0)),
            sigma_per_step=sigma_i * sqrt_dt,
            drift_per_step=-0.5 * sigma_i * sigma_i * DAILY_DT,
            out=ln_obligor,
        )
        observations.append(
            observe(
                name=name,
                unit=unit,
                grid_values=np.exp(ln_obligor),
                mask=calendar_mask(calendar),
            )
        )
        truth[f"loading_global_{name}"] = a_g
        truth[f"loading_europe_{name}"] = a_e
        truth[f"r_squared_{name}"] = r_squared
        truth[f"beta_CREDIT_FACTOR_GLOBAL_{name}"] = sigma_i * a_g / SIGMA_GLOBAL
        truth[f"beta_CREDIT_FACTOR_EUROPE_{name}"] = sigma_i * a_e / SIGMA_EUROPE

    series = tuple(observations)
    return FactorRegressionCase(
        name=CASE_NAME,
        question=(
            "Which loadings on CREDIT_FACTOR_GLOBAL and CREDIT_FACTOR_EUROPE does "
            "each obligor carry, and does the US obligor load on Europe?"
        ),
        series_names=tuple(s.name for s in series),
        window_start=WINDOW_START,
        window_end=GRID_WINDOW_END_DATE,
        truth=truth,
        frame=build_series_frame(CASE_NAME, series),
    )


CREDIT_FACTOR_LOADINGS_CASE: Final[FactorRegressionCase] = _build()
