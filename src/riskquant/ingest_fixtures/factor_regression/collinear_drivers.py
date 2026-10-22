# src/riskquant/ingest_fixtures/factor_regression/collinear_drivers.py
"""
Case ``collinear_drivers``: driver selection when two drivers carry one signal.

Business question
-----------------
German base-load power ``POWER_DE_BASE`` moves with gas. Two gas hubs are
candidate drivers: ``GAS_TTF`` (Dutch hub) and ``GAS_THE`` (German hub).
``GAS_TTF_THERM`` is the TTF price again, quoted per therm instead of per MWh.
Which driver set gives a usable proxy or sensitivity regression?

Data-generating process
-----------------------
::

    ln GAS_TTF       = GBM, sigma 55 %
    ln GAS_THE       = ln GAS_TTF + b_t,  b = AR(1), phi 0.70, sd 0.25 %
    GAS_TTF_THERM    = GAS_TTF / 29.3071           (MWh per therm conversion)
    d ln POWER_DE_BASE = 0.85 * d ln GAS_TTF + idio, idio sigma 30 %

TTF and THE log returns correlate about 0.998: the design with both is
ill-conditioned, the individual slopes are unstable, and only their sum is
well measured. TTF per MWh and TTF per therm differ by a constant factor, so
their log returns coincide up to rounding and the rank gate must refuse the
design.

Calendars
---------
Gas hubs on TARGET (ICE Endex), power on FX (EEX trades every weekday except
the FX holidays).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from riskquant.ingest_fixtures.factor_regression._generators import (
    DAILY_DT,
    ar1_path,
    case_rng,
    correlated_normals,
    log_brownian_path,
)
from riskquant.ingest_fixtures.factor_regression.calendars import (
    GRID_SIZE,
    GRID_START_DATE,
    GRID_WINDOW_END_DATE,
    Calendar,
    calendar_mask,
)
from riskquant.ingest_fixtures.factor_regression.case import (
    FactorRegressionCase,
    build_series_frame,
    observe,
)

__all__ = ["COLLINEAR_DRIVERS_CASE"]

CASE_NAME: Final = "collinear_drivers"
_CASE_SALT: Final = 4

SIGMA_TTF: Final = 0.55
BASIS_PHI: Final = 0.70
BASIS_SD: Final = 0.0025
POWER_GAS_BETA: Final = 0.85
SIGMA_POWER_IDIO: Final = 0.30
MWH_PER_THERM: Final = 29.3071


def _build() -> FactorRegressionCase:
    rng = case_rng(_CASE_SALT)
    n = GRID_SIZE
    sqrt_dt = float(np.sqrt(DAILY_DT))
    z = correlated_normals(rng, n_steps=n, correlation=np.eye(3, dtype=np.float64))

    ln_ttf = np.empty(n, dtype=np.float64)
    log_brownian_path(
        z[:, 0],
        log_s0=float(np.log(35.0)),
        sigma_per_step=SIGMA_TTF * sqrt_dt,
        drift_per_step=-0.5 * SIGMA_TTF**2 * DAILY_DT,
        out=ln_ttf,
    )
    ln_the = ln_ttf + ar1_path(z[:, 1], phi=BASIS_PHI, stationary_sd=BASIS_SD)

    ln_power = np.empty(n, dtype=np.float64)
    log_brownian_path(
        z[:, 2],
        log_s0=float(np.log(90.0)),
        sigma_per_step=SIGMA_POWER_IDIO * sqrt_dt,
        drift_per_step=-0.5 * SIGMA_POWER_IDIO**2 * DAILY_DT,
        out=ln_power,
    )
    ln_power += POWER_GAS_BETA * (ln_ttf - ln_ttf[0])

    ttf = np.exp(ln_ttf)
    target = calendar_mask(Calendar.TARGET)
    series = (
        observe(
            name="POWER_DE_BASE",
            unit="EUR per MWh",
            grid_values=np.exp(ln_power),
            mask=calendar_mask(Calendar.FX),
        ),
        observe(name="GAS_TTF", unit="EUR per MWh", grid_values=ttf, mask=target),
        observe(
            name="GAS_THE",
            unit="EUR per MWh",
            grid_values=np.exp(ln_the),
            mask=target,
        ),
        observe(
            name="GAS_TTF_THERM",
            unit="EUR per therm",
            grid_values=ttf / MWH_PER_THERM,
            mask=target,
        ),
    )
    return FactorRegressionCase(
        name=CASE_NAME,
        question=(
            "Which gas driver set gives a usable regression for POWER_DE_BASE "
            "when two hubs carry one signal and one hub appears in two units?"
        ),
        series_names=tuple(s.name for s in series),
        window_start=GRID_START_DATE,
        window_end=GRID_WINDOW_END_DATE,
        truth={
            "beta_GAS_TTF": POWER_GAS_BETA,
            "beta_GAS_THE": 0.0,
            "beta_sum_gas": POWER_GAS_BETA,
            "basis_ar1_phi": BASIS_PHI,
            "basis_stationary_sd": BASIS_SD,
            "mwh_per_therm": MWH_PER_THERM,
        },
        frame=build_series_frame(CASE_NAME, series),
    )


COLLINEAR_DRIVERS_CASE: Final[FactorRegressionCase] = _build()
