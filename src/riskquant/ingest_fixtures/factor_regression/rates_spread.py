# src/riskquant/ingest_fixtures/factor_regression/rates_spread.py
"""
Case ``rates_spread``: rate levels that cross zero, spreads and trends.

Business question
-----------------
CHF two-year rates fell through zero in 2025. Does the government yield
``CHF_GOVT_2Y`` move one for one with the swap rate ``CHF_SWAP_2Y``? Is the
swap spread mean reverting? Is the covered-bond spread ``CHF_COVERED_2Y`` minus
swap mean reverting around a constant, or only around a trend?

Data-generating process (all in percent)
----------------------------------------
::

    CHF_SWAP_2Y_t    = 1.60 + glide_t + random walk, sd 3 bp per day
                       glide: straight line from 0 to -1.85 over the grid
    CHF_GOVT_2Y_t    = CHF_SWAP_2Y_t - (0.20 + s_t),  s = AR(1), phi 0.85, sd 4 bp
    CHF_COVERED_2Y_t = CHF_SWAP_2Y_t + 0.05 + 0.25 * years_t + c_t,
                       c = AR(1), phi 0.85, sd 3 bp

The swap rate is integrated and crosses zero, so a log return is undefined:
``LOG_RETURN`` must raise ``TransformDomainError`` and a ``LOG``-scale
combination must raise ``ScaleDomainError``. Rates use ``DIFFERENCE`` and
``LEVEL`` scale. ``SWAP - GOVT`` is stationary around a constant.
``COVERED - SWAP`` is stationary around a linear trend (a spread that widens
by 25 bp a year), so only the ``CONSTANT_TREND`` ADF case matches it.

Calendar
--------
Every series on SIX.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from riskquant.ingest_fixtures.factor_regression._generators import (
    DAILY_DT,
    ar1_path,
    case_rng,
    correlated_normals,
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

__all__ = ["RATES_SPREAD_CASE"]

CASE_NAME: Final = "rates_spread"
# Salt probe (recorded, sandbox): over salts 5 to 39, a drifting-walk swap path
# lands in the 10 % rejection tail of its own CONSTANT_TREND ADF null for
# salts 5, 7 and 11, which matches the nominal size; 13 salts never reach zero.
# Salt 6 is the first salt whose I(1) controls stay inside the acceptance
# region and whose swap path crosses zero. A control that rejects by chance
# would teach the opposite of the case.
_CASE_SALT: Final = 6

SWAP_START: Final = 1.60
SWAP_GLIDE: Final = -1.85
SWAP_DAILY_SD: Final = 0.03
SWAP_SPREAD_MEAN: Final = 0.20
SWAP_SPREAD_PHI: Final = 0.85
SWAP_SPREAD_SD: Final = 0.04
COVERED_SPREAD_INTERCEPT: Final = 0.05
COVERED_SPREAD_TREND_PER_YEAR: Final = 0.25
COVERED_SPREAD_PHI: Final = 0.85
COVERED_SPREAD_SD: Final = 0.03


def _build() -> FactorRegressionCase:
    rng = case_rng(_CASE_SALT)
    n = GRID_SIZE
    z = correlated_normals(rng, n_steps=n, correlation=np.eye(3, dtype=np.float64))
    steps = np.arange(n, dtype=np.float64)

    swap = np.cumsum(z[:, 0] * SWAP_DAILY_SD)
    swap += SWAP_START + SWAP_GLIDE * steps / (n - 1)
    swap_spread = SWAP_SPREAD_MEAN + ar1_path(
        z[:, 1], phi=SWAP_SPREAD_PHI, stationary_sd=SWAP_SPREAD_SD
    )
    covered_spread = (
        COVERED_SPREAD_INTERCEPT
        + COVERED_SPREAD_TREND_PER_YEAR * steps * DAILY_DT
        + ar1_path(z[:, 2], phi=COVERED_SPREAD_PHI, stationary_sd=COVERED_SPREAD_SD)
    )

    six = calendar_mask(Calendar.SIX)
    series = (
        observe(name="CHF_SWAP_2Y", unit="percent", grid_values=swap, mask=six),
        observe(
            name="CHF_GOVT_2Y",
            unit="percent",
            grid_values=swap - swap_spread,
            mask=six,
        ),
        observe(
            name="CHF_COVERED_2Y",
            unit="percent",
            grid_values=swap + covered_spread,
            mask=six,
        ),
    )
    swap_observed = series[0].values
    # Fail loud: the domain-error analyses need a swap level at or below zero.
    if not bool((swap_observed <= 0.0).any()):
        raise ValueError(f"{CASE_NAME}: CHF_SWAP_2Y never reaches zero")
    return FactorRegressionCase(
        name=CASE_NAME,
        question=(
            "Does CHF_GOVT_2Y move one for one with CHF_SWAP_2Y, and which CHF "
            "spreads revert to a constant or only to a trend?"
        ),
        series_names=tuple(s.name for s in series),
        window_start=GRID_START_DATE,
        window_end=GRID_WINDOW_END_DATE,
        truth={
            "difference_beta_CHF_SWAP_2Y": 1.0,
            "swap_spread_mean": SWAP_SPREAD_MEAN,
            "swap_spread_ar1_phi": SWAP_SPREAD_PHI,
            "swap_spread_stationary_sd": SWAP_SPREAD_SD,
            "covered_spread_intercept": COVERED_SPREAD_INTERCEPT,
            "covered_spread_trend_per_year": COVERED_SPREAD_TREND_PER_YEAR,
            "covered_spread_ar1_phi": COVERED_SPREAD_PHI,
            "covered_spread_stationary_sd": COVERED_SPREAD_SD,
        },
        frame=build_series_frame(CASE_NAME, series),
    )


RATES_SPREAD_CASE: Final[FactorRegressionCase] = _build()
