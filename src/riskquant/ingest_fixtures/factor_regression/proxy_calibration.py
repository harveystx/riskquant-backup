# src/riskquant/ingest_fixtures/factor_regression/proxy_calibration.py
"""
Case ``proxy_calibration``: calibrate Proxy V1 loadings for a short-history factor.

Business question
-----------------
``SMALLCAP_EQ_EU`` has a short and gappy history: it listed on 2024-06-03 and
does not trade on about 6 % of the later TARGET days. The risk engine cannot
build a historical scenario for it from its own history, so it needs a Proxy
V1 row (D088) of the form::

    X_target = beta_EU * Y(EQ_INDEX_EU, LOG_RETURN, h)
             + beta_US * Y(EQ_INDEX_US, LOG_RETURN, h)

with no intercept. What loadings should that row carry, and at what horizon
can the regression measure them?

Data-generating process
-----------------------
Every weekday has two sessions on a latent half-day grid:

* session 1 ends at the European close and carries 60 % of the daily
  variance;
* session 2 ends at the US close and carries 40 %.

Latent log levels move in both sessions. Index returns correlate 0.75 inside
one session. The target loads on the latent indices::

    d ln SMALLCAP = 0.90 * d ln EQ_INDEX_EU + 0.30 * d ln EQ_INDEX_US + idio

``EQ_INDEX_EU`` and ``SMALLCAP_EQ_EU`` are observed at the European close,
``EQ_INDEX_US`` at the US close. A one-day return of the US index therefore
covers a window shifted by one session against the target. The daily
regression understates the US loading (non-synchronous closes). Over ``h``
days the shift is one session out of ``2h``, so the bias falls like ``1/h``.
``population_loadings`` gives the exact biased loadings at every ``h``.
Overlapping ``h``-day rows need a HAC covariance with lag ``>= h - 1``.

Calendars
---------
``EQ_INDEX_EU`` and the target on TARGET, ``EQ_INDEX_US`` on NYSE.
"""

from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np
import numpy.typing as npt

from riskquant.ingest_fixtures.factor_regression._generators import (
    DAILY_DT,
    case_rng,
    correlated_normals,
)
from riskquant.ingest_fixtures.factor_regression.calendars import (
    GRID_SIZE,
    GRID_START_DATE,
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

__all__ = ["PROXY_CALIBRATION_CASE", "population_loadings"]

CASE_NAME: Final = "proxy_calibration"
_CASE_SALT: Final = 2

BETA_EU: Final = 0.90
BETA_US: Final = 0.30
SIGMA_EQ_INDEX_EU: Final = 0.18
SIGMA_EQ_INDEX_US: Final = 0.17
SIGMA_TARGET_IDIO: Final = 0.20
RHO_INDICES: Final = 0.75
SESSION_1_VARIANCE_SHARE: Final = 0.60
TARGET_LISTING_DATE: Final = date(2024, 6, 3)
TARGET_NO_TRADE_PROBABILITY: Final = 0.06


def _build() -> FactorRegressionCase:
    rng = case_rng(_CASE_SALT)
    n = GRID_SIZE
    corr = np.array(
        [[1.0, RHO_INDICES, 0.0], [RHO_INDICES, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Half-day grid: position 2t is the European close of day t, 2t + 1 the
    # US close of day t.
    z = correlated_normals(rng, n_steps=2 * n, correlation=corr)
    share = np.empty(2 * n, dtype=np.float64)
    share[0::2] = SESSION_1_VARIANCE_SHARE
    share[1::2] = 1.0 - SESSION_1_VARIANCE_SHARE
    session_scale = np.sqrt(share * DAILY_DT)

    latent = np.empty((2 * n, 3), dtype=np.float64, order="F")
    for j, sigma in enumerate((
        SIGMA_EQ_INDEX_EU,
        SIGMA_EQ_INDEX_US,
        SIGMA_TARGET_IDIO,
    )):
        col = latent[:, j]
        np.multiply(z[:, j], session_scale * sigma, out=col)
        col -= 0.5 * sigma * sigma * share * DAILY_DT
        np.cumsum(col, out=col)
    ln_eu = latent[:, 0] + np.log(4800.0)
    ln_us = latent[:, 1] + np.log(5200.0)
    ln_target = BETA_EU * latent[:, 0] + BETA_US * latent[:, 1] + latent[:, 2]
    ln_target += np.log(12.0)

    eu_close = slice(0, 2 * n, 2)
    us_close = slice(1, 2 * n, 2)

    target_days = calendar_mask(Calendar.TARGET)
    traded = rng.random(n) >= TARGET_NO_TRADE_PROBABILITY
    listed = np.zeros(n, dtype=np.bool_)
    listed[grid_index_of(TARGET_LISTING_DATE) :] = True
    series = (
        observe(
            name="SMALLCAP_EQ_EU",
            unit="EUR",
            grid_values=np.exp(ln_target[eu_close]),
            mask=target_days & listed & traded,
        ),
        observe(
            name="EQ_INDEX_EU",
            unit="index points",
            grid_values=np.exp(ln_eu[eu_close]),
            mask=target_days,
        ),
        observe(
            name="EQ_INDEX_US",
            unit="index points",
            grid_values=np.exp(ln_us[us_close]),
            mask=calendar_mask(Calendar.NYSE),
        ),
    )
    return FactorRegressionCase(
        name=CASE_NAME,
        question=(
            "Which Proxy V1 loadings on EQ_INDEX_EU and EQ_INDEX_US reproduce "
            "the log returns of the short-history SMALLCAP_EQ_EU, and at what "
            "horizon can a regression measure them?"
        ),
        series_names=tuple(s.name for s in series),
        window_start=GRID_START_DATE,
        window_end=GRID_WINDOW_END_DATE,
        truth={
            "beta_EQ_INDEX_EU": BETA_EU,
            "beta_EQ_INDEX_US": BETA_US,
            "intercept": 0.0,
            "session_1_variance_share": SESSION_1_VARIANCE_SHARE,
            "target_idio_sigma": SIGMA_TARGET_IDIO,
        },
        frame=build_series_frame(CASE_NAME, series),
    )


def population_loadings(steps: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    """
    Exact population loadings ``(beta_EU(h), beta_US(h))`` of the documented process.

    The regression of ``h``-day target log returns on ``h``-day ``EQ_INDEX_EU``
    and ``EQ_INDEX_US`` log returns has no intercept, so the population
    loadings solve the 2 x 2 normal equations ``M @ beta = r`` with
    ``M = E[x x^T]`` and ``r = E[x y]``. The drift of the process contributes
    mean products of relative size ``sigma**2 * h * dt / 4`` (about 1e-5 at
    ``h = 20``) to these moments; the closed form keeps the covariance part
    only.

    Session bookkeeping
    -------------------
    The target and ``EQ_INDEX_EU`` returns cover ``h`` session-1 and ``h``
    session-2 increments. The ``EQ_INDEX_US`` return covers the same
    session-1 increments and the next ``h`` session-2 increments: it drops the
    first session-2 increment of the target window and adds one after it.
    Increments of different sessions are independent, so each covariance is
    the sum of the per-session covariances over the shared sessions:

    * ``Var(X_EU)`` and ``Var(X_US)`` count ``h`` sessions of each type;
    * ``Cov(X_EU, X_US)`` and ``Cov(Y, X_US)`` count ``h`` session-1 and
      ``h - 1`` session-2 increments (the shared ones);
    * ``Cov(Y, X_EU)`` counts ``h`` sessions of each type.

    The US bias is the missing session-2 term: one of ``2h``, so it falls
    like ``1/h``. As ``h`` grows, ``beta_US(h)`` rises toward ``BETA_US`` from
    below and ``beta_EU(h)`` falls toward ``BETA_EU`` from above.

    Parameters
    ----------
    steps
        ``(n,)`` horizons ``h >= 1`` in grid days.

    Returns
    -------
    ``(n, 2)`` float64, C-order: column 0 is ``beta_EU(h)``, column 1 is
    ``beta_US(h)``.

    Raises
    ------
    ValueError
        ``steps`` is not one-dimensional or holds a value below 1.
    """
    h = np.asarray(steps, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError(f"steps must be one-dimensional, got shape {h.shape}")
    if h.shape[0] and float(h.min()) < 1.0:
        raise ValueError("every step must be >= 1")
    # Per-session second moments of one increment, by session type.
    share = np.array(
        [SESSION_1_VARIANCE_SHARE, 1.0 - SESSION_1_VARIANCE_SHARE], dtype=np.float64
    )
    var_eu = SIGMA_EQ_INDEX_EU * SIGMA_EQ_INDEX_EU * share * DAILY_DT
    var_us = SIGMA_EQ_INDEX_US * SIGMA_EQ_INDEX_US * share * DAILY_DT
    cov_eu_us = RHO_INDICES * SIGMA_EQ_INDEX_EU * SIGMA_EQ_INDEX_US * share * DAILY_DT
    # Session counts: full window (h, h); shared with the US window (h, h - 1).
    n_full = np.stack([h, h], axis=1)
    n_shared = np.stack([h, h - 1.0], axis=1)
    m_eu_eu = n_full @ var_eu
    m_us_us = n_full @ var_us
    m_eu_us = n_shared @ cov_eu_us
    r_eu = n_full @ (BETA_EU * var_eu + BETA_US * cov_eu_us)
    r_us = n_shared @ (BETA_EU * cov_eu_us + BETA_US * var_us)
    det = m_eu_eu * m_us_us - m_eu_us * m_eu_us
    out = np.empty((h.shape[0], 2), dtype=np.float64)
    np.divide(m_us_us * r_eu - m_eu_us * r_us, det, out=out[:, 0])
    np.divide(m_eu_eu * r_us - m_eu_us * r_eu, det, out=out[:, 1])
    return out


PROXY_CALIBRATION_CASE: Final[FactorRegressionCase] = _build()
