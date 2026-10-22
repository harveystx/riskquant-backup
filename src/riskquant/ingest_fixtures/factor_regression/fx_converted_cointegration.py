# src/riskquant/ingest_fixtures/factor_regression/fx_converted_cointegration.py
"""
Case ``fx_converted_cointegration``: does a EUR asset track a USD asset in EUR?

Business question
-----------------
Asset ``XLIST_EUR`` (a euro listing) and asset ``XLIST_USD`` (a US listing of
the same claim) trade on different exchanges in different currencies. Is the
EUR level tied to the USD level converted to EUR with ``USDEUR`` (EUR per
USD)? Is the sector peer ``PEER_EUR`` tied to it in the same way?

Data-generating process (grid step = one weekday)
-------------------------------------------------
::

    ln USDEUR_t    = GBM, sigma 8 %
    ln XLIST_USD_t = GBM, sigma 25 %
    u_t            = AR(1), phi 0.90, stationary sd 1.5 %   (half-life 6.6 days)
    ln XLIST_EUR_t = ln XLIST_USD_t + ln USDEUR_t + PREMIUM + u_t
    d ln PEER_EUR  = 0.70 * (d ln XLIST_USD + d ln USDEUR) + idio, idio sigma 18 %
    ln USDGBP_t    = GBM, sigma 7 %, return correlation 0.60 with USDEUR
    EURGBP_t       = USDGBP_t / USDEUR_t                     (exact cross)

The log combination ``ln XLIST_EUR - ln XLIST_USD - ln USDEUR = PREMIUM + u``
is stationary: the pair is cointegrated with the declared vector
``(1, -1, -1)``. Because the vector is declared, not estimated, the N = 1
MacKinnon table applies (Engle-Granger is backlog). The peer shares 70 % of
the return driver but carries an idiosyncratic random walk: its returns are
correlated and its levels are not cointegrated. This is the trap of a level
regression.

``EURGBP`` is an exact function of two other series. A return regression of
``EURGBP`` on ``USDGBP`` and ``USDEUR`` is an identity, not an estimate. It is
the regression form of the Proxy V1 cross identity
``EURCHF <- +1 USDCHF - 1 USDEUR`` of the canonical proxy model.

Calendars
---------
``XLIST_EUR`` and ``PEER_EUR`` on TARGET, ``XLIST_USD`` on NYSE, the FX
series on FX. The intersection drops every TARGET and NYSE holiday.
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

__all__ = ["FX_CONVERTED_COINTEGRATION_CASE"]

CASE_NAME: Final = "fx_converted_cointegration"
_CASE_SALT: Final = 1

SIGMA_USDEUR: Final = 0.08
SIGMA_USDGBP: Final = 0.07
RHO_USDEUR_USDGBP: Final = 0.60
SIGMA_XLIST_USD: Final = 0.25
SPREAD_PHI: Final = 0.90
SPREAD_SD: Final = 0.015
PREMIUM: Final = 0.02
PEER_LOADING: Final = 0.70
SIGMA_PEER_IDIO: Final = 0.18


def _build() -> FactorRegressionCase:
    rng = case_rng(_CASE_SALT)
    n = GRID_SIZE
    sqrt_dt = float(np.sqrt(DAILY_DT))
    corr = np.array(
        [
            [1.0, RHO_USDEUR_USDGBP, 0.0, 0.0, 0.0],
            [RHO_USDEUR_USDGBP, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    z = correlated_normals(rng, n_steps=n, correlation=corr)

    log_paths = np.empty((n, 4), dtype=np.float64, order="F")
    ln_usdeur, ln_usdgbp, ln_usd, ln_peer = (log_paths[:, j] for j in range(4))
    for out, zj, s0, sigma in (
        (ln_usdeur, z[:, 0], 0.92, SIGMA_USDEUR),
        (ln_usdgbp, z[:, 1], 0.79, SIGMA_USDGBP),
        (ln_usd, z[:, 2], 50.0, SIGMA_XLIST_USD),
    ):
        log_brownian_path(
            zj,
            log_s0=float(np.log(s0)),
            sigma_per_step=sigma * sqrt_dt,
            drift_per_step=-0.5 * sigma * sigma * DAILY_DT,
            out=out,
        )
    spread = ar1_path(z[:, 3], phi=SPREAD_PHI, stationary_sd=SPREAD_SD)
    ln_eur = ln_usd + ln_usdeur + PREMIUM + spread

    # Peer: 70 % of the EUR-converted return driver plus an idiosyncratic walk.
    driver_steps = np.diff(ln_usd + ln_usdeur, prepend=ln_usd[0] + ln_usdeur[0])
    log_brownian_path(
        z[:, 4],
        log_s0=float(np.log(46.0)),
        sigma_per_step=SIGMA_PEER_IDIO * sqrt_dt,
        drift_per_step=-0.5 * SIGMA_PEER_IDIO**2 * DAILY_DT,
        out=ln_peer,
    )
    ln_peer += PEER_LOADING * np.cumsum(driver_steps)

    usdeur = np.exp(ln_usdeur)
    usdgbp = np.exp(ln_usdgbp)
    # Exact cross on levels: one IEEE division per date, no log round trip.
    eurgbp = usdgbp / usdeur

    fx = calendar_mask(Calendar.FX)
    target = calendar_mask(Calendar.TARGET)
    nyse = calendar_mask(Calendar.NYSE)
    series = (
        observe(name="XLIST_EUR", unit="EUR", grid_values=np.exp(ln_eur), mask=target),
        observe(name="XLIST_USD", unit="USD", grid_values=np.exp(ln_usd), mask=nyse),
        observe(name="USDEUR", unit="EUR per USD", grid_values=usdeur, mask=fx),
        observe(name="PEER_EUR", unit="EUR", grid_values=np.exp(ln_peer), mask=target),
        observe(name="USDGBP", unit="GBP per USD", grid_values=usdgbp, mask=fx),
        observe(name="EURGBP", unit="GBP per EUR", grid_values=eurgbp, mask=fx),
    )
    return FactorRegressionCase(
        name=CASE_NAME,
        question=(
            "Is the EUR level of XLIST_EUR tied to the USD level of XLIST_USD "
            "converted to EUR, and is the peer PEER_EUR tied in the same way?"
        ),
        series_names=tuple(s.name for s in series),
        window_start=GRID_START_DATE,
        window_end=GRID_WINDOW_END_DATE,
        truth={
            "cointegration_weight_XLIST_EUR": 1.0,
            "cointegration_weight_XLIST_USD": -1.0,
            "cointegration_weight_USDEUR": -1.0,
            "log_premium": PREMIUM,
            "spread_ar1_phi": SPREAD_PHI,
            "spread_stationary_sd": SPREAD_SD,
            "return_beta_XLIST_USD": 1.0,
            "return_beta_USDEUR": 1.0,
            "peer_return_loading": PEER_LOADING,
            "triangle_beta_USDGBP": 1.0,
            "triangle_beta_USDEUR": -1.0,
        },
        frame=build_series_frame(CASE_NAME, series),
    )


FX_CONVERTED_COINTEGRATION_CASE: Final[FactorRegressionCase] = _build()
