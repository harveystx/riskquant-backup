# src/riskquant/valuation/pricing_models/black76/model.py
"""
Black 76 pricing model primitives.

Scalar Black-76 mathematical primitives.

This module contains the core Black-76 pricing model expressed as
small, allocation-free functions operating on scalar inputs.

Architectural role
------------------
Functions in this module represent Black-76 mathematics only.

They do not know about:

    * portfolios
    * valuation states
    * blocks
    * market-data containers
    * discounting conventions
    * FX conversion
    * settlement mechanics
    * batch execution

They are intended to be:

    * reusable
    * inline-able
    * allocation-free
    * independent of the wider RiskQuant valuation framework

Typical consumers include:

    * product valuation kernels
    * model-evaluation kernels
    * calibration routines
    * implied-volatility solvers
    * research tooling

The functions in this module form the single source of truth for
Black-76 mathematics throughout RiskQuant.
"""

from __future__ import annotations

import math

from riskquant.numba import inline_njit
from riskquant.valuation.pricing_models.math import (
    numba_norm_cdf_abrom_steg,
    numba_norm_pdf,
)

# =======================
# PRICING
# =======================


# @inline_njit
# def _black76_fv_with_kernel_hoists(
#     fwd: float,
#     log_fk: float,
#     strike: float,
#     sqrt_tte: float,
#     half_tte: float,
#     vol: float,
#     cp_mult: int,
# ) -> float:
#     """Undiscounted Black 76 option price. Returns FV, not PV. Caller is responsible
#     for multiplying by the discount factor to get PV."""

#     std_dev = vol * sqrt_tte

#     d1 = (log_fk + vol * vol * half_tte) / std_dev
#     d2 = d1 - std_dev

#     cdf_d1 = numba_norm_cdf_abrom_steg(cp_mult * d1)
#     cdf_d2 = numba_norm_cdf_abrom_steg(cp_mult * d2)

#     fv = cp_mult * (fwd * cdf_d1 - strike * cdf_d2)

#     return fv


@inline_njit
def _black76_fv_kernel_primitive(
    fwd: float,
    log_fk: float,
    strike: float,
    vol_sqrt_tte: float,
    vol2_half_tte: float,
    cp_mult: int,
) -> float:
    """Undiscounted Black 76 option price. Returns FV, not PV. Caller is responsible
    for multiplying by the discount factor to get PV."""

    d1 = (log_fk + vol2_half_tte) / vol_sqrt_tte
    d2 = d1 - vol_sqrt_tte

    cdf_d1 = numba_norm_cdf_abrom_steg(cp_mult * d1)
    cdf_d2 = numba_norm_cdf_abrom_steg(cp_mult * d2)

    fv = cp_mult * (fwd * cdf_d1 - strike * cdf_d2)

    return fv


@inline_njit
def _black76_fv(
    fwd: float,
    strike: float,
    sqrt_tte: float,
    half_tte: float,
    vol: float,
    cp_mult: int,
) -> float:
    """Undiscounted Black 76 option price. Returns FV, not PV. Caller is responsible
    for multiplying by the discount factor to get PV."""

    log_fwd = math.log(fwd)
    log_strike = math.log(strike)
    log_fk = log_fwd - log_strike

    vol_sqrt_tte = vol * sqrt_tte
    vol2_half_tte = vol * vol * half_tte

    return _black76_fv_kernel_primitive(
        fwd,
        log_fk,
        strike,
        vol_sqrt_tte,
        vol2_half_tte,
        cp_mult,
    )


@inline_njit
def _black76_fv_call(
    fwd: float,
    strike: float,
    sqrt_tte: float,
    half_tte: float,
    vol: float,
) -> float:
    """Undiscounted Black 76 call price. Returns FV, not PV. Caller is
    responsible for multiplying by the discount factor to get PV."""
    return _black76_fv(fwd, strike, sqrt_tte, half_tte, vol, 1)


@inline_njit
def _black76_fv_put(
    fwd: float,
    strike: float,
    sqrt_tte: float,
    half_tte: float,
    vol: float,
) -> float:
    """Undiscounted Black 76 put price. Returns FV, not PV. Caller is
    responsible for multiplying by the discount factor to get PV."""
    return _black76_fv(fwd, strike, sqrt_tte, half_tte, vol, -1)


# ======================
# SENSITIVITIES
# ======================

# --- DELTA ---

# TODO: Implement Black-76 delta kernel.

# --- GAMMA ---

# TODO: Implement Black-76 gamma kernel.

# --- VEGA ---


@inline_njit
def _black76_fv_vega(
    fwd: float,
    strike: float,
    sqrt_tte: float,
    half_tte: float,
    vol: float,
) -> float:
    """
    Undiscounted Black-76 vega: dC/dvol = F * phi(d1) * sqrt(T).

    Returns 0.0 for non-positive vol or tte. This makes vega the natural
    quality weight on a calibrated IV: it is 0 for failed solves (sentinel
    vol < 0) and for at-intrinsic zero-vol solves, tiny in flat-vega regions
    (deep ITM/OTM, ultra-short tenors) where the IV is under-determined, and
    large where the inversion is well-conditioned.
    """
    if vol <= 0.0 or sqrt_tte <= 0.0:
        return 0.0

    log_fk = math.log(fwd) - math.log(strike)

    std_dev = vol * sqrt_tte

    d1 = (log_fk + vol * vol * half_tte) / std_dev

    pdf_d1 = numba_norm_pdf(d1)

    return fwd * pdf_d1 * sqrt_tte
