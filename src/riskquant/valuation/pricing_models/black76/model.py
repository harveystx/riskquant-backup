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

Domain contract
---------------
The PRICING primitives assume their preconditions hold: `fwd > 0`,
`strike > 0`, `vol_sqrt_tte > 0`. They contain no guards; the compiler
that builds the block owns those checks.

`_black76_fv_vega` is the one exception, and deliberately so. Its
documented output for a non-positive volatility is part of its
mathematical contract rather than input validation: calibration uses
vega as the quality weight on a solved implied volatility, so the
sentinel and degenerate-vol cases must map to zero weight. See the
function docstring.
"""

from __future__ import annotations

import math

from riskquant.jit import inline_njit
from riskquant.valuation.pricing_models.math import (
    numba_norm_cdf_abrom_steg,
    numba_norm_pdf,
)

# =======================
# PRICING
# =======================


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

    log_fk = math.log(fwd / strike)

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
def _black76_fv_vega_kernel_primitive(
    fwd: float,
    log_fk: float,
    sqrt_tte: float,
    vol_sqrt_tte: float,
    vol2_half_tte: float,
) -> float:
    """
    Undiscounted Black-76 vega on pre-decomposed inputs.

    Preconditions: `vol_sqrt_tte > 0`. The guard belongs to
    `_black76_fv_vega`, which owns the domain contract; this primitive is
    pure arithmetic and performs no checks.
    """

    d1 = (log_fk + vol2_half_tte) / vol_sqrt_tte
    return fwd * numba_norm_pdf(d1) * sqrt_tte


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

    Inputs are otherwise assumed to satisfy the Black-76 kernel
    preconditions (`fwd > 0`, `strike > 0`). Validation of those is the
    caller's responsibility.

    Returns 0.0 for non-positive vol or non-positive sqrt_tte. This makes
    vega the natural quality weight on a calibrated IV:

    - ``vol <= 0`` covers the failure-sentinel case (``IV_SOLVER_FAILED_SENTINEL
      = -1.0``) and the degenerate zero-vol case. The guard is applied BEFORE
      any arithmetic so that a negative sentinel never enters the log or the
      division (which would produce a finite but incorrect positive result
      rather than the contracted 0.0).
    - ``sqrt_tte <= 0`` covers zero or negative time-to-expiry inputs, where
      the option has no time value and vega is exactly zero.

    Contract: vega == 0.0  iff  vol <= 0  or  sqrt_tte <= 0.

    Why the guard lives here rather than at a boundary
    --------------------------------------------------
    This is not input validation. The sentinel is a defined point of this
    function's domain, and mapping it to zero weight is the mathematics the
    calibration consumer contracts for. Pushing it upstream would force every
    caller to re-implement the same branch, and a boundary cannot express
    "IV solve failed" without re-deriving the solver's own result.

    RQ-AUD-002 remediation: previously this function omitted the guard,
    so ``vol = -1.0`` (IV_SOLVER_FAILED_SENTINEL) produced ``vol^2 > 0``,
    making ``d1`` finite and returning a large positive vega (35.2 for
    F = K = 100, T = 1) instead of 0.0. ``vol = 0`` at the money produced
    ``0/0`` and therefore NaN under ``error_model="numpy"``. Both caused
    ``CalibratedOptionQuotes.__post_init__`` to raise on any batch containing
    a non-invertible quote.

    D045/VAL-01 re-confirmed the regression was still live in the shipped
    source despite this docstring. The guard below is the fix; do not remove
    it without also rewriting this contract.
    """

    if vol <= 0.0 or sqrt_tte <= 0.0:
        return 0.0

    log_fk = math.log(fwd / strike)

    vol_sqrt_tte = vol * sqrt_tte

    vol2_half_tte = vol * vol * half_tte

    return _black76_fv_vega_kernel_primitive(
        fwd,
        log_fk,
        sqrt_tte,
        vol_sqrt_tte,
        vol2_half_tte,
    )
