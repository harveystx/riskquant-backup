# src/riskquant/market_data/calibrate/implied_vol/black76/kernels/black76_implied_vol_scalar.py
"""
This module contains the core logic for solving for Black-76 implied volatilities
for individual options, implemented as Numba-jitted functions for maximum performance.
"""

from __future__ import annotations

import numpy as np

from riskquant.constants import IV_SOLVER_FAILED_SENTINEL
from riskquant.jit import inline_njit
from riskquant.valuation.pricing_models.black76.model import _black76_fv_call

_EPSILON = np.finfo(np.float64).eps

_DEFAULT_MAX_ITER = 100

_DEFAULT_ROOT_TOL = 1e-8
_LOWER_VOL = 1e-8
_UPPER_VOL = 10.0


@inline_njit
def _solver_black76_iv_from_undiscounted_call_prices_brent(
    fwd: float,
    strike: float,
    tte: float,
    target_call_price_undiscounted: float,
) -> float:
    # -1. Bind
    _iv_solver_failed = IV_SOLVER_FAILED_SENTINEL

    # 0. Fail fast
    fwd_price_tol = 1e-12 * max(fwd, 1.0)
    if target_call_price_undiscounted > fwd + fwd_price_tol:
        return _iv_solver_failed
    if tte <= 0.0:
        return _iv_solver_failed
    if fwd <= 0.0:
        return _iv_solver_failed
    if strike <= 0.0:
        return _iv_solver_failed

    sqrt_tte = np.sqrt(tte)
    half_tte = 0.5 * tte

    # 1. Intrinsic value check
    intrinsic = max(fwd - strike, 0.0)
    intrinsic_tol = 1e-12 * max(fwd, strike, 1.0)
    if target_call_price_undiscounted < intrinsic - intrinsic_tol:
        return _iv_solver_failed
    if target_call_price_undiscounted <= intrinsic + intrinsic_tol:
        return 0.0

    # 2. Bracket Initialization
    a = _LOWER_VOL
    b = _UPPER_VOL
    fa = (
        _black76_fv_call(
            fwd=fwd,
            strike=strike,
            sqrt_tte=sqrt_tte,
            half_tte=half_tte,
            vol=a,
        )
        - target_call_price_undiscounted
    )
    fb = (
        _black76_fv_call(
            fwd=fwd,
            strike=strike,
            sqrt_tte=sqrt_tte,
            half_tte=half_tte,
            vol=b,
        )
        - target_call_price_undiscounted
    )

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return _iv_solver_failed

    if abs(fa) < abs(fb):  # Ensure b is the best guess
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    eps = _EPSILON

    max_iter = _DEFAULT_MAX_ITER
    vol_tol = _DEFAULT_ROOT_TOL

    # 3. Brent's Iteration Loop
    for _ in range(max_iter):
        # Always maintain 'b' as the best guess.
        #
        # NR zbrent rotation, executed sequentially: a=b, b=c, c=a. Because the
        # trailing ``c = a`` runs AFTER ``a = b``, it resolves to the OLD b, so
        # 'a' and 'c' both take the previous best point. Preserving the OLD a in
        # 'c' instead (the prior bug) silently aliases 'b' and 'c' in the
        # re-bracketed state, yielding xm == 0 and a false convergence that
        # returns a non-repricing vol (often the lower bracket bound).
        if abs(fc) < abs(fb):
            a = b
            fa = fb

            b = c
            fb = fc

            c = a
            fc = fa

        xm = 0.5 * (c - b)
        vol_tol1 = 2.0 * eps * abs(b) + 0.5 * vol_tol

        # Convergence check
        if abs(xm) <= vol_tol1 or fb == 0.0:
            return float(b)

        # Attempt Secant or Inverse Quadratic Interpolation
        if abs(e) >= vol_tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Secant method (only two distinct points available)
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation (three distinct points)
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            p = abs(p)

            # Check if interpolation is acceptable and within bounds
            min1 = 3.0 * xm * q - abs(vol_tol1 * q)
            min2 = abs(e * q)
            if 2.0 * p < (min1 if min1 < min2 else min2):
                e = d
                d = p / q
            else:
                # Interpolation failed, fallback to bisection
                d = xm
                e = d
        else:
            # Bounds tightening too slow, fallback to bisection
            d = xm
            e = d

        a = b
        fa = fb

        # Advance the best guess
        if abs(d) > vol_tol1:
            b += d
        else:
            b += vol_tol1 if xm >= 0.0 else -vol_tol1

        b = float(b)

        # Evaluate the new point
        fb = (
            _black76_fv_call(
                fwd=fwd,
                strike=strike,
                sqrt_tte=sqrt_tte,
                half_tte=half_tte,
                vol=b,
            )
            - target_call_price_undiscounted
        )

        # Re-bracket the root
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c = a
            fc = fa
            d = b - a
            e = d

    # Failed to converge within max_iter
    return _iv_solver_failed
