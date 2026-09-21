# src/riskquant/market_data/curves/zero_curve.py
"""
D080 discount-curve constructor: continuously compounded zero rates -> DF grid.

This is the numeric half of D080. Everything semantic has already happened
upstream in the producer layout plan (`assemble.build_universe_mapping` under
`DiscountingMode.CURVE`): labels resolved to maturities, maturities to the
canonical ACT/365F coordinate `T`, pillars sorted by `T`, duplicates rejected,
and the curve's constructibility (ZERO_RATE + CONTINUOUS + calendar metadata)
proven. What arrives here is numbers.

Mathematics (D080, all float64)
-------------------------------
    z[c, p, s]      D069-normalised decimal zero rate, ALREADY shocked
                    (shocks are additive on the rate, applied upstream in the
                    flat-levels buffer; nothing here shocks anything)
    T[c, p]         (maturity_ord - as_of_ord) / 365.0, strictly increasing, > 0
    lnDF[c, p, s]   = -z[c, p, s] * T[c, p]
    anchor          T = 0, lnDF = 0   (synthetic, unshocked, prepended per curve)
    lnDF(T)         linear in lnDF between pillars on T
    T > T_K         lnDF_K + s_last * (T - T_K), s_last the final segment slope
                    (flat instantaneous forward; forward_last = -s_last)
    DF(T)           = exp(lnDF(T))

Kernel reuse
------------
`interp_linear_on_log_values_linear_extrap_3d_numba_kernel` implements exactly
"linear in the supplied log values on the supplied coordinate, exponentiate at
the end, right tail unclamped". Its FLAT sibling clamps the weight and would
assert zero forward rates beyond the last pillar, which D080 forbids. With the
anchor at `T = 0` and every target `T >= 0`, the unclamped LEFT tail can never
fire, so the linear-extrap kernel is safe on both sides.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.curves.interp_defs import (
    interp_linear_on_log_values_linear_extrap_3d_numba_kernel,
)


def build_zero_curve_discount_factor_grid(
    *,
    pillar_t: npt.NDArray[np.float64],
    pillar_rates: npt.NDArray[np.float64],
    target_t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Build `discount_factor_grid[c, d, s] = DF(as_of, date_d, scenario_s)`.

    Parameters
    ----------
    pillar_t
        `(C, K)` float64, canonical ACT/365F coordinates of the K market pillars
        per curve, strictly increasing and > 0. `K >= 1`.
    pillar_rates
        `(C, K, S)` float64 decimal continuously compounded zero rates, already
        D069-normalised and already scenario-shocked.
    target_t
        `(D,)` float64 canonical coordinates of the run's date universe,
        non-decreasing and >= 0 (`T == 0` is the as-of and yields exactly 1.0).

    Returns
    -------
    `(C, D, S)` float64, C-contiguous, strictly positive.
    """
    c_curves, k_pillars = pillar_t.shape
    if pillar_rates.shape[:2] != (c_curves, k_pillars):
        raise ValueError(
            "zero-curve constructor: pillar_rates shape "
            f"{pillar_rates.shape} does not match pillar_t {pillar_t.shape}."
        )
    n_scen = int(pillar_rates.shape[2]) if pillar_rates.ndim == 3 else 0
    if pillar_rates.ndim != 3:
        raise ValueError("zero-curve constructor: pillar_rates must be (C, K, S).")
    if k_pillars == 0:
        raise ValueError(
            "zero-curve constructor: a constructed curve needs at least one market "
            "pillar; the T = 0 anchor alone is not a curve (D080)."
        )
    if not np.all(np.isfinite(pillar_t)):
        raise ValueError("zero-curve constructor: non-finite pillar coordinate.")
    if not np.all(pillar_t > 0.0):
        raise ValueError(
            "zero-curve constructor: every market pillar must mature after the "
            "as-of (T > 0); the anchor is synthetic (D080)."
        )
    if k_pillars > 1 and not np.all(np.diff(pillar_t, axis=1) > 0.0):
        raise ValueError(
            "zero-curve constructor: pillar coordinates must be strictly increasing "
            "per curve; duplicates and unsorted input are rejected upstream (D080)."
        )
    if not np.all(np.isfinite(pillar_rates)):
        raise ValueError("zero-curve constructor: non-finite zero rate or shock.")
    if not np.all(np.isfinite(target_t)):
        raise ValueError("zero-curve constructor: non-finite target coordinate.")
    if target_t.size and (target_t[0] < 0.0 or np.any(np.diff(target_t) < 0.0)):
        raise ValueError(
            "zero-curve constructor: targets must be non-decreasing and >= 0; a "
            "pre-as-of target is not a supported discounting request (D079)."
        )

    d_dates = int(target_t.shape[0])
    out = np.empty((c_curves, d_dates, n_scen), dtype=np.float64)
    if d_dates == 0 or n_scen == 0 or c_curves == 0:
        return out

    # Anchor prepended: (C, K+1) coordinates, (C, K+1, S) log-DFs. The rate
    # never enters the anchor row; it is exactly 0 for every scenario.
    coords = np.empty((c_curves, k_pillars + 1), dtype=np.float64)
    coords[:, 0] = 0.0
    coords[:, 1:] = pillar_t
    log_df = np.empty((c_curves, k_pillars + 1, n_scen), dtype=np.float64)
    log_df[:, 0, :] = 0.0
    np.multiply(-pillar_rates, pillar_t[:, :, None], out=log_df[:, 1:, :])

    interp_linear_on_log_values_linear_extrap_3d_numba_kernel(
        np.ascontiguousarray(coords),
        np.ascontiguousarray(log_df),
        np.ascontiguousarray(target_t, dtype=np.float64),
        out,
    )
    return out
