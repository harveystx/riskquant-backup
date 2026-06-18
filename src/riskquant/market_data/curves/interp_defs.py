# src/riskquant/market_data/curves/interp_defs.py
"""
INTERP DEFS

Interpolation kernels used by RiskQuant.

Input conventions
-----------------

All kernels operate directly on the values supplied in the ``values``
array.

For kernels operating on log values:

    * ``log_values`` MUST already contain logarithms of the quantity
      being interpolated.
    * The kernel interpolates linearly in log-space.
    * The result is transformed back to original space via ``exp()``.

Example
-------

To interpolate discount factors log-linearly:

    logged_dfs = np.log(discount_factors)

    interp_linear_on_log_values_flat_extrap_3d_numba_kernel(
        tenors,
        logged_dfs,
        targets,
        out,
    )

Passing raw discount factors to a LOG kernel will produce incorrect
results.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    _F64_3D,
    parallel_njit,
)

# --- HELPERS ---

_NB_VOID_3D = nb.void(
    _F64_2D,  # tenors
    _F64_3D,  # values
    _F64_1D,  # targets
    _F64_3D,  # out
)

_NB_VOID_2D = nb.void(
    _F64_2D,  # pillars
    _F64_3D,  # values
    _F64_2D,  # targets
    _F64_2D,  # out
)

# ==============================================================================
# --- LINEAR INTERPOLATION WITH FLAT EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_linear_flat_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zero-allocation 3D Linear Interpolator with Flat Extrapolation."""
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]

            idx = np.searchsorted(tenors[c], target, side="right")
            idx1 = min(max(idx, 1), m_pillars - 1)
            idx0 = idx1 - 1

            t0 = tenors[c, idx0]
            t1 = tenors[c, idx1]

            dt = t1 - t0
            if dt == 0.0:
                dt = 1.0

            w = (target - t0) / dt
            w = min(max(w, 0.0), 1.0)

            for s in range(s_scenarios):
                y0 = values[c, idx0, s]
                y1 = values[c, idx1, s]
                out[c, d, s] = y0 + w * (y1 - y0)


# ==============================================================================
# --- LINEAR INTERPOLATION WITH LINEAR EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_linear_linear_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zero-allocation 3D Linear Interpolator with Linear Extrapolation."""
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]

            idx = np.searchsorted(tenors[c], target, side="right")
            idx1 = min(max(idx, 1), m_pillars - 1)
            idx0 = idx1 - 1

            t0 = tenors[c, idx0]
            t1 = tenors[c, idx1]

            dt = t1 - t0
            if dt == 0.0:
                dt = 1.0

            w = (target - t0) / dt
            # LINEAR EXTRAPOLATION: Deliberately do not clamp w

            for s in range(s_scenarios):
                y0 = values[c, idx0, s]
                y1 = values[c, idx1, s]
                out[c, d, s] = y0 + w * (y1 - y0)


# ==============================================================================
# --- LINEAR INTERPOLATION ON LOG VALUES WITH FLAT EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_linear_on_log_values_flat_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    log_values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Log-linear interpolation of discount factors.

    Interpolates linearly in log-discount-factor space and then exponentiates.

    If discount factors are represented as

        D(T) = exp(-z(T) * T)

    this corresponds to piecewise-linear interpolation of the
    discount-factor exponent (-z(T)T), not necessarily piecewise-linear
    interpolation of zero rates themselves.
    """
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = log_values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]

            idx = np.searchsorted(tenors[c], target, side="right")
            idx1 = min(max(idx, 1), m_pillars - 1)
            idx0 = idx1 - 1

            t0 = tenors[c, idx0]
            t1 = tenors[c, idx1]

            dt = t1 - t0
            if dt == 0.0:
                dt = 1.0

            w = (target - t0) / dt
            w = min(max(w, 0.0), 1.0)

            for s in range(s_scenarios):
                y0 = log_values[c, idx0, s]
                y1 = log_values[c, idx1, s]
                ln_interp = y0 + w * (y1 - y0)
                out[c, d, s] = math.exp(ln_interp)


# ==============================================================================
# --- LOG-LINEAR INTERPOLATION WITH LINEAR EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_linear_on_log_values_linear_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    log_values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 3D Log-Linear Interpolator with Linear Extrapolation.

    Same code shape as the flat-extrap sibling EXCEPT the weight `w =
    (target - t0) / dt` is not clamped to `[0, 1]`. On targets past
    the last pillar, `w > 1` extrapolates linearly in log-space (i.e.
    the implied zero-rate is extrapolated linearly off the last
    pillar segment, then exponentiated). Inside the pillar range the
    behaviour is identical to the flat-extrap kernel.
    """
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = log_values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]

            idx = np.searchsorted(tenors[c], target, side="right")
            idx1 = min(max(idx, 1), m_pillars - 1)
            idx0 = idx1 - 1

            t0 = tenors[c, idx0]
            t1 = tenors[c, idx1]

            dt = t1 - t0
            if dt == 0.0:
                dt = 1.0

            w = (target - t0) / dt

            for s in range(s_scenarios):
                y0 = log_values[c, idx0, s]
                y1 = log_values[c, idx1, s]
                ln_interp = y0 + w * (y1 - y0)
                out[c, d, s] = math.exp(ln_interp)


# ==============================================================================
# --- STEP LEFT (LOCF) INTERPOLATION WITH FLAT EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_locf_flat_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zero-allocation 3D Step Left (LOCF) Interpolator."""
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]
            # side="right" ensures we find the first index > target.
            # Subtracting 1 gives the index <= target (Last Observation Carried Forward).
            idx = np.searchsorted(tenors[c], target, side="right") - 1
            idx = min(max(idx, 0), m_pillars - 1)

            for s in range(s_scenarios):
                out[c, d, s] = values[c, idx, s]


# ==============================================================================
# --- STEP LEFT (LOCF) INTERPOLATION WITH LINEAR EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_locf_linear_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 3D Step Left (LOCF) Interpolator with linear-extrap label.

    A step function has no segment slope to extrapolate along, so on
    a step function "linear extrapolation" reduces to flat
    extrapolation: the last observed value is carried forward (past
    the last pillar) or backward (before the first). The
    implementation is byte-identical to the flat-extrap kernel; both
    are registered separately in `CURVE_MATH_REGISTRY` for taxonomy
    completeness.

    If a future requirement asks for true slope-extrapolation of a
    piecewise-constant function (e.g. extrap using the slope of the
    last two pillar VALUES rather than the last segment), this is the
    place to fork the implementation.
    """
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]
            idx = np.searchsorted(tenors[c], target, side="right") - 1
            idx = min(max(idx, 0), m_pillars - 1)

            for s in range(s_scenarios):
                out[c, d, s] = values[c, idx, s]


# ==============================================================================
# --- STEP RIGHT (NOCB) INTERPOLATION WITH FLAT EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_nocb_flat_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zero-allocation 3D Step Right (NOCB) Interpolator."""
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]
            # side="left" ensures we find the first index >= target (Next Observation Carried Backward).
            idx = np.searchsorted(tenors[c], target, side="left")
            idx = min(max(idx, 0), m_pillars - 1)

            for s in range(s_scenarios):
                out[c, d, s] = values[c, idx, s]


# ==============================================================================
# --- STEP RIGHT (NOCB) INTERPOLATION WITH LINEAR EXTRAPOLATION ---
# ==============================================================================


@parallel_njit(_NB_VOID_3D)
def interp_nocb_linear_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 3D Step Right (NOCB) Interpolator with linear-extrap label.

    Mirror of `interp_step_left_linear_extrap_3d_numba_kernel` — see
    its docstring. A step function has no segment slope, so "linear
    extrapolation" reduces to flat extrapolation; this kernel is
    byte-identical to the flat-extrap step-right sibling and is
    registered separately in `CURVE_MATH_REGISTRY` for taxonomy
    completeness.
    """
    c_curves, m_pillars = tenors.shape
    d_target_dates = targets.shape[0]
    s_scenarios = values.shape[2]

    for c in nb.prange(c_curves):
        for d in range(d_target_dates):
            target = targets[d]
            idx = np.searchsorted(tenors[c], target, side="left")
            idx = min(max(idx, 0), m_pillars - 1)

            for s in range(s_scenarios):
                out[c, d, s] = values[c, idx, s]


# ==============================================================================
# --- 2D SURFACE STRIKE INTERPOLATOR ---
# ==============================================================================


@parallel_njit(_NB_VOID_2D)
def interp_strike_linear_flat_extrap_numba_kernel(
    pillars: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 2D Surface Strike Interpolator.
    Pre-allocates an (N_trades, S_scenarios) array and resolves exactly.
    """
    n_trades, k_strikes = pillars.shape
    s_scenarios = targets.shape[1]

    for n in nb.prange(n_trades):
        for s in range(s_scenarios):
            target = targets[n, s]

            idx = np.searchsorted(pillars[n], target, side="right")
            idx1 = min(max(idx, 1), k_strikes - 1)
            idx0 = idx1 - 1

            x0 = pillars[n, idx0]
            x1 = pillars[n, idx1]

            dx = x1 - x0
            if dx == 0.0:
                dx = 1.0

            w = (target - x0) / dx
            w = min(max(w, 0.0), 1.0)

            y0 = values[n, idx0, s]
            y1 = values[n, idx1, s]

            out[n, s] = y0 + w * (y1 - y0)
