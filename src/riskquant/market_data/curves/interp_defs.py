# src/riskquant/market_data/curves/interp_defs.py
from __future__ import annotations

import math

import numba as nb
import numpy as np

# ==============================================================================
# --- LINEAR INTERPOLATION WITH FLAT EXTRAPOLATION ---
# ==============================================================================


def interp_linear_flat_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Linear Interpolation with Flat Extrapolation.
    Shapes: tenors (c_curves, m_pillars), values (c_curves, m_pillars, s_scenarios), targets (d_targets,)
    Returns: (c_curves, d_targets, s_scenarios)
    """
    c_curves, m_pillars = tenors.shape
    idx1 = np.clip(
        np.sum(tenors[:, None, :] <= targets[None, :, None], axis=2),
        1,
        m_pillars - 1,
    )
    idx0 = idx1 - 1

    c_idx = np.arange(c_curves)[:, None]
    t0, t1 = tenors[c_idx, idx0], tenors[c_idx, idx1]
    y0, y1 = values[c_idx, idx0, :], values[c_idx, idx1, :]

    dt = np.where((t1 - t0) == 0, 1.0, t1 - t0)
    w = (targets[None, :] - t0) / dt

    # FLAT EXTRAPOLATION: Clamp weights to [0.0, 1.0]
    w = np.clip(w, 0.0, 1.0)

    return y0 + w[:, :, None] * (y1 - y0)


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
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


def interp_linear_linear_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Linear Interpolation with Linear Extrapolation.
    Shapes: tenors (c_curves, m_pillars), values (c_curves, m_pillars, s_scenarios), targets (d_targets,)
    Returns: (c_curves, d_targets, s_scenarios)
    """
    c_curves, m_pillars = tenors.shape
    idx1 = np.clip(
        np.sum(tenors[:, None, :] <= targets[None, :, None], axis=2),
        1,
        m_pillars - 1,
    )
    idx0 = idx1 - 1

    c_idx = np.arange(c_curves)[:, None]
    t0, t1 = tenors[c_idx, idx0], tenors[c_idx, idx1]
    y0, y1 = values[c_idx, idx0, :], values[c_idx, idx1, :]

    dt = np.where((t1 - t0) == 0, 1.0, t1 - t0)
    w = (targets[None, :] - t0) / dt

    # LINEAR EXTRAPOLATION: Do not clamp weights
    return y0 + w[:, :, None] * (y1 - y0)


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
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
# --- LOG-LINEAR INTERPOLATION WITH FLAT EXTRAPOLATION ---
# ==============================================================================


def interp_log_linear_flat_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Log-Linear Interpolation with Constant Forward Extrapolation.
    """
    c_curves, m_pillars = tenors.shape
    mask = tenors[:, None, :] <= targets[None, :, None]
    idx1 = np.sum(mask, axis=2)
    idx1 = np.clip(idx1, 1, m_pillars - 1)
    idx0 = idx1 - 1

    c_idx = np.arange(c_curves)[:, None]
    t0 = tenors[c_idx, idx0]
    t1 = tenors[c_idx, idx1]

    y0 = values[c_idx, idx0, :]
    y1 = values[c_idx, idx1, :]

    dt = t1 - t0
    dt = np.where(dt == 0, 1.0, dt)
    w = (targets[None, :] - t0) / dt

    ln_interp = y0 + w[:, :, None] * (y1 - y0)
    return np.exp(ln_interp)


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
def interp_log_linear_flat_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 3D Log-Linear Interpolator.
    Mathematically equivalent to linear interpolation on Zero Rates.
    """
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

            for s in range(s_scenarios):
                y0 = values[c, idx0, s]
                y1 = values[c, idx1, s]
                ln_interp = y0 + w * (y1 - y0)
                out[c, d, s] = math.exp(ln_interp)


# ==============================================================================
# --- LOG-LINEAR INTERPOLATION WITH LINEAR EXTRAPOLATION ---
# ==============================================================================


def interp_log_linear_linear_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Log-Linear Interpolation with Linear Extrapolation.
    """
    c_curves, m_pillars = tenors.shape
    idx1 = np.clip(
        np.sum(tenors[:, None, :] <= targets[None, :, None], axis=2),
        1,
        m_pillars - 1,
    )
    idx0 = idx1 - 1

    c_idx = np.arange(c_curves)[:, None]
    t0, t1 = tenors[c_idx, idx0], tenors[c_idx, idx1]
    y0, y1 = values[c_idx, idx0, :], values[c_idx, idx1, :]

    dt = np.where((t1 - t0) == 0, 1.0, t1 - t0)
    w = (targets[None, :] - t0) / dt

    ln_interp = y0 + w[:, :, None] * (y1 - y0)
    return np.exp(ln_interp)


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
def interp_log_linear_linear_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 3D Log-Linear Interpolator with Linear Extrapolation.
    Mathematically identical to the flat forward logic in log-space.
    """
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

            for s in range(s_scenarios):
                y0 = values[c, idx0, s]
                y1 = values[c, idx1, s]
                ln_interp = y0 + w * (y1 - y0)
                out[c, d, s] = math.exp(ln_interp)


# ==============================================================================
# --- STEP LEFT (LOCF) INTERPOLATION WITH FLAT EXTRAPOLATION ---
# ==============================================================================


def interp_step_left_flat_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Vectorized 3D Step Left (LOCF) Interpolation."""
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] <= targets[None, :, None], axis=2) - 1,
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]
    return values[c_idx, idx, :]


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
def interp_step_left_flat_extrap_3d_numba_kernel(
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


def interp_step_left_linear_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Vectorized 3D Step Left (LOCF) Interpolation with Linear Extrapolation."""
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] <= targets[None, :, None], axis=2) - 1,
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]
    return values[c_idx, idx, :]


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
def interp_step_left_linear_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 3D Step Left (LOCF) Interpolator.
    Mathematically identical to flat extrapolation for step functions.
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


def interp_step_right_flat_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Vectorized 3D Step Right (NOCB) Interpolation."""
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] < targets[None, :, None], axis=2),
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]
    return values[c_idx, idx, :]


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
def interp_step_right_flat_extrap_3d_numba_kernel(
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


def interp_step_right_linear_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Vectorized 3D Step Right (NOCB) Interpolation with Linear Extrapolation."""
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] < targets[None, :, None], axis=2),
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]
    return values[c_idx, idx, :]


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
def interp_step_right_linear_extrap_3d_numba_kernel(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Zero-allocation 3D Step Right (NOCB) Interpolator.
    Mathematically identical to flat extrapolation for step functions.
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


def interp_strike_linear_flat_extrap(
    pillars: np.ndarray,  # Shape: (N_trades, K_strikes)
    values: np.ndarray,  # Shape: (N_trades, K_strikes, S_scenarios)
    targets: np.ndarray,  # Shape: (N_trades, S_scenarios)
) -> np.ndarray:
    """
    Vectorized 1D linear interpolation along the strike/Y-axis with flat extrapolation.
    Used dynamically inside Option Pricers to resolve exact strike vol across scenarios.
    """
    N, K_dim = pillars.shape
    S = targets.shape[1]

    mask = pillars[:, None, :] <= targets[:, :, None]
    idx1 = np.sum(mask, axis=2)
    idx1 = np.clip(idx1, 1, K_dim - 1)
    idx0 = idx1 - 1

    n_idx = np.arange(N)[:, None]
    s_idx = np.arange(S)[None, :]

    x0 = pillars[n_idx, idx0]
    x1 = pillars[n_idx, idx1]

    y0 = values[n_idx, idx0, s_idx]
    y1 = values[n_idx, idx1, s_idx]

    dx = np.where((x1 - x0) == 0, 1.0, x1 - x0)
    w = (targets - x0) / dx

    w = np.clip(w, 0.0, 1.0)
    return y0 + w * (y1 - y0)


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)
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
