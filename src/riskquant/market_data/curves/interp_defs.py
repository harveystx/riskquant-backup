# src/riskquant/market_data/curves/interp_defs.py
from __future__ import annotations

import numpy as np

# --- LINEAR INTERPOLATION WITH FLAT EXTRAPOLATION ---


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


# --- LINEAR INTERPOLATION WITH LINEAR EXTRAPOLATION ---


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


# --- LOG-LINEAR INTERPOLATION WITH FLAT EXTRAPOLATION (ALWAYS USED FOR DISCOUNT FACTORS) ---

# RQ always interps on discount factor curves using log-linear interpolation with flat extrapolation.


def interp_log_linear_flat_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Log-Linear Interpolation with Constant Forward Extrapolation.

    Mathematically equivalent to linear interpolation on Zero Rates.
    To note: values are expected to be in natural log space.
    By allowing the interpolation weight `w` to exceed 1.0 for targets past the
    final tenor, we linearly extrapolate the log discount factor, which perfectly
    preserves the forward rate of the final pillar segment.

    Shapes:
        tenors:     (C_curves, M_pillars)
        values:     (C_curves, M_pillars, S_scenarios)
        targets:    (D_unique_dates,)

    Returns:
        np.ndarray: Exponentiated Discount Factors. Shape (c_curves, d_targets, s_scenarios)
    """
    c_curves, m_pillars = tenors.shape
    d_targets = len(targets)

    # 1. VECTORIZED BOUNDARY SEARCH
    # Expand dims to broadcast: (c_curves, 1, m_pillars) <= (1, d_targets, 1) -> (c_curves, d_targets, m_pillars)
    # Summing across the pillars (axis=2) gives the index of the first tenor > the target.
    mask = tenors[:, None, :] <= targets[None, :, None]
    idx1 = np.sum(mask, axis=2)

    # Clip to safely anchor both interpolation and extrapolation to the known bounds.
    idx1 = np.clip(idx1, 1, m_pillars - 1)
    idx0 = idx1 - 1

    # 2. ADVANCED INTEGER INDEXING
    # Extract bounding tenors and values instantly using a broadcasted column vector
    c_idx = np.arange(c_curves)[:, None]

    t0 = tenors[c_idx, idx0]  # Shape: (c_curves, d_targets)
    t1 = tenors[c_idx, idx1]  # Shape: (c_curves, d_targets)

    y0 = values[c_idx, idx0, :]  # Shape: (c_curves, d_targets, s_scenarios)
    y1 = values[c_idx, idx1, :]  # Shape: (c_curves, d_targets, s_scenarios)

    # 3. CALCULATE WEIGHTS
    dt = t1 - t0
    dt = np.where(dt == 0, 1.0, dt)  # Prevent division-by-zero on sentinels

    # Shape: (c_curves, d_targets)
    w = (targets[None, :] - t0) / dt

    # Note: Deliberately NO np.clip(w, 0.0, 1.0) here.
    # Allowing w > 1.0 natively creates Constant Forward Extrapolation.

    # 4. LOG-LINEAR MATH & EXPONENTIATION
    # Expand weights to (c_curves, d_targets, 1) to broadcast against the Scenarios dimension (s_scenarios)
    ln_interp = y0 + w[:, :, None] * (y1 - y0)

    return np.exp(ln_interp)


# --- LOG-LINEAR INTERPOLATION WITH LINEAR EXTRAPOLATION ---


def interp_log_linear_linear_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Log-Linear Interpolation with Linear Extrapolation.
    To note: values are expected to be in natural log space.
    Extrapolates using linear forward rate.
    Shapes: tenors (c_curves, m_pillars), values (c_curves, m_pillars, s_scenarios), targets (d_targets,)
    Returns: (c_curves, d_targets, s_scenarios) - Exponentiated back to raw values
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
    ln_interp = y0 + w[:, :, None] * (y1 - y0)
    return np.exp(ln_interp)


# --- STEP LEFT (LOCF) INTERPOLATION WITH FLAT EXTRAPOLATION ---


def interp_step_left_flat_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Step Left (LOCF) Interpolation.
    Shapes: tenors (c_curves, m_pillars), values (c_curves, m_pillars, s_scenarios), targets (d_targets,)
    Returns: (c_curves, d_targets, s_scenarios)
    """
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] <= targets[None, :, None], axis=2) - 1,
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]

    return values[c_idx, idx, :]


# --- STEP LEFT (LOCF) INTERPOLATION WITH LINEAR EXTRAPOLATION ---


def interp_step_left_linear_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Step Left (LOCF) Interpolation with Linear Extrapolation.
    Shapes: tenors (c_curves, m_pillars), values (c_curves, m_pillars, s_scenarios), targets (d_targets,)
    Returns: (c_curves, d_targets, s_scenarios)
    """
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] <= targets[None, :, None], axis=2) - 1,
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]

    # For linear extrapolation, we can simply use the last two points to define a slope for extrapolation.
    # However, since this is a step function, the "slope" is zero. Thus, step left with linear extrapolation is effectively the same as step left with flat extrapolation.

    return values[c_idx, idx, :]


# --- STEP RIGHT (NOCB) INTERPOLATION WITH FLAT EXTRAPOLATION ---


def interp_step_right_flat_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Step Right (NOCB) Interpolation.
    Shapes: tenors (c_curves, m_pillars), values (c_curves, m_pillars, s_scenarios), targets (d_targets,)
    Returns: (c_curves, d_targets, s_scenarios)
    """
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] < targets[None, :, None], axis=2),
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]

    return values[c_idx, idx, :]


# --- STEP RIGHT (NOCB) INTERPOLATION WITH LINEAR EXTRAPOLATION ---


def interp_step_right_linear_extrap_3d(
    tenors: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 3D Step Right (NOCB) Interpolation with Linear Extrapolation.
    Shapes: tenors (c_curves, m_pillars), values (c_curves, m_pillars, s_scenarios), targets (d_targets,)
    Returns: (c_curves, d_targets, s_scenarios)
    """
    c_curves, m_pillars = tenors.shape
    idx = np.clip(
        np.sum(tenors[:, None, :] < targets[None, :, None], axis=2),
        0,
        m_pillars - 1,
    )
    c_idx = np.arange(c_curves)[:, None]

    # For linear extrapolation, we can simply use the last two points to define a slope for extrapolation.
    # However, since this is a step function, the "slope" is zero. Thus, step right with linear extrapolation is effectively the same as step right with flat extrapolation.

    return values[c_idx, idx, :]


# --- 2D SURFACE STRIKE INTERPOLATOR ---


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

    # Broadcast pillars to (N, S, K) to compare against targets (N, S, 1)
    mask = pillars[:, None, :] <= targets[:, :, None]
    idx1 = np.sum(mask, axis=2)
    idx1 = np.clip(idx1, 1, K_dim - 1)
    idx0 = idx1 - 1

    n_idx = np.arange(N)[:, None]
    s_idx = np.arange(S)[None, :]

    # Extract bounding pillars
    x0 = pillars[n_idx, idx0]  # (N, S)
    x1 = pillars[n_idx, idx1]  # (N, S)

    # Extract bounding values
    y0 = values[n_idx, idx0, s_idx]  # (N, S)
    y1 = values[n_idx, idx1, s_idx]  # (N, S)

    dx = np.where((x1 - x0) == 0, 1.0, x1 - x0)
    w = (targets - x0) / dx

    # Flat Extrapolation
    w = np.clip(w, 0.0, 1.0)

    return y0 + w * (y1 - y0)
