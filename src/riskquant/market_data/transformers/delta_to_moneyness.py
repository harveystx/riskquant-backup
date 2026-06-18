# src/riskquant/market_data/transformers/delta_to_moneyness.py
"""
DELTA-quoted vol-surface -> MONEYNESS_FWD-quoted vol-surface calibration.

Why MONEYNESS_FWD and not STRIKE
--------------------------------
The Black-Scholes inversion `K = F * exp(-σ * √T * Φ⁻¹(δ) + 0.5 * σ² * T)`
gives strike as a function of the underlying's forward F. Emitting a
STRIKE-quoted shadow surface would force the calibrator to know F per
expiry per scenario — coupling it to FX rows, discount curves, and
forward-curve gathers it has no business reading.

Switching the target axis to MONEYNESS_FWD (K/F) cancels the F factor:

    K/F = exp(-σ * √T * Φ⁻¹(δ) + 0.5 * σ² * T)

The transform depends only on (δ, σ, T). The option-curve kernel
already has a `_Y_KIND_MONEYNESS` dispatch path that computes `y_target
= k / fwd` per scenario; the shadow surface drops in unchanged.

Layout-preserving approximation
-------------------------------
The natural mathematical result is per-(δ, expiry, scenario) pillars,
which would promote `vol_surface_strikes` from `(C, K)` to `(C, D, S,
K)` — a kernel signature change. We avoid that by using a per-surface
scalar reference pair (T_REF, σ_REF):

  T_REF      = tenor at the middle row of the surface's expiry axis.
               Concretely the median tenor in the raw input tenor grid.
  σ_REF      = ATM σ at (T_REF, scenario 0). ATM is the pillar whose
               original delta label is closest to 0.50 (the convention
               for ATM-fwd on a DELTA_CALL-quoted surface).

The shadow pillars are then per-(C, K), shared across expiry and
scenario, matching the existing layout. This is a fixture-grade
approximation that captures the smile's *shape* correctly at T_REF
and at scenario 0; it loses accuracy when trades expire far from
T_REF or under stressed scenarios. Production systems converge to a
proper per-(D, S) pillar grid plus a kernel signature change — that
remains future work.

Numerical floor
---------------
`Φ⁻¹(δ)` blows up as δ -> {0, 1}. The calibrator clamps δ into
`[DELTA_FLOOR, 1 - DELTA_FLOOR]` to keep `ndtri` finite. Production
DELTA-quoted surfaces typically carry pillars in [0.05, 0.95]; the
floor (1e-4) only triggers on pathological inputs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri

from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_DELTA_CALL,
    SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD,
)

_DELTA_FLOOR: float = 1e-4
_ATM_DELTA_TARGET: float = 0.50


def _calibrate_one_surface(
    *,
    deltas: npt.NDArray[np.float64],
    tenors: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Convert one DELTA-quoted surface's pillar coordinates to MONEYNESS_FWD.

    Parameters
    ----------
    deltas
        `(K,)` delta-call pillar labels (e.g. [0.10, 0.25, 0.50, 0.75, 0.90]).
    tenors
        `(M_tenors,)` raw input tenor axis in years (the surface's
        native expiry pillars; NOT the runtime unique-date axis — the
        calibrator runs BEFORE time-slicing).
    values
        `(M_tenors, K, S_scenarios)` raw σ values from the snapshot for
        this surface, sliced from the unflattened `raw_vol_surface_values
        [c]` payload BEFORE any time-slicing or layout swap. The
        calibrator reads scenario 0 only.

    Returns
    -------
    moneyness_pillars
        `(K,)` calibrated K/F pillar coordinates, shared across (expiry,
        scenario) for this surface.
    """
    if deltas.ndim != 1:
        raise ValueError(f"deltas must be 1-D; got {deltas.shape}.")
    k_pillars = deltas.shape[0]
    if k_pillars == 0:
        return np.empty(0, dtype=np.float64)

    if values.shape[1] != k_pillars or tenors.shape[0] != values.shape[0]:
        raise ValueError(
            f"shape mismatch: deltas={deltas.shape}, tenors={tenors.shape}, "
            f"values={values.shape}"
        )

    # ---- T_REF: median tenor ----
    # Median over the surface's *own* tenor axis. Stable under any
    # monotone tenor ordering; doesn't depend on the runtime unique-
    # date axis.
    m_tenors = tenors.shape[0]
    t_ref_idx = m_tenors // 2
    t_ref = float(tenors[t_ref_idx])
    if t_ref <= 0.0:
        # Pathological: zero or negative tenor. Bail to a tiny floor;
        # the kernel's _TTE_FLOOR catches the resulting σ²·T term.
        t_ref = 1e-6

    # ---- σ_REF: ATM σ at scenario 0 ----
    atm_idx = int(np.argmin(np.abs(deltas - _ATM_DELTA_TARGET)))
    sigma_ref = float(values[t_ref_idx, atm_idx, 0])
    if not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
        raise ValueError(
            f"σ_REF resolved to non-positive / non-finite "
            f"({sigma_ref!r}) at tenor index {t_ref_idx}, "
            f"delta-pillar index {atm_idx}. Surface data is malformed."
        )

    # ---- Black-76 inverse mapping: δ -> K/F ----
    deltas_clamped = np.clip(deltas, _DELTA_FLOOR, 1.0 - _DELTA_FLOOR)
    phi_inv = ndtri(deltas_clamped)  # scipy.special.ndtri == Φ⁻¹

    sqrt_t = np.sqrt(t_ref)
    half_sig_sq_t = 0.5 * sigma_ref * sigma_ref * t_ref
    log_moneyness = -sigma_ref * sqrt_t * phi_inv + half_sig_sq_t

    return np.ascontiguousarray(np.exp(log_moneyness), dtype=np.float64)


def calibrate_delta_surfaces_to_moneyness(
    *,
    raw_vol_surface_strikes: npt.NDArray[np.float64],
    raw_vol_surface_tenors: npt.NDArray[np.float64],
    raw_vol_surface_values: npt.NDArray[np.float64],
    surface_y_axis_type_ids: npt.NDArray[np.int8],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
    """
    Calibrate every DELTA_CALL row of a vol-surface batch to MONEYNESS_FWD.

    Returns a NEW `(C_surf, K_strikes)` pillar array and a NEW
    `(C_surf,)` y-axis-type-id array with DELTA_CALL rows rewritten.
    Non-DELTA rows are passed through untouched. The values array
    (`raw_vol_surface_values`) is not mutated — the same σ at the same
    pillar index just acquires a different y-axis label after this
    pass.

    Parameters
    ----------
    raw_vol_surface_strikes
        `(C_surf, K_strikes)` original pillar labels. DELTA_CALL rows
        carry delta values in [0, 1]; other rows carry strike / moneyness
        / log-moneyness coordinates and are passed through.
    raw_vol_surface_tenors
        `(C_surf, M_tenors)` per-surface tenor axes in years.
    raw_vol_surface_values
        `(C_surf, M_tenors, K_strikes, S_scn)` raw σ grid. Only scenario
        0 is read; the rest pass through the builder unchanged.
    surface_y_axis_type_ids
        `(C_surf,)` int8 y-axis-type ids declared by the static config.

    Returns
    -------
    pillars_out
        `(C_surf, K_strikes)` post-calibration pillar coordinates.
    yax_ids_out
        `(C_surf,)` post-calibration y-axis-type ids (DELTA_CALL ->
        MONEYNESS_FWD; others unchanged).
    """
    c_surf = surface_y_axis_type_ids.shape[0]

    if c_surf == 0:
        return (
            raw_vol_surface_strikes.copy(),
            surface_y_axis_type_ids.copy(),
        )

    # Edge allocation; downstream kernels never touch these directly.
    pillars_out = np.ascontiguousarray(raw_vol_surface_strikes, dtype=np.float64).copy()
    yax_ids_out = surface_y_axis_type_ids.copy()

    delta_mask = surface_y_axis_type_ids == SURFACE_Y_AXIS_TYPE_DELTA_CALL
    if not delta_mask.any():
        return pillars_out, yax_ids_out

    delta_idxs = np.flatnonzero(delta_mask)
    for c in delta_idxs:
        pillars_out[c] = _calibrate_one_surface(
            deltas=raw_vol_surface_strikes[c],
            tenors=raw_vol_surface_tenors[c],
            values=raw_vol_surface_values[c],
        )
        yax_ids_out[c] = SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD

    return pillars_out, yax_ids_out
