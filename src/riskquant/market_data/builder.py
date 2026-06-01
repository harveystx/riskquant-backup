# src/riskquant/market_data/builder.py
"""
Market-data matrix builder.

Sits on the producer side of the valuation contract. Takes raw
scenario-shocked levels plus the `UixMapper` describing the run's
global universe, and emits a frozen `MarketDataMatrix` whose every
axis is aligned with the per-trade `_uix` coordinates carried by the
compiled blocks.

Key responsibilities
--------------------
1. FX effective matrix. Reads `UixMapper.fx_effective_pairs` to learn
   which `(fx_uix, invert)` pairs are needed across the portfolio,
   then materialises `fx_spots_effective: (K, S)` from the raw
   `(F, S)` direct-quote scenario matrix. Inverted rows are resolved
   via vectorised `np.reciprocal(out=...)`; direct rows are a single
   block copy. The pricers see zero inversion math.

2. Discount-factor grid. Log-linear interpolation in time, flat
   extrapolation. Routed through the zero-allocation Numba kernel
   with the pre-allocated output buffer.

3. Price-curve grid. Same shape as discount, but interpolation
   method varies per curve and is dispatched via `CURVE_MATH_REGISTRY`
   on `(interp_method, extrap_method)` group masks.

4. Time-to-expiry grid. Act/365.25 day count from `as_of_date` to
   each unique portfolio date, floored at 1e-9 to keep `sqrt(tte)`
   well-defined for same-day expiries.

5. Vol surfaces. Time-slice interpolation onto the unique-date
   universe. Output layout is `[surface, expiry, scenario, pillar]`
   so the strike-axis interpolation in the option-curve kernel can
   walk a contiguous pillar run per (trade, scenario).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.market_data.curves.interp_defs import (
    interp_log_linear_flat_extrap_3d_numba_kernel,
)
from riskquant.market_data.curves.interp_dispatch import CURVE_MATH_REGISTRY
from riskquant.market_data.state import MarketDataMatrix
from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_DELTA_CALL,
)
from riskquant.utils.dates import date_to_ord32

if TYPE_CHECKING:
    from riskquant.portfolio.engine import UixMapper

_ACT_365_25: float = 365.25
_TTE_FLOOR: float = 1e-9


# ==============================================================================
# FX EFFECTIVE
# ==============================================================================


def _build_fx_spots_effective(
    *,
    raw_fx_spots: npt.NDArray[np.float64],
    fx_effective_pairs: npt.NDArray[np.int32],
    num_scenarios: int,
) -> npt.NDArray[np.float64]:
    """
    Materialise the `(K, S)` effective FX matrix from the dedup pair table.

    `fx_effective_pairs[k] == (fx_uix, invert)` — `fx_uix` is a row index
    into `raw_fx_spots` (or `< 0` for an identity-row sentinel), `invert`
    is `0` for direct and `1` for inverted (i.e. `1 / rate`).
    """
    k_pairs = fx_effective_pairs.shape[0]
    out = np.empty((k_pairs, num_scenarios), dtype=np.float64)

    if k_pairs == 0:
        return out

    if fx_effective_pairs.shape[1] != 2:
        raise ValueError(
            f"fx_effective_pairs must have shape (K, 2); got {fx_effective_pairs.shape}"
        )

    fx_uixs = fx_effective_pairs[:, 0]
    invs = fx_effective_pairs[:, 1]

    sentinel_mask = fx_uixs < 0
    direct_mask = (~sentinel_mask) & (invs == 0)
    invert_mask = (~sentinel_mask) & (invs == 1)

    if sentinel_mask.any():
        out[sentinel_mask, :] = 1.0

    if direct_mask.any():
        np.copyto(out[direct_mask, :], raw_fx_spots[fx_uixs[direct_mask]])

    if invert_mask.any():
        np.reciprocal(
            raw_fx_spots[fx_uixs[invert_mask]],
            out=out[invert_mask, :],
        )

    return out


# ==============================================================================
# CURVE GRIDS
# ==============================================================================


def _build_price_curve_grid(
    *,
    raw_tenors: npt.NDArray[np.float64],
    raw_values: npt.NDArray[np.float64],
    target_dates: npt.NDArray[np.int32],
    interp_methods: npt.NDArray[np.int8],
    extrap_methods: npt.NDArray[np.int8],
    num_scenarios: int,
) -> npt.NDArray[np.float64]:
    """
    Group-dispatched 3-D curve interpolation onto the unique-date grid.

    Curves sharing the same `(interp_method, extrap_method)` pair are
    gathered into a contiguous scratch buffer, fed to the dispatched
    kernel, then scatter-assigned back into `out_grid`.

    Scatter contract
    ----------------
    `out_grid[idx]` on the LHS of an assignment is NumPy fancy-index
    setitem and is in-place. `out_grid[idx]` on the RHS (the kernel's
    original signature) would have been a fresh COPY — the kernel
    would have written into the copy, the copy would have been
    discarded, and `out_grid` would have been left as `np.empty()`
    garbage. The scratch + scatter pattern is the correct fix.
    """
    c_curves = raw_values.shape[0]
    d_dates = target_dates.shape[0]

    out_grid = np.empty((c_curves, d_dates, num_scenarios), dtype=np.float64)

    if d_dates == 0 or c_curves == 0:
        return out_grid

    for (interp_id, extrap_id), kernel in CURVE_MATH_REGISTRY.items():
        mask = (interp_methods == interp_id) & (extrap_methods == extrap_id)
        if not mask.any():
            continue

        idx = np.where(mask)[0]
        n_in_group = idx.shape[0]

        # Gather to a contiguous scratch buffer the kernel can write to.
        scratch = np.empty(
            (n_in_group, d_dates, num_scenarios),
            dtype=np.float64,
        )
        kernel(
            np.ascontiguousarray(raw_tenors[idx]),
            np.ascontiguousarray(raw_values[idx]),
            target_dates,
            scratch,
        )
        # Scatter back. Fancy-indexed ASSIGNMENT is in-place.
        out_grid[idx] = scratch

    return out_grid


def _build_discount_factor_grid(
    *,
    raw_tenors: npt.NDArray[np.float64],
    raw_ln_dfs: npt.NDArray[np.float64],
    target_dates: npt.NDArray[np.int32],
    num_scenarios: int,
) -> npt.NDArray[np.float64]:
    """
    Discount-factor grid: log-linear interpolation in time, flat extrap.

    Domain-convention: every discount curve in the system is quoted as
    log(DF) at pillar dates and interpolated linearly on log(DF), which
    is equivalent to linear interpolation on zero rates between pillars.
    The flat extrapolation prevents pathological exponentials at the long
    end and matches the front-office convention.
    """
    c_curves = raw_ln_dfs.shape[0]
    d_dates = target_dates.shape[0]

    out_grid = np.empty((c_curves, d_dates, num_scenarios), dtype=np.float64)

    if d_dates == 0 or c_curves == 0:
        return out_grid

    interp_log_linear_flat_extrap_3d_numba_kernel(
        raw_tenors,
        raw_ln_dfs,
        target_dates,
        out_grid,
    )
    return out_grid


# ==============================================================================
# VOL SURFACES
# ==============================================================================


def _build_vol_surfaces(
    *,
    raw_vol_surface_tenors: npt.NDArray[np.float64],
    raw_vol_surface_strikes: npt.NDArray[np.float64],
    raw_vol_surface_values: npt.NDArray[np.float64],
    surface_interp_x_methods: npt.NDArray[np.int8],
    surface_extrap_x_methods: npt.NDArray[np.int8],
    surface_y_axis_type_ids: npt.NDArray[np.int8],
    target_dates: npt.NDArray[np.int32],
    num_scenarios: int,
) -> npt.NDArray[np.float64]:
    """
    Time-slice every vol surface onto the portfolio's unique-date axis
    and emit the canonical kernel-facing layout.

    Input  : `raw_vol_surface_values` is `(C_surf, M_tenors, K_strikes, S_scn)`.
    Output : `(C_surf, D_dates, S_scn, K_strikes)` — pillar in trailing axis.
    """
    if (surface_y_axis_type_ids == SURFACE_Y_AXIS_TYPE_DELTA_CALL).any():
        raise ValueError(
            "FATAL: Delta-quoted Volatility Surfaces detected in runtime builder. "
            "DELTA_CALL surfaces must be pre-processed and converted to STRIKE or "
            "MONEYNESS by the Calibration module before being passed to the "
            "MarketDataMatrix."
        )

    c_surf = raw_vol_surface_tenors.shape[0] if raw_vol_surface_tenors.ndim > 0 else 0
    k_strikes = (
        raw_vol_surface_strikes.shape[1] if raw_vol_surface_strikes.ndim > 1 else 0
    )
    d_dates = target_dates.shape[0]

    if d_dates == 0 or raw_vol_surface_values.size == 0 or c_surf == 0:
        return np.empty((c_surf, d_dates, num_scenarios, k_strikes), dtype=np.float64)

    c_in, m_tenors, k_in, s_in = raw_vol_surface_values.shape
    if c_in != c_surf or k_in != k_strikes or s_in != num_scenarios:
        raise ValueError(
            f"raw_vol_surface_values shape {raw_vol_surface_values.shape} is "
            f"inconsistent with tenors/strikes/scenarios "
            f"({c_surf}, {m_tenors}, {k_strikes}, {num_scenarios})."
        )

    # Step 1. Fold (pillar, scenario) into one trailing axis and feed
    # the time-interp kernels as a 3-D (c, m_tenors, k * s) array.
    flat_in = raw_vol_surface_values.reshape(
        c_surf, m_tenors, k_strikes * num_scenarios
    )
    time_sliced_flat = _build_price_curve_grid(
        raw_tenors=raw_vol_surface_tenors,
        raw_values=np.ascontiguousarray(flat_in),
        target_dates=target_dates,
        interp_methods=surface_interp_x_methods,
        extrap_methods=surface_extrap_x_methods,
        num_scenarios=k_strikes * num_scenarios,
    )

    # Step 2. Reshape back to (c, d, k, s) and swap to land in the kernel-
    # facing (c, d, s, k) layout. Force C-contiguous because the kernel
    # signature pins `[:, :, :, ::1]`.
    canonical = time_sliced_flat.reshape(
        c_surf, d_dates, k_strikes, num_scenarios
    ).swapaxes(2, 3)
    return np.ascontiguousarray(canonical)


# ==============================================================================
# TOP-LEVEL ENTRY POINT
# ==============================================================================


def build_market_data_matrix(
    *,
    as_of_date: date,
    uix_mapper: UixMapper,
    num_scenarios: int,
    # ---- Raw scalar matrices (F/P/Fwd/Vol, scenarios) ----
    raw_fx_spots: npt.NDArray[np.float64],
    raw_prices: npt.NDArray[np.float64],
    raw_forward_nodes: npt.NDArray[np.float64],
    raw_vol_nodes: npt.NDArray[np.float64],
    # ---- Curve configs ----
    raw_discount_tenors: npt.NDArray[np.float64],
    raw_discount_ln_dfs: npt.NDArray[np.float64],
    raw_price_curve_tenors: npt.NDArray[np.float64],
    raw_price_curve_values: npt.NDArray[np.float64],
    curve_interp_methods: npt.NDArray[np.int8],
    curve_extrap_methods: npt.NDArray[np.int8],
    # ---- Surface configs ----
    raw_vol_surface_tenors: npt.NDArray[np.float64],
    raw_vol_surface_strikes: npt.NDArray[np.float64],
    raw_vol_surface_values: npt.NDArray[np.float64],
    surface_interp_x_methods: npt.NDArray[np.int8],
    surface_extrap_x_methods: npt.NDArray[np.int8],
    surface_y_axis_type_ids: npt.NDArray[np.int8],
) -> MarketDataMatrix:
    """
    Assemble a frozen `MarketDataMatrix` aligned to the run's UIX universes.

    Parameters
    ----------
    uix_mapper
        Source of truth for the FX-effective pair table and the unique-
        date axis. Built once per run by `build_uix_mapper`.

    Returns
    -------
    MarketDataMatrix
        Valuation-ready container; pricers consume directly.
    """
    unique_dates_ord32 = uix_mapper.date_ord32_universe
    d_dates = unique_dates_ord32.shape[0]

    # ---- 1. FX effective matrix ------------------------------------------
    fx_spots_effective = _build_fx_spots_effective(
        raw_fx_spots=raw_fx_spots,
        fx_effective_pairs=uix_mapper.fx_effective_pairs,
        num_scenarios=num_scenarios,
    )

    # ---- 2. Discount factor grid -----------------------------------------
    discount_factor_grid = _build_discount_factor_grid(
        raw_tenors=raw_discount_tenors,
        raw_ln_dfs=raw_discount_ln_dfs,
        target_dates=unique_dates_ord32,
        num_scenarios=num_scenarios,
    )

    # ---- 3. Price curve grid ---------------------------------------------
    price_curve_grid = _build_price_curve_grid(
        raw_tenors=raw_price_curve_tenors,
        raw_values=raw_price_curve_values,
        target_dates=unique_dates_ord32,
        interp_methods=curve_interp_methods,
        extrap_methods=curve_extrap_methods,
        num_scenarios=num_scenarios,
    )

    # ---- 4. TTE grid (act/365.25) ----------------------------------------
    as_of_ord32 = date_to_ord32(as_of_date)
    if d_dates == 0:
        tte_grid = np.empty(0, dtype=np.float64)
    else:
        tte_grid = np.maximum(
            (unique_dates_ord32.astype(np.float64) - float(as_of_ord32)) / _ACT_365_25,
            _TTE_FLOOR,
        )

    # ---- 5. Vol surfaces (canonical [c, d, s, k] layout) -----------------
    vol_surfaces = _build_vol_surfaces(
        raw_vol_surface_tenors=raw_vol_surface_tenors,
        raw_vol_surface_strikes=raw_vol_surface_strikes,
        raw_vol_surface_values=raw_vol_surface_values,
        surface_interp_x_methods=surface_interp_x_methods,
        surface_extrap_x_methods=surface_extrap_x_methods,
        surface_y_axis_type_ids=surface_y_axis_type_ids,
        target_dates=unique_dates_ord32,
        num_scenarios=num_scenarios,
    )

    return MarketDataMatrix(
        num_scenarios=num_scenarios,
        fx_spots_effective=fx_spots_effective,
        prices=raw_prices,
        forward_nodes=raw_forward_nodes,
        vol_nodes=raw_vol_nodes,
        discount_factor_grid=discount_factor_grid,
        price_curve_grid=price_curve_grid,
        tte_grid=tte_grid,
        vol_surface_strikes=raw_vol_surface_strikes,
        vol_surfaces=vol_surfaces,
    )
