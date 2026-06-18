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

5. Vol surfaces. Two-step pipeline:
   (a) DELTA -> MONEYNESS_FWD calibration for DELTA-quoted surfaces
       via `market_data.calibrate.delta_to_moneyness`. Per-surface
       (T_REF, σ_REF) scalars produce expiry/scenario-independent
       moneyness pillars. The conversion is F-independent so the
       calibrator does not need FX or discount-curve context.
   (b) Time-slice interpolation onto the unique-date universe.
       Output layout is `[surface, expiry, scenario, pillar]` so
       the strike-axis interpolation in the option-curve kernel
       can walk a contiguous pillar run per (trade, scenario).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt

from riskquant.constants import DAYS_PER_YR_ACT_365_25, TTE_FLOOR
from riskquant.market_data.curves.interp_defs import (
    interp_linear_on_log_values_flat_extrap_3d_numba_kernel,
)
from riskquant.market_data.curves.interp_dispatch import CURVE_MATH_REGISTRY
from riskquant.market_data.state import MarketDataMatrix
from riskquant.market_data.transformers.delta_to_moneyness import (
    calibrate_delta_surfaces_to_moneyness,
)
from riskquant.universe import UixMapper
from riskquant.utils.dates import date_to_ord32

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

    Scatter contract (CRITICAL)
    ---------------------------
    `out[mask, :] = src` is the NumPy SETITEM form: it scatters `src`
    back into `out` via fancy/boolean-index assignment. This is
    in-place.

    `np.copyto(out[mask, :], src)` is a different code path. `out[mask,
    :]` here is a GETITEM expression; boolean-mask indexing returns a
    COPY of the masked rows, not a view. `np.copyto` then writes into
    that throw-away copy and the actual `out` is never touched. Same
    trap as the original `_build_price_curve_grid` bug — see scatter
    comment in that function.

    The fix is to use `__setitem__` directly:
        out[direct_mask, :] = raw_fx_spots[fx_uixs[direct_mask]]
        out[invert_mask, :] = 1.0 / raw_fx_spots[fx_uixs[invert_mask]]

    Both forms are scatter-assignment back into `out`. The invert path
    is one fused float64 division per scenario per inverted row — the
    `1.0 / x` expression allocates the result, but the allocation is
    K_invert*S which is the size of the final scatter target anyway,
    so no extra peak memory vs the broken `np.reciprocal(..., out=...)`
    form.
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
        out[direct_mask, :] = raw_fx_spots[fx_uixs[direct_mask]]

    if invert_mask.any():
        out[invert_mask, :] = 1.0 / raw_fx_spots[fx_uixs[invert_mask]]

    return out


# ==============================================================================
# SCALAR NODES (UNIVERSE-ALIGNED — item C, option A1)
# ==============================================================================


def _build_scalar_node_levels(
    *,
    raw_scalar_nodes: npt.NDArray[np.float64],
    scalar_node_uix_in_universe: npt.NDArray[np.int32],
    num_market_ids: int,
    num_scenarios: int,
) -> npt.NDArray[np.float64]:
    """
    Scatter dense scalar-node levels into one `market_id_universe`-aligned
    `(num_market_ids, num_scenarios)` matrix.

    The compilers index `prices` / `forward_nodes` / `vol_nodes` with a
    `market_id_universe` UIX (`map_market_ids_to_uix`), so axis-0 of the
    returned array MUST be `len(market_id_universe)`. The dense input rows
    (the price / fwd-node / vol-node sub-block of `flat_levels`, in that
    order) carry their universe position in `scalar_node_uix_in_universe`;
    we scatter each into its row.

    Option A1: one backing array shared by all three pricer-facing scalar
    fields. The price / fwd / vol id sets are disjoint, so no row is written
    twice and each pricer reads only the rows of its own class. Rows of the
    universe that are NOT scalar nodes (FX-spot ids, etc.) are never read by
    a scalar pricer, so they are left as `np.empty` garbage — touching them
    would be a wiring bug the validator already rejects, not a silent
    miss-read.

    Scatter contract: `out[idx_rows, :] = src` is fancy-index SETITEM —
    in-place. Same trap documented in `_build_fx_spots_effective`.
    """
    out = np.empty((num_market_ids, num_scenarios), dtype=np.float64)
    if raw_scalar_nodes.shape[0] == 0:
        return out
    if raw_scalar_nodes.shape[0] != scalar_node_uix_in_universe.shape[0]:
        raise ValueError(
            f"scalar-node row count {raw_scalar_nodes.shape[0]} != universe-index "
            f"count {scalar_node_uix_in_universe.shape[0]}"
        )
    out[scalar_node_uix_in_universe, :] = raw_scalar_nodes
    return out


def _scatter_price_curve_grid_to_universe(
    *,
    dense_grid: npt.NDArray[np.float64],
    price_curve_uix_in_universe: npt.NDArray[np.int32],
    num_market_ids: int,
) -> npt.NDArray[np.float64]:
    """
    Scatter the dense `(C, D, S)` price-curve grid into a
    `(num_market_ids, D, S)` array aligned to `market_id_universe`.

    Forward-curve underlyings resolve to a curve CONTAINER id that lands in
    `market_id_universe`; `compile_vanilla_forward_curve` sets
    `price_curve_uix = map_market_ids_to_uix(...)` and the kernel reads
    `price_curve_grid[price_curve_uix, delivery_uix, s]`. So axis-0 of the
    grid MUST be `len(market_id_universe)` — the exact A1 treatment one axis
    up. `price_curve_uix_in_universe[c]` is the universe row of dense curve
    `c`. Universe rows that are NOT price-curve containers are never read by
    the forward-curve kernel, so they remain `np.empty` garbage (a wiring
    bug there would be rejected by validation, not silently mis-read).
    """
    d_dates = dense_grid.shape[1] if dense_grid.ndim == 3 else 0
    num_scenarios = dense_grid.shape[2] if dense_grid.ndim == 3 else 0
    out = np.empty((num_market_ids, d_dates, num_scenarios), dtype=np.float64)
    if dense_grid.shape[0] == 0:
        return out
    if dense_grid.shape[0] != price_curve_uix_in_universe.shape[0]:
        raise ValueError(
            f"price-curve grid row count {dense_grid.shape[0]} != universe-index "
            f"count {price_curve_uix_in_universe.shape[0]}"
        )
    out[price_curve_uix_in_universe, :, :] = dense_grid
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

    interp_linear_on_log_values_flat_extrap_3d_numba_kernel(
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
    target_dates: npt.NDArray[np.int32],
    num_scenarios: int,
) -> npt.NDArray[np.float64]:
    """
    Time-slice every vol surface onto the portfolio's unique-date axis
    and emit the canonical kernel-facing layout.

    Input  : `raw_vol_surface_values` is `(C_surf, M_tenors, K_strikes, S_scn)`.
    Output : `(C_surf, D_dates, S_scn, K_strikes)` — pillar in trailing axis.

    Pre-calibration contract
    ------------------------
    `build_market_data_matrix` calibrates DELTA-quoted surfaces to
    MONEYNESS_FWD upstream via `delta_to_moneyness.calibrate_delta_
    surfaces_to_moneyness`, which rewrites both `raw_vol_surface_
    strikes` and `surface_y_axis_type_ids`. By the time this function
    runs, no DELTA_CALL rows remain — the kernel's `_Y_KIND_MONEYNESS`
    dispatch path consumes them transparently.
    """
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
    # ---- Raw scalar matrices (F/scalar-nodes, scenarios) ----
    raw_fx_spots: npt.NDArray[np.float64],
    raw_scalar_nodes: npt.NDArray[np.float64],
    scalar_node_uix_in_universe: npt.NDArray[np.int32],
    # ---- Curve configs ----
    raw_discount_tenors: npt.NDArray[np.float64],
    raw_discount_ln_dfs: npt.NDArray[np.float64],
    raw_price_curve_tenors: npt.NDArray[np.float64],
    raw_price_curve_values: npt.NDArray[np.float64],
    curve_interp_methods: npt.NDArray[np.int8],
    curve_extrap_methods: npt.NDArray[np.int8],
    price_curve_uix_in_universe: npt.NDArray[np.int32],
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

    # ---- 0. Scalar-node levels (universe-aligned, item C / A1) -----------
    # One backing array indexed by `market_id_universe`; prices / forward_
    # nodes / vol_nodes alias it (disjoint id classes, each pricer reads
    # only its own rows).
    scalar_levels = _build_scalar_node_levels(
        raw_scalar_nodes=raw_scalar_nodes,
        scalar_node_uix_in_universe=scalar_node_uix_in_universe,
        num_market_ids=uix_mapper.market_id_universe.shape[0],
        num_scenarios=num_scenarios,
    )

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

    # ---- 3. Price curve grid (universe-aligned, item C / A1 for curves) --
    dense_price_curve_grid = _build_price_curve_grid(
        raw_tenors=raw_price_curve_tenors,
        raw_values=raw_price_curve_values,
        target_dates=unique_dates_ord32,
        interp_methods=curve_interp_methods,
        extrap_methods=curve_extrap_methods,
        num_scenarios=num_scenarios,
    )
    price_curve_grid = _scatter_price_curve_grid_to_universe(
        dense_grid=dense_price_curve_grid,
        price_curve_uix_in_universe=price_curve_uix_in_universe,
        num_market_ids=uix_mapper.market_id_universe.shape[0],
    )

    # ---- 4. TTE grid (act/365.25) ----------------------------------------
    as_of_ord32 = date_to_ord32(as_of_date)
    if d_dates == 0:
        tte_grid = np.empty(0, dtype=np.float64)
    else:
        tte_grid = np.maximum(
            (unique_dates_ord32.astype(np.float64) - float(as_of_ord32))
            / DAYS_PER_YR_ACT_365_25,
            TTE_FLOOR,
        )

    # ---- 5. Vol surfaces ------------------------------------------------
    #
    # Step 5a: DELTA -> MONEYNESS_FWD calibration. The calibrator
    # rewrites `raw_vol_surface_strikes` rows whose y-axis-type is
    # DELTA_CALL; non-DELTA rows pass through untouched. The
    # `surface_y_axis_type_ids_calibrated` array is the post-
    # calibration per-surface view: DELTA_CALL slots are rewritten to
    # MONEYNESS_FWD, non-DELTA slots are unchanged. This is the
    # runtime per-surface diagnostic counterpart to the per-trade
    # view on IngestBatch (already MONEYNESS_FWD post-ingest via
    # `surface_y_axis_kind_for_kernel`). Both views agree by
    # construction.
    (
        calibrated_strikes,
        surface_y_axis_type_ids_calibrated,
    ) = calibrate_delta_surfaces_to_moneyness(
        raw_vol_surface_strikes=raw_vol_surface_strikes,
        raw_vol_surface_tenors=raw_vol_surface_tenors,
        raw_vol_surface_values=raw_vol_surface_values,
        surface_y_axis_type_ids=surface_y_axis_type_ids,
    )

    # Step 5b: time-slice onto the unique-date axis. By this point no
    # DELTA_CALL rows remain — `_build_vol_surfaces` no longer needs
    # the y-axis-type ids argument.
    vol_surfaces = _build_vol_surfaces(
        raw_vol_surface_tenors=raw_vol_surface_tenors,
        raw_vol_surface_strikes=calibrated_strikes,
        raw_vol_surface_values=raw_vol_surface_values,
        surface_interp_x_methods=surface_interp_x_methods,
        surface_extrap_x_methods=surface_extrap_x_methods,
        target_dates=unique_dates_ord32,
        num_scenarios=num_scenarios,
    )

    return MarketDataMatrix(
        num_scenarios=num_scenarios,
        fx_spots_effective=fx_spots_effective,
        prices=scalar_levels,
        forward_nodes=scalar_levels,
        vol_nodes=scalar_levels,
        discount_factor_grid=discount_factor_grid,
        price_curve_grid=price_curve_grid,
        tte_grid=tte_grid,
        vol_surface_strikes=calibrated_strikes,
        vol_surfaces=vol_surfaces,
        surface_y_axis_type_ids=surface_y_axis_type_ids_calibrated,
    )
