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
   which `(market_uix, invert)` pairs are needed across the portfolio,
   translates each market row to its dense FX-block row (D071), then
   materialises `fx_spots_effective: (K, S)` from the raw `(F, S)`
   direct-quote scenario matrix. Inverted rows are `1.0 / rate`; direct
   rows are a single block copy. The pricers see zero inversion math.

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
       (T_REF, sigma_REF) scalars produce expiry/scenario-independent
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

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.constants import DAYS_PER_YR_ACT_365_25, DAYS_PER_YR_ACT_365F, TTE_FLOOR
from riskquant.market_data.curves.interp_dispatch import CURVE_MATH_REGISTRY
from riskquant.market_data.curves.zero_curve import (
    build_zero_curve_discount_factor_grid,
)
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
    fx_spot_uix_in_universe: npt.NDArray[np.int32],
    fx_effective_pairs: npt.NDArray[np.int32],
    num_scenarios: int,
) -> npt.NDArray[np.float64]:
    """
    Materialise the `(K, S)` effective FX matrix from the dedup pair table.

    `fx_effective_pairs[k] == (market_uix, invert)` — `market_uix` is a
    `market_id_universe` row, NOT a row of `raw_fx_spots`. It is translated
    to the dense FX-block row through `fx_spot_uix_in_universe` with a
    membership proof (D071). `invert` is `0` for direct and `1` for inverted
    (i.e. `1 / rate`); any other value raises.

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
        out[direct_rows, :] = raw_fx_spots[fx_row[direct_rows]]
        out[invert_rows, :] = 1.0 / raw_fx_spots[fx_row[invert_rows]]

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

    market_uix = fx_effective_pairs[:, 0]
    invs = fx_effective_pairs[:, 1]

    # D071 / MD-01: `fx_effective_pairs[:, 0]` is a `market_id_universe` row.
    # `raw_fx_spots` rows are the DENSE FX block. Translate with a membership
    # proof; a pair whose rf is not a laid-out FX spot fails here.
    if raw_fx_spots.shape[0] != fx_spot_uix_in_universe.shape[0]:
        raise ValueError(
            f"raw_fx_spots rows ({raw_fx_spots.shape[0]}) != fx_spot_uix_in_universe "
            f"entries ({fx_spot_uix_in_universe.shape[0]})."
        )
    fx_row = np.searchsorted(fx_spot_uix_in_universe, market_uix)
    safe = np.minimum(fx_row, max(fx_spot_uix_in_universe.shape[0] - 1, 0))
    if fx_spot_uix_in_universe.shape[0] == 0 or bool(
        np.any(fx_spot_uix_in_universe[safe] != market_uix)
    ):
        raise ValueError(
            "fx_effective_pairs reference market rows that are not FX spot rfs "
            "laid out in the FX block (D071)."
        )
    if bool(np.any((invs != 0) & (invs != 1))):
        raise ValueError("fx_effective_pairs invert flags must be 0 or 1.")

    # Every row is written by exactly one of the two scatters below.
    direct_rows = np.flatnonzero(invs == 0)
    invert_rows = np.flatnonzero(invs == 1)
    if direct_rows.size:
        out[direct_rows, :] = raw_fx_spots[fx_row[direct_rows]]
    if invert_rows.size:
        out[invert_rows, :] = 1.0 / raw_fx_spots[fx_row[invert_rows]]

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
    twice and each pricer reads only the rows of its own class. FX-spot rows
    are scattered by `build_market_data_matrix` after this call (D072). Rows
    that remain `np.empty` (curve / surface containers) are proven unconsumed
    by `_prove_scalar_row_ownership` before the matrix is emitted.

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


def _prove_scalar_row_ownership(
    *,
    uix_mapper: UixMapper,
    written: npt.NDArray[np.bool_],
) -> None:
    """
    D072 / MD-01b: every `prices` row a supported fold gathers must have been
    written by the producer. `np.empty` rows (curve/surface containers) are
    legitimate only while no effective table references them.
    """
    consumed = np.concatenate((
        uix_mapper.spot_effective_pairs[:, 0]
        if uix_mapper.spot_effective_pairs.size
        else np.empty(0, np.int32),
        uix_mapper.efp_effective_quads[:, 0],
        uix_mapper.vol_effective_uix,
    ))
    if consumed.size == 0:
        return
    unowned = ~written[consumed]
    if bool(unowned.any()):
        ids = np.unique(uix_mapper.market_id_universe[consumed[unowned]])[:10].tolist()
        raise ValueError(
            "market ids consumed by the valuation fold have no scalar producer row "
            f"(D072): {ids}. A curve or surface container is not a scalar level."
        )


def _validate_level_domain(
    *,
    uix_mapper: UixMapper,
    raw_fx_spots: npt.NDArray[np.float64],
    fx_spot_uix_in_universe: npt.NDArray[np.int32],
    raw_scalar_nodes: npt.NDArray[np.float64],
    scalar_node_uix_in_universe: npt.NDArray[np.int32],
    scalar_node_strictly_positive: npt.NDArray[np.bool_],
    curve_and_surface_levels: tuple[tuple[str, npt.NDArray[np.float64]], ...],
) -> None:
    """
    AUD-05 / D070: the market-level domain contract, enforced by the producer.

    Every level that can enter `MarketDataMatrix` passes through this function,
    whatever route supplied it (compiled snapshot, direct snapshot, history
    snapshot, flat levels, scenario-shocked levels). Checked before any
    allocation or scatter, so no matrix or effective state ever holds an
    invalid level.

        all levels          finite (no NaN, no +inf, no -inf)
        FX spot rows        > 0   (FX_RATE by config; conversion factor by role)
        VOL scalar rows     > 0   (flag resolved from rf_id -> quote type)
        PRICE/RATE/SPREAD/INDEX: any finite value, including <= 0

    Vectorised over (rows, scenarios); no per-scenario Python loop.
    """
    if raw_fx_spots.shape[0] != fx_spot_uix_in_universe.shape[0]:
        raise ValueError(
            f"raw_fx_spots rows ({raw_fx_spots.shape[0]}) != fx_spot_uix_in_universe "
            f"entries ({fx_spot_uix_in_universe.shape[0]})."
        )
    if not (
        raw_scalar_nodes.shape[0]
        == scalar_node_uix_in_universe.shape[0]
        == scalar_node_strictly_positive.shape[0]
    ):
        raise ValueError(
            f"scalar-node rows ({raw_scalar_nodes.shape[0]}), universe indices "
            f"({scalar_node_uix_in_universe.shape[0]}) and domain flags "
            f"({scalar_node_strictly_positive.shape[0]}) must align."
        )

    def _rf_ids(
        uix_rows: npt.NDArray[np.int32], bad_rows: npt.NDArray[np.bool_]
    ) -> list[int]:
        return [int(x) for x in uix_mapper.market_id_universe[uix_rows[bad_rows]][:10]]

    for name, arr, uix_rows in (
        ("FX spot", raw_fx_spots, fx_spot_uix_in_universe),
        ("scalar node", raw_scalar_nodes, scalar_node_uix_in_universe),
    ):
        if arr.size:
            bad = np.asarray(~np.isfinite(arr).all(axis=1), dtype=np.bool_)
            if bool(bad.any()):
                raise ValueError(
                    f"{name} levels contain non-finite values for rf_ids "
                    f"{_rf_ids(uix_rows, bad)} (AUD-05, D070)."
                )
    if raw_fx_spots.size:
        bad = np.asarray((raw_fx_spots <= 0.0).any(axis=1), dtype=np.bool_)
        if bool(bad.any()):
            raise ValueError(
                "FX spot levels must be strictly positive; offending rf_ids "
                f"{_rf_ids(fx_spot_uix_in_universe, bad)} (AUD-05, D070)."
            )
    if raw_scalar_nodes.size and bool(scalar_node_strictly_positive.any()):
        bad = np.asarray(
            scalar_node_strictly_positive & (raw_scalar_nodes <= 0.0).any(axis=1),
            dtype=np.bool_,
        )
        if bool(bad.any()):
            raise ValueError(
                "VOL levels must be strictly positive; offending rf_ids "
                f"{_rf_ids(scalar_node_uix_in_universe, bad)} (AUD-05, D070)."
            )
    for name, arr in curve_and_surface_levels:
        if arr.size and not bool(np.isfinite(arr).all()):
            raise ValueError(f"{name} contain non-finite values (AUD-05, D070).")


def build_market_data_matrix(
    *,
    as_of_date: date,
    uix_mapper: UixMapper,
    num_scenarios: int,
    discounting_mode: DiscountingMode,
    # ---- Raw scalar matrices (F/scalar-nodes, scenarios) ----
    raw_fx_spots: npt.NDArray[np.float64],
    fx_spot_uix_in_universe: npt.NDArray[np.int32],
    raw_scalar_nodes: npt.NDArray[np.float64],
    scalar_node_uix_in_universe: npt.NDArray[np.int32],
    scalar_node_strictly_positive: npt.NDArray[np.bool_],
    # ---- Curve configs ----
    raw_discount_tenors: npt.NDArray[np.float64],
    raw_discount_levels: npt.NDArray[np.float64],
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
    discounting_mode
        D061, explicit and without a default. `NONE` produces
        `DF(as_of, date) = 1.0` for every discount curve and date in the run's
        universes, reading no curve level and consulting no `CurveInputType`.
        `CURVE` is market-derived discounting and is not implemented: it raises
        before any curve construction. The mode terminates here; nothing
        downstream of the market-data matrix sees it.

    Returns
    -------
    MarketDataMatrix
        Valuation-ready container; pricers consume directly.
    """
    unique_dates_ord32 = uix_mapper.date_ord32_universe
    d_dates = unique_dates_ord32.shape[0]

    # ---- D067: one run as-of ----------------------------------------------
    as_of_ord32 = date_to_ord32(as_of_date)
    if uix_mapper.as_of_ord32 is None or int(uix_mapper.as_of_ord32) != int(
        as_of_ord32
    ):
        raise ValueError(
            f"build_market_data_matrix as_of_date {as_of_date} (ordinal {int(as_of_ord32)}) "
            f"!= UixMapper.as_of_ord32 {uix_mapper.as_of_ord32} (D067)."
        )

    # ---- AUD-05: level domain, before any allocation or scatter ------------
    _validate_level_domain(
        uix_mapper=uix_mapper,
        raw_fx_spots=raw_fx_spots,
        fx_spot_uix_in_universe=fx_spot_uix_in_universe,
        raw_scalar_nodes=raw_scalar_nodes,
        scalar_node_uix_in_universe=scalar_node_uix_in_universe,
        scalar_node_strictly_positive=scalar_node_strictly_positive,
        curve_and_surface_levels=(
            ("discount curve levels", raw_discount_levels),
            ("price curve levels", raw_price_curve_values),
            ("vol surface levels", raw_vol_surface_values),
        ),
    )

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
    # D072: FX spots are scalar levels too (a spot position on an FX rate reads
    # them). They live in their own dense block; scatter it into its universe
    # rows so no FX row is left as `np.empty`.
    if raw_fx_spots.shape[0] != fx_spot_uix_in_universe.shape[0]:
        raise ValueError(
            f"raw_fx_spots rows ({raw_fx_spots.shape[0]}) != fx_spot_uix_in_universe "
            f"entries ({fx_spot_uix_in_universe.shape[0]})."
        )
    if fx_spot_uix_in_universe.size:
        scalar_levels[fx_spot_uix_in_universe, :] = raw_fx_spots
    written = np.zeros(uix_mapper.market_id_universe.shape[0], dtype=np.bool_)
    written[scalar_node_uix_in_universe] = True
    written[fx_spot_uix_in_universe] = True
    _prove_scalar_row_ownership(uix_mapper=uix_mapper, written=written)

    # ---- D079: the grid is spot-starting, so no target may precede the as-of --
    if d_dates and int(unique_dates_ord32[0]) < int(as_of_ord32):
        stale = unique_dates_ord32[unique_dates_ord32 < int(as_of_ord32)][:10]
        raise ValueError(
            f"target date ordinals {stale.tolist()} precede the run as-of "
            f"{int(as_of_ord32)}; discount factors are DF(as_of, date) and a "
            "pre-as-of target is not a supported discounting request (D079)."
        )

    # ---- D061: market-derived discounting is not implemented ----------------
    # The repository does not define the economic meaning of a curve point for
    # discounting (quote convention, compounding, day count, LABEL tenor -> date
    # mapping), and the interpolation kernels' coordinate domain is not the
    # int32 date-ordinal domain of this universe. Producing a grid would mean
    # inventing that convention. A CURVE run fails here, before a kernel is
    # called, instead of raising a Numba TypeError or returning a plausible
    # wrong DF.
    # ---- D061 -> D080: market-derived discounting is live for exactly the
    # curves the layout plan proved constructible (ZERO_RATE + CONTINUOUS +
    # complete maturity metadata, checked in `build_universe_mapping`). A CURVE
    # plan that carries a discount curve with no pillars is a wiring defect,
    # never a silent fallback to DF = 1.
    n_disc_curves = int(uix_mapper.discount_curve_id_universe.shape[0])
    if discounting_mode is DiscountingMode.CURVE and n_disc_curves > 0:
        if raw_discount_tenors.shape != (n_disc_curves, raw_discount_levels.shape[1]):
            raise ValueError(
                "discounting_mode=CURVE: raw_discount_tenors "
                f"{raw_discount_tenors.shape} and raw_discount_levels "
                f"{raw_discount_levels.shape[:2]} disagree with the "
                f"{n_disc_curves}-curve universe (D080 layout plan)."
            )
        if raw_discount_levels.shape[1] == 0:
            raise NotImplementedError(
                "Discount-factor production is not supported (D061): the CURVE "
                "layout plan carries no pillars for a discount curve in the "
                "universe; only a fully declared D080 curve can be constructed."
            )
    if raw_price_curve_values.shape[0] > 0:
        raise NotImplementedError(
            "Price-curve grid production is not supported (D061); curve-sourced "
            "instruments are unregistered (D050)."
        )

    # ---- 1. FX effective matrix ------------------------------------------
    fx_spots_effective = _build_fx_spots_effective(
        raw_fx_spots=raw_fx_spots,
        fx_spot_uix_in_universe=fx_spot_uix_in_universe,
        fx_effective_pairs=uix_mapper.fx_effective_pairs,
        num_scenarios=num_scenarios,
    )

    # ---- 2. Discount factor grid -----------------------------------------
    if discounting_mode is DiscountingMode.NONE:
        # NONE: DF(as_of, date) = 1.0 for every curve and date the run resolved.
        # The curves' own levels (`raw_discount_tenors` / `raw_discount_levels`)
        # are deliberately not read, and no `CurveInputType` is consulted; the
        # linkage stays structural (D062/D066) while the economics say "do not
        # discount" (D061).
        discount_factor_grid = np.ones(
            (n_disc_curves, d_dates, num_scenarios), dtype=np.float64
        )
    else:
        # CURVE (D080): the levels are D069-normalised, already-shocked decimal
        # zero rates on the canonical ACT/365F pillar coordinate the layout plan
        # resolved (maturities are scenario-independent; only the rate varies
        # with s). Targets are the run's date universe on the same coordinate.
        # No unit scaling happens here: D069 ran once at the snapshot boundary.
        target_t = (
            unique_dates_ord32.astype(np.float64) - float(as_of_ord32)
        ) / DAYS_PER_YR_ACT_365F
        discount_factor_grid = build_zero_curve_discount_factor_grid(
            pillar_t=np.ascontiguousarray(raw_discount_tenors, dtype=np.float64),
            pillar_rates=np.ascontiguousarray(raw_discount_levels, dtype=np.float64),
            target_t=np.ascontiguousarray(target_t),
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
        uix_fingerprint=uix_mapper.fingerprint,
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
