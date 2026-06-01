# src/riskquant/valuation/kernels/vanilla_european_option_curve.py
"""
Vanilla European option pricing kernel, curve variant.

Identical CIP-forward + Black-76 math to the node variant, with two
structural differences:

  - The underlying scalar comes from `price_curve_grid[curve_uix, delivery_uix, s]`.
  - Volatility is sourced from a 4D vol surface and requires per-scenario
    strike-axis interpolation along the pillar coordinate dictated by the
    surface's y-axis type.

FX rows are already EFFECTIVE — no inversion math; the strike-FX rate is
loaded once per scenario, used as a divisor for CIP forward synthesis and
as a multiplier for the PV translation back to settle ccy.

Layout contract (CRITICAL)
--------------------------
`vol_surfaces` MUST be `[surface, expiry, scenario, pillar]` (C-contig).
This guarantees that the pillars for a single trade-scenario tuple form
one contiguous run of memory: `vol_surfaces[v, exp, s]` is a 1-D view
suitable for the binary-search interp. The builder is responsible for
emitting this layout; the kernel relies on it for SIMD-friendly reads.

Branch hoisting
---------------
All sentinel checks (`fx_*_uix < 0`, `df_*_uix < 0`) and y-axis-type
selection are resolved once per trade, outside the scenario loop. Within
the scenario loop only the math executes; CPU branch predictors never
see a `y_type` ladder or a sentinel test per scenario.

Vol-surface-uix sentinel handling
---------------------------------
`v_surf_id = vol_surface_uix[i]` may be `INVALID_UIX (< 0)` when the
ingest compiler resolved no vol container for trade i. With
`boundscheck=False`, `vol_surface_strikes[v_surf_id]` and
`vol_surfaces[v_surf_id, exp_id]` on a negative id are undefined
behaviour — out-of-bounds memory.

This kernel hoists `has_vol = v_surf_id >= 0` once per trade, outside
the scenario loop. The sentinel path writes `out_row[:] = 0.0` and skips
the surface gather + Black-76 math entirely.
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant._njit import parallel_njit
from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD,
    SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_SPOT,
    SURFACE_Y_AXIS_TYPE_LOG_STRIKE,
    SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD,
    SURFACE_Y_AXIS_TYPE_MONEYNESS_SPOT,
    SURFACE_Y_AXIS_TYPE_STRIKE,
)
from riskquant.valuation.kernels._math import (
    linear_interp_scalar,
    numba_norm_cdf_abrom_steg,
)

_F64_2D = nb.float64[:, ::1]
_F64_3D = nb.float64[:, :, ::1]
_F64_4D = nb.float64[:, :, :, ::1]
_F64_1D = nb.float64[::1]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]

_TTE_FLOOR: float = 1e-9
_VOL_STD_DEV_FLOOR: float = 1e-12

# Y-axis kind groupings as int8 keys. Resolved once per trade and packed
# into a 0/1/2/3 dispatch tag so the inner loop sees no branch ladder.
_Y_KIND_STRIKE: int = 0  # raw strike: y = k
_Y_KIND_LOG_STRIKE: int = 1  # y = log(k)
_Y_KIND_MONEYNESS: int = 2  # y = k / fwd
_Y_KIND_LOG_MONEYNESS: int = 3  # y = log(k / fwd)


@parallel_njit(
    nb.void(
        _F64_2D,  # out
        _F64_3D,  # price_curve_grid
        _F64_2D,  # vol_surface_strikes
        _F64_4D,  # vol_surfaces  [surface, expiry, scenario, pillar]
        _F64_2D,  # fx_spots_effective
        _F64_3D,  # df_grid
        _F64_1D,  # tte_grid
        _F64_1D,  # qty
        _F64_1D,  # strike
        _I8_1D,  # cp_mult
        _I32_1D,  # price_curve_uix
        _I32_1D,  # vol_surface_uix
        _I8_1D,  # y_axis_type_id
        _I32_1D,  # fx_u_eff_uix
        _I32_1D,  # df_u_uix
        _I32_1D,  # fx_k_eff_uix
        _I32_1D,  # df_k_uix
        _I32_1D,  # delivery_date_uix
        _I32_1D,  # expiry_date_uix
        _I32_1D,  # settle_date_uix
    ),
)
def _kernel_price_vanilla_european_option_curve(
    out,
    price_curve_grid,
    vol_surface_strikes,
    vol_surfaces,
    fx_spots_effective,
    df_grid,
    tte_grid,
    qty,
    strike,
    cp_mult,
    price_curve_uix,
    vol_surface_uix,
    y_axis_type_id,
    fx_u_eff_uix,
    df_u_uix,
    fx_k_eff_uix,
    df_k_uix,
    delivery_date_uix,
    expiry_date_uix,
    settle_date_uix,
) -> None:
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        q = qty[i]
        k = strike[i]
        cp = cp_mult[i]

        p_curve_id = price_curve_uix[i]
        v_surf_id = vol_surface_uix[i]
        y_type = y_axis_type_id[i]

        fx_u_id = fx_u_eff_uix[i]
        df_u_id = df_u_uix[i]
        fx_k_id = fx_k_eff_uix[i]
        df_k_id = df_k_uix[i]

        deliv_id = delivery_date_uix[i]
        exp_id = expiry_date_uix[i]
        stl_id = settle_date_uix[i]

        out_row = out[i]

        # ---- Hoist vol-surface-uix sentinel: no surface -> zero PV ----
        # Resolving this BEFORE the surface gather is critical:
        # vol_surface_strikes[<0] and vol_surfaces[<0, ...] are UB under
        # boundscheck=False. Sentinel-trade rows write zeros and skip.
        has_vol = v_surf_id >= 0

        if not has_vol:
            for s in range(s_scenarios):
                out_row[s] = 0.0
            continue

        # ---- Trade-level scalars (hoist out of scenario loop) ----
        tte = tte_grid[exp_id]
        safe_tte = tte if tte > _TTE_FLOOR else _TTE_FLOOR
        sqrt_tte = math.sqrt(safe_tte)
        log_k = math.log(k)

        # ---- Pre-compute the 2D plane handles ----
        #   pillars      : 1-D, length=K_pillars  (shared across all scenarios)
        #   v_plane      : 2-D, [scenario, pillar] — contiguous run per scenario
        #   p_row        : 1-D, length=S          (price-curve row for delivery)
        pillars = vol_surface_strikes[v_surf_id]
        v_plane = vol_surfaces[v_surf_id, exp_id]
        p_row = price_curve_grid[p_curve_id, deliv_id]

        # ---- Hoist sentinel branches into per-trade booleans ----
        has_fx_u = fx_u_id >= 0
        has_df_u = df_u_id >= 0
        has_fx_k = fx_k_id >= 0
        has_df_k = df_k_id >= 0

        # ---- Hoist y-axis-type ladder into a 0/1/2/3 dispatch tag ----
        if y_type == SURFACE_Y_AXIS_TYPE_STRIKE:
            y_kind = _Y_KIND_STRIKE
        elif y_type == SURFACE_Y_AXIS_TYPE_LOG_STRIKE:
            y_kind = _Y_KIND_LOG_STRIKE
        elif y_type in (
            SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD,
            SURFACE_Y_AXIS_TYPE_MONEYNESS_SPOT,
        ):
            y_kind = _Y_KIND_MONEYNESS
        elif y_type in (
            SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD,
            SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_SPOT,
        ):
            y_kind = _Y_KIND_LOG_MONEYNESS
        else:
            # Unknown y-axis: degrade to strike. Producer validation
            # rejects this case before it reaches the kernel.
            y_kind = _Y_KIND_STRIKE

        for s in range(s_scenarios):
            # ---- CIP forward in strike currency ----
            fwd = p_row[s]
            if has_fx_u:
                fwd *= fx_spots_effective[fx_u_id, s]
            if has_df_u:
                fwd *= df_grid[df_u_id, exp_id, s]

            rate_k = 1.0
            if has_fx_k:
                rate_k = fx_spots_effective[fx_k_id, s]
                fwd /= rate_k
            if has_df_k:
                fwd /= df_grid[df_k_id, exp_id, s]

            # ---- Y-axis target for vol surface lookup ----
            # The branch tag is constant per trade; LLVM should hoist
            # this select out of the SIMD body via loop-unswitching.
            if y_kind == _Y_KIND_STRIKE:
                y_target = k
            elif y_kind == _Y_KIND_LOG_STRIKE:
                y_target = log_k
            elif y_kind == _Y_KIND_MONEYNESS:
                y_target = k / fwd
            else:  # _Y_KIND_LOG_MONEYNESS
                y_target = log_k - math.log(fwd)

            # ---- Strike-axis interpolation ----
            # TODO: QUESTION - DO WE NEED TO DO OTHER TYPES OF INTERPOLATION FOR OTHER
            # Y-AXIS KINDS? E.g. log-log for log-strike?
            # v_plane[s] is a C-contig 1-D run of pillars for THIS scenario.
            sigma = linear_interp_scalar(y_target, pillars, v_plane[s])

            # ---- Black-76 future value ----
            std_dev = sigma * sqrt_tte
            if std_dev < _VOL_STD_DEV_FLOOR:
                std_dev = _VOL_STD_DEV_FLOOR

            d1 = (math.log(fwd) - log_k + 0.5 * sigma * sigma * safe_tte) / std_dev
            d2 = d1 - std_dev

            cdf1 = numba_norm_cdf_abrom_steg(cp * d1)
            cdf2 = numba_norm_cdf_abrom_steg(cp * d2)

            fv = cp * (fwd * cdf1 - k * cdf2)

            # ---- PV discount in strike ccy, FX-translate to settle ccy ----
            if has_df_k:
                fv *= df_grid[df_k_id, stl_id, s]
            fv *= rate_k

            out_row[s] = q * fv
