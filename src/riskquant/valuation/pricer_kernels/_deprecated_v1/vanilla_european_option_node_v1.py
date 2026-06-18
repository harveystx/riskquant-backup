# src/riskquant/valuation/kernels/vanilla_european_option_node.py
"""
Vanilla European option pricing kernel, node variant.

Synthesises the forward in strike currency via Covered Interest Parity,
prices via Black-76, then discounts and FX-translates back to settle ccy.

Math (per trade i, scenario s)
------------------------------
    rate_k = fx_spots_effective[fx_k_eff_uix(i), s]   # k_ccy -> settle_ccy

    # CIP forward in strike currency
    fwd_strike_ccy
        = forward_node[p, s]
          * fx_spots_effective[fx_u_eff_uix(i), s]    # u_ccy -> settle_ccy
          * df_grid[df_u_uix(i), expiry_uix(i), s]    # under-DF at expiry
          / rate_k                                    # settle_ccy -> k_ccy
          / df_grid[df_k_uix(i), expiry_uix(i), s]    # strike-DF at expiry

    # Black-76 future value (cp_mult = +1 call, -1 put)
    sigma = vol_nodes[v, s]
    std_dev = max(sigma * sqrt(tte), eps)
    d1 = (log(fwd / k) + 0.5 * sigma**2 * tte) / std_dev
    d2 = d1 - std_dev
    fv = cp * (fwd * N(cp * d1) - k * N(cp * d2))

    # PV discount in strike ccy, then translate back to settle ccy
    pv_strike_ccy = fv * df_grid[df_k_uix(i), settle_uix(i), s]
    pv_settle_ccy = pv_strike_ccy * rate_k

    out[i, s] = qty[i] * pv_settle_ccy

FX rows are already EFFECTIVE in `MarketDataMatrix.fx_spots_effective`; the
kernel does NO inversion math. The strike-FX rate `rate_k` is loaded once
per scenario and used as a divisor in the forward synthesis and as a
multiplier in the PV translation — equivalent counts to the legacy code
but with the `(1.0 / rate_k) if inv_k else rate_k` gymnastics gone.

Vol-uix sentinel handling
-------------------------
`v_id = vol_uix[i]` may be `INVALID_UIX (< 0)` when the ingest compiler
resolved no vol container for trade i (e.g. a downstream trade that
slipped past validation, or a future-class instrument whose vol path is
not yet wired). With `boundscheck=False`, `vol_nodes[v_id]` on a negative
id is undefined behaviour — out-of-bounds memory.

This kernel hoists `has_vol = v_id >= 0` once per trade, outside the
scenario loop. The sentinel path writes `out_row[:] = 0.0` and skips
the vol gather + Black-76 math entirely. Two scenario loops — one for
the priced path, one for the zero-fill path — keep the SIMD body free
of per-scenario branches.
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant.numba import parallel_njit
from riskquant.valuation.pricer_kernels._deprecated._math_v1 import (
    numba_norm_cdf_abrom_steg,
)

_F64_2D = nb.float64[:, ::1]
_F64_3D = nb.float64[:, :, ::1]
_F64_1D = nb.float64[::1]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]

_TTE_FLOOR: float = 1e-9
_VOL_STD_DEV_FLOOR: float = 1e-12


@parallel_njit(
    nb.void(
        _F64_2D,  # out
        _F64_2D,  # forward_nodes
        _F64_2D,  # vol_nodes
        _F64_2D,  # fx_spots_effective
        _F64_3D,  # df_grid
        _F64_1D,  # sqrt_tte_grid
        _F64_1D,  # half_tte_grid
        _F64_1D,  # qty
        _F64_1D,  # strike
        _I8_1D,  # cp_mult
        _I32_1D,  # price_uix
        _I32_1D,  # vol_uix
        _I32_1D,  # fx_u_eff_uix
        _I32_1D,  # df_u_uix
        _I32_1D,  # fx_k_eff_uix
        _I32_1D,  # df_k_uix
        _I32_1D,  # expiry_date_uix
        _I32_1D,  # settle_date_uix
    ),
)
def _kernel_price_vanilla_european_option_node(
    out,
    forward_nodes,
    vol_nodes,
    fx_spots_effective,
    df_grid,
    sqrt_tte_grid,
    half_tte_grid,
    qty,
    strike,
    cp_mult,
    price_uix,
    vol_uix,
    fx_u_eff_uix,
    df_u_uix,
    fx_k_eff_uix,
    df_k_uix,
    expiry_date_uix,
    settle_date_uix,
) -> None:

    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        # ------------------------------------------------------------------
        # Trade constants
        # ------------------------------------------------------------------

        q = qty[i]
        k = strike[i]
        cp = cp_mult[i]

        p_id = price_uix[i]
        v_id = vol_uix[i]

        fx_u_id = fx_u_eff_uix[i]
        df_u_id = df_u_uix[i]

        fx_k_id = fx_k_eff_uix[i]
        df_k_id = df_k_uix[i]

        exp_id = expiry_date_uix[i]
        stl_id = settle_date_uix[i]

        out_row = out[i]

        # ---- Hoist vol-uix sentinel: no vol container -> zero PV ----
        has_vol = v_id >= 0

        if not has_vol:
            for s in range(s_scenarios):
                out_row[s] = 0.0
            continue

        sqrt_tte = sqrt_tte_grid[exp_id]
        half_tte = half_tte_grid[exp_id]
        log_k = math.log(k)

        # ------------------------------------------------------------------
        # Scenario rows
        # ------------------------------------------------------------------

        p_row = forward_nodes[p_id]
        v_row = vol_nodes[v_id]

        has_fx_u = fx_u_id >= 0
        has_df_u = df_u_id >= 0
        has_fx_k = fx_k_id >= 0
        has_df_k = df_k_id >= 0

        if has_fx_u:
            fx_u_row = fx_spots_effective[fx_u_id]

        if has_fx_k:
            fx_k_row = fx_spots_effective[fx_k_id]

        if has_df_u:
            df_u_exp_row = df_grid[df_u_id, exp_id]

        if has_df_k:
            df_k_exp_row = df_grid[df_k_id, exp_id]
            df_k_stl_row = df_grid[df_k_id, stl_id]

        # ------------------------------------------------------------------
        # Scenario loop
        # ------------------------------------------------------------------

        for s in range(s_scenarios):
            fwd = p_row[s]

            if has_fx_u:
                fwd *= fx_u_row[s]  # pyright: ignore[reportPossiblyUnboundVariable]

            if has_df_u:
                fwd *= df_u_exp_row[s]  # pyright: ignore[reportPossiblyUnboundVariable]

            rate_k = 1.0

            if has_fx_k:
                rate_k = fx_k_row[s]  # pyright: ignore[reportPossiblyUnboundVariable]
                fwd *= 1.0 / rate_k

            if has_df_k:
                fwd *= 1.0 / df_k_exp_row[s]  # pyright: ignore[reportPossiblyUnboundVariable]

            sigma = v_row[s]
            sigma_sq = sigma * sigma

            std_dev = sigma * sqrt_tte
            if std_dev < _VOL_STD_DEV_FLOOR:
                std_dev = _VOL_STD_DEV_FLOOR

            log_fwd = math.log(fwd)

            d1 = (log_fwd - log_k + sigma_sq * half_tte) / std_dev

            d2 = d1 - std_dev

            cdf1 = numba_norm_cdf_abrom_steg(cp * d1)
            cdf2 = numba_norm_cdf_abrom_steg(cp * d2)

            fv = cp * (fwd * cdf1 - k * cdf2)

            if has_df_k:
                fv *= df_k_stl_row[s]  # pyright: ignore[reportPossiblyUnboundVariable]

            fv *= rate_k

            out_row[s] = q * fv
