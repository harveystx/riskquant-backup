# src/riskquant/valuation/kernels/vanilla_forward_curve.py
"""
Vanilla forward pricing kernel, curve variant.

Identical math to the node variant; the only structural change is that the
underlying scalar comes from `price_curve_grid[curve_uix, delivery_uix, s]`
instead of `forward_nodes[uix, s]`.

FX rows are already EFFECTIVE in `MarketDataMatrix.fx_spots_effective`;
no inversion math survives into this kernel.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant._njit import parallel_njit

_F64_2D = nb.float64[:, ::1]
_F64_3D = nb.float64[:, :, ::1]
_F64_1D = nb.float64[::1]
_I32_1D = nb.int32[::1]


@parallel_njit(
    nb.void(
        _F64_2D,  # out
        _F64_3D,  # price_curve_grid
        _F64_2D,  # fx_spots_effective
        _F64_3D,  # df_grid
        _F64_1D,  # qty
        _F64_1D,  # strike
        _I32_1D,  # price_curve_uix
        _I32_1D,  # delivery_date_uix
        _I32_1D,  # fx_u_eff_uix
        _I32_1D,  # df_u_uix
        _I32_1D,  # fx_k_eff_uix
        _I32_1D,  # df_k_uix
        _I32_1D,  # settle_date_uix
    ),
)
def _kernel_price_vanilla_forward_curve(
    out,
    price_curve_grid,
    fx_spots_effective,
    df_grid,
    qty,
    strike,
    price_curve_uix,
    delivery_date_uix,
    fx_u_eff_uix,
    df_u_uix,
    fx_k_eff_uix,
    df_k_uix,
    settle_date_uix,
) -> None:
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        q = qty[i]
        k = strike[i]
        p_curve_id = price_curve_uix[i]
        deliv_id = delivery_date_uix[i]
        fx_u_id = fx_u_eff_uix[i]
        df_u_id = df_u_uix[i]
        fx_k_id = fx_k_eff_uix[i]
        df_k_id = df_k_uix[i]
        stl_id = settle_date_uix[i]

        # Slice the (date, scenario) plane for this trade's underlying curve.
        p_row = price_curve_grid[p_curve_id, deliv_id]
        out_row = out[i]

        for s in range(s_scenarios):
            # UNDERLYING LEG
            leg_u = p_row[s]
            if fx_u_id >= 0:
                leg_u *= fx_spots_effective[fx_u_id, s]
            if df_u_id >= 0:
                leg_u *= df_grid[df_u_id, stl_id, s]

            # STRIKE LEG
            leg_k = k
            if fx_k_id >= 0:
                leg_k *= fx_spots_effective[fx_k_id, s]
            if df_k_id >= 0:
                leg_k *= df_grid[df_k_id, stl_id, s]

            out_row[s] = q * (leg_u - leg_k)
