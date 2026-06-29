# src/riskquant/valuation/kernels/vanilla_forward_node.py
"""
Vanilla forward pricing kernel, node variant.

    leg_u = forward_node * fx_u_effective * df_u(stl)
    leg_k = strike       * fx_k_effective * df_k(stl)
    PV    = qty * (leg_u - leg_k)

FX rows are already EFFECTIVE (direct or pre-inverted) inside
`MarketDataMatrix.fx_spots_effective` — pricer kernels do NO inversion math.
Each FX gather is one indexed read + one multiply per scenario.

Optional legs (FX / DF) are signalled by `INVALID_UIX (< 0)` in the
corresponding `_uix` field. These predicates are trade-level invariants;
Numba's LICM hoists them outside the inner scenario loop, so the inner loop
streams straight-line arithmetic over C-contiguous memory.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import parallel_njit

_F64_2D = nb.float64[:, ::1]
_F64_3D = nb.float64[:, :, ::1]
_F64_1D = nb.float64[::1]
_I32_1D = nb.int32[::1]


@parallel_njit(
    nb.void(
        _F64_2D,  # out
        _F64_2D,  # forward_nodes
        _F64_2D,  # fx_spots_effective
        _F64_3D,  # df_grid
        _F64_1D,  # qty
        _F64_1D,  # strike
        _I32_1D,  # price_uix
        _I32_1D,  # fx_u_eff_uix
        _I32_1D,  # df_u_uix
        _I32_1D,  # fx_k_eff_uix
        _I32_1D,  # df_k_uix
        _I32_1D,  # settle_date_uix
    ),
)
def _kernel_price_vanilla_forward_node(
    out,
    forward_nodes,
    fx_spots_effective,
    df_grid,
    qty,
    strike,
    price_uix,
    fx_u_eff_uix,
    df_u_uix,
    fx_k_eff_uix,
    df_k_uix,
    settle_date_uix,
) -> None:
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        # Trade-level scalars (loop-invariant; Numba hoists the branches below)
        q = qty[i]
        k = strike[i]
        p_id = price_uix[i]
        fx_u_id = fx_u_eff_uix[i]
        df_u_id = df_u_uix[i]
        fx_k_id = fx_k_eff_uix[i]
        df_k_id = df_k_uix[i]
        stl_id = settle_date_uix[i]

        p_row = forward_nodes[p_id]
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
