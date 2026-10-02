# src/riskquant/sensitivities/sensitizer_kernels/vanilla_forward.py
"""
Vanilla forward sensitivity kernel (layer 1; D103, WS-C).

The forward pricing kernel computes, per row, from the folded leg values

    PV = Q * E_fp - Q * K * E_df
    E_fp = F * DF_a * X_a          (asset leg, settlement-currency PV per unit)
    E_df = DF_k * X_k              (strike leg, settlement-currency PV per unit)

with X = 1 for a leg that has no FX. The primitives are the forward-node level
F and the FX levels X_a and X_k; the discount factors are constants of the
level coordinate (D069). PV is bilinear in (F, X_a) and linear in X_k, so the
exact primitive partials are

    w1          = Q * DF_a
    dPV/dF      = w1 * X_a
    dPV/dX_a    = w1 * F
    dPV/dX_k    = -Q * K * DF_k
    d2PV/dF dX_a = w1              (the only nonzero primitive pair)

The kernel writes these per row. Mapping a primitive to its rf id, merging
primitives that share an rf, and the complete pair set belong to the
sensitizer (layer 2). The kernel reads raw, universe-aligned grids: it does not
read the folded `effective_forward_prices`, because dPV/dF and dPV/dX are
partials with respect to different factors (see `sensitizers.spot`).
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _F64_3D_C,
    _I32_1D_C,
    _NB_INT32,
    _NB_INT64,
    _NB_VOID,
    parallel_njit,
)


@parallel_njit(
    _NB_VOID(
        _F64_1D_C,  # out_d_f
        _F64_1D_C,  # out_d_x_a
        _F64_1D_C,  # out_d_x_k
        _F64_1D_C,  # out_d_f_x_a
        _F64_1D_C,  # qty
        _F64_1D_C,  # strike
        _I32_1D_C,  # price_uix
        _I32_1D_C,  # curve_a
        _I32_1D_C,  # date_a
        _I32_1D_C,  # fx_a
        _I32_1D_C,  # curve_k
        _I32_1D_C,  # date_k
        _F64_2D_C,  # forward_nodes
        _F64_3D_C,  # discount_factor_grid
        _F64_2D_C,  # fx_spots_effective
        _NB_INT32,  # invalid_uix
        _NB_INT64,  # scenario
    ),
)
def _kernel_sensitize_vanilla_forward(
    # OUTPUT (all length N)
    out_d_f,
    out_d_x_a,
    out_d_x_k,
    out_d_f_x_a,
    # BLOCK FIELDS
    qty,  # (N,)
    strike,  # (N,)
    # LEG INDEX TABLES (from efp_effective_quads / edf_effective_triples)
    price_uix,  # (N,)  indexes `forward_nodes` axis 0
    curve_a,  # (N,)  asset-leg discount curve
    date_a,  # (N,)  asset-leg discount date
    fx_a,  # (N,)  asset-leg `fx_spots_effective` row, or invalid_uix
    curve_k,  # (N,)  strike-leg discount curve
    date_k,  # (N,)  strike-leg discount date
    # MARKET DATA (raw, UNFOLDED)
    forward_nodes,  # (N_mkt, S)
    discount_factor_grid,  # (C_df, D, S)
    fx_spots_effective,  # (K_fx, S)
    # CONSTANTS
    invalid_uix,
    scenario,
) -> None:
    """
    Write per-row primitive partials into preallocated arrays.

    An asset leg with `fx_a == invalid_uix` has X_a = 1. X_k enters no partial
    (PV is linear in X_k), so the strike-leg FX row is not read here; the
    sensitizer maps it to its rf. Inverted FX legs never reach this kernel
    (D103.3).
    """
    n_rows = qty.shape[0]
    for i in nb.prange(n_rows):
        qty_i = qty[i]
        f_i = forward_nodes[price_uix[i], scenario]
        df_a = discount_factor_grid[curve_a[i], date_a[i], scenario]
        df_k = discount_factor_grid[curve_k[i], date_k[i], scenario]
        x_a = 1.0
        if fx_a[i] != invalid_uix:
            x_a = fx_spots_effective[fx_a[i], scenario]
        w1 = qty_i * df_a
        out_d_f[i] = w1 * x_a
        out_d_x_a[i] = w1 * f_i
        out_d_x_k[i] = -(qty_i * strike[i] * df_k)
        out_d_f_x_a[i] = w1
