# src/riskquant/sensitivities/sensitizer_kernels/spot.py
"""
Spot sensitivity kernel.

Closed-form partials of the spot payoff. The spot pricing kernel
(`valuation.pricer_kernels.spot`) computes

    PV = qty * effective_spot_price
       = qty * spot_price * fx_effective

which is affine in `spot_price` when there is no FX leg, and bilinear in
(`spot_price`, `fx_effective`) when there is. Its exact partials are therefore

    no FX leg:
        dPV/dSpot   = qty
        d2PV/dSpot2 = 0

    direct FX-effective leg:
        dPV/dSpot        = qty * fx
        dPV/dFX          = qty * spot_price
        d2PV/dSpot dFX   = qty
        d2PV/dSpot2      = 0
        d2PV/dFX2        = 0

No bump-and-revalue: these are analytic derivatives, exact to machine
precision.

This kernel does dumb math only. It performs no identity resolution, no
sentinel interpretation, and no allocation -- the caller
(`sensitivities.sensitizers.spot`) has already unpacked the block's identity
graph into flat per-trade uix arrays and preallocated every output. Rows for
trades without an FX leg are written as zeros here and dropped by the caller;
branching on the sentinel is hoisted out of the loop body.
"""

from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _I32_1D_C,
    _NB_INT32,
    _NB_INT64,
    _NB_VOID,
    parallel_njit,
)


@parallel_njit(
    _NB_VOID(
        _F64_1D_C,  # out_spot_first_order
        _F64_1D_C,  # out_fx_first_order
        _F64_1D_C,  # out_cross_second_order
        _F64_1D_C,  # qty
        _I32_1D_C,  # price_uix
        _I32_1D_C,  # fx_eff_uix
        _F64_2D_C,  # prices
        _F64_2D_C,  # fx_spots_effective
        _NB_INT32,  # invalid_uix
        _NB_INT64,  # scenario
    ),
)
def _kernel_sensitize_spot(
    # OUTPUT (all length N)
    out_spot_first_order,
    out_fx_first_order,
    out_cross_second_order,
    # BLOCK FIELDS
    qty,  # (N,)
    price_uix,  # (N,)   indexes `prices` axis 0
    fx_eff_uix,  # (N,)   indexes `fx_spots_effective` axis 0, or invalid_uix
    # MARKET DATA (raw, UNFOLDED -- see module docstring of sensitizers.spot)
    prices,  # (N_mkt, S)
    fx_spots_effective,  # (K_fx,  S)
    # CONSTANTS
    invalid_uix,
    scenario,
) -> None:
    """
    Write per-trade closed-form spot sensitivities into preallocated arrays.

    Trades with `fx_eff_uix == invalid_uix` get `dPV/dSpot = qty` and zero FX
    and cross terms. Second-order own-factor terms are identically zero for
    both branches and are therefore not emitted by this kernel at all.
    """
    n_trades = qty.shape[0]

    for i in nb.prange(n_trades):
        qty_i = qty[i]
        fx_uix_i = fx_eff_uix[i]

        if fx_uix_i == invalid_uix:
            # Affine in spot: dPV/dSpot = qty. No FX, no cross term.
            out_spot_first_order[i] = qty_i
            out_fx_first_order[i] = 0.0
            out_cross_second_order[i] = 0.0
        else:
            # Bilinear in (spot, fx).
            spot_i = prices[price_uix[i], scenario]
            fx_i = fx_spots_effective[fx_uix_i, scenario]

            out_spot_first_order[i] = qty_i * fx_i
            out_fx_first_order[i] = qty_i * spot_i
            out_cross_second_order[i] = qty_i
