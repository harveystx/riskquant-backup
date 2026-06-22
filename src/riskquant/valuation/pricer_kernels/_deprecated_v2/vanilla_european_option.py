# src/riskquant/valuation/kernels/vanilla_european_option.py
"""Pricing kernels for vanilla European options."""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    _I8_1D,
    _I32_1D,
    parallel_njit,
)
from riskquant.valuation.pricing_models.black76.model import (
    _black76_fv_kernel_primitive,
)


@parallel_njit(
    nb.void(
        # OUTPUT
        _F64_2D,  # out
        # MARKET DATA
        _F64_2D,  # forward_prices
        _F64_2D,  # vols
        _F64_2D,  # discount_factors
        _F64_2D,  # fx_spots_effective
        # BLOCK FIELDS (TRADE SPECIFICS)
        _F64_1D,  # qty
        _F64_1D,  # strike
        _I8_1D,  # cp_mult
        _F64_1D,  # sqrt_tte
        _F64_1D,  # half_tte
        # BLOCK FIELDS (LINKAGES / UIX)
        _I32_1D,  # forward_price_uix
        _I32_1D,  # vol_uix
        _I32_1D,  # discount_factor_k_uix
        _I32_1D,  # fx_spot_effective_k2s_uix
    ),
)
def _kernel_price_vanilla_european_option(
    # OUTPUT
    out,  # (N, S)
    # MARKET DATA
    forward_prices,  # (P, S)
    vols,  # (V, S)
    discount_factors,  # (D, S)
    fx_spots_effective,  # (F, S)
    # BLOCK FIELDS (TRADE SPECIFICS)
    qty,  # (N,)
    strike,  # (N,)
    cp_mult,  # (N,)
    sqrt_tte,  # (N,)
    half_tte,  # (N,)
    # BLOCK FIELDS (LINKAGES / UIX)
    forward_price_uix,  # (N,)
    vol_uix,  # (N,)
    discount_factor_k_uix,  # (N,)
    fx_spot_effective_k2s_uix,  # (N,)
) -> None:
    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        # --- TRADE SPECIFICS (FOR ROW) ---
        qty_i = qty[i]
        strike_i = strike[i]
        sqrt_tte_i = sqrt_tte[i]
        half_tte_i = half_tte[i]
        cp_mult_i = cp_mult[i]

        # --- UIX (FOR ROW) ---
        forward_price_id = forward_price_uix[i]
        vol_id = vol_uix[i]
        df_k_id = discount_factor_k_uix[i]
        fx_k2s_id = fx_spot_effective_k2s_uix[i]

        # --- MARKET DATA (FOR ROW) ---
        forwards_row = forward_prices[forward_price_id]
        vols_row = vols[vol_id]
        df_k_row = discount_factors[df_k_id]
        fx_k2s_row = fx_spots_effective[fx_k2s_id]

        # --- HOISTED TRADE-SPECIFIC SCALARS (OUTSIDE SCENARIO LOOP) ---
        log_strike_i = math.log(strike_i)

        # --- PRICING ACROSS SCENARIOS (ACROSS ROW) ---
        for s in range(s_scenarios):
            fwd_s = forwards_row[s]
            vol_s = vols_row[s]

            fv_s = _black76_fv_kernel_primitive(
                fwd_s,
                strike_i,
                log_strike_i,
                sqrt_tte_i,
                half_tte_i,
                vol_s,
                cp_mult_i,
            )

            # PV = FX RATE * DISCOUNT FACTOR * QTY * FV
            pv_s = fx_k2s_row[s] * df_k_row[s] * qty_i * fv_s

            out[i, s] = pv_s
