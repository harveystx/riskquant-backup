# src/riskquant/valuation/pricing_models/black76/kernels.py
"""
High-throughput Black-76 model-evaluation kernels.

This module contains batch-oriented Numba kernels that evaluate the
Black-76 model across many instruments and/or scenarios.

Architectural role
------------------
These kernels are model-evaluation kernels, not product-valuation
kernels.

They operate directly on arrays of Black-76 inputs:

    * forwards
    * strikes
    * volatilities
    * times to expiry

and produce arrays of Black-76 outputs.

They do not know about:

    * portfolios
    * blocks
    * valuation states
    * settlement conventions
    * FX conversion
    * discount-factor construction
    * market-data compilation

Typical consumers include:

    * pricing-model APIs
    * volatility-surface calibration
    * implied-volatility stripping
    * research tooling
    * benchmarking

Product valuation kernels should generally call the scalar functions in
`model.py` directly rather than routing through these batch kernels.

These kernels exist to provide efficient bulk evaluation of the Black-76
model outside the portfolio valuation framework.
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore[import-untyped]

from riskquant.jit import (
    _F64_1D,
    _F64_2D,
    _I8_1D,
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
        _F64_2D,  # forwards
        _F64_2D,  # vols
        # TRADE SPECIFICS
        _F64_1D,  # strike
        _F64_1D,  # sqrt_tte
        _F64_1D,  # half_tte
        _I8_1D,  # cp_mult
    ),
)
def _kernel_black76_fv_batch(
    # OUTPUT
    out,  # (m, S)
    # MARKET DATA
    forwards,  # (m, S)
    vols,  # (m, S)
    # TRADE SPECIFICS
    strike,  # (m, )
    sqrt_tte,  # (m, )
    half_tte,  # (m, )
    cp_mult,  # (m,)
):
    """
    Black 76 pricing kernel. Returns FUTURE VALUE (FV) of the option, not PV.
    The caller is responsible for multiplying by the discount factor to get PV.

    Preconditions
    -------------
    The caller guarantees:

        forward > 0
        strike > 0
        volatility > 0
        sqrt_tte > 0

    Violating these invariants results in undefined behaviour.

    Validation belongs in the compiler layer, not the pricing kernel.

    Arguments
    ---------
    out : np.ndarray, shape (m, S)
        Output array to write the FV of the option for each trade and scenario.
    forwards : np.ndarray, shape (m, S)
        Forward prices for each trade and scenario.
    vols : np.ndarray, shape (m, S)
        Volatilities for each trade and scenario.
    strike : np.ndarray, shape (m,)
        Strike prices for each trade.
    sqrt_tte : np.ndarray, shape (m,)
        Square root of time to expiry for each trade.
    half_tte : np.ndarray, shape (m,)
        Half of the time to expiry for each trade.
    cp_mult : np.ndarray, shape (m,)
        Call/put multiplier for each trade.
    """

    m_trades, s_scenarios = out.shape

    for i in nb.prange(m_trades):
        # --- TRADE SPECIFICS (FOR ROW) ---
        strike_i = strike[i]
        sqrt_tte_i = sqrt_tte[i]
        half_tte_i = half_tte[i]
        cp_mult_i = cp_mult[i]

        # --- MARKET DATA (FOR ROW) ---
        forwards_row = forwards[i]
        vols_row = vols[i]

        # --- HOISTED TRADE-SPECIFIC SCALARS (OUTSIDE SCENARIO LOOP) ---
        log_strike_i = math.log(strike_i)

        # --- PRICING ACROSS SCENARIOS (ACROSS ROW) ---
        for s in range(s_scenarios):
            fwd_s = forwards_row[s]
            vol_s = vols_row[s]

            log_fk_s = math.log(fwd_s) - log_strike_i
            vol_sqrt_tte_s = vol_s * sqrt_tte_i
            vol2_half_tte_s = vol_s * vol_s * half_tte_i

            out[i, s] = _black76_fv_kernel_primitive(
                fwd_s,
                log_fk_s,
                strike_i,
                vol_sqrt_tte_s,
                vol2_half_tte_s,
                cp_mult_i,
            )
