# src/riskquant/market_data/calibrate/implied_vol/black76/kernels/black76_implied_vol_batch.py
"""
This module contains the batch-processing Numba-jitted kernels for solving for Black-76
implied volatilities from large arrays of option market data, including undiscounted calls
and mixed calls/puts via Put-Call parity. The core root-solving logic is implemented in
the `black76_implied_vol_scalar` module.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from riskquant.market_data.calibrate.implied_vol.black76.kernels.black76_implied_vol_scalar import (
    _solver_black76_iv_from_undiscounted_call_prices_brent,
)
from riskquant.numba import _F64_1D, parallel_njit
from riskquant.valuation.pricing_models.black76.model import _black76_fv_vega


@parallel_njit(
    nb.void(
        _F64_1D,  # forward price
        _F64_1D,  # strike
        _F64_1D,  # time to expiry
        _F64_1D,  # market price
        _F64_1D,  # output array for implied vol
        _F64_1D,  # output array for vega at the solved iv
    )
)
def _kernel_black76_iv_from_undiscounted_call_prices(
    fwd_price,  # forward price (n_options,)
    strike,  # strike price (n_options,)
    tte,  # time to expiry (n_options,)
    market_call_price_undiscounted,  # market price (n_options,)
    iv_out,  # output array for implied vol (n_options,)
    vega_out,  # output array for vega at solved iv (n_options,)
):
    n_options = fwd_price.shape[0]

    for i in nb.prange(n_options):
        # Solve using brent's method, which is robust and fast for 1D root-finding
        # (inline numba kernel with no call overhead).
        iv = _solver_black76_iv_from_undiscounted_call_prices_brent(
            fwd_price[i],
            strike[i],
            tte[i],
            market_call_price_undiscounted[i],
        )
        iv_out[i] = iv

        # Fused vega evaluation at the solved iv (reuses the same band of memory,
        # no second pass). The helper returns 0.0 for the failure sentinel
        # (iv < 0) and for at-intrinsic zero-vol solves (iv == 0).
        vega_out[i] = _black76_fv_vega(
            fwd=fwd_price[i],
            strike=strike[i],
            sqrt_tte=np.sqrt(tte[i]),
            half_tte=0.5 * tte[i],
            vol=iv,
        )
