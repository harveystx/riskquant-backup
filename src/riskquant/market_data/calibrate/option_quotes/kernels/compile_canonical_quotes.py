# src/riskquant/market_data/calibrate/option_quotes/kernels/compile_canonical_quotes.py

from __future__ import annotations

import numba as nb

from riskquant.jit import _F64_1D_C, _I8_1D_C, parallel_njit


@parallel_njit(
    nb.void(
        _F64_1D_C,  # forward price
        _F64_1D_C,  # strike
        _I8_1D_C,  # call_put_flag (1=call, 0=put)
        _F64_1D_C,  # discount factor
        _F64_1D_C,  # option price
        _F64_1D_C,  # undiscounted call price out
    )
)
def _kernel_compute_undiscounted_call_price(
    fwd_price,
    strike,
    call_put_flag,
    discount_factor,
    option_price,
    undiscounted_call_price_out,
):
    """
    Compile validated market option quotes into canonical undiscounted
    call-price representation.

    Contract
    --------
    Inputs are assumed valid.

    call_put_flag:
        1 = call
        0 = put

    Output
    ------
    undiscounted_call_price_out[i]

    Transformations
    ---------------
    Call:
        C = option_price / df

    Put:
        C = option_price / df + F - K
    """

    n_quotes = fwd_price.shape[0]

    for i in nb.prange(n_quotes):
        undiscounted_price = option_price[i] / discount_factor[i]

        if call_put_flag[i]:
            undiscounted_call_price_out[i] = undiscounted_price
        else:
            undiscounted_call_price_out[i] = (
                undiscounted_price + fwd_price[i] - strike[i]
            )
