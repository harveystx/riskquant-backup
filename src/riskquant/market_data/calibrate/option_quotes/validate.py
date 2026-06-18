# src/riskquant/market_data/calibrate/option_quotes/validate.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.calibrate.option_quotes.master import (
    ValidatedOptionQuotes,
)

#: Max offending row indices shown in a validation error message.
_PREVIEW = 10

#: Relative tolerance on the no-arbitrage bounds. Strictly a float-noise guard
#: (it absorbs rounding in df * bound), not market bid/ask slack -- validate is
#: the hard edge; dirty feeds must be cleaned upstream.
_ARB_REL_TOL = 1e-9


def _format_violation(message: str, mask: npt.NDArray[np.bool_]) -> str:
    """Append a count and a bounded preview of offending row indices."""
    idx = np.flatnonzero(mask)
    head = ", ".join(str(int(i)) for i in idx[:_PREVIEW])
    more = "" if idx.size <= _PREVIEW else f", ... (+{idx.size - _PREVIEW} more)"
    return f"{message}: {idx.size} quote(s) at rows [{head}{more}]"


def validate_option_quotes(
    fwd_price: npt.ArrayLike,
    strike: npt.ArrayLike,
    tte: npt.ArrayLike,
    call_put_flag: npt.ArrayLike,
    discount_factor: npt.ArrayLike,
    option_price: npt.ArrayLike,
) -> ValidatedOptionQuotes:
    """
    Validate and normalize market option quotes.

    Parameters
    ----------
    fwd_price
        Forward prices.

    strike
        Option strikes.

    tte
        Time-to-expiry in years.

    call_put_flag
        Option type flags.

            1 = call
            0 = put

    discount_factor
        Discount factors.

    option_price
        Discounted market option prices.

    Returns
    -------
    ValidatedOptionQuotes

    Notes
    -----
    This function performs:

        - dtype normalization
        - contiguous array normalization
        - shape validation
        - domain validation
        - no-arbitrage bound validation

    It does NOT perform:

        - put-call parity transformations
        - undiscounting
        - implied volatility calibration
    """

    # ------------------------------------------------------------------
    # Normalize dtypes and memory layout
    # ------------------------------------------------------------------

    fwd_price = np.ascontiguousarray(fwd_price, dtype=np.float64)
    strike = np.ascontiguousarray(strike, dtype=np.float64)
    tte = np.ascontiguousarray(tte, dtype=np.float64)

    call_put_flag = np.ascontiguousarray(
        call_put_flag,
        dtype=np.int8,
    )

    discount_factor = np.ascontiguousarray(
        discount_factor,
        dtype=np.float64,
    )

    option_price = np.ascontiguousarray(
        option_price,
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    n_quotes = fwd_price.shape[0]

    for name, array in (
        ("strike", strike),
        ("tte", tte),
        ("call_put_flag", call_put_flag),
        ("discount_factor", discount_factor),
        ("option_price", option_price),
    ):
        if array.shape[0] != n_quotes:
            raise ValueError(
                f"{name} length ({array.shape[0]}) "
                f"does not match fwd_price length ({n_quotes})"
            )

    # ------------------------------------------------------------------
    # Domain validation
    # ------------------------------------------------------------------

    if np.any(fwd_price <= 0.0):
        raise ValueError("fwd_price must be strictly positive")

    if np.any(strike <= 0.0):
        raise ValueError("strike must be strictly positive")

    if np.any(tte <= 0.0):
        raise ValueError("tte must be strictly positive")

    if np.any(discount_factor <= 0.0):
        raise ValueError("discount_factor must be strictly positive")

    if np.any(option_price < 0.0):
        raise ValueError("option_price cannot be negative")

    invalid_flags = (call_put_flag != 0) & (call_put_flag != 1)

    if np.any(invalid_flags):
        raise ValueError("call_put_flag must contain only 0 (put) or 1 (call)")

    # ------------------------------------------------------------------
    # No-arbitrage bound validation
    # ------------------------------------------------------------------
    #
    # Model-free European bounds on the *discounted* price, per side:
    #     call:  df * max(F - K, 0) <= c <= df * F
    #     put:   df * max(K - F, 0) <= p <= df * K
    # (undiscounted call in [intrinsic, F]; undiscounted put in [intrinsic, K];
    # scale by df). A breach is a broken feed, not dirty data, so we reject
    # loudly here rather than letting it surface as an opaque failed solve.

    is_call = call_put_flag == 1

    intrinsic_undisc = np.where(
        is_call,
        np.maximum(fwd_price - strike, 0.0),
        np.maximum(strike - fwd_price, 0.0),
    )
    lower_bound = discount_factor * intrinsic_undisc
    upper_bound = discount_factor * np.where(is_call, fwd_price, strike)

    arb_tol = _ARB_REL_TOL * np.maximum(upper_bound, 1.0)

    below_lower = option_price < lower_bound - arb_tol
    if np.any(below_lower):
        raise ValueError(
            _format_violation(
                "option_price below discounted intrinsic (lower no-arbitrage bound)",
                below_lower,
            )
        )

    above_upper = option_price > upper_bound + arb_tol
    if np.any(above_upper):
        raise ValueError(
            _format_violation(
                "option_price above discounted forward/strike "
                "(upper no-arbitrage bound)",
                above_upper,
            )
        )

    # ------------------------------------------------------------------
    # Contract construction
    # ------------------------------------------------------------------

    return ValidatedOptionQuotes(
        fwd_price=fwd_price,
        strike=strike,
        tte=tte,
        call_put_flag=call_put_flag,
        discount_factor=discount_factor,
        option_price=option_price,
    )
