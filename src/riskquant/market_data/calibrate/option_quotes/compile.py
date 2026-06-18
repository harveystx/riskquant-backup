# src/riskquant/market_data/calibrate/option_quotes/compile.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.calibrate.option_quotes.kernels.compile_canonical_quotes import (
    _kernel_compute_undiscounted_call_price,
)
from riskquant.market_data.calibrate.option_quotes.master import (
    CanonicalOptionQuotes,
    ValidatedOptionQuotes,
)


def compile_canonical_option_quotes(
    quotes: ValidatedOptionQuotes,
) -> CanonicalOptionQuotes:
    """
    Compile validated option market quotes into the canonical
    calibration representation.

    Input
    -----
    ValidatedOptionQuotes

    Output
    ------
    CanonicalOptionQuotes (F, K, T, C_undiscounted_call) suitable for
    implied-volatility calibration.

    The master is already normalized (dtype + contiguity) and shape-checked by
    ``validate_option_quotes`` and ``ValidatedOptionQuotes.__post_init__``, so
    this path trusts it: no re-cast, no re-validation.
    """
    return _compile_to_canonical(
        fwd_price=quotes.fwd_price,
        strike=quotes.strike,
        tte=quotes.tte,
        call_put_flag=quotes.call_put_flag,
        discount_factor=quotes.discount_factor,
        option_price=quotes.option_price,
    )


def _compile_canonical_option_quotes_arrays(
    fwd_price: npt.NDArray[np.float64],
    strike: npt.NDArray[np.float64],
    tte: npt.NDArray[np.float64],
    call_put_flag: npt.NDArray[np.int8],
    discount_factor: npt.NDArray[np.float64],
    option_price: npt.NDArray[np.float64],
) -> CanonicalOptionQuotes:
    """
    Raw-array compilation seam.

    Normalizes dtype/layout and shape-checks before compiling. Retained for
    direct/raw callers and isolated kernel tests; production callers go through
    ``compile_canonical_option_quotes``, which trusts an already-validated
    master and skips this normalization.
    """
    fwd_price = np.ascontiguousarray(fwd_price, dtype=np.float64)
    strike = np.ascontiguousarray(strike, dtype=np.float64)
    tte = np.ascontiguousarray(tte, dtype=np.float64)
    call_put_flag = np.ascontiguousarray(call_put_flag, dtype=np.int8)
    discount_factor = np.ascontiguousarray(discount_factor, dtype=np.float64)
    option_price = np.ascontiguousarray(option_price, dtype=np.float64)

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
                f"{name} array length ({array.shape[0]}) does not match fwd_price ({n_quotes})"
            )

    return _compile_to_canonical(
        fwd_price=fwd_price,
        strike=strike,
        tte=tte,
        call_put_flag=call_put_flag,
        discount_factor=discount_factor,
        option_price=option_price,
    )


def _compile_to_canonical(
    fwd_price: npt.NDArray[np.float64],
    strike: npt.NDArray[np.float64],
    tte: npt.NDArray[np.float64],
    call_put_flag: npt.NDArray[np.int8],
    discount_factor: npt.NDArray[np.float64],
    option_price: npt.NDArray[np.float64],
) -> CanonicalOptionQuotes:
    """
    Trusted compilation core: assumes inputs are already normalized and
    shape-consistent. Allocates the output band, runs the put-call-parity /
    undiscounting kernel, and constructs the canonical master.
    """
    n_quotes = fwd_price.shape[0]
    undiscounted_call_price = np.empty(n_quotes, dtype=np.float64)

    _kernel_compute_undiscounted_call_price(
        fwd_price,
        strike,
        call_put_flag,
        discount_factor,
        option_price,
        undiscounted_call_price,
    )

    return CanonicalOptionQuotes(
        fwd_price=fwd_price,
        strike=strike,
        tte=tte,
        undiscounted_call_price=undiscounted_call_price,
    )
