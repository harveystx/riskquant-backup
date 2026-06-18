# src/riskquant/market_data/calibrate/implied_vol/black76/calibrate.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.calibrate.implied_vol.black76.kernels.black76_implied_vol_batch import (
    _kernel_black76_iv_from_undiscounted_call_prices,
)
from riskquant.market_data.calibrate.option_quotes.master import (
    CalibratedOptionQuotes,
    CanonicalOptionQuotes,
)


def calibrate_black76_iv_from_undiscounted_call_prices(
    quotes: CanonicalOptionQuotes,
) -> CalibratedOptionQuotes:
    """
    Calibrate Black-76 implied volatilities from canonical (undiscounted-call)
    option quotes.

    Returns a full-length, index-aligned ``CalibratedOptionQuotes`` master.
    Failed inversions carry the ``IV_SOLVER_FAILED_SENTINEL`` value and a
    ``False`` entry in ``success_mask``; ``vega`` carries the sensitivity at the
    solved iv (0.0 for failed and at-intrinsic solves) and acts as the natural
    quality weight. See that master for the full contract.
    """
    iv, vega = _calibrate_black76_iv_arrays(
        fwd_price=quotes.fwd_price,
        strike=quotes.strike,
        tte=quotes.tte,
        market_call_price_undiscounted=quotes.undiscounted_call_price,
    )

    # Branchless contract derivation: solved (>= 0.0, incl. the 0.0 at-intrinsic
    # case) vs the negative failure sentinel. Single source of truth is the iv
    # sign; the master re-checks this invariant on construction.
    success_mask = iv >= 0.0

    return CalibratedOptionQuotes(
        fwd_price=quotes.fwd_price,
        strike=quotes.strike,
        tte=quotes.tte,
        iv=iv,
        vega=vega,
        success_mask=success_mask,
    )


def _calibrate_black76_iv_arrays(
    fwd_price: npt.NDArray[np.float64],
    strike: npt.NDArray[np.float64],
    tte: npt.NDArray[np.float64],
    market_call_price_undiscounted: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Array-level calibration entry point.

    Returns ``(iv, vega)``, both length ``n_options``. Retained as the raw-array
    seam for isolated NumPy/SciPy parity tests of the kernel; production callers
    go through ``calibrate_black76_iv_from_undiscounted_call_prices``.
    """
    fwd_price = np.ascontiguousarray(fwd_price, dtype=np.float64)
    strike = np.ascontiguousarray(strike, dtype=np.float64)
    tte = np.ascontiguousarray(tte, dtype=np.float64)
    market_call_price_undiscounted = np.ascontiguousarray(
        market_call_price_undiscounted, dtype=np.float64
    )

    n_options = fwd_price.shape[0]
    iv_out = np.empty(n_options, dtype=np.float64)
    vega_out = np.empty(n_options, dtype=np.float64)

    _kernel_black76_iv_from_undiscounted_call_prices(
        fwd_price,
        strike,
        tte,
        market_call_price_undiscounted,
        iv_out,
        vega_out,
    )

    return iv_out, vega_out
