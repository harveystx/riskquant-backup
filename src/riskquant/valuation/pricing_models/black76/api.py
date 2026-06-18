# src/riskquant/valuation/pricing_models/black76/api.py
"""
Public API for the Black-76 pricing model.

This module provides user-facing entry points for evaluating the
Black-76 model.

Architectural role
------------------
The API layer sits above the model-evaluation kernels and provides a
stable, convenient interface for callers that wish to evaluate the
Black-76 model directly.

Responsibilities include:

    * allocating output arrays
    * exposing user-friendly function signatures
    * input validation
    * dispatching to optimized batch kernels

The API intentionally hides implementation details such as:

    * Numba kernels
    * memory-layout requirements
    * internal helper functions

Relationship to the valuation framework
---------------------------------------
The functions in this module evaluate the Black-76 model directly and
are independent of RiskQuant's portfolio valuation architecture.

They do not require:

    * Blocks
    * ValuationState
    * Compiled portfolios

Typical consumers include:

    * calibration routines
    * market-data tooling
    * research notebooks
    * standalone pricing workflows
    * testing and validation

Portfolio valuation should generally occur through product-specific
valuation kernels rather than through this API.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from riskquant._arrays import _as_c_contig_bool, _as_c_contig_f64
from riskquant.valuation.pricing_models.black76.kernels import _kernel_black76_fv_batch


def price_black76_fv_batch(
    *,
    forwards: NDArray[np.float64],
    vols: NDArray[np.float64],
    strike: NDArray[np.float64],
    tte: NDArray[np.float64],
    is_call: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Price a batch of options under the Black 76 model. Returns FUTURE VALUE (FV) of the option, not PV.
    The caller is responsible for multiplying by the discount factor to get PV.

    Preconditions
    -------------
    The caller guarantees:

        forward > 0
        strike > 0
        volatility > 0
        tte > 0
    """

    # --- CONTIGUITY ENFORCEMENT ---

    forwards = _as_c_contig_f64(forwards)
    vols = _as_c_contig_f64(vols)
    strike = _as_c_contig_f64(strike)
    tte = _as_c_contig_f64(tte)
    is_call = _as_c_contig_bool(is_call)

    # --- SHAPE VALIDATION ---

    # Market data arrays are 2D (num_options, batch_size).

    if forwards.ndim != 2:
        raise ValueError("forwards must be a 2D array")

    if vols.ndim != 2:
        raise ValueError("vols must be a 2D array")

    # Trade-specific arrays are 1D (num_options,).

    if strike.ndim != 1:
        raise ValueError("strike must be a 1D array")

    if tte.ndim != 1:
        raise ValueError("tte must be a 1D array")

    if is_call.ndim != 1:
        raise ValueError("is_call must be a 1D array")

    m_options, s_scenarios = forwards.shape

    if vols.shape != (m_options, s_scenarios):
        raise ValueError("vols must have the same shape as forwards")

    if strike.shape != (m_options,):
        raise ValueError("strike must have the same number of rows as forwards")

    if tte.shape != (m_options,):
        raise ValueError("tte must have the same number of rows as forwards")

    if is_call.shape != (m_options,):
        raise ValueError("is_call must have the same number of rows as forwards")

    # --- VALUES VALIDATION ---

    if np.any(forwards <= 0.0):
        raise ValueError("forwards must be strictly positive")

    if np.any(vols <= 0.0):
        raise ValueError("vols must be strictly positive")

    if np.any(strike <= 0.0):
        raise ValueError("strike must be strictly positive")

    if np.any(tte <= 0.0):
        raise ValueError("tte must be strictly positive")

    # --- FINITE CHECKS ---

    if not np.all(np.isfinite(forwards)):
        raise ValueError("forwards must be finite")

    if not np.all(np.isfinite(vols)):
        raise ValueError("vols must be finite")

    if not np.all(np.isfinite(strike)):
        raise ValueError("strike must be finite")

    if not np.all(np.isfinite(tte)):
        raise ValueError("tte must be finite")

    # --- KERNEL DISPATCH ---

    cp_mult = np.where(
        is_call,
        np.int8(1),
        np.int8(-1),
    )
    sqrt_tte = np.sqrt(tte)
    half_tte = 0.5 * tte

    out = np.empty((m_options, s_scenarios), dtype=np.float64)

    _kernel_black76_fv_batch(
        # OUTPUT
        out,
        # MARKET DATA
        forwards,
        vols,
        # TRADE SPECIFICS
        strike,
        sqrt_tte,
        half_tte,
        cp_mult,
    )

    return out
