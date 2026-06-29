# src/riskquant/valuation/pricing_models/math.py
"""
Shared bare-metal math helpers for vectorized pricer kernels.

Tiny hot-path functions: `@nb.njit(inline='always')` collapses them into
the surrounding kernel at compile time, so the calling kernel sees fused
code with no call-frame overhead.

Anything that needs `prange` or more than a handful of ops should not
live here.
"""

from __future__ import annotations

import math

from riskquant.jit import inline_njit

# 1 / sqrt(2). Lifted so the kernel never recomputes it.
_INV_SQRT_2: float = 0.7071067811865476

# 1 / sqrt(2 * pi). Lifted so the kernel never recomputes it.
_INV_SQRT_2PI: float = 0.39894228040143267794


@inline_njit
def numba_norm_cdf(x: float) -> float:
    """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


@inline_njit
def numba_norm_pdf(x: float) -> float:
    """Standard normal PDF: phi(x) = exp(-0.5 * x^2) / sqrt(2 * pi)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


@inline_njit
def numba_norm_cdf_abrom_steg(x: float) -> float:

    ax = abs(x)

    t = 1.0 / (1.0 + 0.2316419 * ax)

    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )

    pdf = 0.39894228040143267794 * math.exp(-0.5 * ax * ax)

    cdf = 1.0 - pdf * poly

    if x < 0.0:
        return 1.0 - cdf

    return cdf
