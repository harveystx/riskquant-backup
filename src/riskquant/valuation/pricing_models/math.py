# src/riskquant/valuation/pricing_models/math.py
"""
Shared bare-metal math helpers for vectorized pricer kernels.

Tiny hot-path functions: `@nb.njit(inline='always')` collapses them into
the surrounding kernel at compile time, so the calling kernel sees fused
code with no call-frame overhead.

Anything that needs `prange` or more than a handful of ops should not
live here.

Which normal CDF to use
-----------------------
`numba_norm_cdf_abrom_steg` is the VALUATION PRICE PATH implementation.
`numba_norm_cdf` (erf-based) is retained for research and reference use and
is deliberately NOT used by the pricers. See `numba_norm_cdf_abrom_steg`
for the measured accuracy contract and the reason the erf form is not a
drop-in upgrade.
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
    """
    Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).

    Machine-precision for |x| <~ 7. NOT used by the valuation price path:
    it measures 1.76x slower than the Abramowitz-Stegun form in an otherwise
    identical kernel, and it is not uniformly more accurate -- `math.erf`
    saturates, so this function returns EXACTLY 0.0 at x = -10 where the true
    value is 7.62e-24. Reference and research use only.
    """
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


@inline_njit
def numba_norm_pdf(x: float) -> float:
    """Standard normal PDF: phi(x) = exp(-0.5 * x^2) / sqrt(2 * pi). Exact."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


@inline_njit
def numba_norm_cdf_abrom_steg(x: float) -> float:
    """
    Standard normal CDF via Abramowitz & Stegun 26.2.17. Price-path default.

    Accuracy contract (measured; D045/VAL-10)
    -----------------------------------------
    Absolute error <= 7.5e-8 everywhere. Relative error grows with |x|:

        |x| = 1   ->  3.6e-08
        |x| = 2   ->  3.1e-06
        |x| = 3   ->  5.1e-05
        |x| = 4   ->  4.7e-04
        |x| = 6   ->  3.6e-03
        |x| = 8   ->  1.0e-02

    Consequence for Black-76, measured over K/F in [1, 3], sigma in
    [10%, 60%], T in [1 week, 5 years]:

        absolute price error  <= 2.0e-5 per 100 of forward, everywhere
        relative price error  <= 6e-6   for |d2| <~ 2
                              up to 1.7e-2 for |d2| >~ 7, where the option
                              price is itself below 1e-12 per 100 of forward

    Implied volatility inverted through this pricer is accurate to better
    than 0.1 bp of vol -- three orders inside market quote precision.

    Put-call parity C - P = F - K is EXACT under this approximation: the
    sign branch below satisfies Phi(x) + Phi(-x) == 1 identically, so the
    error cancels in the parity difference.

    Accepted-risk boundary
    ----------------------
    Relative error exceeds 1e-3 for |d2| >~ 4. Revaluation never cares -- the
    price there is economically zero. A consumer that forms a RATIO against
    such a price (a moneyness-scaled diagnostic, an implied-vol inversion at
    extreme strike) inherits the error undivided. That is a constraint on
    consumers, not on this function.

    Do not substitute `numba_norm_cdf` without re-running
    `scripts/demos/valuation/norm_cdf_throughput.py`: it is 1.76x slower and
    underflows to exactly 0.0 at |x| = 10.
    """

    ax = abs(x)

    t = 1.0 / (1.0 + 0.2316419 * ax)

    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )

    pdf = 0.39894228040143267794 * math.exp(-0.5 * ax * ax)

    if x < 0.0:
        return pdf * poly
    return 1.0 - pdf * poly
