# src/riskquant/valuation/kernels/_math.py
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


@inline_njit
def numba_norm_cdf(x: float) -> float:
    """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


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


@inline_njit
def linear_interp_scalar(x: float, xs, ys) -> float:
    """
    Branchless 1D linear interpolation with flat extrapolation.

    Drops the `np.interp` call frame and Numba's bounds-check overhead in
    favour of an inlinable binary search + linear blend. LLVM fuses this
    into the caller's scenario loop.

    Both `xs` and `ys` must be 1-D C-contiguous; `xs` strictly ascending.
    For `x` outside `[xs[0], xs[-1]]` returns the corresponding edge value
    (flat extrapolation), matching `np.interp`'s default semantics and
    avoiding accidental log/division blow-ups on wide-strike moneyness.
    """
    n = xs.shape[0]
    if n == 1:
        return ys[0]

    # Edge clamps — flat extrapolation. Branches are predictable
    # (rare) and the compiler can lift them out of vectorised loops.
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]

    # Binary search for the right-hand pillar. Loop trip count is
    # log2(n) and the compiler unrolls it for small n.
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0 = xs[lo]
    x1 = xs[hi]
    y0 = ys[lo]
    y1 = ys[hi]
    # x1 > x0 by strict-monotone invariant.
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)
