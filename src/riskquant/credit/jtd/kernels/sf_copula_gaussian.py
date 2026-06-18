# src/riskquant/credit/jtd/kernels/sf_copula_gaussian.py
"""
Bare-metal copula JTD loss kernels.

Overview
--------
These kernels realize jump-to-default (JTD) losses for Gaussian and
Student-t copula credit models.

Design goals
------------
- Zero allocation.
- Parallel across obligors.
- Contiguous scenario access.
- Hoist invariant work out of hot loops.
- Scenario-dependent EAD flows directly through loss realization.
- Minimize expensive operations inside the scenario loop.
- Keep the hottest path limited to loads, multiplies, adds, comparisons,
  and stores.

Array layouts
-------------
out             : (M, S) C-contiguous
ead             : (M, S) C-contiguous
latent_factors  : (F, S) C-contiguous

The scenario axis (S) is contiguous everywhere so the inner loop walks
unit-stride memory.

Factor pool conventions
-----------------------
Gaussian kernels
~~~~~~~~~~~~~~~~
Rows referenced by:

    sys_uix
    idio_uix

contain standard-normal draws.

Student-t kernels
~~~~~~~~~~~~~~~~~
Rows referenced by:

    chi_uix

DO NOT contain raw chi-square draws.

Instead they contain:

    inv_sqrt_w = 1 / sqrt(W)

where:

    W ~ χ²

This transformation is performed once during factor generation and stored
directly in the factor pool.

Student-t latent variable
-------------------------
The Student-t latent variable is:

    T = sqrt(ν / W) * Z

where:

    ν = obligor-specific degrees of freedom
    W = scenario-specific chi-square draw
    Z = Gaussian latent variable

The kernel evaluates this as:

    T = sqrt(ν) * (1 / sqrt(W)) * Z

Why precompute 1 / sqrt(W)?
---------------------------
If raw chi-square draws were stored, the kernel would need to compute:

    sqrt(ν) * Z / sqrt(W)

for every obligor-scenario pair.

Large Monte Carlo runs can require hundreds of millions of such
evaluations.

Precomputing:

    1 / sqrt(W)

during factor generation removes a square root from the hottest loop and
reduces the Student-t path to simple multiplication.

Obligor-specific ν
------------------
The degrees-of-freedom parameter is obligor-specific:

    nu.shape == (M,)

Therefore:

    sqrt(ν / W)

cannot be stored directly in the shared factor pool because ν varies by
obligor.

Only:

    1 / sqrt(W)

is precomputed and shared across obligors.

The kernel combines:

    sqrt(ν_i)

with:

    1 / sqrt(W_s)

at runtime.

Loss realization
----------------
For both copulas the realized loss is:

    loss = EAD * LGD * default_indicator

where:

    default_indicator = 1(latent < threshold)

This allows scenario-dependent EAD paths to flow directly through the
simulation without any static loss precomputation.

Performance notes
-----------------
- Parallelized across obligors via Numba prange.
- Scenario axis is contiguous for all arrays.
- All obligor-invariant quantities are hoisted outside the scenario loop.
- Student-t scaling uses precomputed 1/sqrt(W) factors.
- No heap allocation occurs inside either kernel.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    _I32_1D,
    parallel_njit,
)


@parallel_njit(
    nb.void(
        _F64_2D,  # out
        _F64_2D,  # latent_factors
        _F64_2D,  # ead
        _F64_1D,  # lgd
        _F64_1D,  # factor_loading
        _F64_1D,  # thresholds
        _I32_1D,  # sys_uix
        _I32_1D,  # idio_uix
    )
)
def kernel_sf_gaussian_copula_jtd(
    out: npt.NDArray[np.float64],  # (M, S)
    latent_factors: npt.NDArray[np.float64],  # (F, S)
    ead: npt.NDArray[np.float64],  # (M, S)
    lgd: npt.NDArray[np.float64],  # (M,)
    factor_loading: npt.NDArray[np.float64],  # (M,)
    thresholds: npt.NDArray[np.float64],  # (M,)
    sys_uix: npt.NDArray[np.int32],  # (M,)
    idio_uix: npt.NDArray[np.int32],  # (M,)
) -> None:
    """
    Single-factor Gaussian copula JTD loss kernel.
    """
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        factor_loading_i = factor_loading[i]
        threshold_i = thresholds[i]
        loss_scale = lgd[i]

        sqrt_1mb = np.sqrt(1.0 - factor_loading_i * factor_loading_i)

        z_row = latent_factors[sys_uix[i]]
        eps_row = latent_factors[idio_uix[i]]

        ead_row = ead[i]
        out_row = out[i]

        for s in range(s_scenarios):
            latent = factor_loading_i * z_row[s] + sqrt_1mb * eps_row[s]

            out_row[s] = ead_row[s] * loss_scale * (latent < threshold_i)
