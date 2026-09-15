# src/riskquant/credit/jtd/kernels/mf_copula_gaussian.py
"""
Bare-metal multi-factor Gaussian copula JTD loss kernel.

This kernel evaluates Jump-to-Default (JTD) losses under a multi-factor
Gaussian copula model.

Each obligor is exposed to an arbitrary sparse set of systematic credit
factors and one idiosyncratic factor. The systematic exposures are stored in
compressed sparse row (CSR) form to avoid materializing a dense
(obligors × factors) loading matrix.

===============================================================================
MATHEMATICAL MODEL
===============================================================================

For obligor i and scenario s:

    A_i,s
        =
        Σ_j β_i,j Z_j,s
        +
        w_i ε_i,s

where

    A_i,s
        latent asset value

    β_i,j
        loading of obligor i to systematic factor j

    Z_j,s
        realization of systematic factor j in scenario s

    ε_i,s
        realization of the obligor-specific idiosyncratic factor

    w_i
        idiosyncratic loading

The sum runs only over the obligor's non-zero factor exposures.

To preserve unit variance:

    w_i = sqrt(1 - Σ_j β_i,j²)

which is precomputed during ingest and supplied as `idio_weight`.

Default occurs when the latent asset value falls below the obligor's
precomputed threshold:

    default_i,s = 1(A_i,s < threshold_i)

where

    threshold_i = Φ⁻¹(PD_i)

and Φ⁻¹ is the inverse standard-normal CDF.

Realized JTD loss is:

    loss_i,s
        =
        EAD_i,s × LGD_i × default_i,s

===============================================================================
SPARSE FACTOR REPRESENTATION
===============================================================================

Systematic loadings are stored in CSR form:

    loading_indptr
        row pointers for each obligor

    loading_values
        non-zero β values

    systemic_uix
        factor UIX associated with each loading

For obligor i:

    start = loading_indptr[i]
    end   = loading_indptr[i + 1]

the factor exposures are

    loading_values[start:end]

to factors

    systemic_uix[start:end]

This permits O(nnz) storage and computation instead of
O(n_obligors × n_factors).

===============================================================================
KERNEL ALGORITHM
===============================================================================

For each obligor:

1. Initialize the latent accumulator with the idiosyncratic component:

       A_i,s = w_i ε_i,s

2. Accumulate all systematic factor contributions:

       A_i,s += β_i,j Z_j,s

3. Compare against the default threshold:

       default_i,s = (A_i,s < threshold_i)

4. Convert default indicators into realized losses:

       loss_i,s = EAD_i,s × LGD_i × default_i,s

===============================================================================
MEMORY DESIGN
===============================================================================

`out` serves dual purpose:

    Phase 1:
        latent-value accumulator A_i,s

    Phase 2:
        overwritten in-place with realized JTD loss

This avoids allocating a separate latent-value matrix of shape:

    (n_obligors, n_scenarios)

and keeps peak memory usage minimal.

All scenario loops operate over contiguous memory, allowing efficient
vectorized memory access and parallelization across obligors.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _I32_1D_C,
    _I64_1D_C,
    _NB_VOID,
    parallel_njit,
)


@parallel_njit(
    _NB_VOID(
        _F64_2D_C,  # out (M, S)
        _F64_2D_C,  # latent_factors (F, S)
        _F64_2D_C,  # ead (M, S)
        _F64_1D_C,  # lgd (M,)
        _I64_1D_C,  # loading_indptr (M + 1,)
        _F64_1D_C,  # loading_values (nnz,)
        _I32_1D_C,  # systemic_uix (nnz,)
        _I32_1D_C,  # idiosyncratic_uix (M,)
        _F64_1D_C,  # idio_weight (M,)
        _F64_1D_C,  # thresholds (M,)
    )
)
def kernel_mf_gaussian_copula_jtd(
    out: npt.NDArray[np.float64],
    latent_factors: npt.NDArray[np.float64],
    ead: npt.NDArray[np.float64],
    lgd: npt.NDArray[np.float64],
    loading_indptr: npt.NDArray[np.int64],
    loading_values: npt.NDArray[np.float64],
    systemic_uix: npt.NDArray[np.int32],
    idiosyncratic_uix: npt.NDArray[np.int32],
    idio_weight: npt.NDArray[np.float64],
    thresholds: npt.NDArray[np.float64],
) -> None:
    """Multi-factor Gaussian copula JTD loss kernel."""
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        out_row = out[i]

        # Seed the latent accumulator with the idiosyncratic component.
        idio_w = idio_weight[i]
        eps_row = latent_factors[idiosyncratic_uix[i]]
        for s in range(s_scenarios):
            out_row[s] = idio_w * eps_row[s]

        # Accumulate systemic contributions factor-outer: each Z row is read
        # unit-stride and folded into the (unit-stride) accumulator.
        start = loading_indptr[i]
        end = loading_indptr[i + 1]
        for j in range(start, end):
            load_j = loading_values[j]
            z_row = latent_factors[systemic_uix[j]]
            for s in range(s_scenarios):
                out_row[s] += load_j * z_row[s]

        # Finalize: latent -> realized loss, in place.
        threshold_i = thresholds[i]
        loss_scale = lgd[i]
        ead_row = ead[i]
        for s in range(s_scenarios):
            out_row[s] = ead_row[s] * loss_scale * (out_row[s] < threshold_i)
