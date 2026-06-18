# src/riskquant/credit/jtd/kernels/mf_copula_students_t.py
"""
Bare-metal multi-factor Student-t copula JTD loss kernel.

Combines the multi-factor Gaussian latent accumulation (sparse CSR systemic
loadings + idiosyncratic, per `kernel_mf_gaussian_copula_jtd`) with the Student-t
chi-mixing of the single-factor t kernel.

For obligor i and scenario s:

    A_gaussian = w_i * eps_i,s + Σ_j β_i,j Z_j,s        (CSR over the obligor)
    A_t        = sqrt(ν_i) * inv_sqrt_w_s * A_gaussian
    default    = 1(A_t < threshold_i)
    loss       = EAD_i,s * LGD_i * default

The rows referenced by `chi_uix` contain ``inv_sqrt_w = 1 / sqrt(W)`` (W ~ χ²),
precomputed in factor generation, so the hot path stays multiply/add/compare with
no square root. `threshold_i` is the Student-t quantile of PD_i at ν_i, set at
ingest. W is shared across obligors in a scenario, producing tail dependence.

`out` is the dual-purpose accumulator: phase 1 holds the Gaussian latent, phase 2
overwrites it in place with realized loss — no separate latent matrix.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

from riskquant.numba import (
    _F64_1D,
    _F64_2D,
    _I32_1D,
    _I64_1D,
    parallel_njit,
)


@parallel_njit(
    nb.void(
        _F64_2D,  # out (M, S)
        _F64_2D,  # latent_factors (F, S)
        _F64_2D,  # ead (M, S)
        _F64_1D,  # lgd (M,)
        _I64_1D,  # loading_indptr (M + 1,)
        _F64_1D,  # loading_values (nnz,)
        _I32_1D,  # systemic_uix (nnz,)
        _I32_1D,  # idiosyncratic_uix (M,)
        _F64_1D,  # idio_weight (M,)
        _F64_1D,  # thresholds (M,)
        _F64_1D,  # sqrt_nu (M,)
        _I32_1D,  # chi_squared_uix (M,)
    )
)
def kernel_mf_students_t_copula_jtd(
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
    sqrt_nu: npt.NDArray[np.float64],
    chi_squared_uix: npt.NDArray[np.int32],
) -> None:
    """Multi-factor Student-t copula JTD loss kernel."""
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        out_row = out[i]

        # Phase 1a: seed the latent accumulator with the idiosyncratic component.
        idio_w = idio_weight[i]
        eps_row = latent_factors[idiosyncratic_uix[i]]
        for s in range(s_scenarios):
            out_row[s] = idio_w * eps_row[s]

        # Phase 1b: accumulate systemic contributions (CSR slice for obligor i).
        start = loading_indptr[i]
        end = loading_indptr[i + 1]
        for j in range(start, end):
            load_j = loading_values[j]
            z_row = latent_factors[systemic_uix[j]]
            for s in range(s_scenarios):
                out_row[s] += load_j * z_row[s]

        # Phase 2: t-scale by sqrt(ν) * inv_sqrt_w, threshold, realize loss.
        sqrt_nu_i = sqrt_nu[i]
        inv_sqrt_w_row = latent_factors[chi_squared_uix[i]]
        threshold_i = thresholds[i]
        loss_scale = lgd[i]
        ead_row = ead[i]
        for s in range(s_scenarios):
            latent = sqrt_nu_i * inv_sqrt_w_row[s] * out_row[s]
            out_row[s] = ead_row[s] * loss_scale * (latent < threshold_i)
