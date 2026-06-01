# src/riskquant/credit/jtd/kernels/copula.py
"""
Bare-metal copula JTD loss kernels — single-factor and multi-factor.

Zero-allocation, parallel, branchless. Loss is realized per scenario as
`ead * lgd * default_indicator`, so a scenario-dependent EAD flows straight
through with no static precompute. Latent factors are read `(F, S)` with the
scenario axis contiguous.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

_NB = dict(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)


@nb.njit(**_NB)  # type: ignore[misc]
def kernel_sf_gaussian_copula_jtd(
    out: npt.NDArray[np.float64],  # (M, S)
    latent_factors: npt.NDArray[np.float64],  # (F, S)
    ead: npt.NDArray[np.float64],  # (M, S)
    lgd: npt.NDArray[np.float64],  # (M,)
    beta: npt.NDArray[np.float64],  # (M,)
    thresholds: npt.NDArray[np.float64],  # (M,)
    sys_uix: npt.NDArray[np.int32],  # (M,)
    idio_uix: npt.NDArray[np.int32],  # (M,)
) -> None:
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        t_i = thresholds[i]
        beta_i = beta[i]
        sqrt_1mb = np.sqrt(1.0 - beta_i * beta_i)
        lgd_i = lgd[i]

        z_row = latent_factors[sys_uix[i]]
        eps_row = latent_factors[idio_uix[i]]
        ead_row = ead[i]
        out_row = out[i]

        for s in range(s_scenarios):
            a_ = beta_i * z_row[s] + sqrt_1mb * eps_row[s]
            out_row[s] = ead_row[s] * lgd_i * (t_i > a_)


@nb.njit(**_NB)  # type: ignore[misc]
def kernel_sf_t_copula_jtd(
    out: npt.NDArray[np.float64],  # (M, S)
    latent_factors: npt.NDArray[np.float64],  # (F, S)
    ead: npt.NDArray[np.float64],  # (M, S)
    lgd: npt.NDArray[np.float64],  # (M,)
    beta: npt.NDArray[np.float64],  # (M,)
    thresholds: npt.NDArray[np.float64],  # (M,)
    nu: npt.NDArray[np.float64],  # (M,)
    sys_uix: npt.NDArray[np.int32],  # (M,)
    idio_uix: npt.NDArray[np.int32],  # (M,)
    chi_uix: npt.NDArray[np.int32],  # (M,)
) -> None:
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        t_i = thresholds[i]
        beta_i = beta[i]
        sqrt_1mb = np.sqrt(1.0 - beta_i * beta_i)
        lgd_i = lgd[i]
        nu_i = nu[i]

        z_row = latent_factors[sys_uix[i]]
        eps_row = latent_factors[idio_uix[i]]
        w_row = latent_factors[chi_uix[i]]
        ead_row = ead[i]
        out_row = out[i]

        for s in range(s_scenarios):
            # W is a raw chi-square draw, strictly positive (cleaned at
            # factor-generation time); the t-transform is applied ONCE, here.
            m_factor = np.sqrt(nu_i / w_row[s])
            a_ = m_factor * (beta_i * z_row[s] + sqrt_1mb * eps_row[s])
            out_row[s] = ead_row[s] * lgd_i * (t_i > a_)
