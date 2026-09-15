# src/riskquant/credit/jtd/kernels/beta_binomial.py
from __future__ import annotations

import numba as nb
import numpy as np
import numpy.typing as npt

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _I32_1D_C,
    _NB_VOID,
    parallel_njit,
)


@parallel_njit(
    _NB_VOID(
        _F64_2D_C,  # out
        _F64_2D_C,  # latent_factors
        _F64_2D_C,  # ead
        _F64_1D_C,  # lgd
        _I32_1D_C,  # uniform_uix
        _I32_1D_C,  # beta_uix
    )
)
def kernel_beta_binomial_jtd(
    out: npt.NDArray[np.float64],
    latent_factors: npt.NDArray[np.float64],
    ead: npt.NDArray[np.float64],
    lgd: npt.NDArray[np.float64],
    uniform_uix: npt.NDArray[np.int32],
    beta_uix: npt.NDArray[np.int32],
) -> None:

    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        lgd_i = lgd[i]

        u_row = latent_factors[uniform_uix[i]]

        beta_row = latent_factors[beta_uix[i]]

        ead_row = ead[i]
        out_row = out[i]

        for s in range(s_scenarios):
            defaulted = u_row[s] < beta_row[s]

            out_row[s] = ead_row[s] * lgd_i * defaulted
