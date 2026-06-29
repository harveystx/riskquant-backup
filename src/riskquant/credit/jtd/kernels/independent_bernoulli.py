# src/riskquant/credit/jtd/kernels/independent_bernoulli.py


from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

from riskquant.jit import (
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
        _F64_1D,  # pd
        _I32_1D,  # uniform_uix
    )
)
def kernel_independent_bernoulli_jtd(
    out: npt.NDArray[np.float64],
    latent_factors: npt.NDArray[np.float64],
    ead: npt.NDArray[np.float64],
    lgd: npt.NDArray[np.float64],
    pd: npt.NDArray[np.float64],
    uniform_uix: npt.NDArray[np.int32],
):
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        pd_i = pd[i]
        lgd_i = lgd[i]

        u_row = latent_factors[uniform_uix[i]]

        ead_row = ead[i]
        out_row = out[i]

        for s in range(s_scenarios):
            defaulted = u_row[s] < pd_i

            out_row[s] = ead_row[s] * lgd_i * defaulted
