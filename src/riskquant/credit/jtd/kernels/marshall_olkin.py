# src/riskquant/credit/jtd/kernels/marshall_olkin.py

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
        _F64_2D_C,  # out (N, S)
        _F64_2D_C,  # latent_factors (F, S)
        _F64_2D_C,  # ead (N, S)
        _F64_1D_C,  # lgd (N,)
        _F64_1D_C,  # pd_idio (N,) - The decoupled idiosyncratic PD
        _F64_1D_C,  # pd_sys (N,) - The probability of the common shock triggering
        _I32_1D_C,  # uniform_uix (N,) - Obligor-specific idiosyncratic uniform index
        _I32_1D_C,  # systemic_uix (N,) - Shared systemic uniform index
    )
)
def kernel_marshall_olkin_jtd(
    out: npt.NDArray[np.float64],
    latent_factors: npt.NDArray[np.float64],
    ead: npt.NDArray[np.float64],
    lgd: npt.NDArray[np.float64],
    pd_idio: npt.NDArray[np.float64],
    pd_sys: npt.NDArray[np.float64],
    uniform_uix: npt.NDArray[np.int32],
    systemic_uix: npt.NDArray[np.int32],
) -> None:

    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        pd_idio_i = pd_idio[i]
        pd_sys_i = pd_sys[i]
        lgd_i = lgd[i]

        u_row = latent_factors[uniform_uix[i]]
        sys_row = latent_factors[systemic_uix[i]]

        ead_row = ead[i]
        out_row = out[i]

        for s in range(s_scenarios):
            # Branchless logic: bitwise OR (|) forces LLVM to evaluate both conditions
            # without jump instructions, acting as a direct boolean addition.
            defaulted = (u_row[s] < pd_idio_i) | (sys_row[s] < pd_sys_i)

            out_row[s] = ead_row[s] * lgd_i * defaulted
