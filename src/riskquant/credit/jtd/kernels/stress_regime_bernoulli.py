# src/riskquant/credit/jtd/kernels/stress_regime_bernoulli.py


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
        _F64_2D,  # out (N, S)
        _F64_2D,  # latent_factors (N, S)
        _F64_1D,  # stress_regime_indicator (S,) * (stress regime indicator for each scenario s - 1 if stressed, 0 if not)
        _F64_2D,  # ead  (N, S)
        _F64_1D,  # lgd (N,)
        _F64_1D,  # pd (N,)
        _I32_1D,  # uniform_uix (N,)
        _F64_1D,  # stress_pd_multiplier (N,)
    )
)
def kernel_stress_regime_bernoulli_jtd(
    out: npt.NDArray[np.float64],  # (N, S)
    latent_factors: npt.NDArray[np.float64],  # (N, S)
    stress_regime_indicator: npt.NDArray[
        np.float64
    ],  # (S,) stress_regime_indicator ∈ {0, 1}
    ead: npt.NDArray[np.float64],  # (N, S)
    lgd: npt.NDArray[np.float64],  # (N,)
    pd: npt.NDArray[np.float64],  # (N,)
    uniform_uix: npt.NDArray[np.int32],  # (N,)
    stress_pd_multiplier: npt.NDArray[np.float64],  # (N,)
):
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):  # parallel over obligors
        pd_i = pd[i]
        lgd_i = lgd[i]
        stress_pd_mult_i = stress_pd_multiplier[i]

        u_row = latent_factors[uniform_uix[i]]

        ead_row = ead[i]
        out_row = out[i]

        stressed_pd_i = min(pd_i * stress_pd_mult_i, 1.0)

        for s in range(s_scenarios):  # scenarios
            effective_pd = stressed_pd_i if stress_regime_indicator[s] > 0.5 else pd_i

            defaulted = u_row[s] < effective_pd  # Bernoulli draw against effective PD

            out_row[s] = ead_row[s] * lgd_i * defaulted
