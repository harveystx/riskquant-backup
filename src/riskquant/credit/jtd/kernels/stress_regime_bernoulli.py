# src/riskquant/credit/jtd/kernels/stress_regime_bernoulli.py


from __future__ import annotations

import numba as nb  # type: ignore
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
        _F64_1D_C,  # pd (N,)
        _I32_1D_C,  # uniform_uix (N,)
        _I32_1D_C,  # stress_regime_uix (N,) - per-obligor stress-indicator factor
        _F64_1D_C,  # stress_pd_multiplier (N,)
    )
)
def kernel_stress_regime_bernoulli_jtd(
    out: npt.NDArray[np.float64],  # (N, S)
    latent_factors: npt.NDArray[np.float64],  # (F, S)
    ead: npt.NDArray[np.float64],  # (N, S)
    lgd: npt.NDArray[np.float64],  # (N,)
    pd: npt.NDArray[np.float64],  # (N,)
    uniform_uix: npt.NDArray[np.int32],  # (N,)
    stress_regime_uix: npt.NDArray[np.int32],  # (N,)
    stress_pd_multiplier: npt.NDArray[np.float64],  # (N,)
):
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):  # parallel over obligors
        pd_i = pd[i]
        lgd_i = lgd[i]
        stress_pd_mult_i = stress_pd_multiplier[i]

        u_row = latent_factors[uniform_uix[i]]

        # Each obligor reads its own stress-regime indicator row. Obligors sharing
        # a uix share a regime (correlated); distinct uix -> independent
        # stress regimes. All same index -> global stress regime (all obligors share
        # the same regime).
        stress_row = latent_factors[stress_regime_uix[i]]

        ead_row = ead[i]
        out_row = out[i]

        stressed_pd_i = min(pd_i * stress_pd_mult_i, 1.0)

        for s in range(s_scenarios):  # scenarios
            effective_pd = stressed_pd_i if stress_row[s] > 0.5 else pd_i

            defaulted = u_row[s] < effective_pd  # Bernoulli draw against effective PD

            out_row[s] = ead_row[s] * lgd_i * defaulted
