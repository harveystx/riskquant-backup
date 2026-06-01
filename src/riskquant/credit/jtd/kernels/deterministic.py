# src/riskquant/credit/jtd/kernels/deterministic.py
"""
Deterministic JTD loss kernel.

Scenario-independent: default is certain, so loss is `ead * lgd` with the
per-obligor `lgd` scalar broadcast across the scenario axis. The latent-factor
matrix is never read. Worst-case stress is simply `S = 1`.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)  # type: ignore[misc]
def kernel_deterministic_jtd(
    out: npt.NDArray[np.float64],  # (M, S)
    ead: npt.NDArray[np.float64],  # (M, S)
    lgd: npt.NDArray[np.float64],  # (M,)
) -> None:
    m_, s_scenarios = out.shape

    for i in nb.prange(m_):
        lgd_i = lgd[i]
        ead_row = ead[i]
        out_row = out[i]
        for s in range(s_scenarios):
            out_row[s] = ead_row[s] * lgd_i
