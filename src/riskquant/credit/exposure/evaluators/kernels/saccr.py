# src/riskquant/credit/exposure/evaluators/kernels/saccr.py
"""
SA-CCR exposure kernel (scenario-dependent part only).

    RC   = max(V - C, 0)
    mult = min(1, floor + (1-floor) * exp((V - C) / (2*(1-floor)*AddOn)))
    PFE  = mult * AddOn
    EAD  = alpha * (RC + PFE)

``AddOn`` is precomputed per netting set (hedging-set aggregation, scenario-free)
so the kernel only reduces marks and applies the multiplier. The exp is guarded:
when V - C >= 0 the multiplier is already at its cap of 1, so exp is skipped
(no overflow); when AddOn == 0 the multiplier is irrelevant (PFE = 0).
"""

from __future__ import annotations

import math

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

from riskquant.jit import parallel_njit

_F64_2D = nb.float64[:, ::1]
_F64_1D = nb.float64[::1]
_I64_1D = nb.int64[::1]


@parallel_njit(
    nb.void(
        _F64_2D,
        _F64_2D,
        _I64_1D,
        _I64_1D,
        _F64_1D,
        _F64_1D,
        nb.float64,
        nb.float64,
    )
)
def _kernel_saccr(
    out: npt.NDArray[np.float64],  # (n_netting_sets, s_scenarios)
    mtm: npt.NDArray[np.float64],  # (M_total, s_scenarios)
    mtm_row_perm: npt.NDArray[np.int64],  # (M_view,)
    ns_offsets: npt.NDArray[np.int64],  # (n_netting_sets + 1,)
    collateral: npt.NDArray[np.float64],  # (n_netting_sets,)
    add_on: npt.NDArray[np.float64],  # (n_netting_sets,) scenario-free
    alpha: float,
    floor: float,
) -> None:
    n_net_sets, s_scenarios = out.shape
    one_minus_floor = 1.0 - floor
    for n in nb.prange(n_net_sets):
        start = ns_offsets[n]
        end = ns_offsets[n + 1]
        coll = collateral[n]
        add_on_n = add_on[n]
        out_row = out[n]
        denom = 2.0 * one_minus_floor * add_on_n
        for s in range(s_scenarios):
            v = 0.0
            for i in range(start, end):
                v += mtm[mtm_row_perm[i], s]
            v_minus_c = v - coll
            rc = max(v_minus_c, 0.0)
            if add_on_n > 0.0 and v_minus_c < 0.0:
                mult = floor + one_minus_floor * math.exp(v_minus_c / denom)
            else:
                mult = 1.0  # V-C >= 0 -> capped; AddOn == 0 -> PFE == 0 regardless
            out_row[s] = alpha * (rc + mult * add_on_n)
