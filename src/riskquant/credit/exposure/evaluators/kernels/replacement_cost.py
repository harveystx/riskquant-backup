# src/riskquant/credit/exposure/evaluators/kernels/replacement_cost.py
"""
Replacement Cost (RC) exposure model: kernel.

    V  = sum of MtM over the netting set's member trades
    RC = max(V - C, 0)

`_kernel_replacement_cost` computes RC for all netting sets and scenarios
in one parallel pass. The kernel uses positional-or-keyword arguments
(NOT keyword-only): Numba's parsing of the `*,` marker is unreliable
across versions and inconsistent with every other JTD kernel in the
codebase. The evaluator wrapper materializes the SoA fields from the
view before crossing the JIT boundary.
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

from riskquant.jit import (
    _F64_1D_C,
    _F64_2D_C,
    _I64_1D_C,
    parallel_njit,
)



@parallel_njit(
    nb.void(
        _F64_2D_C,  # out
        _F64_2D_C,  # mtm
        _I64_1D_C,  # mtm_row_perm
        _I64_1D_C,  # ns_offsets
        _F64_1D_C,  # collateral
    )
)
def _kernel_replacement_cost(
    out: npt.NDArray[np.float64],  # (n_netting_sets, s_scenarios)
    mtm: npt.NDArray[np.float64],  # (M_total, s_scenarios)
    mtm_row_perm: npt.NDArray[np.int64],  # (M_kept,)
    ns_offsets: npt.NDArray[np.int64],  # (n_netting_sets + 1,)
    collateral: npt.NDArray[np.float64],  # (n_netting_sets,)
) -> None:
    n_net_sets, s_scenarios = out.shape
    for n in nb.prange(n_net_sets):
        start = ns_offsets[n]
        end = ns_offsets[n + 1]
        coll = collateral[n]
        out_row = out[n]
        if end > start:
            row0 = mtm[mtm_row_perm[start]]
            for s in range(s_scenarios):
                out_row[s] = row0[s]
            for i in range(start + 1, end):
                row = mtm[mtm_row_perm[i]]
                for s in range(s_scenarios):
                    out_row[s] += row[s]
        else:
            for s in range(s_scenarios):
                out_row[s] = 0.0
        # RC = max(V - C, 0).
        for s in range(s_scenarios):
            out_row[s] = max(out_row[s] - coll, 0.0)
