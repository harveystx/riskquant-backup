# src/riskquant/credit/exposure/evaluators/replacement_cost.py
"""
Replacement Cost (RC) EAD model: kernel + evaluator.

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

from riskquant.credit.exposure.views.replacement_cost import RcNettingSetView


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    cache=True,
)  # type: ignore[misc]
def _kernel_replacement_cost(
    out: npt.NDArray[np.float64],  # (N, S)
    mtm: npt.NDArray[np.float64],  # (M_total, S)
    mtm_row_perm: npt.NDArray[np.int64],  # (M_kept,)
    ns_offsets: npt.NDArray[np.int64],  # (N + 1,)
    collateral: npt.NDArray[np.float64],  # (N,)
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
        # Branchless floor: RC = max(V - C, 0).
        for s in range(s_scenarios):
            v = out_row[s] - coll
            out_row[s] = v * (v > 0.0)


def evaluate_replacement_cost(
    *,
    view: RcNettingSetView,
    mtm: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    """Unpack the view SoA fields and dispatch to the JIT kernel."""
    _kernel_replacement_cost(
        out,
        mtm,
        view.mtm_row_perm,
        view.netting_set_offsets,
        view.collateral,
    )
