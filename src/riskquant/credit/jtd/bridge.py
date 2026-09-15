# src/riskquant/credit/jtd/bridge.py
"""
Exposure -> Credit bridge.

Aggregates per-netting-set EAD (the output of `run_exposure`) up to per-obligor
EAD, then reindexes the result onto a target obligor-uid order. Pure SoA joins
plus one CSR segment-sum kernel — no row loops in Python, no jagged arrays.

The two joins differ in semantics:

  1. netting-set uid -> obligor uid is a FAIL-FAST lookup: an EadResult row
     that points at an unknown netting set is a bug in the upstream pipeline.

  2. obligor_per_ns -> target obligor index is a SOFT lookup: a netting set
     whose obligor has no row in `obligor_uids_target` is dropped (e.g. when
     the JTD portfolio only models a subset of the obligor universe).
"""

from __future__ import annotations

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

from riskquant._arrays import (
    searchsorted_or_raise,
    searchsorted_soft_join,
)
from riskquant.constants import INVALID_UIX
from riskquant.credit.exposure.results import NettingSetExposurePaths
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.jit import (
    _F64_2D_C,
    _I32_1D_C,
    _I64_1D_C,
    _NB_VOID,
    parallel_njit,
)


@parallel_njit(
    _NB_VOID(
        _F64_2D_C,
        _F64_2D_C,
        _I32_1D_C,
        _I64_1D_C,
    )
)
def _kernel_aggregate_ead(
    out: npt.NDArray[np.float64],  # (n_target, S)
    ead: npt.NDArray[np.float64],  # (n_ns, S)
    ead_perm: npt.NDArray[np.int32],  # (n_ns,)   ead row -> target band order
    target_offsets: npt.NDArray[np.int64],  # (n_target + 1,)
) -> None:
    n_target, s_scenarios = out.shape
    for g in nb.prange(n_target):
        start = target_offsets[g]
        end = target_offsets[g + 1]
        out_row = out[g]
        if end == start:
            for s in range(s_scenarios):
                out_row[s] = 0.0
            continue
        src0 = ead[ead_perm[start]]
        for s in range(s_scenarios):
            out_row[s] = src0[s]
        for i in range(start + 1, end):
            src = ead[ead_perm[i]]
            for s in range(s_scenarios):
                out_row[s] += src[s]


def aggregate_ead_to_obligor(
    *,
    ead_result: NettingSetExposurePaths,
    netting_set_master: CompiledNettingSetMaster,
    obligor_uids_target: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """
    Aggregate per-netting-set EAD up to per-obligor EAD in target-row order.

    Parameters
    ----------
    ead_result
        Output of `run_exposure`; rows aligned to its `uid` (netting-set
        uid) array.
    netting_set_master
        Provides the netting-set uid -> obligor uid linkage. `uid`
        ascending sorted.
    obligor_uids_target
        Target obligor-uid order (typically `jtd_portfolio.obligor_uids`).
        Rows in the output align to this; obligors absent here are
        dropped from the aggregation.
    """
    # Strict: ns uid -> obligor uid.
    ns_pos = searchsorted_or_raise(
        raw=ead_result.uid,
        universe=netting_set_master.uid,
        ctx="aggregate_ead_to_obligor: netting-set uid(s)",
    )
    obligor_per_ns = netting_set_master.obligor_uid[ns_pos]

    # Soft: target obligor index per ns; absences -> INVALID_UIX (-999).
    # We must explicitly probe membership here, because it is valid and
    # expected for some netting sets to belong to obligors not modeled
    # in the target JTD portfolio.
    target_idx = searchsorted_soft_join(
        raw=obligor_per_ns,
        universe=obligor_uids_target,
        sentinel=INVALID_UIX,
    )

    # Drop ns rows whose obligor isn't in the target universe.
    keep = target_idx >= 0
    target_idx = target_idx[keep]
    kept_ead_rows = np.flatnonzero(keep).astype(np.int32, copy=False)

    # CSR build: sort kept ns rows by target obligor, emit offsets.
    order = np.argsort(target_idx, kind="stable")
    ead_perm = np.ascontiguousarray(kept_ead_rows[order], dtype=np.int32)
    target_offsets = np.zeros(obligor_uids_target.shape[0] + 1, dtype=np.int64)
    counts = np.bincount(target_idx, minlength=obligor_uids_target.shape[0])
    np.cumsum(counts.astype(np.int64, copy=False), out=target_offsets[1:])

    s = ead_result.exposure.shape[1]
    out = np.empty((obligor_uids_target.shape[0], s), dtype=np.float64, order="C")
    _kernel_aggregate_ead(
        out=out,
        ead=ead_result.exposure,
        ead_perm=ead_perm,
        target_offsets=target_offsets,
    )
    return out
