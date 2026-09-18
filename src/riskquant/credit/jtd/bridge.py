# src/riskquant/credit/jtd/bridge.py
"""
Exposure -> Credit bridge.

Aggregates per-netting-set EAD (the output of `run_exposure`) up to per-obligor
EAD, then reindexes the result onto a target obligor-uid order. Pure SoA joins
plus one CSR segment-sum kernel — no row loops in Python, no jagged arrays.

Target ordering contract (D-C01)
--------------------------------
`obligor_uids_target` may be in ANY order. It is normally
`jtd_portfolio.obligor_uids`, which is model/band order — the concatenation of
each compiled block's uids — and is therefore NOT sorted ascending. The output
rows are emitted in exactly the caller's target order, because that is the row
order `run_jtd`'s `ead_by_obligor` argument must match.

`searchsorted` requires a sorted-unique universe, so the obligor lookup is
performed against a sorted VIEW of the target and the resulting ranks are mapped
back through the sort permutation. Uniqueness is not a view property, so it is
checked and enforced: duplicate target uids would make row attribution
ambiguous.

The two joins differ in semantics:

  1. netting-set uid -> obligor uid is a FAIL-FAST lookup: an EadResult row
     that points at an unknown netting set is a bug in the upstream pipeline.

  2. obligor_per_ns -> target obligor index is a SOFT lookup: a netting set
     whose obligor has no row in `obligor_uids_target` is dropped (e.g. when
     the JTD portfolio only models a subset of the obligor universe).

The reverse of (2) is NOT soft (D-C03): a target obligor with no netting set at
all is a gap in the inputs, not a legitimate zero exposure, and raises — the
same semantics `align_obligor_ead_to_portfolio` has always enforced on the
obligor-grain path.
"""

from __future__ import annotations

__all__ = [
    "aggregate_ead_to_obligor",
]

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

_PREVIEW = 10


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


def _sorted_target_view(
    obligor_uids_target: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Sorted view of an arbitrarily-ordered target plus its inverse permutation.

    `searchsorted` is only defined against a sorted-unique universe, and the
    target is band order. Sorting a VIEW keeps the caller's row order intact:
    the rank a lookup returns is mapped back to the caller's row through
    ``order``.

    Returns
    -------
    order
        ``(n_target,)`` int64 — ``order[rank]`` is the caller's row index for
        the ``rank``-th smallest target uid.
    target_sorted
        ``(n_target,)`` int64 — the target uids in ascending order.

    Raises
    ------
    ValueError
        If the target contains duplicate uids (row attribution would be
        ambiguous).
    """
    order = np.argsort(obligor_uids_target, kind="stable")
    target_sorted = obligor_uids_target[order]

    if target_sorted.shape[0] > 1:
        repeated = target_sorted[:-1] == target_sorted[1:]
        if repeated.any():
            dups = np.unique(target_sorted[:-1][repeated])[:_PREVIEW].tolist()
            raise ValueError(
                "aggregate_ead_to_obligor: obligor_uids_target contains "
                f"duplicate uid(s) {dups}; each obligor must occupy exactly one "
                "target row."
            )

    return order, target_sorted


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
        Target obligor-uid order (typically `jtd_portfolio.obligor_uids`, which
        is band order and NOT sorted). Any order is accepted and preserved
        exactly; uids must be unique. Netting sets whose obligor is absent from
        this target are dropped.

    Returns
    -------
    npt.NDArray[np.float64]
        ``(n_target, S)`` C-contiguous EAD, row *i* belonging to
        ``obligor_uids_target[i]``.

    Raises
    ------
    ValueError
        If the target contains duplicate uids; if an EAD row references a
        netting set absent from the master; or if a target obligor has no
        netting set at all (D-C03 — a gap in the inputs, not a zero exposure).
    """
    n_target = int(obligor_uids_target.shape[0])

    # Target order is free; uniqueness is not. Sort a view for the lookup and
    # keep the permutation that maps a rank back to the caller's row.
    order, target_sorted = _sorted_target_view(obligor_uids_target)

    # Strict: ns uid -> obligor uid.
    ns_pos = searchsorted_or_raise(
        raw=ead_result.uid,
        universe=netting_set_master.uid,
        ctx="aggregate_ead_to_obligor: netting-set uid(s)",
    )
    obligor_per_ns = netting_set_master.obligor_uid[ns_pos]

    # Soft: rank of each netting set's obligor within the SORTED target view;
    # absences -> INVALID_UIX (-999). It is valid and expected for some netting
    # sets to belong to obligors not modelled in the target JTD portfolio.
    rank = searchsorted_soft_join(
        raw=obligor_per_ns,
        universe=target_sorted,
        sentinel=INVALID_UIX,
    )

    # Drop ns rows whose obligor isn't in the target universe, then map the
    # surviving ranks back onto the caller's row order.
    keep = rank >= 0
    target_idx = order[rank[keep]]
    kept_ead_rows = np.flatnonzero(keep).astype(np.int32, copy=False)

    # CSR build: sort kept ns rows by target obligor, emit offsets.
    csr_order = np.argsort(target_idx, kind="stable")
    ead_perm = np.ascontiguousarray(kept_ead_rows[csr_order], dtype=np.int32)
    counts = np.bincount(target_idx, minlength=n_target)

    # D-C03: a modelled obligor with no netting set is an input gap. Zeroing it
    # would silently remove the obligor from EL/VaR/ES with no diagnostic.
    if n_target and not bool(counts.all()):
        missing = obligor_uids_target[np.flatnonzero(counts == 0)]
        raise ValueError(
            "aggregate_ead_to_obligor: "
            f"{missing.size} target obligor(s) have no netting set in the "
            f"netting-set master: {missing[:_PREVIEW].tolist()}. A modelled "
            "obligor with no exposure is a gap in the inputs, not a zero "
            "exposure; supply its netting set(s) or drop it from the target."
        )

    target_offsets = np.zeros(n_target + 1, dtype=np.int64)
    np.cumsum(counts.astype(np.int64, copy=False), out=target_offsets[1:])

    s = ead_result.exposure.shape[1]
    out = np.empty((n_target, s), dtype=np.float64, order="C")
    _kernel_aggregate_ead(
        out=out,
        ead=ead_result.exposure,
        ead_perm=ead_perm,
        target_offsets=target_offsets,
    )
    return out
