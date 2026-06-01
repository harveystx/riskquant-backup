# src/riskquant/credit/exposure/compile.py

"""
Exposure compiler: raw trade -> netting-set CSR views.

PURE PROJECTION KERNEL.

-----------------------------------------------------------------------

HARD INVARIANTS (ENFORCED)

- Every trade MUST map to a netting set uid
- Every netting set uid MUST exist in the master
- No implicit singleton construction
- No partial joins
- No silent fallback logic

-----------------------------------------------------------------------

DESIGN

- Structure = CSR over netting sets
- Attributes = strict projection from master
- Join primitive = centralized (searchsorted_or_raise)

-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.constants import NO_NETTING_SET_ID
from riskquant.credit._uid_join import searchsorted_or_raise
from riskquant.credit.exposure.compiled import CompiledExposure
from riskquant.credit.exposure.views.replacement_cost import RcNettingSetView
from riskquant.credit.netting_set.compiled import CompiledNettingSetMaster

# =====================================================================
# Validation
# =====================================================================


def _validate_inputs(
    trade_uids: npt.NDArray[np.int64],
    trade_netting_set_id: npt.NDArray[np.int64],
) -> None:
    if trade_uids.dtype != np.int64:
        raise TypeError("trade_uids must be int64")

    if trade_netting_set_id.dtype != np.int64:
        raise TypeError("trade_netting_set_id must be int64")

    if trade_uids.ndim != 1 or trade_netting_set_id.ndim != 1:
        raise ValueError("inputs must be 1D")

    if trade_uids.shape[0] != trade_netting_set_id.shape[0]:
        raise ValueError("trade_uids and trade_netting_set_id must align")


def _validate_master(master: CompiledNettingSetMaster) -> None:
    if master.uid.dtype != np.int64:
        raise TypeError("netting_set_master.uid must be int64")

    if master.net_collateral.dtype != np.float64:
        raise TypeError("netting_set_master.net_collateral must be float64")

    if master.uid.ndim != 1:
        raise ValueError("netting_set_master.uid must be 1D")

    if master.uid.size >= 2 and not np.all(master.uid[:-1] < master.uid[1:]):
        raise ValueError("netting_set_master.uid must be strictly ascending and unique")


def _validate_no_missing_netting_sets(
    trade_netting_set_id: npt.NDArray[np.int64],
) -> None:
    if np.any(trade_netting_set_id == NO_NETTING_SET_ID):
        raise ValueError(
            "NO_NETTING_SET_ID detected. "
            "All trades must be assigned to a valid or singleton netting set upstream."
        )


# =====================================================================
# CSR construction (structure only)
# =====================================================================


def _build_csr(
    netting_set_per_row: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.int64],  # netting_set_uids
    npt.NDArray[np.int64],  # offsets
    npt.NDArray[np.int64],  # mtm_row_perm
]:
    """
    Deterministic grouping of trades by netting set.

    Guarantees:
    - netting_set_uids sorted ascending
    - stable ordering within each netting set
    """

    order = np.argsort(netting_set_per_row, kind="stable")

    mtm_row_perm = np.ascontiguousarray(order, dtype=np.int64)
    sorted_ns = netting_set_per_row[order]

    netting_set_uids, counts = np.unique(sorted_ns, return_counts=True)
    netting_set_uids = np.ascontiguousarray(netting_set_uids, dtype=np.int64)

    offsets = np.empty(netting_set_uids.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    return netting_set_uids, offsets, mtm_row_perm


# =====================================================================
# Attribute projection (STRICT)
# =====================================================================


def _project_netting_set_attributes(
    netting_set_uids: npt.NDArray[np.int64],
    master: CompiledNettingSetMaster,
) -> tuple[
    npt.NDArray[np.float64],  # collateral
    npt.NDArray[np.bool_],  # is_collateralized
]:
    """
    Strict projection from master.

    Invariant:
    - Every netting_set_uid MUST exist in master.uid
    """

    pos = searchsorted_or_raise(
        target=netting_set_uids,
        superset_sorted=master.uid,
        ctx="compile_exposure: netting_set_uids",
    )

    collateral = np.ascontiguousarray(
        master.net_collateral[pos],
        dtype=np.float64,
    )

    is_collateralized = collateral > 0.0

    return collateral, is_collateralized


# =====================================================================
# Public API
# =====================================================================


def compile_exposure(
    *,
    trade_uids: npt.NDArray[np.int64],
    trade_netting_set_id: npt.NDArray[np.int64],
    netting_set_master: CompiledNettingSetMaster,
) -> CompiledExposure:
    """
    Compile exposure into a Replacement Cost netting-set CSR view.

    This function is a PURE projection kernel:
    - no mutation
    - no inference
    - no fallback logic

    Upstream responsibilities:
    - assign singleton netting sets
    - ensure full master coverage
    """

    # ---- Validate inputs ----
    _validate_inputs(trade_uids, trade_netting_set_id)
    _validate_master(netting_set_master)
    _validate_no_missing_netting_sets(trade_netting_set_id)

    # ---- Build structure ----
    netting_set_uids, offsets, mtm_row_perm = _build_csr(trade_netting_set_id)

    # ---- Project attributes ----
    collateral, is_collateralized = _project_netting_set_attributes(
        netting_set_uids,
        netting_set_master,
    )

    # ---- Construct view ----
    rc_view = RcNettingSetView(
        netting_set_uids=netting_set_uids,
        netting_set_offsets=offsets,
        mtm_row_perm=mtm_row_perm,
        collateral=collateral,
        is_collateralized=is_collateralized,
    )

    return CompiledExposure(replacement_cost=rc_view)
