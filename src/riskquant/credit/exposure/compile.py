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

from riskquant._arrays import searchsorted_or_raise
from riskquant.constants import (
    CEM_PFE_COEFFICIENT_DEFAULT,
    CEM_PFE_FLOOR_DEFAULT,
    NO_ID,
)
from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors
from riskquant.credit.exposure.compiled import CompiledExposure
from riskquant.credit.exposure.saccr.aggregate import compute_saccr_add_on
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig
from riskquant.credit.exposure.views.current_exposure_method import (
    CemNettingSetView,
)
from riskquant.credit.exposure.views.replacement_cost import RcNettingSetView
from riskquant.credit.exposure.views.saccr import SaccrNettingSetView
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster

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
    if np.any(trade_netting_set_id == NO_ID):
        raise ValueError(
            "NO_ID detected. "
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
) -> npt.NDArray[np.float64]:
    """
    Strict projection from master.

    Invariant:
    - Every netting_set_uid MUST exist in master.uid
    """

    pos = searchsorted_or_raise(
        raw=netting_set_uids,
        universe=master.uid,
        ctx="compile_exposure: netting_set_uids",
    )

    collateral = np.ascontiguousarray(
        master.net_collateral[pos],
        dtype=np.float64,
    )

    return collateral


# =====================================================================
# Public API
# =====================================================================


def compile_exposure_rc(
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
    collateral = _project_netting_set_attributes(
        netting_set_uids,
        netting_set_master,
    )

    # ---- Construct view ----
    rc_view = RcNettingSetView(
        netting_set_uids=netting_set_uids,
        netting_set_offsets=offsets,
        mtm_row_perm=mtm_row_perm,
        collateral=collateral,
    )

    return CompiledExposure(replacement_cost=rc_view)


def compile_exposure_cem(
    *,
    trade_uids: npt.NDArray[np.int64],
    trade_netting_set_id: npt.NDArray[np.int64],
    trade_effective_notional: npt.NDArray[np.float64],
    trade_supervisory_factor: npt.NDArray[np.float64],
    netting_set_master: CompiledNettingSetMaster,
    pfe_floor: float = CEM_PFE_FLOOR_DEFAULT,
    pfe_coefficient: float = CEM_PFE_COEFFICIENT_DEFAULT,
) -> CompiledExposure:
    """
    Compile exposure into a Current Exposure Method netting-set CSR view.

    Identical projection to `compile_exposure_rc` (RC), plus the two per-trade add-on
    scalars — ``effective_notional`` and ``supervisory_factor`` (resolved at the
    edge in ``add_on.resolve``) — permuted into the same CSR order as the MtM
    bands. Kept separable rather than pre-multiplied so the SA-CCR upgrade is a
    scalar swap. Pure projection: no mutation, no inference, no fallback.

    Parameters
    ----------
    trade_effective_notional, trade_supervisory_factor
        Per-trade, aligned to ``trade_uids`` (portfolio order). Their product is
        the trade's gross add-on; this compiler only reorders them.
    """
    _validate_inputs(trade_uids, trade_netting_set_id)
    _validate_master(netting_set_master)
    _validate_no_missing_netting_sets(trade_netting_set_id)

    for name, arr in (
        ("trade_effective_notional", trade_effective_notional),
        ("trade_supervisory_factor", trade_supervisory_factor),
    ):
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must be float64")
        if arr.shape != trade_uids.shape:
            raise ValueError(f"{name} must align with trade_uids")

    if not (0.0 <= pfe_floor <= 1.0):
        raise ValueError("pfe_floor must lie in [0, 1].")
    if pfe_coefficient < 0.0:
        raise ValueError("pfe_coefficient must be >= 0.")

    netting_set_uids, offsets, mtm_row_perm = _build_csr(trade_netting_set_id)

    collateral = _project_netting_set_attributes(
        netting_set_uids,
        netting_set_master,
    )

    # Reorder both add-on scalars into CSR (netting-set-contiguous) order so the
    # evaluator forms the product over a band directly, no gather.
    effective_notional = np.ascontiguousarray(
        trade_effective_notional[mtm_row_perm], dtype=np.float64
    )
    supervisory_factor = np.ascontiguousarray(
        trade_supervisory_factor[mtm_row_perm], dtype=np.float64
    )

    cem_view = CemNettingSetView(
        netting_set_uids=netting_set_uids,
        netting_set_offsets=offsets,
        mtm_row_perm=mtm_row_perm,
        collateral=collateral,
        effective_notional=effective_notional,
        supervisory_factor=supervisory_factor,
        pfe_floor=pfe_floor,
        pfe_coefficient=pfe_coefficient,
    )

    return CompiledExposure(current_exposure_method=cem_view)


def compile_exposure_saccr(
    *,
    trade_uids: npt.NDArray[np.int64],
    trade_netting_set_id: npt.NDArray[np.int64],
    trade_effective_notional: npt.NDArray[np.float64],
    trade_subclass_uix: npt.NDArray[np.int32],
    netting_set_master: CompiledNettingSetMaster,
    add_on_factors: CompiledAddOnFactors,
    saccr_config: CompiledSaccrConfig,
) -> CompiledExposure:
    """
    Compile exposure into a SA-CCR netting-set view.

    Same CSR projection as RC/CEM for the mark reduction, plus the hedging-set
    add-on aggregation (`compute_saccr_add_on`, vectorized, scenario-free) folded
    to one add-on per netting set. The kernel then applies the supervisory
    multiplier and alpha. Pure projection: no mutation, no fallback.
    """
    _validate_inputs(trade_uids, trade_netting_set_id)
    _validate_master(netting_set_master)
    _validate_no_missing_netting_sets(trade_netting_set_id)

    for name, arr in (("trade_effective_notional", trade_effective_notional),):
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must be float64")
        if arr.shape != trade_uids.shape:
            raise ValueError(f"{name} must align with trade_uids")
    if trade_subclass_uix.dtype != np.int32:
        raise TypeError("trade_subclass_uix must be int32")
    if trade_subclass_uix.shape != trade_uids.shape:
        raise ValueError("trade_subclass_uix must align with trade_uids")

    netting_set_uids, offsets, mtm_row_perm = _build_csr(trade_netting_set_id)
    collateral = _project_netting_set_attributes(netting_set_uids, netting_set_master)

    add_on = compute_saccr_add_on(
        netting_set_uids=netting_set_uids,
        trade_netting_set_id=trade_netting_set_id,
        trade_effective_notional=trade_effective_notional,
        trade_subclass_uix=trade_subclass_uix,
        add_on_factors=add_on_factors,
        saccr_config=saccr_config,
    )

    saccr_view = SaccrNettingSetView(
        netting_set_uids=netting_set_uids,
        netting_set_offsets=offsets,
        mtm_row_perm=mtm_row_perm,
        collateral=collateral,
        add_on=add_on,
        alpha=saccr_config.alpha,
        multiplier_floor=saccr_config.multiplier_floor,
    )
    return CompiledExposure(saccr=saccr_view)
