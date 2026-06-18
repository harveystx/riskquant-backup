# src/riskquant/credit/masters/cross_master/validate.py
"""
Layered cross-master validators for the credit package.

Single-master validators (`obligors/validate.py`, `netting_set/validate.py`)
enforce intra-table invariants — dtype, range, uniqueness, sortedness,
self-references. They are intentionally unaware of other masters: an
obligor master that imports the netting-set master, or vice versa, would
couple two edges that should compile independently.

Layered invariants — facts that hold *across* tables — live here. The
canonical layered invariant is the foreign-key relationship from
`CompiledNettingSetMaster.obligor_uid` into `CompiledObligorCredit.uid`:
every netting-set row must point at an obligor that exists on the master.

This package is consumed by the top-level pipeline orchestrator (a future
`compile_credit_pipeline(...)` entry point) and by any caller that holds
both masters and wants to fail fast before they reach a downstream join.
Pure vectorized predicates on already-compiled SoA — no row loops, no
pandas, no string lookups.
"""

from __future__ import annotations

from typing import Final

from riskquant._arrays import searchsorted_membership
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit

_PREVIEW: Final[int] = 10


def validate_netting_set_obligor_uids_on_master(
    *,
    netting_set_master: CompiledNettingSetMaster,
    obligor_master: CompiledObligorCredit,
) -> None:
    """
    Check that every netting set's owning obligor exists on the obligor master.

    Raises
    ------
    ValueError
        If any `netting_set_master.obligor_uid` value is absent from
        `obligor_master.uid`. The error message previews up to ten
        offending (ns_uid, obligor_uid) pairs.
    """
    matched = searchsorted_membership(
        universe=obligor_master.uid,
        raw=netting_set_master.obligor_uid,
    )
    if matched.all():
        return

    bad_pairs = list(
        zip(
            netting_set_master.uid[~matched][:_PREVIEW].tolist(),
            netting_set_master.obligor_uid[~matched][:_PREVIEW].tolist(),
            strict=True,
        )
    )
    n_bad = int((~matched).sum())
    raise ValueError(
        f"cross_master: {n_bad} netting set(s) reference obligor uid(s) "
        f"absent from the obligor master. First {len(bad_pairs)} "
        f"(ns_uid, obligor_uid) pair(s): {bad_pairs}"
    )


def validate_cross_masters(
    *,
    obligor_master: CompiledObligorCredit,
    netting_set_master: CompiledNettingSetMaster,
) -> None:
    """
    Single-call entry point: every cross-master invariant, fail-fast.

    Today: the netting-set -> obligor foreign key. Future layered checks
    (rating taxonomy join, sector taxonomy join, parent-obligor closure
    against an external corporate hierarchy) land here too.
    """
    validate_netting_set_obligor_uids_on_master(
        netting_set_master=netting_set_master,
        obligor_master=obligor_master,
    )
