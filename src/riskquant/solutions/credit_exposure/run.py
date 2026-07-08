# src/riskquant/solutions/credit_exposure/run.py
"""
Credit-exposure entry point: trade table in, per-obligor EAD out.

``run_credit_exposure`` wraps the whole pipeline — build masters, resolve the
priced notional and supervisory factor, compile the chosen exposure view, run the
engine, aggregate netting-set EAD up to the obligor — behind one call, returning
the typed `CreditExposureResult` (Contract B). Deterministic: ``mtm`` is one
current mark per trade, so the exposure is a single "scenario".
"""

from __future__ import annotations

__all__ = [
    "run_credit_exposure",
]

import numpy as np
import pandas as pd

from riskquant.credit.exposure.add_on.ingest import compile_add_on_factors
from riskquant.credit.exposure.add_on.resolve import (
    resolve_quantity_price_effective_notional,
    resolve_supervisory_factors,
)
from riskquant.credit.exposure.compile import compile_exposure_cem, compile_exposure_rc
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.exposure.models import CURRENT_EXPOSURE_METHOD_ID
from riskquant.credit.jtd.bridge import aggregate_ead_to_obligor
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.solutions.credit_exposure.catalog import exposure_model_entry
from riskquant.solutions.credit_exposure.schemas import (
    CreditExposureRequest,
    CreditExposureResult,
    ExposureProvenance,
    NettingSetExposureRow,
    ObligorExposure,
)

_ENGINE_VERSION = "riskquant-credit-exposure/0.1.0"


def _coerce(request: CreditExposureRequest | dict) -> CreditExposureRequest:
    if isinstance(request, CreditExposureRequest):
        return request
    return CreditExposureRequest.model_validate(request)


def run_credit_exposure(
    request: CreditExposureRequest | dict,
    *,
    obligor_uids_target: np.ndarray | None = None,
) -> CreditExposureResult:
    """Run one credit-exposure request end to end.

    ``obligor_uids_target`` fixes the obligor row order (e.g. a JTD portfolio's
    ``obligor_uids``); it defaults to the ascending distinct obligors in the book.
    """
    req = _coerce(request)
    entry = exposure_model_entry(req.model)

    # --- netting-set master (ascending by uid, the master invariant) ---
    ns_uid = np.array([n.netting_set_uid for n in req.netting_sets], dtype=np.int64)
    ns_obligor = np.array([n.obligor_uid for n in req.netting_sets], dtype=np.int64)
    ns_collateral = np.array(
        [n.net_collateral for n in req.netting_sets], dtype=np.float64
    )
    order = np.argsort(ns_uid, kind="stable")
    master = CompiledNettingSetMaster(
        uid=np.ascontiguousarray(ns_uid[order]),
        obligor_uid=np.ascontiguousarray(ns_obligor[order]),
        net_collateral=np.ascontiguousarray(ns_collateral[order]),
    )

    # --- trade SoA ---
    trade_uids = np.array([t.uid for t in req.trades], dtype=np.int64)
    trade_ns = np.array([t.netting_set_id for t in req.trades], dtype=np.int64)
    mtm = np.ascontiguousarray(
        np.array([t.mtm for t in req.trades], dtype=np.float64).reshape(-1, 1)
    )  # (M, S=1) deterministic

    # --- compile the chosen view ---
    if entry.exposure_model_id == CURRENT_EXPOSURE_METHOD_ID:
        add_on = compile_add_on_factors(
            pd.DataFrame(
                {
                    "subclass_uix": [
                        a.subclass_uix for a in (req.add_on_factors or [])
                    ],
                    "factor": [a.factor for a in (req.add_on_factors or [])],
                }
            )
        )
        volume = np.array([t.volume for t in req.trades], dtype=np.float64)
        current_price = np.array(
            [t.current_price for t in req.trades], dtype=np.float64
        )
        subclass = np.array([t.asset_subclass_uix for t in req.trades], dtype=np.int32)
        effective_notional = resolve_quantity_price_effective_notional(
            volume=volume, current_price=current_price
        )
        supervisory_factor = resolve_supervisory_factors(subclass, add_on)
        cem_overrides = {}
        if req.pfe_floor is not None:
            cem_overrides["pfe_floor"] = req.pfe_floor
        if req.pfe_coefficient is not None:
            cem_overrides["pfe_coefficient"] = req.pfe_coefficient
        compiled = compile_exposure_cem(
            trade_uids=trade_uids,
            trade_netting_set_id=trade_ns,
            trade_effective_notional=effective_notional,
            trade_supervisory_factor=supervisory_factor,
            netting_set_master=master,
            **cem_overrides,
        )
    else:  # replacement cost
        compiled = compile_exposure_rc(
            trade_uids=trade_uids,
            trade_netting_set_id=trade_ns,
            netting_set_master=master,
        )

    # --- run + aggregate ---
    ns_paths = run_exposure(exposure=compiled, mtm=mtm)

    target = (
        np.ascontiguousarray(np.asarray(obligor_uids_target, dtype=np.int64))
        if obligor_uids_target is not None
        else np.unique(master.obligor_uid)
    )
    ead_by_obligor = aggregate_ead_to_obligor(
        ead_result=ns_paths, netting_set_master=master, obligor_uids_target=target
    )  # (N_obl, S=1)

    # --- assemble result (Contract B) ---
    obligor_rows = [
        ObligorExposure(uid=int(u), ead=float(e))
        for u, e in zip(target, ead_by_obligor[:, 0], strict=True)
    ]
    # netting-set detail: join each result netting set back to its obligor.
    ns_obligor_of = master.obligor_uid[np.searchsorted(master.uid, ns_paths.uid)]
    netting_set_rows = [
        NettingSetExposureRow(
            netting_set_uid=int(ns), obligor_uid=int(ob), ead=float(e)
        )
        for ns, ob, e in zip(
            ns_paths.uid, ns_obligor_of, ns_paths.exposure[:, 0], strict=True
        )
    ]

    return CreditExposureResult(
        obligor_ead=obligor_rows,
        netting_set_ead=netting_set_rows,
        provenance=ExposureProvenance(
            engine_version=_ENGINE_VERSION,
            model=req.model,
            n_trades=req.n_trades,
            n_netting_sets=len(req.netting_sets),
            n_obligors=int(target.size),
        ),
    )
