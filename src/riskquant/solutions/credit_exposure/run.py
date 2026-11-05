# src/riskquant/solutions/credit_exposure/run.py
"""
Credit-exposure entry point: trade table in, per-obligor EAD out.

``run_credit_exposure`` is a thin orchestrator, mirroring
``solutions.credit_var.run``: it converts the validated request into flat tables,
hands them to the model recipe (`recipes.assemble_exposure_inputs`) for all
model-specific assembly, runs the engine, aggregates netting-set EAD up to the
obligor, and shapes the typed `CreditExposureResult` (Contract B). Deterministic:
``mtm`` is one current mark per trade, so the exposure is a single "scenario".
"""

from __future__ import annotations

__all__ = [
    "run_credit_exposure",
]

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import polars as pl
from pydantic import BaseModel

from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.jtd.bridge import aggregate_ead_to_obligor
from riskquant.solutions.credit_exposure.recipes import assemble_exposure_inputs
from riskquant.solutions.credit_exposure.schemas import (
    CreditExposureRequest,
    CreditExposureResult,
    ExposureProvenance,
    NettingSetExposureRow,
    ObligorExposure,
)

_ENGINE_VERSION = "riskquant-credit-exposure/0.1.0"


def _coerce(request: CreditExposureRequest | dict[str, Any]) -> CreditExposureRequest:
    if isinstance(request, CreditExposureRequest):
        return request
    return CreditExposureRequest.model_validate(request)


# --------------------------------------------------------------------------
# Request -> flat tables (the recipes' authoring contract)
# --------------------------------------------------------------------------


# Explicit schemas (F-23): a frame built from row dicts infers its dtypes from
# the first 100 rows, so 100 trades with ``volume=None`` then a float raise
# ComputeError, and 100 int marks then 1.5 truncate to 1. Each frame is built
# column by column with the dtype of its row-model field.
_TRADES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "trade_uid": pl.Int64(),
    "netting_set_uid": pl.Int64(),
    "mtm": pl.Float64(),
    "volume": pl.Float64(),
    "current_price": pl.Float64(),
    "asset_subclass_uix": pl.Int64(),
    "direction": pl.Utf8(),
    "remaining_maturity_years": pl.Float64(),
}
_NETTING_SETS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "netting_set_uid": pl.Int64(),
    "obligor_uid": pl.Int64(),
    "net_collateral": pl.Float64(),
}
_ADD_ON_SCHEMA: Final[dict[str, pl.DataType]] = {
    "subclass_uix": pl.Int64(),
    "factor": pl.Float64(),
}
_HEDGING_SETS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "subclass_uix": pl.Int64(),
    "hedging_set_uix": pl.Int64(),
}
_CORRELATIONS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "hedging_set_uix": pl.Int64(),
    "rho": pl.Float64(),
}


def _frame(
    rows: Sequence[BaseModel], schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    """Rows of one request table as a frame with the stated ``schema``."""
    return pl.DataFrame(
        {name: [getattr(row, name) for row in rows] for name in schema},
        schema=dict(schema),
    )


def _trades_frame(req: CreditExposureRequest) -> pl.DataFrame:
    return _frame(req.trades, _TRADES_SCHEMA)


def _netting_sets_frame(req: CreditExposureRequest) -> pl.DataFrame:
    return _frame(req.netting_sets, _NETTING_SETS_SCHEMA)


def _add_on_frame(req: CreditExposureRequest) -> pl.DataFrame | None:
    if not req.add_on_factors:
        return None
    return _frame(req.add_on_factors, _ADD_ON_SCHEMA)


def _hedging_sets_frame(req: CreditExposureRequest) -> pl.DataFrame | None:
    if not req.subclass_hedging_sets:
        return None
    return _frame(req.subclass_hedging_sets, _HEDGING_SETS_SCHEMA)


def _correlations_frame(req: CreditExposureRequest) -> pl.DataFrame | None:
    if not req.hedging_set_correlations:
        return None
    return _frame(req.hedging_set_correlations, _CORRELATIONS_SCHEMA)


def _params(req: CreditExposureRequest) -> dict[str, float]:
    """The scalar knobs, present only when the author set them (engine defaults
    otherwise)."""
    out: dict[str, float] = {}
    for key in (
        "pfe_floor",
        "pfe_coefficient",
        "saccr_alpha",
        "saccr_multiplier_floor",
    ):
        value = getattr(req, key)
        if value is not None:
            out[key] = float(value)
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_credit_exposure(
    request: CreditExposureRequest | dict[str, Any],
    *,
    obligor_uids_target: np.ndarray | None = None,
) -> CreditExposureResult:
    """Run one credit-exposure request end to end.

    ``obligor_uids_target`` fixes the obligor row order (e.g. a JTD portfolio's
    ``obligor_uids``); it defaults to the ascending distinct obligors in the book.
    """
    req = _coerce(request)

    # --- model-specific assembly (the recipe layer) ---
    inputs = assemble_exposure_inputs(
        model_key=req.model,
        trades=_trades_frame(req),
        netting_sets=_netting_sets_frame(req),
        params=_params(req),
        add_on_factors=_add_on_frame(req),
        subclass_hedging_sets=_hedging_sets_frame(req),
        hedging_set_correlations=_correlations_frame(req),
    )
    master = inputs.netting_set_master

    # --- run + aggregate ---
    mtm = np.ascontiguousarray(
        np.array([t.mtm for t in req.trades], dtype=np.float64).reshape(-1, 1)
    )  # (M, S=1) deterministic
    ns_paths = run_exposure(exposure=inputs.compiled, mtm=mtm)

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
        ObligorExposure(obligor_uid=int(u), ead=float(e))
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
