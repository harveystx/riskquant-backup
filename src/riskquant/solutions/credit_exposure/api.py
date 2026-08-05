# src/riskquant/solutions/credit_exposure/api.py
"""Public surface for credit-exposure (mirror of ``solutions.credit_var.api``)."""

from __future__ import annotations

__all__ = [
    "EXPOSURE_MODEL_CATALOG",
    "AddOnFactorRow",
    "CreditExposureRequest",
    "CreditExposureResult",
    "ExposureModelEntry",
    "ExposureProvenance",
    "HedgingSetRhoRow",
    "NettingSetExposureRow",
    "NettingSetRow",
    "ObligorExposure",
    "SubclassHedgingSetRow",
    "TradeRow",
    "attach_exposure_ead",
    "exposure_model_entry",
    "run_credit_exposure",
    "supported_exposure_models",
]

from riskquant.solutions.credit_exposure.catalog import (
    EXPOSURE_MODEL_CATALOG,
    ExposureModelEntry,
    exposure_model_entry,
    supported_exposure_models,
)
from riskquant.solutions.credit_exposure.handoff import attach_exposure_ead
from riskquant.solutions.credit_exposure.run import run_credit_exposure
from riskquant.solutions.credit_exposure.schemas import (
    AddOnFactorRow,
    CreditExposureRequest,
    CreditExposureResult,
    ExposureProvenance,
    HedgingSetRhoRow,
    NettingSetExposureRow,
    NettingSetRow,
    ObligorExposure,
    SubclassHedgingSetRow,
    TradeRow,
)
