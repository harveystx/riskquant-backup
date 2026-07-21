# src/riskquant/solutions/credit_exposure/api.py
"""Public surface for credit-exposure (mirror of ``solutions.credit_var.api``)."""

from __future__ import annotations

__all__ = [
    # entry point
    "run_credit_exposure",
    # request / result contract
    "CreditExposureRequest",
    "CreditExposureResult",
    "TradeRow",
    "NettingSetRow",
    "AddOnFactorRow",
    "SubclassHedgingSetRow",
    "HedgingSetRhoRow",
    "ObligorExposure",
    "NettingSetExposureRow",
    "ExposureProvenance",
    # exposure-model catalog
    "EXPOSURE_MODEL_CATALOG",
    "ExposureModelEntry",
    "exposure_model_entry",
    "supported_exposure_models",
    # credit-VaR handoff
    "attach_exposure_ead",
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
