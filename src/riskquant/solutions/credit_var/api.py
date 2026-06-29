# src/riskquant/solutions/credit_var/api.py
"""
Public API for the credit-VaR solution.

This module is THE entry surface for the solution: every symbol a consumer
(notebook, Emissary, the xlwings tool) is meant to use is re-exported here, and
nothing else in `riskquant.solutions.credit_var.*` should be imported from
outside the package. The other modules (`run`, `schemas`, `catalog`, `recipes`)
are supportive internals — they may change shape freely as long as this surface
holds.

Why a module and not the package `__init__`: in `src/riskquant`, every
`__init__.py` is empty, without exception. A solution that wants a single import
line gets it from a named `api` module instead, so the "one stable surface" need
is met without weakening the empty-init rule.

Import as::

    from riskquant.solutions.credit_var.api import run_credit_var, CreditVarRequest
"""

from __future__ import annotations

__all__ = [
    # entry point
    "run_credit_var",
    # request / result contract
    "CreditVarRequest",
    "CreditVarResult",
    "ObligorRow",
    "LoadingRow",
    "GroupRow",
    "RiskSummary",
    "ModelTail",
    "RunProvenance",
    "ObligorContribution",
    "LossHistogram",
    "StressTestResult",
    "StressTestObligor",
    # model catalog (the front-end menu + param specs)
    "MODEL_CATALOG",
    "ModelEntry",
    "ParamSpec",
    "model_entry",
    "supported_model_keys",
]

from riskquant.solutions.credit_var.catalog import (
    MODEL_CATALOG,
    ModelEntry,
    ParamSpec,
    model_entry,
    supported_model_keys,
)
from riskquant.solutions.credit_var.run import run_credit_var
from riskquant.solutions.credit_var.schemas import (
    CreditVarRequest,
    CreditVarResult,
    GroupRow,
    LoadingRow,
    LossHistogram,
    ModelTail,
    ObligorContribution,
    ObligorRow,
    RiskSummary,
    RunProvenance,
    StressTestObligor,
    StressTestResult,
)
