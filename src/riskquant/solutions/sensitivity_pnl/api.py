# src/riskquant/solutions/sensitivity_pnl/api.py
"""Public surface for sensitivity PnL (mirror of ``solutions.credit_exposure.api``)."""

from __future__ import annotations

#: Entry points          `SensitivityPnlCalculator` (compile once, price many),
#:                       `run_sensitivity_pnl`, `write_sensitivity_pnl`
#: Book / market / result `SensitivityBookSpec`, `FactorAxis`, `ScenarioShocks`,
#:                       `ScenarioLevels`, `MarketScenarios`,
#:                       `SensitivityPnlRequest`, `SensitivityPnlResult`,
#:                       `SensitivityPnlProvenance`
#: Policy enums          `DuplicatePolicy`, `DisplacementConvention`
#: Errors                `SensitivityPnlInputError`, `InputDefect`
__all__ = [
    "DisplacementConvention",
    "DuplicatePolicy",
    "FactorAxis",
    "InputDefect",
    "MarketScenarios",
    "ScenarioLevels",
    "ScenarioShocks",
    "SensitivityBookSpec",
    "SensitivityPnlCalculator",
    "SensitivityPnlInputError",
    "SensitivityPnlProvenance",
    "SensitivityPnlRequest",
    "SensitivityPnlResult",
    "run_sensitivity_pnl",
    "write_sensitivity_pnl",
]

from riskquant.sensitivity_pnl.compile import DuplicatePolicy
from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.solutions.sensitivity_pnl.calculator import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.errors import (
    InputDefect,
    SensitivityPnlInputError,
)
from riskquant.solutions.sensitivity_pnl.handoff import write_sensitivity_pnl
from riskquant.solutions.sensitivity_pnl.market import (
    FactorAxis,
    MarketScenarios,
    ScenarioLevels,
    ScenarioShocks,
)
from riskquant.solutions.sensitivity_pnl.run import run_sensitivity_pnl
from riskquant.solutions.sensitivity_pnl.schemas import (
    SensitivityBookSpec,
    SensitivityPnlProvenance,
    SensitivityPnlRequest,
    SensitivityPnlResult,
)
