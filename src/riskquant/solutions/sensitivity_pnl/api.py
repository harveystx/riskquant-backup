# src/riskquant/solutions/sensitivity_pnl/api.py
"""Public surface for sensitivity PnL (mirror of ``solutions.credit_exposure.api``)."""

from __future__ import annotations

#: Entry points          `SensitivityPnlCalculator` (compile once, price many),
#:                       `run_sensitivity_pnl`, `write_sensitivity_pnl`
#: Book / market / result `SensitivityBookSpec`, `FactorAxis`, `ScenarioShocks`,
#:                       `ScenarioLevels`, `MarketScenarios`,
#:                       `SensitivityPnlRequest`, `SensitivityPnlResult`,
#:                       `SensitivityPnlProvenance`
#: Grid handoff           `PnlGrid`, `allocate_pnl_grid` -- the shared buffer
#:                       `write_sensitivity_pnl` / `calculator.run_into_grid`
#:                       write into; needed by any caller using that pattern,
#:                       so it belongs at the seam, not only in core.
#: Policy enums          `DuplicatePolicy`, `DisplacementConvention`
#: Quote-base constants  `QUOTE_BASE_NATURAL`, `QUOTE_BASE_PER_PERCENT`,
#:                       `QUOTE_BASE_PER_BASIS_POINT`, `QUOTE_BASE_PER_VOL_POINT`
#: Errors                `SensitivityPnlInputError`, `InputDefect`
__all__ = [
    "QUOTE_BASE_NATURAL",
    "QUOTE_BASE_PER_BASIS_POINT",
    "QUOTE_BASE_PER_PERCENT",
    "QUOTE_BASE_PER_VOL_POINT",
    "DisplacementConvention",
    "DuplicatePolicy",
    "FactorAxis",
    "InputDefect",
    "MarketScenarios",
    "PnlGrid",
    "ScenarioLevels",
    "ScenarioShocks",
    "SensitivityBookSpec",
    "SensitivityPnlCalculator",
    "SensitivityPnlInputError",
    "SensitivityPnlProvenance",
    "SensitivityPnlRequest",
    "SensitivityPnlResult",
    "allocate_pnl_grid",
    "assess_characteristic_displacement",
    "assess_taylor_validity",
    "build_sensitivity_book_spec",
    "derive_taylor_declaration",
    "run_sensitivity_pnl",
    "write_sensitivity_pnl",
]

from riskquant.sensitivity_pnl.compile import DuplicatePolicy
from riskquant.sensitivity_pnl.conventions import (
    QUOTE_BASE_NATURAL,
    QUOTE_BASE_PER_BASIS_POINT,
    QUOTE_BASE_PER_PERCENT,
    QUOTE_BASE_PER_VOL_POINT,
    DisplacementConvention,
)
from riskquant.sensitivity_pnl.grid import PnlGrid, allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.book_adapter import (
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
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
from riskquant.solutions.sensitivity_pnl.taylor_validity import (
    assess_characteristic_displacement,
    assess_taylor_validity,
)
