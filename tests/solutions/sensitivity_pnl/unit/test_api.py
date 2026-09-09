# tests/solutions/sensitivity_pnl/unit/test_api.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.api`, the public entry
point named in the module's own docstring as "the entry point in riskquant to
using the underlying riskquant.sensitivity_pnl package."

The failure this file exists to catch is the exact one found and fixed
earlier in this package's history: `api.py` importing a name from a module
that does not exist on disk, which raises `ModuleNotFoundError` the instant
anything does `from riskquant.solutions.sensitivity_pnl.api import
anything` -- a failure invisible to every OTHER test in this suite, because
none of them import through `api` specifically; they import each concrete
module directly. This file is the one place that imports `api` itself and
checks its `__all__` against what actually landed in the module namespace.
"""

from __future__ import annotations

import importlib


class TestPublicSurface:
    def test_the_module_itself_imports_without_error(self) -> None:
        """The regression this file exists for: api.py must not reference a
        name from a module that is missing on disk."""
        importlib.import_module("riskquant.solutions.sensitivity_pnl.api")

    def test_every_name_in_all_is_actually_importable_from_the_module(self) -> None:
        api = importlib.import_module("riskquant.solutions.sensitivity_pnl.api")
        missing = [name for name in api.__all__ if not hasattr(api, name)]
        assert missing == []

    def test_all_has_no_duplicate_entries(self) -> None:
        api = importlib.import_module("riskquant.solutions.sensitivity_pnl.api")
        assert len(api.__all__) == len(set(api.__all__))

    def test_expected_entry_points_are_present(self) -> None:
        from riskquant.solutions.sensitivity_pnl import api

        for name in (
            "run_sensitivity_pnl",
            "write_sensitivity_pnl",
            "SensitivityPnlCalculator",
        ):
            assert name in api.__all__

    def test_expected_book_and_market_types_are_present(self) -> None:
        from riskquant.solutions.sensitivity_pnl import api

        for name in (
            "SensitivityBookSpec",
            "FactorAxis",
            "ScenarioShocks",
            "ScenarioLevels",
            "MarketScenarios",
            "SensitivityPnlRequest",
            "SensitivityPnlResult",
            "SensitivityPnlProvenance",
        ):
            assert name in api.__all__

    def test_expected_policy_enums_are_present(self) -> None:
        from riskquant.solutions.sensitivity_pnl import api

        assert "DuplicatePolicy" in api.__all__
        assert "DisplacementConvention" in api.__all__

    def test_expected_error_types_are_present(self) -> None:
        from riskquant.solutions.sensitivity_pnl import api

        assert "SensitivityPnlInputError" in api.__all__
        assert "InputDefect" in api.__all__

    def test_expected_taylor_validity_entry_points_are_present(self) -> None:
        from riskquant.solutions.sensitivity_pnl import api

        assert "assess_taylor_validity" in api.__all__
        assert "assess_characteristic_displacement" in api.__all__

    def test_error_types_are_reexported_from_the_same_module_as_defined(
        self,
    ) -> None:
        from riskquant.solutions.sensitivity_pnl import api, errors

        assert api.SensitivityPnlInputError is errors.SensitivityPnlInputError
        assert api.InputDefect is errors.InputDefect

    def test_calculator_is_reexported_from_the_same_class_as_defined(self) -> None:
        from riskquant.solutions.sensitivity_pnl import api, calculator

        assert api.SensitivityPnlCalculator is calculator.SensitivityPnlCalculator

    def test_duplicate_policy_is_the_same_enum_as_the_core_package(self) -> None:
        """A caller checking `isinstance(x, api.DuplicatePolicy)` against a
        value from riskquant.sensitivity_pnl.compile must get True -- a
        re-imported or re-defined enum would silently break that."""
        from riskquant.sensitivity_pnl.compile import DuplicatePolicy as CoreEnum
        from riskquant.solutions.sensitivity_pnl.api import DuplicatePolicy as ApiEnum

        assert ApiEnum is CoreEnum

    def test_displacement_convention_is_the_same_enum_as_the_core_package(
        self,
    ) -> None:
        from riskquant.sensitivity_pnl.conventions import (
            DisplacementConvention as CoreEnum,
        )
        from riskquant.solutions.sensitivity_pnl.api import (
            DisplacementConvention as ApiEnum,
        )

        assert ApiEnum is CoreEnum
