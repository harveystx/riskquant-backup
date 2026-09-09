# tests/solutions/sensitivity_pnl/unit/test_taylor_validity_facade.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.taylor_validity` -- the
caller-key-aware wrapper around `riskquant.sensitivity_pnl.validity`.

Two things this facade exists for, each with its own test class:
`assess_taylor_validity` applying the SAME axis-binding and market-validation
guarantees `calculator.run` gives, and `assess_characteristic_displacement`
translating the core layer's compiled `active_factor_slot` back into the
caller's own factor keys -- correctly under BOTH the direct and slot
encodings, since the translation logic differs between them.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.solutions.sensitivity_pnl.calculator import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError
from riskquant.solutions.sensitivity_pnl.market import FactorAxis
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityBookSpec
from riskquant.solutions.sensitivity_pnl.taylor_validity import (
    CharacteristicDisplacementReport,
    TaylorValidityReport,
    assess_characteristic_displacement,
    assess_taylor_validity,
)

ABSOLUTE = int(DisplacementConvention.ABSOLUTE)


def _calculator(axis_keys) -> SensitivityPnlCalculator:
    axis = FactorAxis.from_keys(list(axis_keys))
    sens = pl.DataFrame(
        {
            "position_id": [1, 1],
            "factor_id": [axis_keys[0], axis_keys[1]],
            "first_order": [10.0, 0.0],
            "second_order": [0.0, 20.0],  # pure gamma on the second factor
        }
    )
    meta = pl.DataFrame(
        {
            "factor_id": list(axis_keys),
            "displacement_convention_id": [ABSOLUTE] * len(axis_keys),
            "quote_base": [1.0] * len(axis_keys),
        }
    )
    return SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sens, factor_meta=meta, factor_axis=axis, position_uids=[1]
        )
    )


class TestAssessTaylorValidity:
    def test_returns_a_correctly_shaped_grid(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 4), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        assert isinstance(report, TaylorValidityReport)
        assert report.grid.num_positions == 1
        assert report.grid.num_scenarios == 4

    def test_position_uids_property_matches_the_grid(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 2), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        assert report.position_uids.tolist() == [1]

    def test_pure_gamma_factor_gives_nonzero_nonlinearity(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 1), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        assert report.grid.quadratic_term[0, 0] > 0.0

    def test_rejects_a_mismatched_axis(self) -> None:
        calc = _calculator([1, 2])
        wrong_axis = FactorAxis.from_keys([2, 1])
        market = wrong_axis.shocks(np.ones((2, 1), dtype=np.float32))
        with pytest.raises(SensitivityPnlInputError, match="does not match"):
            assess_taylor_validity(calculator=calc, market=market)

    def test_a_rebuilt_axis_with_the_same_keys_is_accepted(self) -> None:
        calc = _calculator([1, 2])
        rebuilt = FactorAxis.from_keys([1, 2])
        market = rebuilt.shocks(np.ones((2, 1), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        assert report.grid.num_positions == 1

    def test_log_convention_non_positive_shocked_level_is_rejected(self) -> None:
        """Same P0 guard `calculator.run` relies on must protect this path."""
        axis = FactorAxis.from_keys([1])
        sens = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [1],
                "first_order": [10.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [int(DisplacementConvention.LOG)],
                "quote_base": [1.0],
            }
        )
        calc = SensitivityPnlCalculator.compile(
            SensitivityBookSpec(
                sensitivities=sens,
                factor_meta=meta,
                factor_axis=axis,
                position_uids=[1],
            )
        )
        market = axis.levels(
            base_levels=np.array([2.0]),
            shocked_levels=np.ascontiguousarray(np.array([[-1.0]])),
        )
        with pytest.raises(SensitivityPnlInputError, match="non-positive"):
            assess_taylor_validity(calculator=calc, market=market)

    def test_custom_threshold_is_passed_through_to_the_summary(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 1), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market, threshold=0.9)
        assert report.summary.threshold == 0.9

    def test_provenance_is_populated(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 3), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        assert report.provenance.num_positions == 1
        assert report.provenance.num_scenarios == 3


class TestNonlinearityFractionFor:
    def test_returns_the_correct_row(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 3), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        row = report.nonlinearity_fraction_for(1)
        assert row.shape == (3,)

    def test_matches_the_grids_own_computation(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 3), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        row = report.nonlinearity_fraction_for(1)
        assert np.array_equal(row, report.grid.nonlinearity_fraction()[0])

    def test_unknown_uid_raises_key_error(self) -> None:
        calc = _calculator([1, 2])
        market = calc.axis.shocks(np.ones((2, 1), dtype=np.float32))
        report = assess_taylor_validity(calculator=calc, market=market)
        with pytest.raises(KeyError, match="999"):
            report.nonlinearity_fraction_for(999)


class TestAssessCharacteristicDisplacement:
    def test_returns_a_report_with_matching_lengths(self) -> None:
        calc = _calculator([1, 2])
        report = assess_characteristic_displacement(calc)
        assert isinstance(report, CharacteristicDisplacementReport)
        assert (
            report.position_uids.shape[0]
            == report.x_star.shape[0]
            == len(report.factor_keys)
        )

    def test_translates_direct_encoded_keys_correctly(self) -> None:
        """axis_keys=[1, 2] are sorted integers -> DIRECT encoding."""
        calc = _calculator([1, 2])
        report = assess_characteristic_displacement(calc)
        assert set(report.factor_keys) <= {1, 2}
        assert all(isinstance(k, int) for k in report.factor_keys)

    def test_translates_slot_encoded_string_keys_correctly(self) -> None:
        calc = _calculator(["EQ_B", "EQ_A"])  # unsorted strings -> SLOT
        report = assess_characteristic_displacement(calc)
        assert set(report.factor_keys) <= {"EQ_A", "EQ_B"}
        assert all(isinstance(k, str) for k in report.factor_keys)

    def test_pure_delta_entry_has_infinite_x_star(self) -> None:
        calc = _calculator([1, 2])  # factor 1: delta only, factor 2: gamma only
        report = assess_characteristic_displacement(calc)
        by_key = dict(zip(report.factor_keys, report.x_star.tolist(), strict=True))
        assert np.isinf(by_key[1])

    def test_pure_gamma_entry_with_zero_delta_has_x_star_zero(self) -> None:
        calc = _calculator([1, 2])
        report = assess_characteristic_displacement(calc)
        by_key = dict(zip(report.factor_keys, report.x_star.tolist(), strict=True))
        assert by_key[2] == 0.0

    def test_needs_no_market_argument_at_all(self) -> None:
        """Scenario-free by construction -- calling it costs nothing beyond
        the compiled book already sitting in the calculator."""
        import inspect

        params = inspect.signature(assess_characteristic_displacement).parameters
        assert list(params) == ["calculator"]


class TestSortedByBreakdown:
    def test_orders_ascending_by_x_star(self) -> None:
        calc = _calculator([1, 2])
        report = assess_characteristic_displacement(calc).sorted_by_breakdown()
        assert report.x_star.tolist() == sorted(report.x_star.tolist())

    def test_infinite_x_star_sorts_last(self) -> None:
        calc = _calculator([1, 2])
        report = assess_characteristic_displacement(calc).sorted_by_breakdown()
        assert np.isinf(report.x_star[-1])

    def test_position_uids_and_factor_keys_stay_aligned_with_x_star_after_sort(
        self,
    ) -> None:
        calc = _calculator([1, 2])
        original = assess_characteristic_displacement(calc)
        sorted_report = original.sorted_by_breakdown()
        # rebuild the (key, x_star) pairing from the original and confirm the
        # sorted report reproduces the exact same pairing, just reordered.
        original_pairs = set(
            zip(original.factor_keys, original.x_star.tolist(), strict=True)
        )
        sorted_pairs = set(
            zip(sorted_report.factor_keys, sorted_report.x_star.tolist(), strict=True)
        )
        assert original_pairs == sorted_pairs

    def test_does_not_mutate_the_original_report(self) -> None:
        calc = _calculator([1, 2])
        original = assess_characteristic_displacement(calc)
        original_order = list(original.factor_keys)
        original.sorted_by_breakdown()
        assert list(original.factor_keys) == original_order
