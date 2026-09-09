# tests/solutions/sensitivity_pnl/unit/test_calculator.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.calculator`.

`SensitivityPnlCalculator`'s central claim is that a cached book is validated
exactly as thoroughly on tick 5000 as on tick 1, and that the one thing
caching can newly get wrong -- pricing against a market the book was not
compiled for -- is caught by comparing axes by value. This file tests that
claim directly (repeated `.run()` calls, an axis-mismatch on a LATER call, not
just the first), plus `run_into_grid`'s extra row-set/scenario-count checks
and `_axis_mismatch`'s message-construction logic in isolation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.calculator import (
    SensitivityPnlCalculator,
    _axis_mismatch,
)
from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError
from riskquant.solutions.sensitivity_pnl.market import FactorAxis
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityBookSpec

ABSOLUTE = int(DisplacementConvention.ABSOLUTE)


def _calculator(*, axis_keys=(1, 2), position_uids=(1,)) -> SensitivityPnlCalculator:
    axis = FactorAxis.from_keys(list(axis_keys))
    sens = pl.DataFrame(
        {
            "position_id": [1],
            "factor_id": [axis_keys[0]],
            "first_order": [10.0],
            "second_order": [0.0],
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
            sensitivities=sens,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=list(position_uids),
        )
    )


class TestCompile:
    def test_compile_returns_a_calculator_with_the_declared_axis(self) -> None:
        calc = _calculator()
        assert calc.axis.num_factors == 2

    def test_compile_raises_for_a_malformed_spec_before_compiling(self) -> None:
        axis = FactorAxis.from_keys([1])
        bad_sens = pl.DataFrame(
            {
                "position_id": [1.5],  # float id
                "factor_id": [1],
                "first_order": [1.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError):
            SensitivityPnlCalculator.compile(
                SensitivityBookSpec(
                    sensitivities=bad_sens, factor_meta=meta, factor_axis=axis
                )
            )


class TestRun:
    def test_run_returns_a_result_with_the_correct_shape(self) -> None:
        calc = _calculator()
        market = calc.axis.shocks(np.zeros((2, 5), dtype=np.float32))
        result = calc.run(market)
        assert result.pnl.shape == (1, 5)

    def test_repeated_run_calls_each_validate_independently(self) -> None:
        """The central claim: a market with a bad value on the SECOND call
        must be rejected even though the FIRST call succeeded -- caching the
        book must not weaken validation on any later tick."""
        calc = _calculator()
        good_market = calc.axis.shocks(np.zeros((2, 3), dtype=np.float32))
        calc.run(good_market)  # tick 1: fine

        bad = np.zeros((2, 3), dtype=np.float32)
        bad[0, 0] = np.nan
        with pytest.raises(SensitivityPnlInputError, match="non-finite"):
            calc.axis.shocks(bad)  # tick 2: rejected at construction

    def test_axis_mismatch_is_rejected_on_a_later_run_not_just_the_first(
        self,
    ) -> None:
        calc = _calculator()
        calc.run(calc.axis.shocks(np.zeros((2, 3), dtype=np.float32)))  # tick 1 fine
        reordered = FactorAxis.from_keys([2, 1])
        with pytest.raises(SensitivityPnlInputError, match="different row order"):
            calc.run(reordered.shocks(np.zeros((2, 3), dtype=np.float32)))

    def test_each_run_returns_a_freshly_allocated_block(self) -> None:
        calc = _calculator()
        market = calc.axis.shocks(np.zeros((2, 2), dtype=np.float32))
        result1 = calc.run(market)
        result2 = calc.run(market)
        assert not np.shares_memory(result1.pnl, result2.pnl)

    def test_book_reused_defaults_to_true(self) -> None:
        calc = _calculator()
        market = calc.axis.shocks(np.zeros((2, 2), dtype=np.float32))
        result = calc.run(market)
        assert result.provenance.book_reused is True

    def test_book_reused_can_be_overridden_to_false(self) -> None:
        calc = _calculator()
        market = calc.axis.shocks(np.zeros((2, 2), dtype=np.float32))
        result = calc.run(market, book_reused=False)
        assert result.provenance.book_reused is False

    def test_a_rebuilt_axis_with_identical_keys_is_accepted(self) -> None:
        """matches() compares by value, so rebuilding the axis every tick from
        the same key column is not an error."""
        calc = _calculator()
        rebuilt = FactorAxis.from_keys([1, 2])
        result = calc.run(rebuilt.shocks(np.zeros((2, 2), dtype=np.float32)))
        assert result.pnl.shape == (1, 2)


class TestRunIntoGrid:
    def test_writes_into_the_registered_block(self) -> None:
        calc = _calculator()
        grid = allocate_pnl_grid(
            block_position_uids=[calc.position_uids], num_scenarios=3
        )
        market = calc.axis.shocks(np.zeros((2, 3), dtype=np.float32))
        calc.run_into_grid(market, grid=grid, block_index=0)
        assert grid.pnl.shape == (1, 3)

    def test_rejects_a_scenario_count_mismatch_against_the_grid(self) -> None:
        calc = _calculator()
        grid = allocate_pnl_grid(
            block_position_uids=[calc.position_uids], num_scenarios=5
        )
        market = calc.axis.shocks(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(SensitivityPnlInputError, match="scenario"):
            calc.run_into_grid(market, grid=grid, block_index=0)

    def test_rejects_a_block_registered_with_the_wrong_row_set(self) -> None:
        calc = _calculator(position_uids=(1,))
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1, 2], dtype=np.int64)], num_scenarios=3
        )
        market = calc.axis.shocks(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(SensitivityPnlInputError, match="position_uids"):
            calc.run_into_grid(market, grid=grid, block_index=0)

    def test_writes_into_the_correct_block_of_a_multi_block_grid(self) -> None:
        calc = _calculator()
        grid = allocate_pnl_grid(
            block_position_uids=[
                np.array([100, 200], dtype=np.int64),
                calc.position_uids,
            ],
            num_scenarios=2,
        )
        grid.block_view(0)[:] = -1.0
        market = calc.axis.shocks(np.ones((2, 2), dtype=np.float32))
        calc.run_into_grid(market, grid=grid, block_index=1)
        assert bool(np.all(grid.block_view(0) == -1.0))  # untouched
        assert not bool(np.any(np.isnan(grid.block_view(1))))

    def test_returns_provenance_not_a_result(self) -> None:
        from riskquant.solutions.sensitivity_pnl.schemas import SensitivityPnlProvenance

        calc = _calculator()
        grid = allocate_pnl_grid(
            block_position_uids=[calc.position_uids], num_scenarios=2
        )
        market = calc.axis.shocks(np.zeros((2, 2), dtype=np.float32))
        provenance = calc.run_into_grid(market, grid=grid, block_index=0)
        assert isinstance(provenance, SensitivityPnlProvenance)


class TestIntrospectionProperties:
    def test_position_uids(self) -> None:
        calc = _calculator(position_uids=(1, 5, 9))
        assert calc.position_uids.tolist() == [1, 5, 9]

    def test_factor_keys(self) -> None:
        calc = _calculator(axis_keys=(1, 2))
        assert calc.factor_keys.to_list() == [1, 2]

    def test_num_positions(self) -> None:
        calc = _calculator(position_uids=(1, 2, 3))
        assert calc.num_positions == 3

    def test_num_active_factors_reflects_only_touched_factors(self) -> None:
        """The book uses only axis_keys[0]; the second factor in the axis is
        never referenced by a sensitivity."""
        calc = _calculator(axis_keys=(1, 2))
        assert calc.num_active_factors == 1

    def test_num_first_order_entries(self) -> None:
        calc = _calculator()
        assert calc.num_first_order_entries == 1

    def test_num_cross_entries_is_zero_with_no_cross_sensitivities(self) -> None:
        calc = _calculator()
        assert calc.num_cross_entries == 0


class TestAxisMismatchMessage:
    def test_differing_dtype_is_named_explicitly(self) -> None:
        compiled = FactorAxis.from_keys([1, 2])
        supplied = FactorAxis.from_keys(["A", "B"])
        message = _axis_mismatch(compiled=compiled, supplied=supplied)
        assert "dtype" in message.lower() or "String" in message

    def test_differing_length_is_named_explicitly(self) -> None:
        compiled = FactorAxis.from_keys([1, 2])
        supplied = FactorAxis.from_keys([1, 2, 3])
        message = _axis_mismatch(compiled=compiled, supplied=supplied)
        assert "3 factors" in message
        assert "2" in message

    def test_reordered_axis_names_the_first_differing_row(self) -> None:
        compiled = FactorAxis.from_keys(["A", "B", "C"])
        supplied = FactorAxis.from_keys(["A", "C", "B"])
        message = _axis_mismatch(compiled=compiled, supplied=supplied)
        assert "row 1" in message
        assert "'C'" in message and "'B'" in message

    def test_reordered_axis_message_does_not_falsely_flag_matching_prefix(
        self,
    ) -> None:
        """The first N-1 rows agree; only the differing row should be named,
        not row 0."""
        compiled = FactorAxis.from_keys(["A", "B", "C", "D"])
        supplied = FactorAxis.from_keys(["A", "B", "D", "C"])
        message = _axis_mismatch(compiled=compiled, supplied=supplied)
        assert "row 2" in message

    def test_identical_axes_fall_through_to_the_generic_message(self) -> None:
        """This function is only ever called when matches() has already
        returned False, but calling it directly on identical axes must not
        crash -- it should reach the final generic fallback."""
        compiled = FactorAxis.from_keys([1, 2])
        supplied = FactorAxis.from_keys([1, 2])
        message = _axis_mismatch(compiled=compiled, supplied=supplied)
        assert "differs from the compiled one" in message
