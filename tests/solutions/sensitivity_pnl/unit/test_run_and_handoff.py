# tests/solutions/sensitivity_pnl/unit/test_run_and_handoff.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.run` and `.handoff` --
the one-shot entry points. Both are thin compositions of
`SensitivityPnlCalculator`, so the tests here focus on what is specific to
each wrapper rather than re-proving the calculator's own behaviour: `run`'s
`book_reused=False` stamp, and `handoff`'s extra `position_uids`-required and
row-set/scenario-count checks that only make sense for a caller-owned grid.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.sensitivity_pnl.grid import allocate_pnl_grid
from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError
from riskquant.solutions.sensitivity_pnl.handoff import write_sensitivity_pnl
from riskquant.solutions.sensitivity_pnl.market import FactorAxis
from riskquant.solutions.sensitivity_pnl.run import run_sensitivity_pnl
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityPnlRequest

ABSOLUTE = int(DisplacementConvention.ABSOLUTE)


def _request(*, position_uids=None) -> SensitivityPnlRequest:
    axis = FactorAxis.from_keys([1])
    return SensitivityPnlRequest(
        sensitivities=pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [1],
                "first_order": [10.0],
                "second_order": [0.0],
            }
        ),
        factor_meta=pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        ),
        market=axis.shocks(np.array([[1.0, 2.0, 3.0]], dtype=np.float32)),
        position_uids=position_uids,
    )


class TestRunSensitivityPnl:
    def test_returns_a_correctly_shaped_result(self) -> None:
        result = run_sensitivity_pnl(_request(position_uids=[1]))
        assert result.pnl.shape == (1, 3)

    def test_computes_the_correct_numbers(self) -> None:
        result = run_sensitivity_pnl(_request(position_uids=[1]))
        assert result.pnl[0].tolist() == pytest.approx([10.0, 20.0, 30.0])

    def test_provenance_reports_book_reused_false(self) -> None:
        """A one-shot request compiles the book exactly once and never
        reuses it -- book_reused must be False, not the calculator's own
        default of True."""
        result = run_sensitivity_pnl(_request(position_uids=[1]))
        assert result.provenance.book_reused is False

    def test_raises_sensitivity_pnl_input_error_for_a_malformed_request(
        self,
    ) -> None:
        request = _request(position_uids=[1])
        bad_sens = pl.DataFrame(
            {
                "position_id": [1.5],
                "factor_id": [1],
                "first_order": [1.0],
                "second_order": [0.0],
            }
        )
        from dataclasses import replace

        request = replace(request, sensitivities=bad_sens)
        with pytest.raises(SensitivityPnlInputError):
            run_sensitivity_pnl(request)

    def test_works_without_declared_position_uids_when_entries_are_present(
        self,
    ) -> None:
        result = run_sensitivity_pnl(_request(position_uids=None))
        assert result.pnl.shape == (1, 3)

    def test_result_row_order_matches_position_uids_order(self) -> None:
        axis = FactorAxis.from_keys([1])
        request = SensitivityPnlRequest(
            sensitivities=pl.DataFrame(
                {
                    "position_id": [5, 2],
                    "factor_id": [1, 1],
                    "first_order": [1.0, 2.0],
                    "second_order": [0.0, 0.0],
                }
            ),
            factor_meta=pl.DataFrame(
                {
                    "factor_id": [1],
                    "displacement_convention_id": [ABSOLUTE],
                    "quote_base": [1.0],
                }
            ),
            market=axis.shocks(np.array([[10.0]], dtype=np.float32)),
        )
        result = run_sensitivity_pnl(request)
        assert result.position_uids.tolist() == [2, 5]  # ascending


class TestWriteSensitivityPnl:
    def test_requires_position_uids(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1], dtype=np.int64)], num_scenarios=3
        )
        with pytest.raises(SensitivityPnlInputError, match="position_uids"):
            write_sensitivity_pnl(
                _request(position_uids=None), grid=grid, block_index=0
            )

    def test_writes_the_correct_numbers_into_the_registered_block(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1], dtype=np.int64)], num_scenarios=3
        )
        write_sensitivity_pnl(_request(position_uids=[1]), grid=grid, block_index=0)
        assert grid.pnl[0].tolist() == pytest.approx([10.0, 20.0, 30.0])

    def test_writes_into_the_correct_block_of_a_multi_block_grid(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[
                np.array([100], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ],
            num_scenarios=3,
        )
        grid.block_view(0)[:] = -1.0
        write_sensitivity_pnl(_request(position_uids=[1]), grid=grid, block_index=1)
        assert bool(np.all(grid.block_view(0) == -1.0))
        assert grid.block_view(1)[0].tolist() == pytest.approx([10.0, 20.0, 30.0])

    def test_rejects_a_row_set_mismatch_with_the_registered_block(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1, 2], dtype=np.int64)], num_scenarios=3
        )
        with pytest.raises(SensitivityPnlInputError, match="position_uids"):
            write_sensitivity_pnl(_request(position_uids=[1]), grid=grid, block_index=0)

    def test_rejects_a_scenario_count_mismatch_with_the_grid(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1], dtype=np.int64)], num_scenarios=99
        )
        with pytest.raises(SensitivityPnlInputError, match="scenario"):
            write_sensitivity_pnl(_request(position_uids=[1]), grid=grid, block_index=0)

    def test_returns_provenance_with_book_reused_false(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1], dtype=np.int64)], num_scenarios=3
        )
        provenance = write_sensitivity_pnl(
            _request(position_uids=[1]), grid=grid, block_index=0
        )
        assert provenance.book_reused is False

    def test_returns_provenance_not_a_pnl_matrix(self) -> None:
        from riskquant.solutions.sensitivity_pnl.schemas import SensitivityPnlProvenance

        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1], dtype=np.int64)], num_scenarios=3
        )
        result = write_sensitivity_pnl(
            _request(position_uids=[1]), grid=grid, block_index=0
        )
        assert isinstance(result, SensitivityPnlProvenance)
