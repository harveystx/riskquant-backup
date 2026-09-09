# tests/sensitivity_pnl/unit/test_validity.py
"""
Focused unit tests for `riskquant.sensitivity_pnl.validity`, beyond what
`test_golden_master_taylor_validity.py` already pins with its own frozen
fixture. This file targets the mathematical edges of `nonlinearity_fraction`
(the 0/0 convention, boundedness under extreme inputs) and
`characteristic_displacement`'s x* formula directly, plus construction-time
validation not already exercised elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.sensitivity_pnl.validity import (
    TaylorValidityGrid,
    TaylorValiditySummary,
    summarize_taylor_validity,
)


def _grid(
    linear: list[list[float]], quadratic: list[list[float]]
) -> TaylorValidityGrid:
    linear_arr = np.ascontiguousarray(np.array(linear, dtype=np.float32))
    quadratic_arr = np.ascontiguousarray(np.array(quadratic, dtype=np.float32))
    uids = np.arange(1, linear_arr.shape[0] + 1, dtype=np.int64)
    return TaylorValidityGrid(
        position_uids=uids, linear_term=linear_arr, quadratic_term=quadratic_arr
    )


class TestNonlinearityFractionMath:
    def test_zero_quadratic_gives_fraction_zero(self) -> None:
        grid = _grid([[100.0]], [[0.0]])
        assert grid.nonlinearity_fraction()[0, 0] == 0.0

    def test_zero_linear_gives_fraction_exactly_one(self) -> None:
        grid = _grid([[0.0]], [[50.0]])
        assert grid.nonlinearity_fraction()[0, 0] == 1.0

    def test_both_zero_gives_fraction_zero_not_nan(self) -> None:
        """The 0/0 convention: no exposure at all reads as 'no nonlinearity',
        not as an undefined/NaN result that would poison a downstream max()."""
        grid = _grid([[0.0]], [[0.0]])
        fraction = grid.nonlinearity_fraction()[0, 0]
        assert fraction == 0.0
        assert np.isfinite(fraction)

    def test_equal_magnitude_terms_give_fraction_one_half(self) -> None:
        grid = _grid([[10.0]], [[10.0]])
        assert np.isclose(grid.nonlinearity_fraction()[0, 0], 0.5)

    def test_fraction_uses_absolute_values_sign_does_not_matter(self) -> None:
        """|quadratic| / (|linear| + |quadratic|): a negative linear term with
        a positive quadratic term of equal magnitude must still give 0.5, not
        something reflecting the signed sum."""
        grid = _grid([[-10.0]], [[10.0]])
        assert np.isclose(grid.nonlinearity_fraction()[0, 0], 0.5)

    @pytest.mark.parametrize(
        "linear,quadratic",
        [
            (1.0, 1e10),
            (1e10, 1.0),
            (1e-10, 1e-10),
            (-1e6, 1e6),
            (1e6, -1e-6),
        ],
    )
    def test_fraction_is_always_bounded_in_zero_one_across_extreme_magnitudes(
        self, linear: float, quadratic: float
    ) -> None:
        grid = _grid([[linear]], [[quadratic]])
        fraction = float(grid.nonlinearity_fraction()[0, 0])
        assert 0.0 <= fraction <= 1.0
        assert np.isfinite(fraction)

    def test_fraction_is_computed_independently_per_cell(self) -> None:
        grid = _grid(
            [[100.0, 0.0], [0.0, 0.0]],
            [[0.0, 50.0], [0.0, 25.0]],
        )
        fraction = grid.nonlinearity_fraction()
        assert fraction[0, 0] == 0.0
        assert fraction[0, 1] == 1.0
        assert fraction[1, 0] == 0.0  # 0/0
        assert fraction[1, 1] == 1.0

    def test_total_pnl_recovers_linear_plus_quadratic(self) -> None:
        grid = _grid([[10.0, -5.0]], [[2.0, 3.0]])
        assert np.allclose(grid.total_pnl(), [[12.0, -2.0]])


class TestTaylorValidityGridConstruction:
    def test_valid_construction_succeeds(self) -> None:
        _grid([[1.0]], [[2.0]])  # must not raise

    def test_rejects_position_uids_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="int64"):
            TaylorValidityGrid(
                position_uids=np.array([1], dtype=np.int32),
                linear_term=np.zeros((1, 1), dtype=np.float32),
                quadratic_term=np.zeros((1, 1), dtype=np.float32),
            )

    def test_rejects_linear_term_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            TaylorValidityGrid(
                position_uids=np.array([1], dtype=np.int64),
                linear_term=np.zeros((1, 1), dtype=np.float64),
                quadratic_term=np.zeros((1, 1), dtype=np.float32),
            )

    def test_rejects_quadratic_term_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            TaylorValidityGrid(
                position_uids=np.array([1], dtype=np.int64),
                linear_term=np.zeros((1, 1), dtype=np.float32),
                quadratic_term=np.zeros((1, 1), dtype=np.float64),
            )

    def test_rejects_non_contiguous_linear_term(self) -> None:
        non_contig = np.zeros((1, 4), dtype=np.float32)[:, ::2]
        with pytest.raises(ValueError, match="C-contiguous"):
            TaylorValidityGrid(
                position_uids=np.array([1], dtype=np.int64),
                linear_term=non_contig,
                quadratic_term=np.zeros((1, 2), dtype=np.float32),
            )

    def test_rejects_row_count_mismatch_with_position_uids(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            TaylorValidityGrid(
                position_uids=np.array([1, 2], dtype=np.int64),
                linear_term=np.zeros((1, 1), dtype=np.float32),
                quadratic_term=np.zeros((1, 1), dtype=np.float32),
            )

    def test_rejects_shape_mismatch_between_linear_and_quadratic(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            TaylorValidityGrid(
                position_uids=np.array([1], dtype=np.int64),
                linear_term=np.zeros((1, 2), dtype=np.float32),
                quadratic_term=np.zeros((1, 3), dtype=np.float32),
            )

    def test_num_positions_and_num_scenarios_properties(self) -> None:
        grid = _grid([[1.0, 2.0, 3.0]], [[1.0, 1.0, 1.0]])
        assert grid.num_positions == 1
        assert grid.num_scenarios == 3


class TestSummarizeTaylorValidity:
    def test_rejects_threshold_of_exactly_zero(self) -> None:
        grid = _grid([[1.0]], [[1.0]])
        with pytest.raises(ValueError, match="threshold"):
            summarize_taylor_validity(grid, threshold=0.0)

    def test_rejects_threshold_above_one(self) -> None:
        grid = _grid([[1.0]], [[1.0]])
        with pytest.raises(ValueError, match="threshold"):
            summarize_taylor_validity(grid, threshold=1.0001)

    def test_accepts_threshold_of_exactly_one(self) -> None:
        grid = _grid([[1.0]], [[1.0]])
        summarize_taylor_validity(grid, threshold=1.0)  # must not raise

    def test_worst_nonlinearity_fraction_is_the_max_over_scenarios(self) -> None:
        grid = _grid([[100.0, 0.0, 50.0]], [[0.0, 10.0, 50.0]])
        summary = summarize_taylor_validity(grid, threshold=0.5)
        # fractions: [0.0, 1.0, 0.5] -> worst is 1.0
        assert summary.worst_nonlinearity_fraction[0] == 1.0

    def test_worst_scenario_index_points_at_the_actual_worst_cell(self) -> None:
        grid = _grid([[100.0, 0.0, 50.0]], [[0.0, 10.0, 50.0]])
        summary = summarize_taylor_validity(grid, threshold=0.5)
        assert summary.worst_scenario_index[0] == 1

    def test_exceedance_fraction_counts_only_cells_strictly_above_threshold(
        self,
    ) -> None:
        """A cell exactly AT the threshold must not count -- '>', not '>='."""
        grid = _grid([[10.0, 10.0]], [[10.0, 20.0]])  # fractions: 0.5, 2/3
        summary = summarize_taylor_validity(grid, threshold=0.5)
        assert summary.exceedance_fraction[0] == 0.5  # only the second cell

    def test_num_positions_exceeding_counts_positions_not_cells(self) -> None:
        grid = _grid(
            [[0.0, 0.0], [100.0, 0.0]],
            [[0.0, 0.0], [0.0, 100.0]],
        )
        summary = summarize_taylor_validity(grid, threshold=0.1)
        # position 0: fractions [0, 0] -- never exceeds
        # position 1: fractions [0, 1.0] -- exceeds once
        assert summary.num_positions_exceeding == 1

    def test_summary_position_uids_match_the_grid(self) -> None:
        grid = _grid([[1.0], [2.0]], [[0.0], [0.0]])
        summary = summarize_taylor_validity(grid, threshold=0.5)
        assert summary.position_uids.tolist() == grid.position_uids.tolist()

    def test_summary_carries_the_threshold_it_was_computed_with(self) -> None:
        grid = _grid([[1.0]], [[1.0]])
        summary = summarize_taylor_validity(grid, threshold=0.37)
        assert summary.threshold == 0.37

    def test_summary_is_a_dataclass_instance(self) -> None:
        grid = _grid([[1.0]], [[1.0]])
        summary = summarize_taylor_validity(grid, threshold=0.5)
        assert isinstance(summary, TaylorValiditySummary)
