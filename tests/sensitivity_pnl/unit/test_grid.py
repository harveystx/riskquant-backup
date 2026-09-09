# tests/sensitivity_pnl/unit/test_grid.py
"""
Unit tests for `riskquant.sensitivity_pnl.grid`.

`allocate_pnl_grid` replaced a `np.concatenate`-based `attach_pnl` specifically
to eliminate a third-array allocation; `PnlGrid.block_view` is only useful if
it is genuinely a zero-copy view, so several tests below assert on
`np.shares_memory` directly rather than merely checking the returned values are
numerically correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.sensitivity_pnl.grid import PnlGrid, allocate_pnl_grid


class TestAllocatePnlGrid:
    def test_single_block_basic_shape(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1, 2, 3], dtype=np.int64)],
            num_scenarios=5,
        )
        assert grid.pnl.shape == (3, 5)
        assert grid.trade_uids.tolist() == [1, 2, 3]
        assert grid.num_blocks == 1
        assert grid.num_scenarios == 5

    def test_multiple_blocks_are_concatenated_in_the_given_order(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[
                np.array([10, 20], dtype=np.int64),
                np.array([30, 40, 50], dtype=np.int64),
            ],
            num_scenarios=4,
        )
        assert grid.trade_uids.tolist() == [10, 20, 30, 40, 50]
        assert grid.pnl.shape == (5, 4)
        assert grid.num_blocks == 2

    def test_pnl_dtype_is_float32(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1], dtype=np.int64)], num_scenarios=1
        )
        assert grid.pnl.dtype == np.float32

    def test_pnl_is_c_contiguous(self) -> None:
        grid = allocate_pnl_grid(
            block_position_uids=[np.array([1, 2], dtype=np.int64)], num_scenarios=3
        )
        assert grid.pnl.flags.c_contiguous

    def test_rejects_an_empty_block_list(self) -> None:
        with pytest.raises(ValueError, match="at least one block"):
            allocate_pnl_grid(block_position_uids=[], num_scenarios=5)

    @pytest.mark.parametrize("num_scenarios", [0, -1, -100])
    def test_rejects_non_positive_num_scenarios(self, num_scenarios: int) -> None:
        with pytest.raises(ValueError, match="num_scenarios must be positive"):
            allocate_pnl_grid(
                block_position_uids=[np.array([1], dtype=np.int64)],
                num_scenarios=num_scenarios,
            )

    def test_rejects_a_non_int64_uid_array(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D int64"):
            allocate_pnl_grid(
                block_position_uids=[np.array([1, 2], dtype=np.int32)],
                num_scenarios=5,
            )

    def test_rejects_a_2d_uid_array(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D int64"):
            allocate_pnl_grid(
                block_position_uids=[np.array([[1], [2]], dtype=np.int64)],
                num_scenarios=5,
            )

    def test_rejects_duplicate_uids_within_one_block(self) -> None:
        with pytest.raises(ValueError, match="double-counted"):
            allocate_pnl_grid(
                block_position_uids=[np.array([1, 1, 2], dtype=np.int64)],
                num_scenarios=5,
            )

    def test_rejects_a_uid_shared_across_two_blocks(self) -> None:
        with pytest.raises(ValueError, match="double-counted"):
            allocate_pnl_grid(
                block_position_uids=[
                    np.array([1, 2], dtype=np.int64),
                    np.array([2, 3], dtype=np.int64),
                ],
                num_scenarios=5,
            )

    def test_error_reports_the_actual_colliding_uid(self) -> None:
        with pytest.raises(ValueError, match=r"First 10: \[7\]"):
            allocate_pnl_grid(
                block_position_uids=[
                    np.array([7], dtype=np.int64),
                    np.array([7], dtype=np.int64),
                ],
                num_scenarios=5,
            )

    def test_an_empty_block_is_permitted_alongside_a_nonempty_one(self) -> None:
        """A zero-length block is not the same as an empty block LIST -- the
        latter is rejected, the former (e.g. a full-reval block with no
        positions this run) is a legitimate degenerate case."""
        grid = allocate_pnl_grid(
            block_position_uids=[
                np.empty(0, dtype=np.int64),
                np.array([1, 2], dtype=np.int64),
            ],
            num_scenarios=3,
        )
        assert grid.pnl.shape == (2, 3)
        assert grid.num_blocks == 2
        assert grid.block_view(0).shape == (0, 3)


class TestPnlGridPostInit:
    def _make(self, **overrides):
        fields = {
            "trade_uids": np.array([1, 2], dtype=np.int64),
            "pnl": np.zeros((2, 3), dtype=np.float32),
            "block_offsets": np.array([0, 2], dtype=np.int32),
        }
        fields.update(overrides)
        return PnlGrid(**fields)

    def test_valid_construction_succeeds(self) -> None:
        self._make()

    def test_rejects_non_float32_pnl(self) -> None:
        with pytest.raises(ValueError, match="must be float32"):
            self._make(pnl=np.zeros((2, 3), dtype=np.float64))

    def test_rejects_non_contiguous_pnl(self) -> None:
        pnl = np.zeros((2, 6), dtype=np.float32)[:, ::2]
        assert not pnl.flags.c_contiguous
        with pytest.raises(ValueError, match="C-contiguous"):
            self._make(pnl=pnl)

    def test_rejects_trade_uids_pnl_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="row mismatch"):
            self._make(trade_uids=np.array([1, 2, 3], dtype=np.int64))

    def test_rejects_block_offsets_not_terminating_at_m_total(self) -> None:
        with pytest.raises(ValueError, match="do not terminate at M_total"):
            self._make(block_offsets=np.array([0, 1], dtype=np.int32))


class TestPnlGridAccessors:
    def _grid(self) -> PnlGrid:
        return allocate_pnl_grid(
            block_position_uids=[
                np.array([1, 2], dtype=np.int64),
                np.array([3, 4, 5], dtype=np.int64),
            ],
            num_scenarios=4,
        )

    def test_block_view_returns_the_correct_slice(self) -> None:
        grid = self._grid()
        assert grid.block_view(0).shape == (2, 4)
        assert grid.block_view(1).shape == (3, 4)

    def test_block_view_is_a_zero_copy_view_not_a_copy(self) -> None:
        grid = self._grid()
        view = grid.block_view(1)
        assert np.shares_memory(view, grid.pnl)

    def test_writing_through_block_view_mutates_the_shared_grid(self) -> None:
        """`allocate_pnl_grid` uses `np.empty`, not `np.zeros` -- deliberately
        uninitialized, since every cell is expected to be written before being
        read. So block 0 is seeded with a KNOWN value first, rather than
        assumed to start at 0.0, and the real invariant under test is that
        writing block 1 leaves block 0's own values untouched."""
        grid = self._grid()
        grid.block_view(0)[:] = -1.0
        grid.block_view(1)[:] = 42.0
        assert bool(np.all(grid.pnl[2:5] == 42.0))
        assert bool(
            np.all(grid.pnl[0:2] == -1.0)
        )  # block 0 untouched by block 1's write

    def test_block_view_is_c_contiguous(self) -> None:
        grid = self._grid()
        assert grid.block_view(0).flags.c_contiguous
        assert grid.block_view(1).flags.c_contiguous

    @pytest.mark.parametrize("bad_index", [-1, 2, 100])
    def test_block_view_rejects_an_out_of_range_index(self, bad_index: int) -> None:
        grid = self._grid()
        with pytest.raises(IndexError):
            grid.block_view(bad_index)

    def test_block_uids_returns_the_correct_slice(self) -> None:
        grid = self._grid()
        assert grid.block_uids(0).tolist() == [1, 2]
        assert grid.block_uids(1).tolist() == [3, 4, 5]

    def test_block_uids_is_a_view(self) -> None:
        grid = self._grid()
        assert np.shares_memory(grid.block_uids(1), grid.trade_uids)

    @pytest.mark.parametrize("bad_index", [-1, 2])
    def test_block_uids_rejects_an_out_of_range_index(self, bad_index: int) -> None:
        grid = self._grid()
        with pytest.raises(IndexError):
            grid.block_uids(bad_index)

    def test_to_pnl_matrix_is_zero_copy(self) -> None:
        grid = self._grid()
        matrix = grid.to_pnl_matrix()
        assert matrix.pnl is grid.pnl
        assert matrix.trade_uids is grid.trade_uids

    def test_to_pnl_matrix_reflects_the_whole_grid_not_one_block(self) -> None:
        grid = self._grid()
        matrix = grid.to_pnl_matrix()
        assert matrix.pnl.shape[0] == 5
        assert matrix.trade_uids.tolist() == [1, 2, 3, 4, 5]

    def test_num_blocks(self) -> None:
        assert self._grid().num_blocks == 2

    def test_num_scenarios(self) -> None:
        assert self._grid().num_scenarios == 4
