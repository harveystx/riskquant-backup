# tests/sensitivity_pnl/unit/test_scheduling.py
"""
Unit tests for `riskquant.sensitivity_pnl.scheduling`.

`balanced_chunk_offsets` is the fix for a real, measured production issue —
prange load imbalance on heavy-tailed books swinging scaling between 0.88x and
2.13x across identical runs (see the module docstring). These tests pin the
STRUCTURAL guarantees the fix depends on: offsets start at 0, are strictly
increasing, terminate at M, and the resulting cost-per-chunk is genuinely
balanced on a skewed book — not just "the function returns without raising."
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.compile import compile_sensitivity_book
from riskquant.sensitivity_pnl.conventions import DISPLACEMENT_ABSOLUTE_ID
from riskquant.sensitivity_pnl.scheduling import (
    DEFAULT_CHUNKS_PER_THREAD,
    PER_ROW_SCATTER_COST,
    balanced_chunk_offsets,
    default_num_chunks,
    row_costs,
)


def _book(
    *, entries_per_position: list[int], cross_per_position: list[int] | None = None
):
    """A book with `entries_per_position[m]` diagonal entries and
    `cross_per_position[m]` cross entries for position m+1, on a shared factor
    pool sized to fit the largest row."""
    cross_per_position = cross_per_position or [0] * len(entries_per_position)
    num_factors = max(max(entries_per_position, default=0), 2) + 2
    ids = np.arange(num_factors, dtype=np.int32)
    rng = np.random.default_rng(0)

    diag_rows = []
    cross_rows = []
    for position_index, (n_diag, n_cross) in enumerate(
        zip(entries_per_position, cross_per_position, strict=True)
    ):
        uid = position_index + 1
        chosen = rng.choice(num_factors, n_diag, replace=False)
        for factor_id in chosen:
            diag_rows.append((uid, int(factor_id), 1.0, 0.0))
        for _ in range(n_cross):
            pair = rng.choice(num_factors, 2, replace=False)
            cross_rows.append((uid, int(pair[0]), int(pair[1]), 1.0))

    sensitivities = pl.DataFrame(
        diag_rows or [],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.Int32,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    cross = pl.DataFrame(
        cross_rows or [],
        schema={
            "position_id": pl.Int64,
            "factor_id_i": pl.Int32,
            "factor_id_j": pl.Int32,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.full(
                num_factors, int(DISPLACEMENT_ABSOLUTE_ID), dtype=np.int8
            ),
            "quote_base": np.ones(num_factors),
        }
    )
    return compile_sensitivity_book(
        sensitivities=sensitivities,
        factor_meta=meta,
        level_universe_factor_ids=ids,
        cross_sensitivities=cross,
        extra_position_uids=np.arange(1, len(entries_per_position) + 1, dtype=np.int64),
    )


class TestRowCosts:
    def test_matches_the_documented_formula(self) -> None:
        """cost(m) = nnz(m) + 2*nnz_cross(m) + PER_ROW_SCATTER_COST"""
        book = _book(entries_per_position=[3, 0, 5], cross_per_position=[1, 2, 0])
        cost = row_costs(book)
        expected = np.array(
            [
                3 + 2 * 1 + PER_ROW_SCATTER_COST,
                0 + 2 * 2 + PER_ROW_SCATTER_COST,
                5 + 2 * 0 + PER_ROW_SCATTER_COST,
            ]
        )
        assert np.array_equal(cost, expected)

    def test_a_row_with_zero_entries_still_has_positive_cost(self) -> None:
        """PER_ROW_SCATTER_COST guarantees cost > 0 for every row, including a
        declared-but-empty position — required for the cost prefix sum used
        by balanced_chunk_offsets to be strictly increasing."""
        book = _book(entries_per_position=[0, 0, 0])
        cost = row_costs(book)
        assert bool(np.all(cost > 0.0))
        assert np.array_equal(cost, np.full(3, PER_ROW_SCATTER_COST))

    def test_output_shape_matches_num_positions(self) -> None:
        book = _book(entries_per_position=[1, 2, 3, 4])
        assert row_costs(book).shape == (4,)

    def test_output_dtype_is_float64(self) -> None:
        book = _book(entries_per_position=[1])
        assert row_costs(book).dtype == np.float64


class TestDefaultNumChunks:
    def test_zero_for_zero_positions(self) -> None:
        assert default_num_chunks(num_positions=0, thread_ceiling=8) == 0

    def test_capped_at_one_chunk_per_position(self) -> None:
        """More chunks than positions would create empty chunks."""
        assert default_num_chunks(num_positions=3, thread_ceiling=64) == 3

    def test_uses_thread_ceiling_times_chunks_per_thread_when_it_fits(self) -> None:
        assert (
            default_num_chunks(
                num_positions=10_000, thread_ceiling=4, chunks_per_thread=8
            )
            == 32
        )

    def test_default_chunks_per_thread_constant_is_used_when_not_overridden(
        self,
    ) -> None:
        result = default_num_chunks(num_positions=10_000, thread_ceiling=4)
        assert result == 4 * DEFAULT_CHUNKS_PER_THREAD

    def test_thread_ceiling_of_zero_is_treated_as_one(self) -> None:
        """A pathological ceiling of 0 must not collapse chunk count to 0 for
        a nonempty book -- max(1, thread_ceiling) is the floor."""
        assert (
            default_num_chunks(num_positions=100, thread_ceiling=0, chunks_per_thread=8)
            == 8
        )

    def test_rejects_non_positive_chunks_per_thread(self) -> None:
        with pytest.raises(ValueError, match="chunks_per_thread"):
            default_num_chunks(num_positions=10, thread_ceiling=4, chunks_per_thread=0)

    def test_rejects_negative_chunks_per_thread(self) -> None:
        with pytest.raises(ValueError, match="chunks_per_thread"):
            default_num_chunks(num_positions=10, thread_ceiling=4, chunks_per_thread=-1)

    def test_single_position_gives_a_single_chunk(self) -> None:
        assert default_num_chunks(num_positions=1, thread_ceiling=64) == 1


class TestBalancedChunkOffsets:
    def test_zero_positions_gives_the_degenerate_single_offset(self) -> None:
        book = _book(entries_per_position=[])
        offsets = balanced_chunk_offsets(book=book, num_chunks=8)
        assert offsets.tolist() == [0]

    def test_num_chunks_le_1_gives_a_single_chunk_spanning_everything(self) -> None:
        book = _book(entries_per_position=[1, 2, 3])
        for num_chunks in (0, 1, -5):
            offsets = balanced_chunk_offsets(book=book, num_chunks=num_chunks)
            assert offsets.tolist() == [0, 3]

    def test_offsets_start_at_zero_and_terminate_at_m(self) -> None:
        book = _book(entries_per_position=[5, 1, 1, 1, 20, 1, 1, 1, 1, 1])
        offsets = balanced_chunk_offsets(book=book, num_chunks=32)
        assert int(offsets[0]) == 0
        assert int(offsets[-1]) == book.num_positions

    def test_offsets_are_strictly_increasing(self) -> None:
        """An empty chunk (a repeated offset) would consume a prange slot for
        no work -- the docstring's own stated invariant."""
        book = _book(entries_per_position=[5, 1, 1, 1, 20, 1, 1, 1, 1, 1])
        offsets = balanced_chunk_offsets(book=book, num_chunks=32)
        assert bool(np.all(np.diff(offsets) > 0))

    def test_offsets_dtype_is_int32(self) -> None:
        book = _book(entries_per_position=[1, 2, 3])
        offsets = balanced_chunk_offsets(book=book, num_chunks=4)
        assert offsets.dtype == np.int32

    def test_offsets_are_c_contiguous(self) -> None:
        book = _book(entries_per_position=[1, 2, 3])
        offsets = balanced_chunk_offsets(book=book, num_chunks=4)
        assert offsets.flags.c_contiguous

    def test_num_chunks_capped_at_num_positions(self) -> None:
        """Requesting more chunks than positions cannot create positions;
        C <= min(num_chunks, num_positions) is the documented cap."""
        book = _book(entries_per_position=[1, 1, 1])
        offsets = balanced_chunk_offsets(book=book, num_chunks=1000)
        assert offsets.shape[0] - 1 <= 3

    def test_a_single_very_heavy_row_still_yields_a_valid_partition(self) -> None:
        """One position with almost all the entries: chunk cuts colliding on
        it must merge rather than emit an empty chunk."""
        book = _book(entries_per_position=[1000, 1, 1, 1, 1])
        offsets = balanced_chunk_offsets(book=book, num_chunks=16)
        assert int(offsets[0]) == 0
        assert int(offsets[-1]) == 5
        assert bool(np.all(np.diff(offsets) > 0))

    def test_uniform_book_yields_roughly_equal_sized_chunks(self) -> None:
        """A book with identical per-row cost should partition into chunks of
        roughly equal ROW COUNT too, since cost is proportional to count here."""
        book = _book(entries_per_position=[4] * 32)
        offsets = balanced_chunk_offsets(book=book, num_chunks=8)
        sizes = np.diff(offsets)
        assert sizes.max() - sizes.min() <= 1  # near-perfectly even

    def test_chunk_cost_is_balanced_on_a_skewed_book(self) -> None:
        """The actual point of this module: a book where 1 of 20 positions
        holds as much work as the other 19 combined must still partition into
        chunks whose COST (not row count) is roughly even."""
        entries = [1] * 19 + [19]
        book = _book(entries_per_position=entries)
        cost = row_costs(book)
        offsets = balanced_chunk_offsets(book=book, num_chunks=8)
        chunk_costs = np.array(
            [
                cost[offsets[i] : offsets[i + 1]].sum()
                for i in range(offsets.shape[0] - 1)
            ]
        )
        # Perfect balance would be total/8; allow generous slack since a
        # single row cannot be split, but it must not be wildly skewed.
        target = cost.sum() / (offsets.shape[0] - 1)
        assert chunk_costs.max() < target * 2.5

    def test_chunking_never_drops_or_duplicates_a_row(self) -> None:
        """Every position index 0..M-1 must be covered by exactly one chunk."""
        book = _book(entries_per_position=[3, 7, 1, 1, 12, 1, 1, 1, 4, 4, 4])
        offsets = balanced_chunk_offsets(book=book, num_chunks=5)
        covered = np.zeros(book.num_positions, dtype=np.int64)
        for i in range(offsets.shape[0] - 1):
            covered[offsets[i] : offsets[i + 1]] += 1
        assert bool(np.all(covered == 1))
