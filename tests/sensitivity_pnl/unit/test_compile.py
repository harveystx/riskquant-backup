# tests/sensitivity_pnl/unit/test_compile.py
"""
Unit tests for `riskquant.sensitivity_pnl.compile`.

`compile_sensitivity_book` is the only place in the package where strings
resolve to integers, quote conventions fold into coefficients, and a caller's
raw edge frames become the CSR arrays the kernels dereference under
`boundscheck=False`. The module docstring calls out five owned policy
decisions (label resolution, strict factor-universe join, active-factor
compaction, quote folding, duplicate rejection); this file has one section per
decision, plus the array primitives (`_csr_offsets`, `_collapse_duplicates`)
tested directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.compile import (
    DuplicatePolicy,
    _collapse_duplicates,
    _csr_offsets,
    compile_sensitivity_book,
)
from riskquant.sensitivity_pnl.conventions import DisplacementConvention

ABSOLUTE = int(DisplacementConvention.ABSOLUTE)
RELATIVE = int(DisplacementConvention.RELATIVE)
LOG = int(DisplacementConvention.LOG)


def _meta(factor_ids, conventions=None, quote_bases=None) -> pl.DataFrame:
    n = len(factor_ids)
    return pl.DataFrame(
        {
            "factor_id": factor_ids,
            "displacement_convention_id": np.array(
                conventions or [ABSOLUTE] * n, dtype=np.int8
            ),
            "quote_base": quote_bases or [1.0] * n,
        }
    )


def _sens(rows, dtype_factor=pl.Int32) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "position_id": pl.Int64,
            "factor_id": dtype_factor,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )


def _cross(rows, dtype_factor=pl.Int32) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "position_id": pl.Int64,
            "factor_id_i": dtype_factor,
            "factor_id_j": dtype_factor,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )


# =========================================================================
# BASIC COMPILATION
# =========================================================================


class TestBasicCompilation:
    def test_a_single_position_single_factor_book_compiles(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
        )
        assert book.num_positions == 1
        assert book.nnz == 1
        assert book.first_order.tolist() == [5.0]

    def test_position_uids_are_returned_sorted_ascending(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens(
                [(30, 1, 1.0, 0.0), (10, 1, 2.0, 0.0), (20, 1, 3.0, 0.0)]
            ),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
        )
        assert book.position_uids.tolist() == [10, 20, 30]

    def test_active_factor_ids_are_the_union_of_diagonal_and_cross_factors(
        self,
    ) -> None:
        ids = np.array([1, 2, 3, 4], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1, 2, 3, 4]),
            level_universe_factor_ids=ids,
            cross_sensitivities=_cross([(1, 2, 3, 1.0)]),
        )
        # factor 4 is in the level universe and factor_meta but touched by
        # nothing -- must NOT appear in the compacted active-factor set.
        assert book.active_factor_ids.tolist() == [1, 2, 3]

    def test_rows_are_sorted_into_csr_order_by_position_then_factor(self) -> None:
        ids = np.array([1, 2, 3], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens(
                [(1, 3, 30.0, 0.0), (1, 1, 10.0, 0.0), (1, 2, 20.0, 0.0)]
            ),
            factor_meta=_meta([1, 2, 3]),
            level_universe_factor_ids=ids,
        )
        # factor_uix must be ascending within the single row.
        assert book.factor_uix.tolist() == sorted(book.factor_uix.tolist())
        assert book.first_order.tolist() == [10.0, 20.0, 30.0]

    def test_row_order_of_the_output_matches_ascending_position_uid(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(5, 1, 1.0, 0.0), (2, 1, 2.0, 0.0)]),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
        )
        assert book.position_uids.tolist() == [2, 5]
        # position 2's entry (value 2.0) must be in row 0, position 5's in row 1
        row0 = book.first_order[book.position_offsets[0] : book.position_offsets[1]]
        assert row0.tolist() == [2.0]


# =========================================================================
# STRING LABEL RESOLUTION
# =========================================================================


class TestLabelResolution:
    def test_string_position_and_factor_labels_resolve_via_lookups(self) -> None:
        ids = np.array([0, 1], dtype=np.int32)
        sens = pl.DataFrame(
            {
                "position_id": ["POS_A"],
                "factor_id": ["EQ_ACME"],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        book = compile_sensitivity_book(
            sensitivities=sens,
            factor_meta=_meta(["EQ_ACME"]),
            level_universe_factor_ids=ids,
            position_id_lookup={"POS_A": 100},
            factor_id_lookup={"EQ_ACME": 0},
        )
        assert book.position_uids.tolist() == [100]

    def test_unmapped_position_label_raises(self) -> None:
        ids = np.array([0], dtype=np.int32)
        sens = pl.DataFrame(
            {
                "position_id": ["UNKNOWN"],
                "factor_id": [0],
                "first_order": [1.0],
                "second_order": [0.0],
            }
        )
        with pytest.raises(Exception):  # noqa: B017 -- polars replace_strict raises on a miss
            compile_sensitivity_book(
                sensitivities=sens,
                factor_meta=_meta([0]),
                level_universe_factor_ids=ids,
                position_id_lookup={"KNOWN": 1},
            )

    def test_no_lookup_means_the_column_is_cast_directly(self) -> None:
        """When `factor_id_lookup` is None, factor_id must already be
        (castable to) an integer -- this is the pre-integer feed path."""
        ids = np.array([7], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 7, 5.0, 0.0)]),
            factor_meta=_meta([7]),
            level_universe_factor_ids=ids,
        )
        assert book.active_factor_ids.tolist() == [7]


# =========================================================================
# STRICT FACTOR-UNIVERSE JOIN
# =========================================================================


class TestStrictFactorUniverseJoin:
    def test_a_sensitivity_to_a_factor_outside_the_level_universe_raises(
        self,
    ) -> None:
        """The module docstring calls this 'the single most dangerous failure
        mode': a sensitivity to a factor with no scenario row would silently
        contribute zero risk forever."""
        ids = np.array([1], dtype=np.int32)  # universe does NOT include 2
        with pytest.raises(ValueError):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 2, 5.0, 0.0)]),
                factor_meta=_meta([1, 2]),
                level_universe_factor_ids=ids,
            )

    def test_a_cross_factor_outside_the_level_universe_raises(self) -> None:
        ids = np.array([1, 2], dtype=np.int32)  # universe does NOT include 3
        with pytest.raises(ValueError):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=_meta([1, 2, 3]),
                level_universe_factor_ids=ids,
                cross_sensitivities=_cross([(1, 1, 3, 1.0)]),
            )

    def test_a_factor_in_the_universe_but_missing_from_factor_meta_raises(
        self,
    ) -> None:
        ids = np.array([1, 2], dtype=np.int32)
        with pytest.raises(ValueError):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 2, 5.0, 0.0)]),
                factor_meta=_meta([1]),  # factor 2 has no meta row
                level_universe_factor_ids=ids,
            )

    def test_level_universe_factor_ids_must_be_sorted_and_unique(self) -> None:
        with pytest.raises(ValueError, match="sorted ascending and unique"):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=_meta([1, 2]),
                level_universe_factor_ids=np.array([2, 1], dtype=np.int32),
            )

    def test_level_universe_factor_ids_with_a_duplicate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sorted ascending and unique"):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=_meta([1]),
                level_universe_factor_ids=np.array([1, 1], dtype=np.int32),
            )


# =========================================================================
# QUOTE-BASE FOLDING
# =========================================================================


class TestQuoteBaseFolding:
    def test_first_order_scales_by_quote_base_to_the_first_power(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1], quote_bases=[100.0]),
            level_universe_factor_ids=ids,
        )
        assert book.first_order.tolist() == [500.0]

    def test_second_order_scales_by_quote_base_squared(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 0.0, 4.0)]),
            factor_meta=_meta([1], quote_bases=[100.0]),
            level_universe_factor_ids=ids,
        )
        assert book.second_order.tolist() == [4.0 * 100.0 * 100.0]

    def test_cross_second_order_scales_by_the_product_of_the_two_quote_bases(
        self,
    ) -> None:
        ids = np.array([1, 2], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([]),
            factor_meta=_meta([1, 2], quote_bases=[10.0, 1000.0]),
            level_universe_factor_ids=ids,
            cross_sensitivities=_cross([(1, 1, 2, 2.0)]),
        )
        assert book.cross_second_order.tolist() == [2.0 * 10.0 * 1000.0]

    def test_quote_base_of_one_leaves_coefficients_unchanged(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 7.5, -2.5)]),
            factor_meta=_meta([1], quote_bases=[1.0]),
            level_universe_factor_ids=ids,
        )
        assert book.first_order.tolist() == [7.5]
        assert book.second_order.tolist() == [-2.5]

    def test_non_positive_quote_base_is_rejected(self) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(
            ValueError, match="quote_base must be finite and strictly positive"
        ):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=_meta([1], quote_bases=[0.0]),
                level_universe_factor_ids=ids,
            )

    def test_negative_quote_base_is_rejected(self) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(
            ValueError, match="quote_base must be finite and strictly positive"
        ):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=_meta([1], quote_bases=[-100.0]),
                level_universe_factor_ids=ids,
            )

    def test_infinite_quote_base_is_rejected(self) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(
            ValueError, match="quote_base must be finite and strictly positive"
        ):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=_meta([1], quote_bases=[np.inf]),
                level_universe_factor_ids=ids,
            )

    def test_unknown_displacement_convention_id_in_factor_meta_is_rejected(
        self,
    ) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(ValueError, match="unknown displacement_convention_id"):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=_meta([1], conventions=[99]),
                level_universe_factor_ids=ids,
            )

    def test_the_house_wide_factor_table_may_carry_extra_untouched_factors(
        self,
    ) -> None:
        """factor_meta may be the whole house table; rows for factors the
        book never touches must be silently ignored, not raise."""
        ids = np.array([1, 2, 3], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1, 2, 3]),  # 2, 3 are untouched by the book
            level_universe_factor_ids=ids,
        )
        assert book.active_factor_ids.tolist() == [1]

    def test_duplicate_factor_id_rows_in_factor_meta_are_rejected(self) -> None:
        ids = np.array([1], dtype=np.int32)
        meta = pl.DataFrame(
            {
                "factor_id": [1, 1],
                "displacement_convention_id": np.array(
                    [ABSOLUTE, RELATIVE], dtype=np.int8
                ),
                "quote_base": [1.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="duplicate factor_id rows"):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0)]),
                factor_meta=meta,
                level_universe_factor_ids=ids,
            )


# =========================================================================
# DUPLICATE POLICY
# =========================================================================


class TestDuplicatePolicy:
    def test_raise_is_the_default_and_fires_on_a_repeated_diagonal_key(
        self,
    ) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(ValueError, match="duplicate"):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 5.0, 0.0), (1, 1, 3.0, 0.0)]),
                factor_meta=_meta([1]),
                level_universe_factor_ids=ids,
            )

    def test_raise_fires_on_a_repeated_cross_key(self) -> None:
        ids = np.array([1, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="duplicate"):
            compile_sensitivity_book(
                sensitivities=_sens([]),
                factor_meta=_meta([1, 2]),
                level_universe_factor_ids=ids,
                cross_sensitivities=_cross([(1, 1, 2, 5.0), (1, 1, 2, 3.0)]),
            )

    def test_sum_policy_adds_repeated_diagonal_coefficients_exactly(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 1.0), (1, 1, 3.0, 2.0)]),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
            duplicate_policy=DuplicatePolicy.SUM,
        )
        assert book.nnz == 1
        assert book.first_order.tolist() == [8.0]
        assert book.second_order.tolist() == [3.0]

    def test_sum_policy_adds_repeated_cross_coefficients(self) -> None:
        ids = np.array([1, 2], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([]),
            factor_meta=_meta([1, 2]),
            level_universe_factor_ids=ids,
            cross_sensitivities=_cross([(1, 1, 2, 5.0), (1, 1, 2, 3.0)]),
            duplicate_policy=DuplicatePolicy.SUM,
        )
        assert book.nnz_cross == 1
        assert book.cross_second_order.tolist() == [8.0]

    def test_sum_policy_does_not_affect_non_duplicated_rows(self) -> None:
        ids = np.array([1, 2], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0), (1, 2, 9.0, 0.0)]),
            factor_meta=_meta([1, 2]),
            level_universe_factor_ids=ids,
            duplicate_policy=DuplicatePolicy.SUM,
        )
        assert book.nnz == 2
        assert sorted(book.first_order.tolist()) == [5.0, 9.0]

    def test_cross_ij_and_ji_are_the_same_key_and_collide_under_raise(self) -> None:
        """(i, j) and (j, i) refer to the SAME unordered pair -- a caller
        supplying both is a duplicate, not two distinct cross terms."""
        ids = np.array([1, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="duplicate"):
            compile_sensitivity_book(
                sensitivities=_sens([]),
                factor_meta=_meta([1, 2]),
                level_universe_factor_ids=ids,
                cross_sensitivities=_cross([(1, 1, 2, 5.0), (1, 2, 1, 3.0)]),
            )


# =========================================================================
# CROSS-BLOCK SPECIFIC VALIDATION
# =========================================================================


class TestCrossBlockValidation:
    def test_i_equal_to_j_is_rejected(self) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(ValueError, match="factor_id_i == factor_id_j"):
            compile_sensitivity_book(
                sensitivities=_sens([]),
                factor_meta=_meta([1]),
                level_universe_factor_ids=ids,
                cross_sensitivities=_cross([(1, 1, 1, 5.0)]),
            )

    def test_no_cross_sensitivities_gives_an_empty_but_well_formed_cross_block(
        self,
    ) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
        )
        assert book.nnz_cross == 0
        assert book.cross_offsets.tolist() == [0, 0]

    def test_a_position_with_only_cross_entries_and_no_diagonal_ones_is_valid(
        self,
    ) -> None:
        ids = np.array([1, 2], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([]),
            factor_meta=_meta([1, 2]),
            level_universe_factor_ids=ids,
            cross_sensitivities=_cross([(1, 1, 2, 5.0)]),
        )
        assert book.nnz == 0
        assert book.nnz_cross == 1
        assert book.num_positions == 1


# =========================================================================
# extra_position_uids
# =========================================================================


class TestExtraPositionUids:
    def test_a_position_with_no_sensitivities_holds_its_row_as_zeros(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
            extra_position_uids=np.array([999], dtype=np.int64),
        )
        assert book.position_uids.tolist() == [1, 999]
        row = int(np.searchsorted(book.position_uids, 999))
        lo, hi = book.position_offsets[row], book.position_offsets[row + 1]
        assert hi == lo  # zero entries for this row

    def test_extra_position_uids_overlapping_an_existing_position_is_a_no_op(
        self,
    ) -> None:
        """Declaring a UID that already carries sensitivities must not create
        a duplicate row -- position_uids is a set union, not a concatenation."""
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
            extra_position_uids=np.array([1], dtype=np.int64),
        )
        assert book.position_uids.tolist() == [1]
        assert book.num_positions == 1

    def test_none_is_the_default_and_adds_no_extra_rows(self) -> None:
        ids = np.array([1], dtype=np.int32)
        book = compile_sensitivity_book(
            sensitivities=_sens([(1, 1, 5.0, 0.0)]),
            factor_meta=_meta([1]),
            level_universe_factor_ids=ids,
        )
        assert book.num_positions == 1


# =========================================================================
# FINITENESS
# =========================================================================


class TestFinitenessValidation:
    def test_nan_first_order_is_rejected(self) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(ValueError, match="non-finite"):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, float("nan"), 0.0)]),
                factor_meta=_meta([1]),
                level_universe_factor_ids=ids,
            )

    def test_inf_second_order_is_rejected(self) -> None:
        ids = np.array([1], dtype=np.int32)
        with pytest.raises(ValueError, match="non-finite"):
            compile_sensitivity_book(
                sensitivities=_sens([(1, 1, 0.0, float("inf"))]),
                factor_meta=_meta([1]),
                level_universe_factor_ids=ids,
            )

    def test_negative_inf_cross_second_order_is_rejected(self) -> None:
        ids = np.array([1, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="non-finite"):
            compile_sensitivity_book(
                sensitivities=_sens([]),
                factor_meta=_meta([1, 2]),
                level_universe_factor_ids=ids,
                cross_sensitivities=_cross([(1, 1, 2, float("-inf"))]),
            )


# =========================================================================
# ARRAY PRIMITIVES (tested directly, no compiler round trip)
# =========================================================================


class TestCsrOffsets:
    def test_empty_rows_gives_all_zero_offsets(self) -> None:
        offsets = _csr_offsets(rows=np.empty(0, dtype=np.int64), num_rows=3)
        assert offsets.tolist() == [0, 0, 0, 0]

    def test_one_entry_per_row(self) -> None:
        offsets = _csr_offsets(rows=np.array([0, 1, 2], dtype=np.int64), num_rows=3)
        assert offsets.tolist() == [0, 1, 2, 3]

    def test_uneven_distribution(self) -> None:
        # row 0: 3 entries, row 1: 0 entries, row 2: 1 entry
        offsets = _csr_offsets(rows=np.array([0, 0, 0, 2], dtype=np.int64), num_rows=3)
        assert offsets.tolist() == [0, 3, 3, 4]

    def test_output_dtype_is_int32(self) -> None:
        offsets = _csr_offsets(rows=np.array([0], dtype=np.int64), num_rows=1)
        assert offsets.dtype == np.int32

    def test_zero_rows_gives_a_single_zero_offset(self) -> None:
        offsets = _csr_offsets(rows=np.empty(0, dtype=np.int64), num_rows=0)
        assert offsets.tolist() == [0]


class TestCollapseDuplicates:
    def test_no_duplicates_returns_every_index_unchanged(self) -> None:
        keys = (np.array([0, 1, 2], dtype=np.int64),)
        values = (np.array([1.0, 2.0, 3.0]),)
        keep, collapsed = _collapse_duplicates(
            keys=keys, values=values, policy=DuplicatePolicy.RAISE, ctx="test"
        )
        assert keep.tolist() == [0, 1, 2]
        assert collapsed[0].tolist() == [1.0, 2.0, 3.0]

    def test_raise_policy_reports_the_correct_duplicate_count(self) -> None:
        keys = (np.array([0, 0, 0, 1], dtype=np.int64),)
        values = (np.array([1.0, 2.0, 3.0, 4.0]),)
        with pytest.raises(ValueError, match="2 duplicate"):
            _collapse_duplicates(
                keys=keys, values=values, policy=DuplicatePolicy.RAISE, ctx="test"
            )

    def test_sum_policy_sums_within_each_group(self) -> None:
        keys = (np.array([0, 0, 1, 1, 1], dtype=np.int64),)
        values = (np.array([1.0, 2.0, 10.0, 20.0, 30.0]),)
        keep, collapsed = _collapse_duplicates(
            keys=keys, values=values, policy=DuplicatePolicy.SUM, ctx="test"
        )
        assert keep.tolist() == [0, 2]  # group-start indices
        assert collapsed[0].tolist() == [3.0, 60.0]

    def test_sum_policy_sums_multiple_value_columns_independently(self) -> None:
        keys = (np.array([0, 0, 1], dtype=np.int64),)
        values = (
            np.array([1.0, 2.0, 100.0]),
            np.array([10.0, 20.0, 1000.0]),
        )
        _keep, collapsed = _collapse_duplicates(
            keys=keys, values=values, policy=DuplicatePolicy.SUM, ctx="test"
        )
        assert collapsed[0].tolist() == [3.0, 100.0]
        assert collapsed[1].tolist() == [30.0, 1000.0]

    def test_multi_column_keys_only_group_when_every_key_column_matches(
        self,
    ) -> None:
        # rows (0,0) and (0,1) share the first key column but not the second
        keys = (
            np.array([0, 0, 0], dtype=np.int64),
            np.array([0, 1, 1], dtype=np.int64),
        )
        values = (np.array([1.0, 2.0, 3.0]),)
        keep, collapsed = _collapse_duplicates(
            keys=keys, values=values, policy=DuplicatePolicy.SUM, ctx="test"
        )
        assert keep.tolist() == [0, 1]  # (0,0) alone; (0,1) groups rows 1&2
        assert collapsed[0].tolist() == [1.0, 5.0]

    def test_empty_input_returns_empty_output(self) -> None:
        keys = (np.empty(0, dtype=np.int64),)
        values = (np.empty(0),)
        keep, collapsed = _collapse_duplicates(
            keys=keys, values=values, policy=DuplicatePolicy.RAISE, ctx="test"
        )
        assert keep.shape == (0,)
        assert collapsed[0].shape == (0,)
