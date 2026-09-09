# tests/sensitivity_pnl/unit/test_contracts.py
"""
Unit tests for `riskquant.sensitivity_pnl.contracts`.

`CompiledSensitivityBook.__post_init__` is described in its own module
docstring as "a memory-safety boundary, not a courtesy": the kernels run under
`boundscheck=False`, so every index this validator lets through is an index a
kernel will dereference without a second check. These tests build a
`CompiledSensitivityBook` DIRECTLY (never through the compiler), one field
mutated per test, so each individual validation branch is pinned in isolation
rather than only reachable through whatever malformed inputs the compiler
happens to be able to produce.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook


def _valid_fields() -> dict[str, np.ndarray]:
    """
    A minimal, fully valid book: 2 positions, 4 active factors, one diagonal
    entry per position, two cross entries (so every array below has at least
    2 elements -- needed for the contiguity tests, which build a
    genuinely-strided view by taking every other element of a padded array,
    a construction that degenerates to "trivially contiguous" for a
    single-element array).

        position 0 (uid 10): factor slot 0                                -> nnz row [0, 1)
        position 1 (uid 20): factor slot 1, cross(slot 0,2), cross(slot 1,3) -> nnz row [1, 2)
    """
    return {
        "position_uids": np.array([10, 20], dtype=np.int64),
        "active_factor_ids": np.array([100, 200, 300, 400], dtype=np.int32),
        "active_level_uix": np.array([0, 1, 2, 3], dtype=np.int32),
        "displacement_convention_ids": np.array([1, 1, 1, 1], dtype=np.int8),
        "position_offsets": np.array([0, 1, 2], dtype=np.int32),
        "factor_uix": np.array([0, 1], dtype=np.int32),
        "first_order": np.array([5.0, 6.0], dtype=np.float64),
        "second_order": np.array([0.0, 0.0], dtype=np.float64),
        "cross_offsets": np.array([0, 0, 2], dtype=np.int32),
        "cross_factor_uix_i": np.array([0, 1], dtype=np.int32),
        "cross_factor_uix_j": np.array([2, 3], dtype=np.int32),
        "cross_second_order": np.array([1.5, 2.5], dtype=np.float64),
    }


def _book(**overrides) -> CompiledSensitivityBook:
    fields = _valid_fields()
    fields.update(overrides)
    return CompiledSensitivityBook(**fields)


def test_the_valid_fixture_itself_constructs_without_error() -> None:
    """Every other test in this file mutates ONE field away from this baseline
    -- if the baseline itself were invalid, every rejection test below would
    be meaningless (rejected for the wrong reason)."""
    book = _book()
    assert book.num_positions == 2
    assert book.num_active_factors == 4
    assert book.nnz == 2
    assert book.nnz_cross == 2


# =========================================================================
# DTYPES
# =========================================================================


class TestDtypeValidation:
    def test_position_uids_must_be_int64(self) -> None:
        with pytest.raises(ValueError, match="position_uids must be int64"):
            _book(position_uids=np.array([10, 20], dtype=np.int32))

    def test_active_factor_ids_must_be_int32(self) -> None:
        with pytest.raises(ValueError, match="active-factor axis must be int32"):
            _book(active_factor_ids=np.array([100, 200, 300], dtype=np.int64))

    def test_active_level_uix_must_be_int32(self) -> None:
        with pytest.raises(ValueError, match="active-factor axis must be int32"):
            _book(active_level_uix=np.array([0, 1, 2], dtype=np.int64))

    def test_displacement_convention_ids_must_be_int8(self) -> None:
        with pytest.raises(
            ValueError, match="displacement_convention_ids must be int8"
        ):
            _book(displacement_convention_ids=np.array([1, 1, 1], dtype=np.int32))

    def test_position_offsets_must_be_int32(self) -> None:
        with pytest.raises(ValueError, match="CSR offsets must be int32"):
            _book(position_offsets=np.array([0, 1, 2], dtype=np.int64))

    def test_cross_offsets_must_be_int32(self) -> None:
        with pytest.raises(ValueError, match="CSR offsets must be int32"):
            _book(cross_offsets=np.array([0, 1, 1], dtype=np.int64))

    def test_factor_uix_must_be_int32(self) -> None:
        with pytest.raises(ValueError, match="factor index arrays must be int32"):
            _book(factor_uix=np.array([0, 1], dtype=np.int64))

    def test_cross_factor_uix_i_must_be_int32(self) -> None:
        with pytest.raises(ValueError, match="factor index arrays must be int32"):
            _book(cross_factor_uix_i=np.array([0], dtype=np.int64))

    def test_cross_factor_uix_j_must_be_int32(self) -> None:
        with pytest.raises(ValueError, match="factor index arrays must be int32"):
            _book(cross_factor_uix_j=np.array([2], dtype=np.int64))

    def test_first_order_must_be_float64(self) -> None:
        with pytest.raises(ValueError, match="coefficient arrays must be float64"):
            _book(first_order=np.array([5.0, 6.0], dtype=np.float32))

    def test_second_order_must_be_float64(self) -> None:
        with pytest.raises(ValueError, match="coefficient arrays must be float64"):
            _book(second_order=np.array([0.0, 0.0], dtype=np.float32))

    def test_cross_second_order_must_be_float64(self) -> None:
        with pytest.raises(ValueError, match="coefficient arrays must be float64"):
            _book(cross_second_order=np.array([1.5], dtype=np.float32))


# =========================================================================
# SHAPES
# =========================================================================


class TestShapeValidation:
    def test_active_level_uix_length_must_match_active_factor_ids(self) -> None:
        with pytest.raises(
            ValueError, match="active-factor axis arrays are misaligned"
        ):
            _book(active_level_uix=np.array([0, 1], dtype=np.int32))

    def test_displacement_convention_ids_length_must_match_active_factor_ids(
        self,
    ) -> None:
        with pytest.raises(
            ValueError, match="active-factor axis arrays are misaligned"
        ):
            _book(displacement_convention_ids=np.array([1, 1], dtype=np.int8))

    def test_first_order_length_must_match_nnz(self) -> None:
        with pytest.raises(ValueError, match="diagonal entry arrays are misaligned"):
            _book(first_order=np.array([5.0], dtype=np.float64))

    def test_second_order_length_must_match_nnz(self) -> None:
        with pytest.raises(ValueError, match="diagonal entry arrays are misaligned"):
            _book(second_order=np.array([0.0], dtype=np.float64))

    def test_cross_factor_uix_j_length_must_match_nnzx(self) -> None:
        with pytest.raises(ValueError, match="cross entry arrays are misaligned"):
            _book(cross_factor_uix_j=np.array([2], dtype=np.int32))

    def test_cross_second_order_length_must_match_nnzx(self) -> None:
        with pytest.raises(ValueError, match="cross entry arrays are misaligned"):
            _book(cross_second_order=np.array([1.5], dtype=np.float64))

    def test_position_offsets_wrong_length_is_caught_by_csr_validation(self) -> None:
        """position_offsets must have length M+1; a wrong length is reported
        by the CSR-structure check, not the alignment check above -- pinned
        here so a refactor cannot silently drop this branch entirely."""
        with pytest.raises(ValueError, match=r"offsets must have length M\+1"):
            _book(position_offsets=np.array([0, 1], dtype=np.int32))


# =========================================================================
# CONTIGUITY
# =========================================================================


class TestContiguityValidation:
    @pytest.mark.parametrize(
        "field_name",
        [
            "active_factor_ids",
            "active_level_uix",
            "displacement_convention_ids",
            "position_offsets",
            "factor_uix",
            "first_order",
            "second_order",
            "cross_offsets",
            "cross_factor_uix_i",
            "cross_factor_uix_j",
            "cross_second_order",
        ],
    )
    def test_every_array_must_be_c_contiguous(self, field_name: str) -> None:
        fields = _valid_fields()
        array = fields[field_name]
        # Build a non-contiguous view of the same logical values by padding
        # and striding, preserving dtype and values exactly.
        padded = np.empty(array.shape[0] * 2, dtype=array.dtype)
        padded[::2] = array
        non_contiguous = padded[::2]
        assert non_contiguous.shape == array.shape
        assert not non_contiguous.flags.c_contiguous
        fields[field_name] = non_contiguous
        with pytest.raises(ValueError, match=f"{field_name} must be C-contiguous"):
            CompiledSensitivityBook(**fields)


# =========================================================================
# CSR STRUCTURE (the memory-safety boundary)
# =========================================================================


class TestCsrStructureValidation:
    def test_position_offsets_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="diagonal offsets must start at 0"):
            _book(position_offsets=np.array([1, 1, 2], dtype=np.int32))

    def test_cross_offsets_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="cross offsets must start at 0"):
            _book(cross_offsets=np.array([1, 1, 1], dtype=np.int32))

    def test_position_offsets_must_be_non_decreasing(self) -> None:
        """[0, 5, 0] terminates at the right NNZ by coincidence but would make
        the kernel walk range(0, 5) over a 2-entry array -- this is the exact
        failure mode the module docstring calls out."""
        with pytest.raises(ValueError, match="diagonal offsets must be non-decreasing"):
            _book(
                position_offsets=np.array([0, 2, 1], dtype=np.int32),
                first_order=np.array([5.0, 6.0], dtype=np.float64),
                second_order=np.array([0.0, 0.0], dtype=np.float64),
                factor_uix=np.array([0, 1], dtype=np.int32),
            )

    def test_position_offsets_must_terminate_at_nnz(self) -> None:
        with pytest.raises(ValueError, match="diagonal offsets terminate at"):
            _book(position_offsets=np.array([0, 1, 3], dtype=np.int32))

    def test_cross_offsets_must_terminate_at_nnzx(self) -> None:
        with pytest.raises(ValueError, match="cross offsets terminate at"):
            _book(cross_offsets=np.array([0, 0, 0], dtype=np.int32))

    def test_zero_positions_with_nonzero_diagonal_nnz_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="diagonal block has entries but no rows"):
            _book(
                position_uids=np.empty(0, dtype=np.int64),
                active_factor_ids=np.empty(0, dtype=np.int32),
                active_level_uix=np.empty(0, dtype=np.int32),
                displacement_convention_ids=np.empty(0, dtype=np.int8),
                position_offsets=np.array([0], dtype=np.int32),
                cross_offsets=np.array([0], dtype=np.int32),
                cross_factor_uix_i=np.empty(0, dtype=np.int32),
                cross_factor_uix_j=np.empty(0, dtype=np.int32),
                cross_second_order=np.empty(0, dtype=np.float64),
                # first_order/second_order/factor_uix left non-empty on purpose
            )

    def test_zero_positions_and_zero_entries_is_accepted(self) -> None:
        book = _book(
            position_uids=np.empty(0, dtype=np.int64),
            active_factor_ids=np.empty(0, dtype=np.int32),
            active_level_uix=np.empty(0, dtype=np.int32),
            displacement_convention_ids=np.empty(0, dtype=np.int8),
            position_offsets=np.array([0], dtype=np.int32),
            factor_uix=np.empty(0, dtype=np.int32),
            first_order=np.empty(0, dtype=np.float64),
            second_order=np.empty(0, dtype=np.float64),
            cross_offsets=np.array([0], dtype=np.int32),
            cross_factor_uix_i=np.empty(0, dtype=np.int32),
            cross_factor_uix_j=np.empty(0, dtype=np.int32),
            cross_second_order=np.empty(0, dtype=np.float64),
        )
        assert book.num_positions == 0
        assert book.nnz == 0


# =========================================================================
# INDEX RANGE (the actual memory-safety guard)
# =========================================================================


class TestIndexRangeValidation:
    def test_factor_uix_out_of_range_is_rejected_even_with_nonzero_nnz(self) -> None:
        with pytest.raises(ValueError, match="factor_uix out of range"):
            _book(factor_uix=np.array([0, 99], dtype=np.int32))

    def test_factor_uix_negative_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="factor_uix out of range"):
            _book(factor_uix=np.array([-1, 1], dtype=np.int32))

    def test_cross_factor_uix_i_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cross_factor_uix_i out of range"):
            _book(
                cross_factor_uix_i=np.array([99, 1], dtype=np.int32),
                cross_factor_uix_j=np.array([2, 3], dtype=np.int32),
            )

    def test_cross_factor_uix_j_out_of_range_is_rejected(self) -> None:
        """
        Regression pin: earlier this exact check was skipped whenever nnz == 0
        (a cross-only book), because the range check used to short-circuit on
        an empty DIAGONAL block rather than checking the cross block on its
        own terms. `_require_index_range` here must fire regardless of nnz.
        """
        with pytest.raises(ValueError, match="cross_factor_uix_j out of range"):
            _book(
                position_offsets=np.array([0, 0, 0], dtype=np.int32),
                factor_uix=np.empty(0, dtype=np.int32),
                first_order=np.empty(0, dtype=np.float64),
                second_order=np.empty(0, dtype=np.float64),
                cross_offsets=np.array([0, 0, 1], dtype=np.int32),
                cross_factor_uix_i=np.array([0], dtype=np.int32),
                cross_factor_uix_j=np.array([9999], dtype=np.int32),
                cross_second_order=np.array([1.5], dtype=np.float64),
            )

    def test_active_level_uix_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="active_level_uix must be non-negative"):
            _book(active_level_uix=np.array([0, -1, 2, 3], dtype=np.int32))

    def test_active_level_uix_has_no_upper_bound_check_here(self) -> None:
        """
        Deliberately NOT checked in this module: the upper bound depends on
        the caller's levels/displacement matrix, which this class does not
        hold. A large active_level_uix value is valid at construction and is
        the engine's responsibility to bound against the actual buffer size.
        """
        book = _book(active_level_uix=np.array([0, 1, 2, 10_000_000], dtype=np.int32))
        assert int(book.active_level_uix.max()) == 10_000_000


# =========================================================================
# SEMANTICS
# =========================================================================


class TestSemanticValidation:
    def test_unknown_displacement_convention_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown displacement_convention_id"):
            _book(displacement_convention_ids=np.array([1, 1, 1, 99], dtype=np.int8))

    def test_zero_is_not_a_valid_convention_id(self) -> None:
        with pytest.raises(ValueError, match="unknown displacement_convention_id"):
            _book(displacement_convention_ids=np.array([0, 1, 1, 1], dtype=np.int8))

    def test_cross_pair_with_i_equal_to_j_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="i < j"):
            _book(
                cross_factor_uix_i=np.array([1, 1], dtype=np.int32),
                cross_factor_uix_j=np.array([1, 3], dtype=np.int32),
            )

    def test_cross_pair_with_i_greater_than_j_is_rejected(self) -> None:
        """Canonicalisation (i < j) is the compiler's job; the contract
        rejects an un-canonicalised pair rather than silently accepting it."""
        with pytest.raises(ValueError, match="i < j"):
            _book(
                cross_factor_uix_i=np.array([2, 1], dtype=np.int32),
                cross_factor_uix_j=np.array([0, 3], dtype=np.int32),
            )

    def test_no_cross_entries_skips_the_i_less_than_j_check_entirely(self) -> None:
        book = _book(
            position_offsets=np.array([0, 1, 2], dtype=np.int32),
            cross_offsets=np.array([0, 0, 0], dtype=np.int32),
            cross_factor_uix_i=np.empty(0, dtype=np.int32),
            cross_factor_uix_j=np.empty(0, dtype=np.int32),
            cross_second_order=np.empty(0, dtype=np.float64),
        )
        assert book.nnz_cross == 0


# =========================================================================
# PROPERTIES
# =========================================================================


class TestProperties:
    def test_num_positions(self) -> None:
        assert _book().num_positions == 2

    def test_num_active_factors(self) -> None:
        assert _book().num_active_factors == 4

    def test_nnz(self) -> None:
        assert _book().nnz == 2

    def test_nnz_cross(self) -> None:
        assert _book().nnz_cross == 2

    def test_max_level_uix_is_the_maximum_active_level_uix(self) -> None:
        book = _book(active_level_uix=np.array([0, 7, 3, 2], dtype=np.int32))
        assert book.max_level_uix == 7

    def test_max_level_uix_is_minus_one_with_no_active_factors(self) -> None:
        book = _book(
            active_factor_ids=np.empty(0, dtype=np.int32),
            active_level_uix=np.empty(0, dtype=np.int32),
            displacement_convention_ids=np.empty(0, dtype=np.int8),
            position_offsets=np.array([0, 0, 0], dtype=np.int32),
            factor_uix=np.empty(0, dtype=np.int32),
            first_order=np.empty(0, dtype=np.float64),
            second_order=np.empty(0, dtype=np.float64),
            cross_offsets=np.array([0, 0, 0], dtype=np.int32),
            cross_factor_uix_i=np.empty(0, dtype=np.int32),
            cross_factor_uix_j=np.empty(0, dtype=np.int32),
            cross_second_order=np.empty(0, dtype=np.float64),
        )
        assert book.max_level_uix == -1  # unaffected by the cross-count change

    def test_the_book_is_frozen(self) -> None:
        book = _book()
        with pytest.raises((AttributeError, Exception)):
            book.position_uids = np.array([1, 2], dtype=np.int64)  # type: ignore[misc]

    def test_replace_produces_an_independently_valid_book(self) -> None:
        """dataclasses.replace should re-run __post_init__ and therefore still
        catch a defect introduced via the replacement."""
        book = _book()
        with pytest.raises(ValueError, match="position_uids must be int64"):
            replace(book, position_uids=np.array([10, 20], dtype=np.int32))
