# tests/sensitivity_pnl/unit/test_batch_contract.py
"""
Unit tests for `riskquant.sensitivity_pnl._batch_contract`.

Every function here is a memory-safety gate for a kernel running under
`boundscheck=False` (`check_2d_batch_source`), or the guard against the LOG-
convention P0 (a non-positive shocked level poisoning a PnL grid with a NaN
that then sorts OUT of a lower-tail VaR — see `require_positive_levels_for_log_
factors`'s own docstring for the full failure chain). These are tested here in
complete isolation from either streaming engine, with fabricated inputs rather
than a compiled book, specifically so a defect in this module cannot hide
behind a coincidentally-correct engine-level test.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.sensitivity_pnl._batch_contract import (
    check_2d_batch_source,
    derive_min_source_rows,
    derive_positive_level_rows,
    require_positive_levels_for_log_factors,
    validate_base_levels_for_ratio_factors,
)
from riskquant.sensitivity_pnl.compile import compile_sensitivity_book
from riskquant.sensitivity_pnl.conventions import (
    DISPLACEMENT_ABSOLUTE_ID,
    DISPLACEMENT_LOG_ID,
    DISPLACEMENT_RELATIVE_ID,
)


def _tiny_book(convention_ids: list[int]):
    """A book with one diagonal entry per factor, factor_id == row index."""
    import polars as pl

    n = len(convention_ids)
    ids = np.arange(n, dtype=np.int32)
    sensitivities = pl.DataFrame(
        {
            "position_id": [1] * n,
            "factor_id": ids.tolist(),
            "first_order": [10.0] * n,
            "second_order": [0.0] * n,
        }
    )
    meta = pl.DataFrame(
        {
            "factor_id": ids,
            "displacement_convention_id": np.array(convention_ids, dtype=np.int8),
            "quote_base": np.ones(n),
        }
    )
    return compile_sensitivity_book(
        sensitivities=sensitivities, factor_meta=meta, level_universe_factor_ids=ids
    )


class TestDeriveMinSourceRows:
    def test_equals_max_level_uix_plus_one(self) -> None:
        book = _tiny_book([int(DISPLACEMENT_ABSOLUTE_ID)] * 3)
        assert derive_min_source_rows(book) == book.max_level_uix + 1

    def test_zero_for_a_book_with_no_active_factors(self) -> None:
        import polars as pl

        empty_sens = pl.DataFrame(
            schema={
                "position_id": pl.Int64,
                "factor_id": pl.Int32,
                "first_order": pl.Float64,
                "second_order": pl.Float64,
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": np.array([0], dtype=np.int32),
                "displacement_convention_id": np.array(
                    [int(DISPLACEMENT_ABSOLUTE_ID)], dtype=np.int8
                ),
                "quote_base": [1.0],
            }
        )
        book = compile_sensitivity_book(
            sensitivities=empty_sens,
            factor_meta=meta,
            level_universe_factor_ids=np.array([0], dtype=np.int32),
            extra_position_uids=np.array([1], dtype=np.int64),
        )
        assert book.num_active_factors == 0
        assert derive_min_source_rows(book) == 0


class TestDerivePositiveLevelRows:
    def test_selects_only_log_factors(self) -> None:
        book = _tiny_book(
            [
                int(DISPLACEMENT_ABSOLUTE_ID),
                int(DISPLACEMENT_RELATIVE_ID),
                int(DISPLACEMENT_LOG_ID),
            ]
        )
        uix, factor_ids = derive_positive_level_rows(book)
        assert uix.tolist() == [2]  # row index of the LOG factor
        assert factor_ids.tolist() == [2]  # factor_id happens to equal row here

    def test_empty_when_no_log_factors(self) -> None:
        book = _tiny_book(
            [int(DISPLACEMENT_ABSOLUTE_ID), int(DISPLACEMENT_RELATIVE_ID)]
        )
        uix, factor_ids = derive_positive_level_rows(book)
        assert uix.shape == (0,)
        assert factor_ids.shape == (0,)

    def test_output_dtype_is_int32(self) -> None:
        book = _tiny_book([int(DISPLACEMENT_LOG_ID)])
        uix, factor_ids = derive_positive_level_rows(book)
        assert uix.dtype == np.int32
        assert factor_ids.dtype == np.int32

    def test_output_arrays_are_c_contiguous(self) -> None:
        book = _tiny_book([int(DISPLACEMENT_LOG_ID), int(DISPLACEMENT_LOG_ID)])
        uix, factor_ids = derive_positive_level_rows(book)
        assert uix.flags.c_contiguous
        assert factor_ids.flags.c_contiguous


class TestValidateBaseLevelsForRatioFactors:
    def _call(
        self, *, base_levels, min_source_rows=1, convention=DISPLACEMENT_RELATIVE_ID
    ):
        book = _tiny_book([int(convention)])
        validate_base_levels_for_ratio_factors(
            book=book, base_levels=base_levels, min_source_rows=min_source_rows
        )

    def test_accepts_a_valid_positive_base_level(self) -> None:
        self._call(base_levels=np.array([100.0]))  # must not raise

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="float64"):
            self._call(base_levels=np.array([100.0], dtype=np.float32))

    def test_rejects_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            self._call(base_levels=np.array([[100.0]]))

    def test_rejects_non_contiguous(self) -> None:
        base = np.array([100.0, 200.0, 300.0])[::2]
        assert not base.flags.c_contiguous
        with pytest.raises(ValueError, match="C-contiguous"):
            self._call(base_levels=base)

    def test_rejects_too_few_rows(self) -> None:
        with pytest.raises(ValueError, match="dereferences row"):
            self._call(base_levels=np.array([100.0]), min_source_rows=2)

    def test_rejects_zero_base_level_for_relative(self) -> None:
        with pytest.raises(ValueError, match="strictly positive base level"):
            self._call(base_levels=np.array([0.0]), convention=DISPLACEMENT_RELATIVE_ID)

    def test_rejects_negative_base_level_for_log(self) -> None:
        with pytest.raises(ValueError, match="strictly positive base level"):
            self._call(base_levels=np.array([-5.0]), convention=DISPLACEMENT_LOG_ID)

    def test_rejects_nan_base_level(self) -> None:
        with pytest.raises(ValueError, match="strictly positive base level"):
            self._call(base_levels=np.array([np.nan]))

    def test_rejects_infinite_base_level(self) -> None:
        with pytest.raises(ValueError, match="strictly positive base level"):
            self._call(base_levels=np.array([np.inf]))

    def test_absolute_convention_never_checked_for_positivity(self) -> None:
        """A zero or negative base level is fine under ABSOLUTE — x = L - L0
        never divides by it."""
        self._call(base_levels=np.array([0.0]), convention=DISPLACEMENT_ABSOLUTE_ID)
        self._call(base_levels=np.array([-5.0]), convention=DISPLACEMENT_ABSOLUTE_ID)

    def test_error_names_the_offending_factor_id(self) -> None:
        with pytest.raises(ValueError, match=r"factor_id\(s\) \[0\]"):
            self._call(base_levels=np.array([0.0]))


class TestRequirePositiveLevelsForLogFactors:
    def _log_uix_and_ids(self, n: int = 1):
        return (
            np.arange(n, dtype=np.int32),
            np.arange(n, dtype=np.int32),
        )

    def test_no_rows_means_no_check_at_all(self) -> None:
        """Zero LOG factors: must not even look at `shocked_levels`."""
        require_positive_levels_for_log_factors(
            positive_level_uix=np.empty(0, dtype=np.int32),
            positive_level_factor_ids=np.empty(0, dtype=np.int32),
            shocked_levels=np.array([[-1.0]]),  # would fail if actually checked
            batch_len=1,
            levels_offset=0,
        )

    def test_accepts_all_positive_levels(self) -> None:
        uix, ids = self._log_uix_and_ids(1)
        require_positive_levels_for_log_factors(
            positive_level_uix=uix,
            positive_level_factor_ids=ids,
            shocked_levels=np.array([[1.0, 2.0, 3.0]]),
            batch_len=3,
            levels_offset=0,
        )

    def test_rejects_exact_zero(self) -> None:
        uix, ids = self._log_uix_and_ids(1)
        with pytest.raises(ValueError, match="strictly positive shocked level"):
            require_positive_levels_for_log_factors(
                positive_level_uix=uix,
                positive_level_factor_ids=ids,
                shocked_levels=np.array([[1.0, 0.0, 3.0]]),
                batch_len=3,
                levels_offset=0,
            )

    def test_rejects_negative(self) -> None:
        uix, ids = self._log_uix_and_ids(1)
        with pytest.raises(ValueError, match="strictly positive shocked level"):
            require_positive_levels_for_log_factors(
                positive_level_uix=uix,
                positive_level_factor_ids=ids,
                shocked_levels=np.array([[1.0, -0.5, 3.0]]),
                batch_len=3,
                levels_offset=0,
            )

    def test_respects_batch_len_and_levels_offset_window(self) -> None:
        """A non-positive value OUTSIDE the [offset, offset+batch_len) window
        must not trip the check — the batch legitimately does not see it."""
        uix, ids = self._log_uix_and_ids(1)
        shocked = np.array([[1.0, 1.0, -99.0, 1.0]])  # poison at column 2
        require_positive_levels_for_log_factors(
            positive_level_uix=uix,
            positive_level_factor_ids=ids,
            shocked_levels=shocked,
            batch_len=2,
            levels_offset=0,  # window is columns [0, 2) -- poison at 2 is outside
        )
        with pytest.raises(ValueError, match="strictly positive shocked level"):
            require_positive_levels_for_log_factors(
                positive_level_uix=uix,
                positive_level_factor_ids=ids,
                shocked_levels=shocked,
                batch_len=2,
                levels_offset=1,  # window is columns [1, 3) -- now it is in range
            )

    def test_checks_every_log_row_not_just_the_first(self) -> None:
        uix = np.array([0, 1, 2], dtype=np.int32)
        ids = np.array([100, 101, 102], dtype=np.int32)
        shocked = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, -1.0],  # only the THIRD row is poisoned
            ]
        )
        with pytest.raises(ValueError, match="factor_id 102"):
            require_positive_levels_for_log_factors(
                positive_level_uix=uix,
                positive_level_factor_ids=ids,
                shocked_levels=shocked,
                batch_len=2,
                levels_offset=0,
            )

    def test_error_names_the_worst_offending_factor_when_several_are_bad(
        self,
    ) -> None:
        uix = np.array([0, 1], dtype=np.int32)
        ids = np.array([7, 8], dtype=np.int32)
        # factor_id 7's row dips to -1.0, factor_id 8's row dips to -5.0 (worse)
        shocked = np.array([[-1.0, 1.0], [-5.0, 1.0]])
        with pytest.raises(ValueError, match="factor_id 8"):
            require_positive_levels_for_log_factors(
                positive_level_uix=uix,
                positive_level_factor_ids=ids,
                shocked_levels=shocked,
                batch_len=2,
                levels_offset=0,
            )

    def test_error_message_reports_the_actual_offending_value(self) -> None:
        uix, ids = self._log_uix_and_ids(1)
        with pytest.raises(ValueError, match=r"-2\.5"):
            require_positive_levels_for_log_factors(
                positive_level_uix=uix,
                positive_level_factor_ids=ids,
                shocked_levels=np.array([[-2.5]]),
                batch_len=1,
                levels_offset=0,
            )


class TestCheck2dBatchSource:
    def _good_kwargs(self, **overrides):
        kwargs = {
            "source": np.zeros((3, 10), dtype=np.float32),
            "dtype": np.float32,
            "name": "x",
            "min_source_rows": 3,
            "batch_len": 5,
            "batch_size": 5,
            "source_offset": 0,
            "scenario_offset": 0,
            "out_num_scenarios": 10,
        }
        kwargs.update(overrides)
        return kwargs

    def test_accepts_a_well_formed_source(self) -> None:
        check_2d_batch_source(**self._good_kwargs())  # must not raise

    def test_rejects_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            check_2d_batch_source(
                **self._good_kwargs(source=np.zeros(10, dtype=np.float32))
            )

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="must be float32"):
            check_2d_batch_source(
                **self._good_kwargs(source=np.zeros((3, 10), dtype=np.float64))
            )

    def test_dtype_message_names_the_requested_dtype_not_a_hardcoded_one(self) -> None:
        with pytest.raises(ValueError, match="must be float64"):
            check_2d_batch_source(
                **self._good_kwargs(
                    source=np.zeros((3, 10), dtype=np.float32), dtype=np.float64
                )
            )

    def test_rejects_non_contiguous(self) -> None:
        source = np.zeros((3, 20), dtype=np.float32)[:, ::2]
        assert not source.flags.c_contiguous
        with pytest.raises(ValueError, match="C-contiguous"):
            check_2d_batch_source(
                **self._good_kwargs(source=source, out_num_scenarios=10)
            )

    def test_rejects_too_few_factor_rows(self) -> None:
        with pytest.raises(ValueError, match="factor rows"):
            check_2d_batch_source(
                **self._good_kwargs(
                    source=np.zeros((2, 10), dtype=np.float32), min_source_rows=3
                )
            )

    @pytest.mark.parametrize("batch_len", [0, -1, 6])
    def test_rejects_batch_len_outside_valid_range(self, batch_len: int) -> None:
        with pytest.raises(ValueError, match="batch_len must lie in"):
            check_2d_batch_source(**self._good_kwargs(batch_len=batch_len))

    def test_accepts_batch_len_equal_to_batch_size(self) -> None:
        check_2d_batch_source(**self._good_kwargs(batch_len=5, batch_size=5))

    def test_rejects_negative_source_offset(self) -> None:
        with pytest.raises(ValueError, match="scenario axis"):
            check_2d_batch_source(**self._good_kwargs(source_offset=-1))

    def test_rejects_source_offset_plus_batch_len_overrunning_source(self) -> None:
        with pytest.raises(ValueError, match="scenario axis"):
            check_2d_batch_source(
                **self._good_kwargs(source_offset=8, batch_len=5, batch_size=5)
            )

    def test_accepts_source_offset_landing_exactly_at_the_end(self) -> None:
        check_2d_batch_source(**self._good_kwargs(source_offset=5, batch_len=5))

    def test_rejects_negative_scenario_offset(self) -> None:
        with pytest.raises(ValueError, match="out scenario axis"):
            check_2d_batch_source(**self._good_kwargs(scenario_offset=-1))

    def test_rejects_scenario_offset_plus_batch_len_overrunning_out(self) -> None:
        with pytest.raises(ValueError, match="out scenario axis"):
            check_2d_batch_source(
                **self._good_kwargs(
                    scenario_offset=8, batch_len=5, out_num_scenarios=10
                )
            )

    def test_accepts_scenario_offset_landing_exactly_at_the_end(self) -> None:
        check_2d_batch_source(
            **self._good_kwargs(scenario_offset=5, batch_len=5, out_num_scenarios=10)
        )

    def test_error_message_includes_the_caller_supplied_name(self) -> None:
        with pytest.raises(ValueError, match="my_special_buffer"):
            check_2d_batch_source(
                **self._good_kwargs(
                    name="my_special_buffer", source=np.zeros(10, dtype=np.float32)
                )
            )
