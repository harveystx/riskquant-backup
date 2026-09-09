# tests/solutions/sensitivity_pnl/unit/test_market.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.market`.

This module exists to make one specific failure unrepresentable: a `(K, S)`
scenario matrix whose row order silently disagrees with the factor universe a
book was compiled against. The module's own docstring cites a measured
consequence — a three-factor book given its shock rows in reverse order
returned 123 instead of 321, with no warning. Every test below is either
pinning that the axis/matrix pairing genuinely cannot be expressed apart, or
exercising one of the two key encodings (DIRECT vs SLOT) and their
diagnostic-only difference.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError
from riskquant.solutions.sensitivity_pnl.market import (
    FactorAxis,
    ScenarioLevels,
    ScenarioShocks,
)

# =========================================================================
# FactorAxis.from_keys -- encoding selection
# =========================================================================


class TestFactorAxisEncodingSelection:
    def test_sorted_unique_integers_use_direct_encoding(self) -> None:
        axis = FactorAxis.from_keys([10, 20, 30])
        assert axis.slot_map is None
        assert axis.universe_ids.tolist() == [10, 20, 30]

    def test_unsorted_integers_use_slot_encoding(self) -> None:
        axis = FactorAxis.from_keys([30, 10, 20])
        assert axis.slot_map is not None
        assert axis.universe_ids.tolist() == [0, 1, 2]

    def test_string_keys_always_use_slot_encoding(self) -> None:
        axis = FactorAxis.from_keys(["EQ_A", "EQ_B"])
        assert axis.slot_map is not None
        assert axis.universe_ids.tolist() == [0, 1]

    def test_slot_map_maps_each_key_to_its_row_position(self) -> None:
        axis = FactorAxis.from_keys(["EQ_C", "EQ_A", "EQ_B"])
        assert axis.slot_map == {"EQ_C": 0, "EQ_A": 1, "EQ_B": 2}

    def test_single_key_uses_direct_encoding(self) -> None:
        axis = FactorAxis.from_keys([42])
        assert axis.slot_map is None

    def test_universe_ids_dtype_is_always_int32(self) -> None:
        assert FactorAxis.from_keys([1, 2, 3]).universe_ids.dtype == np.int32
        assert FactorAxis.from_keys(["a", "b"]).universe_ids.dtype == np.int32

    def test_accepts_a_polars_series_directly(self) -> None:
        axis = FactorAxis.from_keys(pl.Series("factor_id", [1, 2, 3]))
        assert axis.num_factors == 3

    def test_accepts_a_numpy_array(self) -> None:
        axis = FactorAxis.from_keys(np.array([5, 6, 7], dtype=np.int64))
        assert axis.num_factors == 3


# =========================================================================
# FactorAxis.from_keys -- validation
# =========================================================================


class TestFactorAxisKeyValidation:
    def test_rejects_an_empty_key_set(self) -> None:
        with pytest.raises(SensitivityPnlInputError, match="empty"):
            FactorAxis.from_keys([])

    def test_rejects_null_keys(self) -> None:
        series = pl.Series("factor_id", [1, None, 3])
        with pytest.raises(SensitivityPnlInputError, match="null factor key"):
            FactorAxis.from_keys(series)

    def test_rejects_duplicate_keys(self) -> None:
        with pytest.raises(SensitivityPnlInputError, match="duplicate factor key"):
            FactorAxis.from_keys([1, 2, 2, 3])

    def test_rejects_float_keys(self) -> None:
        with pytest.raises(SensitivityPnlInputError, match="dtype"):
            FactorAxis.from_keys([1.5, 2.5])

    def test_rejects_boolean_keys(self) -> None:
        with pytest.raises(SensitivityPnlInputError):
            FactorAxis.from_keys(pl.Series("factor_id", [True, False]))

    def test_multiple_key_defects_are_reported_together(self) -> None:
        series = pl.Series("factor_id", [1, 1, None])
        with pytest.raises(SensitivityPnlInputError) as excinfo:
            FactorAxis.from_keys(series)
        sources = {d.source for d in excinfo.value.defects}
        assert len(excinfo.value.defects) >= 2
        assert sources == {"market.factor_keys"}

    def test_duplicate_error_names_the_row_indices(self) -> None:
        with pytest.raises(SensitivityPnlInputError) as excinfo:
            FactorAxis.from_keys([1, 2, 2, 3, 3])
        defect = excinfo.value.defects[0]
        assert defect.total_rows == 4


# =========================================================================
# FactorAxis properties and methods
# =========================================================================


class TestFactorAxisProperties:
    def test_num_factors(self) -> None:
        assert FactorAxis.from_keys([1, 2, 3]).num_factors == 3

    def test_matches_is_true_for_the_same_keys_in_the_same_order(self) -> None:
        a = FactorAxis.from_keys(["X", "Y"])
        b = FactorAxis.from_keys(["X", "Y"])
        assert a.matches(b)

    def test_matches_compares_by_value_not_identity(self) -> None:
        a = FactorAxis.from_keys([1, 2, 3])
        b = FactorAxis.from_keys([1, 2, 3])
        assert a is not b
        assert a.matches(b)

    def test_matches_is_false_for_a_reordered_axis(self) -> None:
        a = FactorAxis.from_keys(["X", "Y"])
        b = FactorAxis.from_keys(["Y", "X"])
        assert not a.matches(b)

    def test_matches_is_false_for_a_different_width(self) -> None:
        a = FactorAxis.from_keys([1, 2])
        b = FactorAxis.from_keys([1, 2, 3])
        assert not a.matches(b)

    def test_matches_is_false_for_different_keys_of_the_same_length(self) -> None:
        a = FactorAxis.from_keys([1, 2, 3])
        b = FactorAxis.from_keys([1, 2, 4])
        assert not a.matches(b)


class TestFactorAxisRelabel:
    def test_direct_encoding_leaves_the_frame_untouched(self) -> None:
        axis = FactorAxis.from_keys([10, 20, 30])
        frame = pl.DataFrame({"factor_id": [10, 20], "value": [1.0, 2.0]})
        result = axis.relabel(frame, ("factor_id",))
        assert result["factor_id"].to_list() == [10, 20]

    def test_slot_encoding_rewrites_the_key_column_to_row_positions(self) -> None:
        axis = FactorAxis.from_keys(["B", "A", "C"])
        frame = pl.DataFrame({"factor_id": ["A", "C", "B"], "value": [1.0, 2.0, 3.0]})
        result = axis.relabel(frame, ("factor_id",))
        assert result["factor_id"].to_list() == [1, 2, 0]

    def test_relabel_only_touches_columns_present_in_the_frame(self) -> None:
        axis = FactorAxis.from_keys(["A", "B"])
        frame = pl.DataFrame({"factor_id_i": ["A"], "factor_id_j": ["B"]})
        result = axis.relabel(frame, ("factor_id", "factor_id_i", "factor_id_j"))
        assert result["factor_id_i"].to_list() == [0]
        assert result["factor_id_j"].to_list() == [1]

    def test_relabel_with_no_matching_columns_returns_the_frame_unchanged(
        self,
    ) -> None:
        axis = FactorAxis.from_keys(["A", "B"])
        frame = pl.DataFrame({"other_column": [1, 2]})
        result = axis.relabel(frame, ("factor_id",))
        assert result.columns == ["other_column"]

    def test_relabel_output_column_dtype_is_int32(self) -> None:
        axis = FactorAxis.from_keys(["A", "B"])
        frame = pl.DataFrame({"factor_id": ["A", "B"]})
        result = axis.relabel(frame, ("factor_id",))
        assert result.schema["factor_id"] == pl.Int32


class TestFactorAxisMarketFactories:
    def test_shocks_binds_a_displacement_matrix(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        market = axis.shocks(np.zeros((2, 5)))
        assert isinstance(market, ScenarioShocks)
        assert market.axis is axis

    def test_shocks_converts_to_float32_c_contiguous(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        market = axis.shocks(np.zeros((2, 5), dtype=np.float64))
        assert market.displacement.dtype == np.float32
        assert market.displacement.flags.c_contiguous

    def test_levels_binds_base_and_shocked_levels(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        market = axis.levels(
            base_levels=np.array([1.0, 2.0]),
            shocked_levels=np.zeros((2, 3)),
        )
        assert isinstance(market, ScenarioLevels)
        assert market.axis is axis

    def test_levels_converts_to_float64_c_contiguous(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        market = axis.levels(
            base_levels=np.array([1.0, 2.0], dtype=np.float32),
            shocked_levels=np.zeros((2, 3), dtype=np.float32),
        )
        assert market.base_levels.dtype == np.float64
        assert market.shocked_levels.dtype == np.float64
        assert market.shocked_levels.flags.c_contiguous


# =========================================================================
# ScenarioShocks
# =========================================================================


class TestScenarioShocksValidation:
    def test_valid_construction(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        market = axis.shocks(np.zeros((2, 3), dtype=np.float32))
        assert market.num_scenarios == 3

    def test_rejects_1d_matrix(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="2-D"):
            ScenarioShocks(axis=axis, displacement=np.zeros(5, dtype=np.float32))

    def test_rejects_wrong_row_count(self) -> None:
        axis = FactorAxis.from_keys([1, 2, 3])
        with pytest.raises(SensitivityPnlInputError, match="rows against"):
            ScenarioShocks(axis=axis, displacement=np.zeros((2, 5), dtype=np.float32))

    def test_rejects_zero_scenarios(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="zero scenarios"):
            ScenarioShocks(axis=axis, displacement=np.zeros((2, 0), dtype=np.float32))

    def test_rejects_non_float32_dtype(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="dtype"):
            ScenarioShocks(axis=axis, displacement=np.zeros((2, 3), dtype=np.float64))

    def test_rejects_non_contiguous_matrix(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        matrix = np.zeros((2, 10), dtype=np.float32)[:, ::2]
        assert not matrix.flags.c_contiguous
        with pytest.raises(SensitivityPnlInputError, match="c_contiguous"):
            ScenarioShocks(axis=axis, displacement=matrix)

    def test_rejects_nan(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        matrix = np.zeros((2, 3), dtype=np.float32)
        matrix[0, 1] = np.nan
        with pytest.raises(SensitivityPnlInputError, match="non-finite"):
            ScenarioShocks(axis=axis, displacement=matrix)

    def test_rejects_inf(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        matrix = np.zeros((2, 3), dtype=np.float32)
        matrix[1, 0] = np.inf
        with pytest.raises(SensitivityPnlInputError, match="non-finite"):
            ScenarioShocks(axis=axis, displacement=matrix)

    def test_num_scenarios_is_the_column_count(self) -> None:
        axis = FactorAxis.from_keys([1])
        market = axis.shocks(np.zeros((1, 17), dtype=np.float32))
        assert market.num_scenarios == 17


class TestScenarioShocksConstructors:
    def test_from_matrix_builds_the_axis_and_binds_in_one_step(self) -> None:
        market = ScenarioShocks.from_matrix(
            factor_keys=["A", "B"], displacement=np.zeros((2, 4))
        )
        assert market.axis.num_factors == 2
        assert market.num_scenarios == 4

    def test_from_frame_treats_every_non_key_column_as_a_scenario(self) -> None:
        frame = pl.DataFrame({"factor_id": [1, 2], "s0": [0.1, 0.2], "s1": [0.3, 0.4]})
        market = ScenarioShocks.from_frame(frame)
        assert market.num_scenarios == 2
        assert market.axis.num_factors == 2

    def test_from_frame_preserves_column_order_as_scenario_order(self) -> None:
        frame = pl.DataFrame(
            {"factor_id": [1], "z_scenario": [9.0], "a_scenario": [1.0]}
        )
        market = ScenarioShocks.from_frame(frame)
        assert market.displacement[0].tolist() == pytest.approx([9.0, 1.0])

    def test_from_frame_rejects_missing_key_column(self) -> None:
        frame = pl.DataFrame({"not_factor_id": [1], "s0": [0.1]})
        with pytest.raises(SensitivityPnlInputError, match="factor_id"):
            ScenarioShocks.from_frame(frame)

    def test_from_frame_rejects_no_scenario_columns(self) -> None:
        frame = pl.DataFrame({"factor_id": [1, 2]})
        with pytest.raises(SensitivityPnlInputError, match="no scenario columns"):
            ScenarioShocks.from_frame(frame)

    def test_from_frame_honours_custom_factor_key_column_name(self) -> None:
        frame = pl.DataFrame({"my_key": [1, 2], "s0": [0.1, 0.2]})
        market = ScenarioShocks.from_frame(frame, factor_key_column="my_key")
        assert market.axis.num_factors == 2


# =========================================================================
# ScenarioLevels
# =========================================================================


class TestScenarioLevelsValidation:
    def test_valid_construction(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        market = axis.levels(
            base_levels=np.array([100.0, 200.0]), shocked_levels=np.zeros((2, 3))
        )
        assert market.num_scenarios == 3

    def test_rejects_base_levels_wrong_shape(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="base_levels"):
            ScenarioLevels(
                axis=axis,
                base_levels=np.array([100.0], dtype=np.float64),
                shocked_levels=np.ascontiguousarray(np.zeros((2, 3))),
            )

    def test_rejects_base_levels_2d(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="base_levels"):
            ScenarioLevels(
                axis=axis,
                base_levels=np.zeros((2, 1), dtype=np.float64),
                shocked_levels=np.ascontiguousarray(np.zeros((2, 3))),
            )

    def test_rejects_nan_base_level(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="non-finite"):
            ScenarioLevels(
                axis=axis,
                base_levels=np.array([1.0, np.nan]),
                shocked_levels=np.ascontiguousarray(np.zeros((2, 3))),
            )

    def test_rejects_shocked_levels_wrong_row_count(self) -> None:
        axis = FactorAxis.from_keys([1, 2, 3])
        with pytest.raises(SensitivityPnlInputError, match="rows against"):
            ScenarioLevels(
                axis=axis,
                base_levels=np.array([1.0, 2.0, 3.0]),
                shocked_levels=np.ascontiguousarray(np.zeros((2, 3))),
            )

    def test_rejects_shocked_levels_non_float64(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="dtype"):
            ScenarioLevels(
                axis=axis,
                base_levels=np.array([1.0, 2.0]),
                shocked_levels=np.ascontiguousarray(np.zeros((2, 3), dtype=np.float32)),
            )

    def test_rejects_nan_in_shocked_levels(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        shocked = np.zeros((2, 3))
        shocked[0, 0] = np.nan
        with pytest.raises(SensitivityPnlInputError, match="non-finite"):
            ScenarioLevels(
                axis=axis,
                base_levels=np.array([1.0, 2.0]),
                shocked_levels=np.ascontiguousarray(shocked),
            )

    def test_base_levels_check_runs_before_shocked_levels_finiteness(self) -> None:
        axis = FactorAxis.from_keys([1, 2])
        with pytest.raises(SensitivityPnlInputError, match="base_levels"):
            ScenarioLevels(
                axis=axis,
                base_levels=np.array([1.0]),
                shocked_levels=np.ascontiguousarray(np.zeros((2, 3))),
            )

    def test_num_scenarios_is_the_shocked_levels_column_count(self) -> None:
        axis = FactorAxis.from_keys([1])
        market = axis.levels(
            base_levels=np.array([1.0]), shocked_levels=np.zeros((1, 9))
        )
        assert market.num_scenarios == 9


class TestScenarioLevelsConstructors:
    def test_from_matrix(self) -> None:
        market = ScenarioLevels.from_matrix(
            factor_keys=[1, 2],
            base_levels=np.array([1.0, 2.0]),
            shocked_levels=np.zeros((2, 3)),
        )
        assert market.axis.num_factors == 2

    def test_from_frame_reads_base_level_column(self) -> None:
        frame = pl.DataFrame(
            {"factor_id": [1, 2], "base_level": [100.0, 200.0], "s0": [105.0, 210.0]}
        )
        market = ScenarioLevels.from_frame(frame)
        assert market.base_levels.tolist() == [100.0, 200.0]
        assert market.shocked_levels[:, 0].tolist() == [105.0, 210.0]

    def test_from_frame_rejects_missing_base_level_column(self) -> None:
        frame = pl.DataFrame({"factor_id": [1], "s0": [1.0]})
        with pytest.raises(SensitivityPnlInputError, match="base_level"):
            ScenarioLevels.from_frame(frame)

    def test_from_frame_honours_custom_base_level_column_name(self) -> None:
        frame = pl.DataFrame({"factor_id": [1], "t0": [50.0], "s0": [55.0]})
        market = ScenarioLevels.from_frame(frame, base_level_column="t0")
        assert market.base_levels.tolist() == [50.0]

    def test_from_frame_excludes_base_level_column_from_scenario_columns(
        self,
    ) -> None:
        frame = pl.DataFrame(
            {"factor_id": [1], "base_level": [50.0], "s0": [51.0], "s1": [52.0]}
        )
        market = ScenarioLevels.from_frame(frame)
        assert market.num_scenarios == 2
