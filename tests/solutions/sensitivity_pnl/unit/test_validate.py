# tests/solutions/sensitivity_pnl/unit/test_validate.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.validate`.

This is the largest, most defect-rich module in the package: its own docstring
lists five specific failure modes it exists to catch that the compiler and
engine either miss or report in terms a caller cannot act on (null ids
silently becoming a sentinel UID, float ids silently truncating, key-space
mismatches reaching `replace_strict` as an opaque error, coverage gaps named
in an internal slot index instead of the caller's own key, and row-set
drift). Each gets its own test class here, exercised directly against
`validate_book_spec` / `validate_market_against_book` rather than only through
a full compile-and-run round trip.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivity_pnl.compile import compile_sensitivity_book
from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.solutions.sensitivity_pnl.errors import SensitivityPnlInputError
from riskquant.solutions.sensitivity_pnl.market import FactorAxis
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityBookSpec
from riskquant.solutions.sensitivity_pnl.validate import (
    validate_book_spec,
    validate_market_against_book,
)

ABSOLUTE = int(DisplacementConvention.ABSOLUTE)
LOG = int(DisplacementConvention.LOG)


def _spec(
    *,
    sensitivities: pl.DataFrame | None = None,
    factor_meta: pl.DataFrame | None = None,
    axis_keys: list | None = None,
    position_uids=None,
    cross_sensitivities: pl.DataFrame | None = None,
) -> SensitivityBookSpec:
    axis = FactorAxis.from_keys(axis_keys if axis_keys is not None else [1])
    return SensitivityBookSpec(
        sensitivities=sensitivities
        if sensitivities is not None
        else pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [1],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        ),
        factor_meta=factor_meta
        if factor_meta is not None
        else pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        ),
        factor_axis=axis,
        position_uids=position_uids,
        cross_sensitivities=cross_sensitivities,
    )


def _validate(spec: SensitivityBookSpec) -> None:
    from riskquant.solutions.sensitivity_pnl.pipeline import scope_factor_meta

    validate_book_spec(spec, factor_meta=scope_factor_meta(spec))


class TestValidateBookSpecHappyPath:
    def test_a_well_formed_spec_passes(self) -> None:
        _validate(_spec())  # must not raise

    def test_a_well_formed_spec_with_cross_terms_passes(self) -> None:
        cross = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id_i": [1],
                "factor_id_j": [1],
                "cross_second_order": [1.0],
            }
        )
        # needs a second factor for i != j downstream, but validate.py itself
        # does not enforce i != j (that is the compiler's job) -- this frame
        # is well-formed from validate.py's own perspective.
        _validate(_spec(cross_sensitivities=cross))


class TestNullAndFloatIdColumns:
    def test_null_position_id_is_reported(self) -> None:
        frame = pl.DataFrame(
            {
                "position_id": [1, None],
                "factor_id": [1, 1],
                "first_order": [5.0, 5.0],
                "second_order": [0.0, 0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="null value"):
            _validate(_spec(sensitivities=frame))

    def test_float_position_id_is_reported(self) -> None:
        """position_id = 2.7 casts to 2 without complaint downstream -- caught
        here as a dtype defect before it ever reaches the compiler."""
        frame = pl.DataFrame(
            {
                "position_id": [2.7],
                "factor_id": [1],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="must be integer"):
            _validate(_spec(sensitivities=frame))

    def test_null_value_column_is_reported(self) -> None:
        frame = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [1],
                "first_order": pl.Series([None], dtype=pl.Float64),
                "second_order": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="null value"):
            _validate(_spec(sensitivities=frame))

    def test_non_finite_value_is_reported(self) -> None:
        frame = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [1],
                "first_order": [float("inf")],
                "second_order": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="non-finite"):
            _validate(_spec(sensitivities=frame))

    def test_non_numeric_value_column_is_reported(self) -> None:
        frame = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [1],
                "first_order": ["not a number"],
                "second_order": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="must be numeric"):
            _validate(_spec(sensitivities=frame))


class TestKeySpaceMismatch:
    def test_string_factor_id_against_integer_axis_is_reported(self) -> None:
        frame = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": ["not_an_int"],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="factor axis uses"):
            _validate(_spec(sensitivities=frame, axis_keys=[1, 2]))

    def test_integer_factor_id_against_string_axis_is_reported(self) -> None:
        frame = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [1],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="factor axis uses"):
            _validate(_spec(sensitivities=frame, axis_keys=["A", "B"]))


class TestMissingColumns:
    def test_missing_sensitivities_column_is_reported(self) -> None:
        frame = pl.DataFrame(
            {"position_id": [1], "factor_id": [1], "first_order": [5.0]}
        )  # second_order missing
        with pytest.raises(SensitivityPnlInputError, match="missing column"):
            _validate(_spec(sensitivities=frame))

    def test_missing_factor_meta_column_is_reported(self) -> None:
        meta = pl.DataFrame(
            {"factor_id": [1], "displacement_convention_id": [ABSOLUTE]}
        )
        with pytest.raises(SensitivityPnlInputError, match="missing column"):
            _validate(_spec(factor_meta=meta))


class TestFactorCoverage:
    def test_factor_absent_from_axis_is_reported(self) -> None:
        """A sensitivity to a factor with no scenario row would silently
        contribute zero risk in every scenario -- the module's own
        description of the most dangerous case."""
        frame = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [2],  # axis only has factor 1
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="no row on the factor axis"):
            _validate(_spec(sensitivities=frame, axis_keys=[1]))

    def test_factor_absent_from_factor_meta_is_reported_in_callers_own_key(
        self,
    ) -> None:
        """The one that matters most under string labels: the compiler
        resolves to internal slots BEFORE validating, so a naive error would
        name a slot index the caller has never seen. This layer names the
        caller's own key."""
        frame = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": ["EQ_MISSING"],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": ["EQ_OTHER"],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [1.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError) as excinfo:
            _validate(
                _spec(
                    sensitivities=frame,
                    factor_meta=meta,
                    axis_keys=["EQ_MISSING", "EQ_OTHER"],
                )
            )
        text = str(excinfo.value)
        assert "EQ_MISSING" in text
        assert "no row in factor_meta" in text

    def test_the_house_wide_factor_table_with_extra_rows_does_not_raise(
        self,
    ) -> None:
        """factor_meta may carry factors the axis does not -- scope_factor_meta
        drops them before validate_book_spec ever sees them."""
        meta = pl.DataFrame(
            {
                "factor_id": [1, 2, 3],
                "displacement_convention_id": [ABSOLUTE] * 3,
                "quote_base": [1.0, 1.0, 1.0],
            }
        )
        _validate(_spec(factor_meta=meta, axis_keys=[1]))  # must not raise


class TestFactorMetaValidation:
    def test_duplicate_factor_id_rows_are_reported(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1, 1],
                "displacement_convention_id": [ABSOLUTE, LOG],
                "quote_base": [1.0, 1.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="duplicate factor row"):
            _validate(_spec(factor_meta=meta))

    def test_unknown_convention_id_is_reported(self) -> None:
        meta = pl.DataFrame(
            {"factor_id": [1], "displacement_convention_id": [99], "quote_base": [1.0]}
        )
        with pytest.raises(SensitivityPnlInputError, match="null or an id outside"):
            _validate(_spec(factor_meta=meta))

    def test_null_convention_id_is_reported(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": pl.Series([None], dtype=pl.Int8),
                "quote_base": [1.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="null or an id outside"):
            _validate(_spec(factor_meta=meta))

    def test_non_integer_convention_dtype_is_reported(self) -> None:
        meta = pl.DataFrame(
            {"factor_id": [1], "displacement_convention_id": [1.5], "quote_base": [1.0]}
        )
        with pytest.raises(
            SensitivityPnlInputError, match="the convention is an integer"
        ):
            _validate(_spec(factor_meta=meta))

    def test_zero_quote_base_is_reported(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="non-positive"):
            _validate(_spec(factor_meta=meta))

    def test_negative_quote_base_is_reported(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [-5.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="non-positive"):
            _validate(_spec(factor_meta=meta))

    def test_null_quote_base_is_reported(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": pl.Series([None], dtype=pl.Float64),
            }
        )
        with pytest.raises(
            SensitivityPnlInputError, match="null, non-finite, or non-positive"
        ):
            _validate(_spec(factor_meta=meta))

    def test_infinite_quote_base_is_reported(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": [float("inf")],
            }
        )
        with pytest.raises(
            SensitivityPnlInputError, match="null, non-finite, or non-positive"
        ):
            _validate(_spec(factor_meta=meta))

    def test_non_numeric_quote_base_dtype_is_reported(self) -> None:
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": ["one"],
            }
        )
        with pytest.raises(
            SensitivityPnlInputError, match="positive scalar per factor"
        ):
            _validate(_spec(factor_meta=meta))

    def test_integer_quote_base_of_zero_is_reported(self) -> None:
        """quote_base as an integer column, not float -- a different code
        path in _validate_quote_base_column (the else branch)."""
        meta = pl.DataFrame(
            {
                "factor_id": [1],
                "displacement_convention_id": [ABSOLUTE],
                "quote_base": pl.Series([0], dtype=pl.Int64),
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="non-positive"):
            _validate(_spec(factor_meta=meta))

    def test_no_rows_covering_the_market_is_reported(self) -> None:
        meta = pl.DataFrame(
            schema={
                "factor_id": pl.Int64,
                "displacement_convention_id": pl.Int8,
                "quote_base": pl.Float64,
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="no rows covering"):
            _validate(_spec(factor_meta=meta))


class TestRowSetValidation:
    def test_no_declared_uids_and_no_rows_is_reported(self) -> None:
        empty_sens = pl.DataFrame(
            schema={
                "position_id": pl.Int64,
                "factor_id": pl.Int64,
                "first_order": pl.Float64,
                "second_order": pl.Float64,
            }
        )
        with pytest.raises(
            SensitivityPnlInputError, match="no position_uids were declared"
        ):
            _validate(_spec(sensitivities=empty_sens))

    def test_no_declared_uids_with_rows_present_is_fine(self) -> None:
        _validate(_spec(position_uids=None))  # must not raise

    def test_malformed_position_uids_wrong_ndim_is_reported(self) -> None:
        with pytest.raises(SensitivityPnlInputError, match="1-D sequence"):
            _validate(_spec(position_uids=np.array([[1, 2]])))

    def test_malformed_position_uids_non_integer_dtype_is_reported(self) -> None:
        with pytest.raises(SensitivityPnlInputError, match="1-D sequence"):
            _validate(_spec(position_uids=np.array([1.5, 2.5])))

    def test_duplicate_declared_uids_is_reported(self) -> None:
        with pytest.raises(SensitivityPnlInputError, match="duplicate UID"):
            _validate(_spec(position_uids=[1, 1, 2]))

    def test_undeclared_position_referenced_by_sensitivities_is_reported(self) -> None:
        """The row-set-drift case: a sensitivity for a position outside the
        declared set would silently widen the output matrix."""
        frame = pl.DataFrame(
            {
                "position_id": [1, 99],
                "factor_id": [1, 1],
                "first_order": [5.0, 5.0],
                "second_order": [0.0, 0.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="absent from position_uids"):
            _validate(_spec(sensitivities=frame, position_uids=[1]))

    def test_a_declared_position_with_no_sensitivities_is_fine(self) -> None:
        """The whole point of declaring position_uids: a hedged position with
        zero rows still gets validated cleanly and holds its row downstream."""
        _validate(_spec(position_uids=[1, 999]))  # must not raise

    def test_undeclared_position_referenced_only_by_cross_is_reported(self) -> None:
        cross = pl.DataFrame(
            {
                "position_id": [77],
                "factor_id_i": [1],
                "factor_id_j": [1],
                "cross_second_order": [1.0],
            }
        )
        with pytest.raises(SensitivityPnlInputError, match="absent from position_uids"):
            _validate(_spec(cross_sensitivities=cross, position_uids=[1]))


class TestValidateMarketAgainstBook:
    def _book(self, *, convention: int):
        ids = np.array([0], dtype=np.int32)
        sens = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": [0],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": [0],
                "displacement_convention_id": [convention],
                "quote_base": [1.0],
            }
        )
        return compile_sensitivity_book(
            sensitivities=sens, factor_meta=meta, level_universe_factor_ids=ids
        )

    def test_shocks_market_is_never_checked(self) -> None:
        """validate_market_against_book only inspects ScenarioLevels -- a
        ScenarioShocks market has no 'level' concept to check."""
        book = self._book(convention=LOG)
        axis = FactorAxis.from_keys([0])
        market = axis.shocks(np.array([[-999.0]], dtype=np.float32))
        validate_market_against_book(book=book, axis=axis, market=market)  # no raise

    def test_log_factor_with_negative_shocked_level_is_rejected(self) -> None:
        book = self._book(convention=LOG)
        axis = FactorAxis.from_keys([0])
        market = axis.levels(
            base_levels=np.array([2.0]),
            shocked_levels=np.ascontiguousarray(np.array([[-1.0]])),
        )
        with pytest.raises(SensitivityPnlInputError, match="non-positive"):
            validate_market_against_book(book=book, axis=axis, market=market)

    def test_log_factor_with_zero_shocked_level_is_rejected(self) -> None:
        book = self._book(convention=LOG)
        axis = FactorAxis.from_keys([0])
        market = axis.levels(
            base_levels=np.array([2.0]),
            shocked_levels=np.ascontiguousarray(np.array([[0.0]])),
        )
        with pytest.raises(SensitivityPnlInputError, match="non-positive"):
            validate_market_against_book(book=book, axis=axis, market=market)

    def test_log_factor_with_positive_shocked_level_is_accepted(self) -> None:
        book = self._book(convention=LOG)
        axis = FactorAxis.from_keys([0])
        market = axis.levels(
            base_levels=np.array([2.0]),
            shocked_levels=np.ascontiguousarray(np.array([[3.0]])),
        )
        validate_market_against_book(book=book, axis=axis, market=market)  # no raise

    def test_absolute_convention_is_never_checked_for_level_positivity(self) -> None:
        book = self._book(convention=ABSOLUTE)
        axis = FactorAxis.from_keys([0])
        market = axis.levels(
            base_levels=np.array([2.0]),
            shocked_levels=np.ascontiguousarray(np.array([[-999.0]])),
        )
        validate_market_against_book(book=book, axis=axis, market=market)  # no raise

    def test_error_names_the_callers_own_factor_key(self) -> None:
        """Under SLOT encoding the engine's own guard names a compiled slot
        index; this one names the caller's actual key."""
        sens = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id": ["MY_LOG_FACTOR"],
                "first_order": [5.0],
                "second_order": [0.0],
            }
        )
        meta = pl.DataFrame(
            {
                "factor_id": ["MY_LOG_FACTOR"],
                "displacement_convention_id": [LOG],
                "quote_base": [1.0],
            }
        )
        axis = FactorAxis.from_keys(["MY_LOG_FACTOR"])
        book = compile_sensitivity_book(
            sensitivities=axis.relabel(sens, ("factor_id",)),
            factor_meta=axis.relabel(meta, ("factor_id",)),
            level_universe_factor_ids=axis.universe_ids,
        )
        market = axis.levels(
            base_levels=np.array([2.0]),
            shocked_levels=np.ascontiguousarray(np.array([[-1.0]])),
        )
        with pytest.raises(SensitivityPnlInputError, match="MY_LOG_FACTOR"):
            validate_market_against_book(book=book, axis=axis, market=market)
