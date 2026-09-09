# tests/solutions/sensitivity_pnl/unit/test_schemas.py
"""
Unit tests for `riskquant.solutions.sensitivity_pnl.schemas`.

Covers the three pinned contracts: `SensitivityBookSpec` / `SensitivityPnlRequest`
(book + market, and their `book_spec()` projection), `SensitivityPnlProvenance`
(a Pydantic model with `extra="forbid"`), and `SensitivityPnlResult` (the
`row_for` accessor's deliberately narrow contract -- a view, not a copy, and
the only row-level accessor on the class).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from riskquant.pnl.results import PnlMatrix
from riskquant.sensitivity_pnl.compile import DuplicatePolicy
from riskquant.solutions.sensitivity_pnl.market import FactorAxis
from riskquant.solutions.sensitivity_pnl.schemas import (
    SensitivityBookSpec,
    SensitivityPnlProvenance,
    SensitivityPnlRequest,
    SensitivityPnlResult,
)


def _minimal_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "position_id": [1],
            "factor_id": [1],
            "first_order": [5.0],
            "second_order": [0.0],
        }
    )


def _minimal_meta() -> pl.DataFrame:
    return pl.DataFrame(
        {"factor_id": [1], "displacement_convention_id": [1], "quote_base": [1.0]}
    )


class TestSensitivityBookSpec:
    def test_duplicate_policy_defaults_to_raise(self) -> None:
        spec = SensitivityBookSpec(
            sensitivities=_minimal_frame(),
            factor_meta=_minimal_meta(),
            factor_axis=FactorAxis.from_keys([1]),
        )
        assert spec.duplicate_policy is DuplicatePolicy.RAISE

    def test_position_uids_defaults_to_none(self) -> None:
        spec = SensitivityBookSpec(
            sensitivities=_minimal_frame(),
            factor_meta=_minimal_meta(),
            factor_axis=FactorAxis.from_keys([1]),
        )
        assert spec.position_uids is None

    def test_cross_sensitivities_defaults_to_none(self) -> None:
        spec = SensitivityBookSpec(
            sensitivities=_minimal_frame(),
            factor_meta=_minimal_meta(),
            factor_axis=FactorAxis.from_keys([1]),
        )
        assert spec.cross_sensitivities is None

    def test_is_frozen(self) -> None:
        spec = SensitivityBookSpec(
            sensitivities=_minimal_frame(),
            factor_meta=_minimal_meta(),
            factor_axis=FactorAxis.from_keys([1]),
        )
        with pytest.raises((AttributeError, Exception)):
            spec.duplicate_policy = DuplicatePolicy.SUM  # type: ignore[misc]


class TestSensitivityPnlRequest:
    def test_num_scenarios_delegates_to_the_market(self) -> None:
        axis = FactorAxis.from_keys([1])
        request = SensitivityPnlRequest(
            sensitivities=_minimal_frame(),
            factor_meta=_minimal_meta(),
            market=axis.shocks(np.zeros((1, 7), dtype=np.float32)),
        )
        assert request.num_scenarios == 7

    def test_book_spec_carries_over_every_book_field(self) -> None:
        axis = FactorAxis.from_keys([1])
        cross = pl.DataFrame(
            {
                "position_id": [1],
                "factor_id_i": [1],
                "factor_id_j": [1],
                "cross_second_order": [1.0],
            }
        )
        request = SensitivityPnlRequest(
            sensitivities=_minimal_frame(),
            factor_meta=_minimal_meta(),
            market=axis.shocks(np.zeros((1, 3), dtype=np.float32)),
            position_uids=[1, 2],
            cross_sensitivities=cross,
            duplicate_policy=DuplicatePolicy.SUM,
        )
        spec = request.book_spec()
        assert spec.sensitivities is request.sensitivities
        assert spec.factor_meta is request.factor_meta
        assert spec.position_uids == [1, 2]
        assert spec.cross_sensitivities is cross
        assert spec.duplicate_policy is DuplicatePolicy.SUM

    def test_book_spec_takes_its_axis_from_the_market_not_a_separate_field(
        self,
    ) -> None:
        """The whole point: a request never states the axis twice, so the
        two can never disagree."""
        axis = FactorAxis.from_keys([1, 2])
        request = SensitivityPnlRequest(
            sensitivities=_minimal_frame(),
            factor_meta=_minimal_meta(),
            market=axis.shocks(np.zeros((2, 3), dtype=np.float32)),
        )
        assert request.book_spec().factor_axis is axis


class TestSensitivityPnlProvenance:
    def _kwargs(self, **overrides):
        kwargs = {
            "engine_version": "1.0",
            "market_contract": "displacement",
            "book_reused": False,
            "factor_key_encoding": "direct",
            "num_positions": 1,
            "num_scenarios": 1,
            "num_market_factors": 1,
            "num_active_factors": 1,
            "num_first_order_entries": 1,
            "num_cross_entries": 0,
            "num_positions_without_sensitivities": 0,
            "batch_size": 1,
        }
        kwargs.update(overrides)
        return kwargs

    def test_valid_construction(self) -> None:
        SensitivityPnlProvenance(**self._kwargs())  # must not raise

    def test_extra_fields_are_forbidden(self) -> None:
        """model_config = ConfigDict(extra='forbid'): a typo'd field name must
        raise loudly rather than silently be dropped."""
        with pytest.raises(ValidationError):
            SensitivityPnlProvenance(**self._kwargs(extra_field_typo=123))

    def test_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SensitivityPnlProvenance(**self._kwargs(num_positions="not an int"))

    def test_missing_required_field_is_rejected(self) -> None:
        kwargs = self._kwargs()
        del kwargs["batch_size"]
        with pytest.raises(ValidationError):
            SensitivityPnlProvenance(**kwargs)

    def test_is_a_pydantic_model(self) -> None:
        from pydantic import BaseModel

        assert isinstance(SensitivityPnlProvenance(**self._kwargs()), BaseModel)


class TestSensitivityPnlResult:
    def _result(self) -> SensitivityPnlResult:
        from riskquant.solutions.sensitivity_pnl.schemas import (
            SensitivityPnlProvenance as Prov,
        )

        matrix = PnlMatrix(
            trade_uids=np.array([10, 20, 30], dtype=np.int64),
            pnl=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        )
        provenance = Prov(
            engine_version="1.0",
            market_contract="displacement",
            book_reused=False,
            factor_key_encoding="direct",
            num_positions=3,
            num_scenarios=2,
            num_market_factors=1,
            num_active_factors=1,
            num_first_order_entries=1,
            num_cross_entries=0,
            num_positions_without_sensitivities=0,
            batch_size=2,
        )
        return SensitivityPnlResult(pnl_matrix=matrix, provenance=provenance)

    def test_position_uids_reads_through_to_the_matrix(self) -> None:
        result = self._result()
        assert result.position_uids.tolist() == [10, 20, 30]

    def test_pnl_reads_through_to_the_matrix(self) -> None:
        result = self._result()
        assert result.pnl.shape == (3, 2)

    def test_row_for_returns_the_correct_row(self) -> None:
        result = self._result()
        assert result.row_for(20).tolist() == [3.0, 4.0]

    def test_row_for_returns_a_view_not_a_copy(self) -> None:
        result = self._result()
        row = result.row_for(10)
        assert np.shares_memory(row, result.pnl_matrix.pnl)

    def test_row_for_unknown_uid_raises_key_error(self) -> None:
        result = self._result()
        with pytest.raises(KeyError, match="99"):
            result.row_for(99)

    def test_row_for_uid_below_the_range_raises_key_error(self) -> None:
        result = self._result()
        with pytest.raises(KeyError):
            result.row_for(1)

    def test_row_for_uid_above_the_range_raises_key_error(self) -> None:
        result = self._result()
        with pytest.raises(KeyError):
            result.row_for(1000)

    def test_row_for_error_message_reports_the_valid_range(self) -> None:
        result = self._result()
        with pytest.raises(KeyError, match=r"10\.\.30"):
            result.row_for(99)
