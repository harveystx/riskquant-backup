# tests/solutions/scenario_models/test_forecast_period.py
"""
D090 — parametric forecast-period representation.

Expected values are written out literally (counts, unit codes, dumped dicts);
nothing here derives an expectation from the code under test.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.config.proxy_model.compiled import CompiledProxyModel
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.simlab.shocks.column_binding import bind_term_columns
from riskquant.solutions.run_identity.hashing import hash_parametric_risk_data
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
    resolve_parametric_binding_horizon,
)

_K = 2
_INT64_MAX = 9_223_372_036_854_775_807  # literal oracle, not np.iinfo


_BUSINESS_DAY_CODE = np.int8(1)  # literal oracle for the V1 unit code


def _data(count: Any = 1, unit: Any = _BUSINESS_DAY_CODE) -> ParametricRiskData:
    return ParametricRiskData(
        factor_ids=np.arange(_K, dtype=np.int32),
        cholesky_matrix=np.ascontiguousarray(np.eye(_K, dtype=np.float64)),
        volatilities=np.ascontiguousarray(np.full(_K, 0.1, dtype=np.float64)),
        effective_period_count=count,
        effective_period_unit=unit,
        chart_ids=np.full(_K, SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )


def _raw(horizon: Any = None, *, drop: bool = False) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "model": "parametric",
        "total_scenarios": 5,
        "seed": 3,
        "distribution": {"kind": "standard_normal"},
    }
    if not drop:
        raw["forecast_horizon"] = (
            {"count": 1, "unit": "BUSINESS_DAY"} if horizon is None else horizon
        )
    return raw


# ---- the single vocabulary --------------------------------------------------


def test_business_day_code_is_int8_one() -> None:
    assert FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID.dtype == np.dtype(np.int8)
    assert int(FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID) == 1


# ---- configuration: H_requested ------------------------------------------------


@pytest.mark.parametrize("count", [1, 5, 250, _INT64_MAX])
def test_requested_period_accepts_positive_counts(count: int) -> None:
    cfg = parse_scenario_model_config(_raw({"count": count, "unit": "BUSINESS_DAY"}))
    assert cfg.forecast_horizon.count == count
    assert cfg.forecast_horizon.unit == "BUSINESS_DAY"


@pytest.mark.parametrize(
    "horizon",
    [
        {"count": 0, "unit": "BUSINESS_DAY"},
        {"count": -1, "unit": "BUSINESS_DAY"},
        {"count": True, "unit": "BUSINESS_DAY"},
        {"count": 1.0, "unit": "BUSINESS_DAY"},
        {"count": "1", "unit": "BUSINESS_DAY"},
        {"count": _INT64_MAX + 1, "unit": "BUSINESS_DAY"},
        {"count": 5, "unit": "WEEK"},
        {"count": 5, "unit": "CALENDAR_DAY"},
        {"count": 5, "unit": "business_day"},
        {"count": 5, "unit": 1},
        {"count": 5},
        {"unit": "BUSINESS_DAY"},
        {"count": 5, "unit": "BUSINESS_DAY", "step": 1},
        5,
    ],
)
def test_requested_period_rejects_malformed_input(horizon: Any) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_raw(horizon))


def test_requested_period_is_required() -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_raw(drop=True))


def test_matrix_configs_do_not_accept_a_forecast_period() -> None:
    """D090 is parametric-only; matrix worlds keep their date-axis horizon."""
    horizon = {"count": 1, "unit": "BUSINESS_DAY"}
    for raw in (
        {"model": "historical", "horizon": 2},
        {"model": "sbb", "horizon": 2, "total_scenarios": 6, "seed": 1},
    ):
        assert parse_scenario_model_config(raw).horizon == 2
        with pytest.raises(ValidationError):
            parse_scenario_model_config({**raw, "forecast_horizon": horizon})


# ---- data: H_effective --------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 7, np.int64(7), np.int32(3), _INT64_MAX])
def test_effective_period_accepts_positive_integers(count: Any) -> None:
    data = _data(count=count)
    assert int(data.effective_period_count) == int(count)


@pytest.mark.parametrize(
    ("count", "exc"),
    [
        (0, ValueError),
        (-3, ValueError),
        (True, TypeError),
        (np.bool_(True), TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (_INT64_MAX + 1, ValueError),
    ],
)
def test_effective_period_count_is_validated(count: Any, exc: type) -> None:
    with pytest.raises(exc):
        _data(count=count)


@pytest.mark.parametrize(
    ("unit", "exc"),
    [
        (np.int8(0), ValueError),
        (np.int8(2), ValueError),
        (np.int8(-1), ValueError),
        (1, TypeError),
        (np.int16(1), TypeError),
        ("BUSINESS_DAY", TypeError),
        (None, TypeError),
    ],
)
def test_effective_period_unit_is_validated(unit: Any, exc: type) -> None:
    with pytest.raises(exc):
        _data(unit=unit)


def test_effective_period_is_required() -> None:
    with pytest.raises(TypeError):
        ParametricRiskData(  # type: ignore[call-arg]
            factor_ids=np.arange(_K, dtype=np.int32),
            cholesky_matrix=np.ascontiguousarray(np.eye(_K, dtype=np.float64)),
            volatilities=np.ascontiguousarray(np.full(_K, 0.1, dtype=np.float64)),
            chart_ids=np.full(_K, SHIFT_LOG_RETURN_ID, dtype=np.int8),
        )


# ---- composition gate: H_requested == H_effective --------------------------------------


def test_equal_periods_reduce_to_the_count() -> None:
    cfg = parse_scenario_model_config(_raw({"count": 7, "unit": "BUSINESS_DAY"}))
    h = resolve_parametric_binding_horizon(cfg, _data(count=7))
    assert h == 7
    assert type(h) is int


@pytest.mark.parametrize(("requested", "effective"), [(1, 2), (5, 1), (10, 11)])
def test_different_counts_are_rejected(requested: int, effective: int) -> None:
    cfg = parse_scenario_model_config(
        _raw({"count": requested, "unit": "BUSINESS_DAY"})
    )
    with pytest.raises(ValueError, match="D090"):
        resolve_parametric_binding_horizon(cfg, _data(count=effective))


def test_different_units_are_rejected() -> None:
    """V1 admits one unit, so both boundaries already reject any other; the
    gate itself compares units too. A data object carrying another code is
    forced past construction here to exercise that comparison."""
    cfg = parse_scenario_model_config(_raw({"count": 3, "unit": "BUSINESS_DAY"}))
    data = _data(count=3)
    object.__setattr__(data, "effective_period_unit", np.int8(2))
    with pytest.raises(ValueError, match="D090"):
        resolve_parametric_binding_horizon(cfg, data)


def test_gate_requires_parametric_data() -> None:
    cfg = parse_scenario_model_config(_raw())
    matrix = RiskFactorMatrix(
        dates=np.arange(3, dtype=np.int32),
        factor_ids=np.arange(_K, dtype=np.int32),
        values=np.asfortranarray(np.full((3, _K), 100.0)),
    )
    with pytest.raises(TypeError):
        resolve_parametric_binding_horizon(cfg, matrix)


def test_building_the_generator_applies_the_same_gate() -> None:
    cfg = parse_scenario_model_config(_raw({"count": 5, "unit": "BUSINESS_DAY"}))
    build_scenario_generator(cfg, data=_data(count=5))
    with pytest.raises(ValueError, match="D090"):
        build_scenario_generator(cfg, data=_data(count=4))


# ---- binding reduction: h = count(H_W), only after equality --------------------------------


def test_binder_receives_the_reduced_integer_horizon() -> None:
    cfg = parse_scenario_model_config(_raw({"count": 10, "unit": "BUSINESS_DAY"}))
    h = resolve_parametric_binding_horizon(cfg, _data(count=10))
    cols = bind_term_columns(
        term_driver_rf_ids=np.array([1, 0], dtype=np.int32),
        term_chart_ids=np.array([1, 1], dtype=np.int8),
        requested_horizon=h,
        world_column_driver_rf_ids=np.array([0, 1], dtype=np.int32),
        world_column_chart_ids=np.array([1, 1], dtype=np.int8),
        world_horizon=10,
    )
    assert cols.tolist() == [1, 0]


def test_a_mismatch_never_yields_a_binding_horizon() -> None:
    cfg = parse_scenario_model_config(_raw({"count": 10, "unit": "BUSINESS_DAY"}))
    with pytest.raises(ValueError, match="D090"):
        resolve_parametric_binding_horizon(cfg, _data(count=9))


# ---- identity -----------------------------------------------------------------------------


def test_effective_count_enters_the_data_digest() -> None:
    assert hash_parametric_risk_data(_data(count=1)) == hash_parametric_risk_data(
        _data(count=1)
    )
    assert hash_parametric_risk_data(_data(count=1)) != hash_parametric_risk_data(
        _data(count=2)
    )


def test_count_representation_does_not_change_the_digest() -> None:
    assert hash_parametric_risk_data(_data(count=5)) == hash_parametric_risk_data(
        _data(count=np.int64(5))
    )


def test_effective_unit_enters_the_data_digest() -> None:
    """The hash reads the unit code; a different code is a different digest.
    A plain namespace stands in for a future unit V1 does not admit."""
    base = _data(count=4)
    arrays = {
        "factor_ids": base.factor_ids,
        "cholesky_matrix": base.cholesky_matrix,
        "volatilities": base.volatilities,
        "effective_period_count": 4,
        "chart_ids": base.chart_ids,
    }
    one = SimpleNamespace(**arrays, effective_period_unit=np.int8(1))
    two = SimpleNamespace(**arrays, effective_period_unit=np.int8(2))
    assert hash_parametric_risk_data(one) == hash_parametric_risk_data(base)
    assert hash_parametric_risk_data(one) != hash_parametric_risk_data(two)


def test_requested_period_enters_the_scenario_config_identity() -> None:
    five = parse_scenario_model_config(_raw({"count": 5, "unit": "BUSINESS_DAY"}))
    six = parse_scenario_model_config(_raw({"count": 6, "unit": "BUSINESS_DAY"}))
    assert five.model_dump(mode="json")["forecast_horizon"] == {
        "count": 5,
        "unit": "BUSINESS_DAY",
    }
    assert five.model_dump(mode="json") != six.model_dump(mode="json")


def test_the_proxy_operator_carries_no_horizon() -> None:
    """proxy_model_digest hashes exactly these five arrays (D088 §7)."""
    assert [f.name for f in fields(CompiledProxyModel)] == [
        "row_target_rf_ids",
        "row_term_offsets",
        "term_driver_rf_ids",
        "term_chart_ids",
        "term_loadings",
    ]
