# tests/solutions/scenario_models/test_parametric_chart_contract.py
"""
D091 — parametric column chart, scale and dependence contract (data side).

Expected values are literal: chart codes 1/2/3, the 1e-9 acceptance bound,
explicit matrices. Nothing derives an expectation from the code under test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from riskquant.simlab.scenarios.parametric_data import (
    PARAMETRIC_DRIVER_CHART_IDS,
    ParametricRiskData,
)
from riskquant.simlab.shocks.column_binding import bind_term_columns
from riskquant.solutions.run_identity.hashing import hash_parametric_risk_data
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
    resolve_parametric_binding_horizon,
)

_LOG, _SIMPLE, _DIFF = np.int8(1), np.int8(2), np.int8(3)  # literal ShiftType codes


def _data(
    charts: Any = None, chol: Any = None, k: int = 2
) -> ParametricRiskData:
    return ParametricRiskData(
        factor_ids=np.arange(10, 10 + k, dtype=np.int32),
        cholesky_matrix=(
            np.ascontiguousarray(np.eye(k, dtype=np.float64)) if chol is None else chol
        ),
        volatilities=np.ascontiguousarray(np.full(k, 0.1, dtype=np.float64)),
        effective_period_count=1,
        effective_period_unit=np.int8(1),
        chart_ids=np.full(k, _LOG, dtype=np.int8) if charts is None else charts,
    )


# ---- chart vocabulary and shape ---------------------------------------------------


def test_driver_chart_codes_are_log_simple_difference() -> None:
    assert PARAMETRIC_DRIVER_CHART_IDS == (1, 2, 3)


@pytest.mark.parametrize("codes", [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 3, 2]])
def test_each_driver_chart_and_mixed_charts_are_accepted(codes: list[int]) -> None:
    data = _data(charts=np.array(codes, dtype=np.int8), k=3)
    assert data.chart_ids.tolist() == codes


@pytest.mark.parametrize("bad", [0, -1, 4, 5])
def test_non_driver_chart_codes_are_rejected(bad: int) -> None:
    """0 is invalid, -1 is NO_SHIFT and 4 is ABSOLUTE_BASE_RETURN (D091 cl. 1)."""
    with pytest.raises(ValueError):
        _data(charts=np.array([1, bad], dtype=np.int8))


@pytest.mark.parametrize(
    ("charts", "exc"),
    [
        (np.array([1, 1], dtype=np.int32), TypeError),
        (np.array([1, 1], dtype=np.int64), TypeError),
        (np.array([1, 1], dtype=np.uint8), TypeError),
        (np.array([[1, 1]], dtype=np.int8), TypeError),
        ([1, 1], TypeError),
        (np.array([1], dtype=np.int8), ValueError),
        (np.array([1, 1, 1], dtype=np.int8), ValueError),
        (np.array([1, 9, 1, 9], dtype=np.int8)[::2], ValueError),  # strided view
    ],
)
def test_chart_ids_shape_dtype_and_layout_are_validated(charts: Any, exc: type) -> None:
    with pytest.raises(exc):
        _data(charts=charts)


def test_chart_ids_are_required() -> None:
    with pytest.raises(TypeError):
        ParametricRiskData(  # type: ignore[call-arg]
            factor_ids=np.arange(2, dtype=np.int32),
            cholesky_matrix=np.ascontiguousarray(np.eye(2)),
            volatilities=np.ascontiguousarray(np.full(2, 0.1)),
            effective_period_count=1,
            effective_period_unit=np.int8(1),
        )


# ---- dependence factor: R = L Lᵀ with unit diagonal ---------------------------------


def test_a_correlation_cholesky_factor_is_accepted_unchanged() -> None:
    chol = np.ascontiguousarray(np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]])))
    data = _data(chol=chol)
    assert data.cholesky_matrix is chol


def test_no_triangularity_is_required() -> None:
    """A rotation has unit rows and is not lower triangular."""
    c, s = np.cos(0.3), np.sin(0.3)
    rotation = np.ascontiguousarray(np.array([[c, -s], [s, c]]))
    assert _data(chol=rotation).cholesky_matrix is rotation


def test_no_strict_positive_definiteness_is_required() -> None:
    """Unit rows, singular R = [[1, 1], [1, 1]]: positive semidefinite only."""
    singular = np.ascontiguousarray(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert _data(chol=singular).cholesky_matrix is singular


def test_a_covariance_factor_is_rejected() -> None:
    """Cholesky of diag(4, 1): row norms 2 and 1, so diag(L Lᵀ) != 1."""
    cov_factor = np.ascontiguousarray(np.array([[2.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="correlation factor"):
        _data(chol=cov_factor)


@pytest.mark.parametrize(("excess", "accepted"), [(5e-10, True), (5e-9, False)])
def test_unit_row_norm_is_accepted_within_one_e_minus_nine(
    excess: float, accepted: bool
) -> None:
    """Row sum of squares 1 + excess against the absolute bound 1e-9."""
    chol = np.ascontiguousarray(np.array([[np.sqrt(1.0 + excess), 0.0], [0.0, 1.0]]))
    if accepted:
        assert _data(chol=chol).cholesky_matrix is chol
    else:
        with pytest.raises(ValueError, match="correlation factor"):
            _data(chol=chol)


# ---- identity -------------------------------------------------------------------------------


def test_chart_ids_enter_the_data_digest() -> None:
    log = _data(charts=np.array([1, 1], dtype=np.int8))
    also_log = _data(charts=np.array([1, 1], dtype=np.int8))
    diff = _data(charts=np.array([1, 3], dtype=np.int8))
    assert hash_parametric_risk_data(log) == hash_parametric_risk_data(also_log)
    assert hash_parametric_risk_data(log) != hash_parametric_risk_data(diff)


# ---- the generator is chart-blind ------------------------------------------------------------


def _shocks(data: ParametricRiskData) -> np.ndarray:
    cfg = parse_scenario_model_config({
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": 40,
        "seed": 5,
        "distribution": {"kind": "student_t", "df": 5.0},
    })
    gen = build_scenario_generator(cfg, data=data)
    out = [
        np.array(b.shocks, copy=True)
        for b in gen.iter_batches(
            chart_ids=np.array([1, 1], dtype=np.int8),
            driver_rf_ids=data.factor_ids,
            batch_size=16,
        )
    ]
    return np.vstack(out)


def test_chart_metadata_does_not_change_generated_values() -> None:
    log = _shocks(_data(charts=np.array([1, 1], dtype=np.int8)))
    diff = _shocks(_data(charts=np.array([3, 3], dtype=np.int8)))
    assert np.array_equal(log.view(np.uint32), diff.view(np.uint32))


# ---- binding: (factor_ids[i], chart_ids[i]) at h = count(H_W) --------------------------------


def _world_keys(data: ParametricRiskData, h: int, driver: list[int], chart: list[int]):
    return bind_term_columns(
        term_driver_rf_ids=np.array(driver, dtype=np.int32),
        term_chart_ids=np.array(chart, dtype=np.int8),
        requested_horizon=h,
        world_column_driver_rf_ids=data.factor_ids,
        world_column_chart_ids=data.chart_ids,
        world_horizon=h,
    )


def test_declared_driver_charts_bind_to_their_columns() -> None:
    data = _data(charts=np.array([1, 3, 2], dtype=np.int8), k=3)
    cfg = parse_scenario_model_config({
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": 4,
        "distribution": {"kind": "standard_normal"},
    })
    h = resolve_parametric_binding_horizon(cfg, data)
    cols = _world_keys(data, h, driver=[12, 10, 11], chart=[2, 1, 3])
    assert cols.tolist() == [2, 0, 1]


def test_an_undeclared_chart_of_a_factor_is_a_missing_column() -> None:
    """Factor 11 declares DIFFERENCE; asking for its LOG column fails."""
    data = _data(charts=np.array([1, 3], dtype=np.int8))
    with pytest.raises(ValueError):
        _world_keys(data, 1, driver=[11], chart=[1])
