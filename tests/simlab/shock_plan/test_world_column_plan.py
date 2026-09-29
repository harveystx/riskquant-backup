# tests/simlab/shock_plan/test_world_column_plan.py
"""
`build_world_column_plan`: validation of a world request and its gather plan.

A world request is unique `(driver_rf_id, chart_id)` pairs (D088 §4). Charts
are the V1 driver charts and the native ABSOLUTE_BASE_RETURN column; NO_SHIFT
requests no column (§13.4). Expected values are literal.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.simlab.shock_plan.plan import WORLD_COLUMN_CHART_IDS, build_world_column_plan

LOG, SIMPLE, DIFF, ABS, NO_SHIFT = 1, 2, 3, 4, -1
IDS = np.array([12, 4002, 9007], dtype=np.int32)  # world columns, sorted


def _plan(drivers, charts, ids=IDS):
    return build_world_column_plan(
        driver_rf_ids=np.array(drivers, dtype=np.int32),
        chart_ids=np.array(charts, dtype=np.int8),
        factor_ids=ids,
    )


def test_world_column_charts_are_the_v1_charts_and_abs() -> None:
    assert WORLD_COLUMN_CHART_IDS == frozenset({1, 2, 3, 4})


def test_request_order_is_recovered_by_the_gather_map() -> None:
    """Request [9007, 12, 4002] -> data columns [2, 0, 1]; column k of every
    batch is request pair k."""
    plan = _plan([9007, 12, 4002], [LOG, DIFF, ABS])
    assert plan.cols_sorted.tolist() == [0, 1, 2]
    assert plan.inverse_permutation.tolist() == [2, 0, 1]
    assert plan.cols_sorted[plan.inverse_permutation].tolist() == [2, 0, 1]
    assert plan.shift_types_sorted[plan.inverse_permutation].tolist() == [LOG, DIFF, ABS]


def test_one_driver_under_two_charts_is_two_columns() -> None:
    plan = _plan([4002, 4002], [LOG, SIMPLE])
    assert plan.cols_sorted.tolist() == [1, 1]
    assert plan.shift_types_sorted[plan.inverse_permutation].tolist() == [LOG, SIMPLE]


def test_an_empty_request_is_an_empty_plan() -> None:
    plan = _plan([], [])
    assert plan.cols_sorted.size == 0 and plan.inverse_permutation.size == 0


@pytest.mark.parametrize("chart", [NO_SHIFT, 0, 5, -2])
def test_a_chart_no_world_column_carries_is_refused(chart) -> None:
    with pytest.raises(ValueError, match="no world column carries"):
        _plan([12], [chart])


def test_a_duplicate_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="each \\(d, tau\\) pair once"):
        _plan([12, 4002, 12], [LOG, LOG, LOG])


def test_a_driver_that_is_not_a_world_column_is_refused() -> None:
    with pytest.raises(KeyError, match="missing"):
        _plan([12, 77], [LOG, LOG])


@pytest.mark.parametrize(
    ("drivers", "charts", "ids", "exc"),
    [
        (np.array([12], np.int64), np.array([LOG], np.int8), IDS, TypeError),
        (np.array([12], np.int32), np.array([LOG], np.int32), IDS, TypeError),
        (np.array([12], np.int32), np.array([LOG], np.int8), IDS.astype(np.int64), TypeError),
        (np.array([[12]], np.int32), np.array([LOG], np.int8), IDS, TypeError),
        (np.array([12, 4002], np.int32), np.array([LOG], np.int8), IDS, ValueError),
    ],
    ids=["driver_dtype", "chart_dtype", "ids_dtype", "driver_rank", "length"],
)
def test_malformed_requests_are_refused(drivers, charts, ids, exc) -> None:
    with pytest.raises(exc):
        build_world_column_plan(driver_rf_ids=drivers, chart_ids=charts, factor_ids=ids)
