# tests/solutions/sensitivity_pnl/test_golden_master_taylor_validity.py
"""
Golden master for `riskquant.solutions.sensitivity_pnl.taylor_validity`.

The core-tier module (`riskquant.sensitivity_pnl.validity`) already has its own
two-layer golden master, hand-computed fixture included. This file does not
repeat that arithmetic — it tests the TRANSLATION layer this facade exists for:
axis binding and rejection identical to `calculator.run`'s guarantees, caller-
facing factor keys under BOTH axis encodings, and the report's own accessors.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
    SensitivityPnlInputError,
    assess_characteristic_displacement,
    assess_taylor_validity,
)

FACTOR_KEYS = ["EQ_A", "EQ_B", "FX_C"]

SENSITIVITY_ROWS = [
    (1, "EQ_A", 100.0, 40.0),
    (1, "EQ_B", -50.0, 0.0),
    (2, "FX_C", 10.0, 5000.0),  # deliberately gamma-heavy -> should register strain
]
CROSS_ROWS = [(1, "EQ_A", "EQ_B", 5.0)]


def build_calculator(axis: FactorAxis) -> SensitivityPnlCalculator:
    sensitivities = pl.DataFrame(
        SENSITIVITY_ROWS,
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    cross = pl.DataFrame(
        CROSS_ROWS,
        schema={
            "position_id": pl.Int64,
            "factor_id_i": pl.String,
            "factor_id_j": pl.String,
            "cross_second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": FACTOR_KEYS,
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 3,
            "quote_base": [1.0] * 3,
        }
    )
    return SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sensitivities,
            factor_meta=meta,
            factor_axis=axis,
            cross_sensitivities=cross,
            position_uids=[1, 2],
        )
    )


def displacement_market(axis: FactorAxis, *, seed: int = 9, num_scenarios: int = 500):
    rng = np.random.default_rng(seed)
    disp = np.ascontiguousarray(
        rng.normal(0.0, 0.5, size=(3, num_scenarios)).astype(np.float32)
    )
    return axis.shocks(disp)


# =========================================================================
# assess_taylor_validity
# =========================================================================


def test_report_summary_matches_the_core_grid() -> None:
    """The facade must not silently transform the numbers, only their keys."""
    axis = FactorAxis.from_keys(FACTOR_KEYS)  # string keys -> SLOT encoding
    calculator = build_calculator(axis)
    market = displacement_market(axis)

    report = assess_taylor_validity(
        calculator=calculator, market=market, threshold=0.25
    )

    assert report.provenance.market_contract == "displacement"
    assert report.provenance.factor_key_encoding == "slot"
    assert np.array_equal(report.position_uids, np.array([1, 2], dtype=np.int64))
    assert report.grid.num_positions == 2
    assert report.grid.num_scenarios == 500

    expected_fraction = report.grid.nonlinearity_fraction()
    assert np.array_equal(
        report.summary.worst_nonlinearity_fraction, expected_fraction.max(axis=1)
    )


def test_nonlinearity_fraction_for_returns_the_correct_row() -> None:
    axis = FactorAxis.from_keys(FACTOR_KEYS)
    calculator = build_calculator(axis)
    market = displacement_market(axis)
    report = assess_taylor_validity(calculator=calculator, market=market)

    row = report.nonlinearity_fraction_for(2)
    assert row.shape == (500,)
    assert np.array_equal(row, report.grid.nonlinearity_fraction()[1])


def test_nonlinearity_fraction_for_rejects_an_unknown_uid() -> None:
    axis = FactorAxis.from_keys(FACTOR_KEYS)
    calculator = build_calculator(axis)
    report = assess_taylor_validity(
        calculator=calculator, market=displacement_market(axis)
    )
    with pytest.raises(KeyError, match="999"):
        report.nonlinearity_fraction_for(999)


def test_mismatched_axis_is_rejected_before_anything_is_allocated() -> None:
    axis = FactorAxis.from_keys(FACTOR_KEYS)
    calculator = build_calculator(axis)
    reordered = FactorAxis.from_keys(["EQ_B", "EQ_A", "FX_C"])
    with pytest.raises(SensitivityPnlInputError, match="does not match"):
        assess_taylor_validity(
            calculator=calculator,
            market=reordered.shocks(np.zeros((3, 4), dtype=np.float32)),
        )


def test_a_rebuilt_axis_with_the_same_keys_is_accepted() -> None:
    """Axes compared by value, not identity — mirrors the calculator's own rule."""
    axis = FactorAxis.from_keys(FACTOR_KEYS)
    calculator = build_calculator(axis)
    rebuilt = FactorAxis.from_keys(list(FACTOR_KEYS))
    report = assess_taylor_validity(
        calculator=calculator, market=displacement_market(rebuilt)
    )
    assert report.grid.num_positions == 2


def test_log_convention_non_positive_shocked_level_is_rejected() -> None:
    """The same P0 guard protecting `calculator.run` protects this path too —
    and fires here at the SOLUTIONS tier (`validate_market_against_book`,
    naming the caller's own factor key) before the request ever reaches the
    core engine's own guard. A silent NaN here would poison
    `nonlinearity_fraction` exactly as it would poison a PnL grid."""
    axis = FactorAxis.from_keys(["LOGFAC"])
    sensitivities = pl.DataFrame(
        [(1, "LOGFAC", 10.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.String,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": ["LOGFAC"],
            "displacement_convention_id": [int(DisplacementConvention.LOG)],
            "quote_base": [1.0],
        }
    )
    calculator = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sensitivities,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=[1],
        )
    )
    market = axis.levels(
        base_levels=np.array([2.0]),
        shocked_levels=np.ascontiguousarray(np.array([[-1.0]])),
    )
    with pytest.raises(SensitivityPnlInputError, match="non-positive shocked level"):
        assess_taylor_validity(calculator=calculator, market=market)


# =========================================================================
# assess_characteristic_displacement
# =========================================================================


def test_characteristic_displacement_reports_caller_keys_under_slot_encoding() -> None:
    axis = FactorAxis.from_keys(FACTOR_KEYS)  # string keys -> SLOT
    calculator = build_calculator(axis)
    assert calculator.axis.slot_map is not None

    report = assess_characteristic_displacement(calculator)
    by_key = dict(zip(report.factor_keys, report.x_star.tolist(), strict=True))
    # factor EQ_A on position 1: x* = 2*100/40 = 5.0
    assert by_key["EQ_A"] == pytest.approx(5.0)
    # factor EQ_B on position 1: second_order == 0 -> no breakdown scale
    assert np.isinf(by_key["EQ_B"])
    # factor FX_C on position 2: x* = 2*10/5000 = 0.004
    assert by_key["FX_C"] == pytest.approx(0.004)
    assert all(isinstance(key, str) for key in report.factor_keys)


def test_characteristic_displacement_reports_caller_keys_under_direct_encoding() -> (
    None
):
    axis = FactorAxis.from_keys([100, 200, 300])  # sorted ints -> DIRECT
    assert axis.slot_map is None
    sensitivities = pl.DataFrame(
        [(1, 100, 100.0, 40.0), (1, 200, -50.0, 0.0)],
        schema={
            "position_id": pl.Int64,
            "factor_id": pl.Int64,
            "first_order": pl.Float64,
            "second_order": pl.Float64,
        },
        orient="row",
    )
    meta = pl.DataFrame(
        {
            "factor_id": [100, 200, 300],
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 3,
            "quote_base": [1.0] * 3,
        }
    )
    calculator = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sensitivities,
            factor_meta=meta,
            factor_axis=axis,
            position_uids=[1],
        )
    )
    report = assess_characteristic_displacement(calculator)
    by_key = dict(zip(report.factor_keys, report.x_star.tolist(), strict=True))
    assert by_key[100] == pytest.approx(5.0)
    assert np.isinf(by_key[200])


def test_sorted_by_breakdown_orders_ascending_with_inf_last() -> None:
    axis = FactorAxis.from_keys(FACTOR_KEYS)
    calculator = build_calculator(axis)
    report = assess_characteristic_displacement(calculator).sorted_by_breakdown()
    assert report.x_star.tolist() == sorted(report.x_star.tolist())
    assert report.factor_keys[0] == "FX_C"  # x* = 0.004, the smallest
    assert np.isinf(report.x_star[-1])  # EQ_B's x* = inf, sorted last


def test_characteristic_displacement_needs_no_market_at_all() -> None:
    """Scenario-free by construction: calling it costs nothing beyond the
    compiled book already sitting in the calculator."""
    axis = FactorAxis.from_keys(FACTOR_KEYS)
    calculator = build_calculator(axis)
    report = assess_characteristic_displacement(calculator)
    assert (
        report.position_uids.shape[0]
        == report.x_star.shape[0]
        == len(report.factor_keys)
    )
