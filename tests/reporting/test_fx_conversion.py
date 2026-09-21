# tests/reporting/test_fx_conversion.py
"""
Tests for the reporting-currency conversion / aggregation layer.

Covers the convention choice itself (settlement P&L converted at the SCENARIO
FX rate), the direct / inverse / identity FX routes, multi-currency
aggregation ordering, and the fail-fast boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.constants import INVALID_UIX
from riskquant.reporting.fx_conversion import (
    build_fx_conversion_plan,
    convert_scenario_pnl_to_reporting_currency,
)


def _f64(rows) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float64))


# =============================================================================
# FX routing
# =============================================================================


def test_direct_pair_is_resolved_and_multiplies() -> None:
    """USD -> EUR with a quoted USDEUR factor multiplies by the rate."""
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD"], reporting_ccy="EUR", fx_factor_names=["USDEUR"]
    )
    assert plan.fx_row_index.tolist() == [0]
    assert plan.invert.tolist() == [0]

    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=_f64([[100.0, -200.0]]),
        fx_levels=_f64([[1.08, 1.10]]),
        plan=plan,
        base_settlement_values=np.zeros(
            np.asarray(_f64([[100.0, -200.0]])).shape[0], dtype=np.float64
        ),
        base_fx_levels=np.ones(
            np.asarray(_f64([[1.08, 1.10]])).shape[0], dtype=np.float64
        ),
    )
    np.testing.assert_allclose(out.converted_pnl, [[108.0, -220.0]])


def test_inverse_pair_is_resolved_and_divides() -> None:
    """EUR -> USD with only USDEUR quoted uses the reciprocal."""
    plan = build_fx_conversion_plan(
        settlement_ccys=["EUR"], reporting_ccy="USD", fx_factor_names=["USDEUR"]
    )
    assert plan.invert.tolist() == [1]

    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=_f64([[108.0, -220.0]]),
        fx_levels=_f64([[1.08, 1.10]]),
        plan=plan,
        base_settlement_values=np.zeros(
            np.asarray(_f64([[108.0, -220.0]])).shape[0], dtype=np.float64
        ),
        base_fx_levels=np.ones(
            np.asarray(_f64([[1.08, 1.10]])).shape[0], dtype=np.float64
        ),
    )
    np.testing.assert_allclose(out.converted_pnl, [[100.0, -200.0]])


def test_same_currency_is_an_explicit_identity() -> None:
    """Identity is explicit routing, not a 1.0 fallback for a missing rate."""
    plan = build_fx_conversion_plan(
        settlement_ccys=["EUR"], reporting_ccy="EUR", fx_factor_names=[]
    )
    assert plan.fx_row_index.tolist() == [int(INVALID_UIX)]
    assert bool(plan.is_identity[0])

    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=_f64([[5.0, -7.0]]),
        fx_levels=np.empty((0, 2), dtype=np.float64),
        plan=plan,
        base_settlement_values=np.zeros(
            np.asarray(_f64([[5.0, -7.0]])).shape[0], dtype=np.float64
        ),
        base_fx_levels=np.ones(
            np.asarray(np.empty((0, 2), dtype=np.float64)).shape[0], dtype=np.float64
        ),
    )
    np.testing.assert_allclose(out.converted_pnl, [[5.0, -7.0]])


# =============================================================================
# The convention decision (the load-bearing test)
# =============================================================================


def _l0_reporting_oracle(v0: float, vs: float, q0: float, qs: float) -> float:
    """Independent L0 reference: ordinary MTM reporting P&L, W = V*q."""
    return vs * qs - v0 * q0


def _convert_one(*, dv, qs, v0, q0, settle="EUR", report="USD", pair="EURUSD"):
    plan = build_fx_conversion_plan(
        settlement_ccys=[settle], reporting_ccy=report, fx_factor_names=[pair]
    )
    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=np.array([[dv]], dtype=np.float64),
        fx_levels=np.array([[qs]], dtype=np.float64),
        plan=plan,
        base_settlement_values=np.array([v0], dtype=np.float64),
        base_fx_levels=np.array([q0], dtype=np.float64),
    )
    return float(out.converted_pnl[0, 0])


def test_ordinary_mtm_value_and_fx_move() -> None:
    """D081 §39.1: V0=100, Vs=110 EUR; q 1.10 -> 1.20 USD/EUR; dW = 22."""
    assert _convert_one(dv=10.0, qs=1.20, v0=100.0, q0=1.10) == pytest.approx(
        _l0_reporting_oracle(100.0, 110.0, 1.10, 1.20)
    )
    assert _l0_reporting_oracle(100.0, 110.0, 1.10, 1.20) == pytest.approx(22.0)


def test_ordinary_mtm_fx_only_translates_the_base_value() -> None:
    """D081 §39.2: unchanged EUR value, moving FX -> dW = V0*(qs-q0) = 10.
    This is the regression pin against the former D014 form (dV*qs = 0)."""
    assert _convert_one(dv=0.0, qs=1.20, v0=100.0, q0=1.10) == pytest.approx(10.0)


def test_ordinary_mtm_value_only_is_q_times_dv() -> None:
    """D081 §39.3: q constant -> dW = q*dV, no translation term."""
    assert _convert_one(dv=10.0, qs=1.10, v0=100.0, q0=1.10) == pytest.approx(11.0)


def test_ordinary_mtm_zero_movement_is_zero() -> None:
    assert _convert_one(dv=0.0, qs=1.10, v0=100.0, q0=1.10) == 0.0


def test_ordinary_mtm_zero_base_value_isolates_scenario_translation() -> None:
    """D081 §39.5: V0 = 0 -> dW = dV*qs (the only case where D014 was exact)."""
    assert _convert_one(dv=10.0, qs=1.20, v0=0.0, q0=1.10) == pytest.approx(12.0)


def test_ordinary_mtm_negative_value_follows_ordinary_multiplication() -> None:
    """D081 §39.6: a short EUR position LOSES when EURUSD rises."""
    assert _convert_one(dv=0.0, qs=1.20, v0=-100.0, q0=1.10) == pytest.approx(-10.0)
    assert _convert_one(dv=-10.0, qs=1.20, v0=-100.0, q0=1.10) == pytest.approx(
        _l0_reporting_oracle(-100.0, -110.0, 1.10, 1.20)
    )


def test_ordinary_mtm_reciprocal_pair_is_invariant() -> None:
    """D081 §39.7: quoting USDEUR instead of EURUSD changes nothing."""
    direct = _convert_one(dv=10.0, qs=1.20, v0=100.0, q0=1.10, pair="EURUSD")
    recip = _convert_one(dv=10.0, qs=1.0 / 1.20, v0=100.0, q0=1.0 / 1.10, pair="USDEUR")
    assert direct == pytest.approx(recip) == pytest.approx(22.0)


def test_ordinary_mtm_brent_internal_fx_cancels() -> None:
    """
    D081 §39.8 (the former D014 example, now under ordinary MTM).

    Brent: Q=-50, priced USD, SETTLED EUR, reported USD; V_eur = Q*B*F with
    F = USDEUR. The USD value is W = V_eur / F = Q*B: a USD-priced asset has
    NO USDEUR exposure for a USD reporter, so ordinary MTM must return exactly
    Q*(B1-B0) and the internal FX factor must cancel. The former D014 form
    (dV*qs) returned -40.5 here -- the wrong sign on a short into a falling
    price -- because it omitted V0*(qs-q0).
    """
    qty, b0, f0 = -50.0, 82.30, 1.08
    b1, f1 = 81.481104, 1.101817
    v0, v1 = qty * b0 * f0, qty * b1 * f1
    produced = _convert_one(dv=v1 - v0, qs=f1, v0=v0, q0=f0, pair="USDEUR")
    exact = _l0_reporting_oracle(v0, v1, 1.0 / f0, 1.0 / f1)
    assert produced == pytest.approx(exact)
    assert produced == pytest.approx(qty * (b1 - b0))  # +40.9448, no FX term
    assert produced > 0.0  # short position, price fell


def test_missing_base_inputs_fail_closed() -> None:
    """The D014 form cannot be requested: base inputs are required keywords."""
    plan = build_fx_conversion_plan(
        settlement_ccys=["EUR"], reporting_ccy="USD", fx_factor_names=["EURUSD"]
    )
    with pytest.raises(TypeError):
        convert_scenario_pnl_to_reporting_currency(  # type: ignore[call-arg]
            settlement_pnl=np.array([[10.0]]), fx_levels=np.array([[1.2]]), plan=plan
        )
    with pytest.raises(ValueError, match="base_fx_levels"):
        convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=np.array([[10.0]]),
            fx_levels=np.array([[1.2]]),
            plan=plan,
            base_settlement_values=np.array([100.0]),
            base_fx_levels=np.array([0.0]),
        )


def test_static_fx_makes_the_two_conventions_coincide() -> None:
    """With FX unchanged the distinction vanishes -- a control for the above."""
    qty, brent_base, fx = -50.0, 82.30, 1.08
    brent_shocked = 80.670354

    settlement_pnl_eur = qty * brent_shocked * fx - qty * brent_base * fx
    plan = build_fx_conversion_plan(
        settlement_ccys=["EUR"], reporting_ccy="USD", fx_factor_names=["USDEUR"]
    )
    produced = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=_f64([[settlement_pnl_eur]]),
        fx_levels=_f64([[fx]]),
        plan=plan,
        base_settlement_values=np.zeros(
            np.asarray(_f64([[settlement_pnl_eur]])).shape[0], dtype=np.float64
        ),
        base_fx_levels=np.ones(np.asarray(_f64([[fx]])).shape[0], dtype=np.float64),
    )
    convention_b = qty * brent_shocked - qty * brent_base
    assert float(produced.converted_pnl[0, 0]) == pytest.approx(convention_b, abs=1e-9)


# =============================================================================
# Multi-currency aggregation
# =============================================================================


def test_multi_currency_aggregation_converts_before_summing() -> None:
    """
    USD -> EUR, GBP -> EUR, EUR -> EUR in one plan. The portfolio total must
    equal the sum of CONVERTED rows, never a conversion of a mixed-currency
    sum.
    """
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD", "GBP", "EUR"],
        reporting_ccy="EUR",
        fx_factor_names=["USDEUR", "GBPEUR"],
    )
    assert plan.fx_row_index.tolist() == [0, 1, int(INVALID_UIX)]
    assert plan.invert.tolist() == [0, 0, 0]

    settlement_pnl = _f64([[100.0, 50.0], [10.0, -20.0], [7.0, 7.0]])
    fx_levels = _f64([[1.08, 1.10], [1.15, 1.20]])

    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=settlement_pnl,
        fx_levels=fx_levels,
        plan=plan,
        base_settlement_values=np.zeros(settlement_pnl.shape[0], dtype=np.float64),
        base_fx_levels=np.ones(fx_levels.shape[0], dtype=np.float64),
    )
    expected = np.array([
        [100.0 * 1.08, 50.0 * 1.10],
        [10.0 * 1.15, -20.0 * 1.20],
        [7.0, 7.0],
    ])
    np.testing.assert_allclose(out.converted_pnl, expected)
    np.testing.assert_allclose(out.portfolio_pnl, expected.sum(axis=0))

    # The naive "sum then convert" reading would be different and wrong.
    naive = settlement_pnl.sum(axis=0) * fx_levels[0]
    assert not np.allclose(out.portfolio_pnl, naive)


def test_offsetting_positions_and_zero_pnl() -> None:
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD", "USD"],
        reporting_ccy="EUR",
        fx_factor_names=["USDEUR"],
    )
    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=_f64([[100.0, 0.0], [-100.0, 0.0]]),
        fx_levels=_f64([[1.08, 1.08]]),
        plan=plan,
        base_settlement_values=np.zeros(
            np.asarray(_f64([[100.0, 0.0], [-100.0, 0.0]])).shape[0], dtype=np.float64
        ),
        base_fx_levels=np.ones(
            np.asarray(_f64([[1.08, 1.08]])).shape[0], dtype=np.float64
        ),
    )
    np.testing.assert_allclose(out.portfolio_pnl, [0.0, 0.0])


def test_result_shape_dtype_and_currency_label() -> None:
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD", "EUR"],
        reporting_ccy="USD",
        fx_factor_names=["USDEUR"],
    )
    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=_f64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        fx_levels=_f64([[1.08, 1.09, 1.10]]),
        plan=plan,
        base_settlement_values=np.zeros(
            np.asarray(_f64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).shape[0],
            dtype=np.float64,
        ),
        base_fx_levels=np.ones(
            np.asarray(_f64([[1.08, 1.09, 1.10]])).shape[0], dtype=np.float64
        ),
    )
    assert out.converted_pnl.shape == (2, 3)
    assert out.portfolio_pnl.shape == (3,)
    assert out.converted_pnl.dtype == np.float64
    assert out.converted_pnl.flags["C_CONTIGUOUS"]
    assert out.portfolio_pnl.flags["C_CONTIGUOUS"]
    assert out.reporting_ccy == "USD"
    assert out.num_positions == 2
    assert out.num_scenarios == 3


# =============================================================================
# Fail fast
# =============================================================================


def test_missing_fx_pair_raises() -> None:
    with pytest.raises(ValueError, match="No FX factor available"):
        build_fx_conversion_plan(
            settlement_ccys=["JPY"],
            reporting_ccy="USD",
            fx_factor_names=["USDEUR"],
        )


def test_invalid_currency_codes_raise() -> None:
    with pytest.raises(ValueError):
        build_fx_conversion_plan(
            settlement_ccys=["usd"], reporting_ccy="EUR", fx_factor_names=["USDEUR"]
        )
    with pytest.raises(ValueError):
        build_fx_conversion_plan(
            settlement_ccys=["USD"], reporting_ccy="EU", fx_factor_names=["USDEUR"]
        )
    with pytest.raises(TypeError):
        build_fx_conversion_plan(
            settlement_ccys=[None],  # type: ignore[list-item]
            reporting_ccy="EUR",
            fx_factor_names=["USDEUR"],
        )


def test_duplicate_fx_factor_name_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate FX factor name"):
        build_fx_conversion_plan(
            settlement_ccys=["USD"],
            reporting_ccy="EUR",
            fx_factor_names=["USDEUR", "USDEUR"],
        )


def test_scenario_count_mismatch_raises() -> None:
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD"], reporting_ccy="EUR", fx_factor_names=["USDEUR"]
    )
    with pytest.raises(ValueError, match="scenario"):
        convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=_f64([[1.0, 2.0, 3.0]]),
            fx_levels=_f64([[1.08, 1.09]]),
            plan=plan,
            base_settlement_values=np.zeros(
                np.asarray(_f64([[1.0, 2.0, 3.0]])).shape[0], dtype=np.float64
            ),
            base_fx_levels=np.ones(
                np.asarray(_f64([[1.08, 1.09]])).shape[0], dtype=np.float64
            ),
        )


def test_position_count_mismatch_raises() -> None:
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD"], reporting_ccy="EUR", fx_factor_names=["USDEUR"]
    )
    with pytest.raises(ValueError, match="position"):
        convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=_f64([[1.0], [2.0]]),
            fx_levels=_f64([[1.08]]),
            plan=plan,
            base_settlement_values=np.zeros(
                np.asarray(_f64([[1.0], [2.0]])).shape[0], dtype=np.float64
            ),
            base_fx_levels=np.ones(
                np.asarray(_f64([[1.08]])).shape[0], dtype=np.float64
            ),
        )


@pytest.mark.parametrize("bad_rate", [0.0, -1.08, np.nan, np.inf])
def test_unusable_fx_rate_raises(bad_rate: float) -> None:
    """A non-finite or non-positive rate is rejected, never silently used."""
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD"], reporting_ccy="EUR", fx_factor_names=["USDEUR"]
    )
    with pytest.raises(ValueError):
        convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=_f64([[1.0]]),
            fx_levels=_f64([[bad_rate]]),
            plan=plan,
            base_settlement_values=np.zeros(
                np.asarray(_f64([[1.0]])).shape[0], dtype=np.float64
            ),
            base_fx_levels=np.ones(
                np.asarray(_f64([[bad_rate]])).shape[0], dtype=np.float64
            ),
        )


def test_non_finite_settlement_pnl_raises() -> None:
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD"], reporting_ccy="EUR", fx_factor_names=["USDEUR"]
    )
    with pytest.raises(ValueError, match="finite"):
        convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=_f64([[np.nan]]),
            fx_levels=_f64([[1.08]]),
            plan=plan,
            base_settlement_values=np.zeros(
                np.asarray(_f64([[np.nan]])).shape[0], dtype=np.float64
            ),
            base_fx_levels=np.ones(
                np.asarray(_f64([[1.08]])).shape[0], dtype=np.float64
            ),
        )


def test_float32_pnl_is_rejected_at_the_boundary() -> None:
    """The float32 -> float64 promotion must be explicit at the caller."""
    plan = build_fx_conversion_plan(
        settlement_ccys=["USD"], reporting_ccy="EUR", fx_factor_names=["USDEUR"]
    )
    with pytest.raises(TypeError, match="float64"):
        convert_scenario_pnl_to_reporting_currency(
            settlement_pnl=np.asarray([[1.0]], dtype=np.float32),
            fx_levels=_f64([[1.08]]),
            plan=plan,
            base_settlement_values=np.zeros(
                np.asarray(np.asarray([[1.0]], dtype=np.float32)).shape[0],
                dtype=np.float64,
            ),
            base_fx_levels=np.ones(
                np.asarray(_f64([[1.08]])).shape[0], dtype=np.float64
            ),
        )


# =============================================================================
# D081 §9 — FULL / Taylor reconciliation identity
# =============================================================================
def test_reporting_transformation_preserves_the_full_vs_taylor_settlement_error():
    """
    dW_FULL - dW_T = q_s * (dV_FULL - dV_T): the base-value translation term
    V0*(q_s - q0) is identical on both sides and cancels EXACTLY, so reporting
    conversion cannot mask or manufacture a Taylor approximation error. Both
    settlement value and FX move here; nothing trivial cancels by accident.
    """
    plan = build_fx_conversion_plan(
        settlement_ccys=["EUR", "EUR"], reporting_ccy="USD", fx_factor_names=["EURUSD"]
    )
    v0 = np.array([100.0, -250.0])
    q0 = np.array([1.10])
    qs = np.array([[1.20, 1.05, 0.95]])
    dv_full = np.array([[10.0, -4.0, 7.5], [-30.0, 12.0, 1.0]])
    dv_taylor = np.array([
        [9.4, -4.6, 7.9],
        [-31.2, 11.1, 0.4],
    ])  # deliberately different
    kw = dict(fx_levels=qs, plan=plan, base_settlement_values=v0, base_fx_levels=q0)
    w_full = convert_scenario_pnl_to_reporting_currency(settlement_pnl=dv_full, **kw)
    w_tay = convert_scenario_pnl_to_reporting_currency(settlement_pnl=dv_taylor, **kw)
    # independent right-hand side: q_s * (dV_FULL - dV_T), no reporting code
    rhs = qs * (dv_full - dv_taylor)
    np.testing.assert_allclose(
        w_full.converted_pnl - w_tay.converted_pnl, rhs, rtol=0, atol=1e-12
    )
    # and the translation term is present on both sides (not trivially zero)
    assert np.all(np.abs(v0[:, None] * (qs - q0)) > 0.0)


# =============================================================================
# D081 §6 — Architecture A: reporting FX enters exactly once, at the reporting layer
# =============================================================================
def test_reporting_fx_is_applied_exactly_once_and_never_as_a_taylor_factor():
    """
    Architecture A guard. Take a book whose settlement valuation does NOT depend
    on the reporting pair and move ONLY that pair. Then:
      * the settlement Taylor P&L must be exactly zero (the reporting FX is not a
        settlement factor, so it must not have been Taylor-expanded);
      * the reporting P&L must be exactly V0*(q_s - q0) -- the FX effect enters
        once, through the exact translation, never a second time.
    The same pair MAY legitimately be a settlement factor for another position
    (a quanto), in which case its settlement Greek is a valuation dependency;
    that case is covered by the D019 demo's hand loop, not forbidden here.
    """
    import polars as pl

    from riskquant.sensitivity_pnl.conventions import DisplacementConvention
    from riskquant.solutions.sensitivity_pnl.api import (
        FactorAxis,
        SensitivityBookSpec,
        SensitivityPnlCalculator,
    )

    equity_rf, reporting_fx_rf = 1, 2
    axis = FactorAxis.from_keys([equity_rf, reporting_fx_rf])
    spec = SensitivityBookSpec(
        sensitivities=pl.DataFrame({
            "position_id": [7],
            "factor_id": [equity_rf],  # NO reporting-FX row: V does not depend on it
            "first_order": [10.0],
            "second_order": [0.0],
        }),
        factor_meta=pl.DataFrame({
            "factor_id": [equity_rf, reporting_fx_rf],
            "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * 2,
            "quote_base": [1.0, 1.0],
        }),
        factor_axis=axis,
        position_uids=[7],
    )
    calc = SensitivityPnlCalculator.compile(spec)
    base = np.array([100.0, 1.10])
    shocked = np.array([
        [100.0, 100.0],
        [1.20, 0.90],
    ])  # equity flat; reporting FX moves
    dv = np.asarray(
        calc.run(axis.levels(base_levels=base, shocked_levels=shocked)).pnl,
        dtype=np.float64,
    )
    assert np.all(
        dv == 0.0
    )  # reporting FX was NOT Taylor-expanded as a settlement factor

    plan = build_fx_conversion_plan(
        settlement_ccys=["EUR"], reporting_ccy="USD", fx_factor_names=["EURUSD"]
    )
    out = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=dv,
        fx_levels=shocked[1:2],
        plan=plan,
        base_settlement_values=np.array([500.0]),
        base_fx_levels=base[1:2],
    )
    np.testing.assert_allclose(
        out.converted_pnl, 500.0 * (shocked[1:2] - 1.10), atol=1e-12
    )
