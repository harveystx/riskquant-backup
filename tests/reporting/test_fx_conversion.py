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
    )
    np.testing.assert_allclose(out.converted_pnl, [[5.0, -7.0]])


# =============================================================================
# The convention decision (the load-bearing test)
# =============================================================================


def test_moving_fx_distinguishes_the_two_conventions() -> None:
    """
    Economic distinction, not just a number.

    Brent: Q=-50, priced USD, SETTLED EUR, reported USD.
        V_eur = Q * B * F           (F = USDEUR)

    Convention (a), implemented here -- convert the settlement-currency P&L
    at the SCENARIO rate:
        dV_usd = (Q*B1*F1 - Q*B0*F0) / F1

    Convention (b), rejected -- convert shocked and base VALUES separately
    then difference:
        dV_usd = Q*B1 - Q*B0
    In (b) the FX factor cancels algebraically, so the position shows NO FX
    exposure at all -- flatly contradicting the sensitivity book, which
    carries dV/dUSDEUR = -4115 for this trade.

    With FX moving the two disagree in SIGN, so this is a correctness choice.
    """
    qty, brent_base, fx_base = -50.0, 82.30, 1.08
    brent_shocked, fx_shocked = 81.481104, 1.101817

    settlement_pnl_eur = qty * brent_shocked * fx_shocked - qty * brent_base * fx_base

    plan = build_fx_conversion_plan(
        settlement_ccys=["EUR"], reporting_ccy="USD", fx_factor_names=["USDEUR"]
    )
    produced = convert_scenario_pnl_to_reporting_currency(
        settlement_pnl=_f64([[settlement_pnl_eur]]),
        fx_levels=_f64([[fx_shocked]]),
        plan=plan,
    )
    convention_a = settlement_pnl_eur / fx_shocked
    convention_b = qty * brent_shocked - qty * brent_base

    assert float(produced.converted_pnl[0, 0]) == pytest.approx(convention_a, abs=1e-9)
    assert float(produced.converted_pnl[0, 0]) == pytest.approx(-40.536, abs=1e-3)

    # The two conventions genuinely disagree, and in sign.
    assert convention_a < 0.0 < convention_b
    assert float(produced.converted_pnl[0, 0]) != pytest.approx(convention_b, abs=1e-3)


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
        settlement_pnl=settlement_pnl, fx_levels=fx_levels, plan=plan
    )
    expected = np.array(
        [
            [100.0 * 1.08, 50.0 * 1.10],
            [10.0 * 1.15, -20.0 * 1.20],
            [7.0, 7.0],
        ]
    )
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
        )
