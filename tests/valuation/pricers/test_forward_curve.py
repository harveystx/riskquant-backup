# tests/valuation/pricers/test_forward_curve.py
from __future__ import annotations

import numpy as np
import pytest

from riskquant.portfolio.blocks.forward_block import ForwardCurveBlock
from riskquant.market_data.state import MarketDataMatrix
from riskquant.valuation.pricers.forward_curve import price_forward_curve_block
from riskquant.constants import (
    NO_FX_RF_REQUIRED,
    NO_DISCOUNT_CURVE_ID,
    NO_DATE_UIX,
)

# =========================================================================
# CONSTANTS
# =========================================================================
N_SCENARIOS = 4


# =========================================================================
# HELPERS
# =========================================================================


def _make_forward_curve_block(
    *,
    n: int,
    qty: np.ndarray,
    strike: np.ndarray,
    price_curve_uix: np.ndarray,
    delivery_date_uix: np.ndarray,
    settlement_date_uix: np.ndarray | None = None,
    fx_under_to_settle_uix: np.ndarray | None = None,
    fx_strike_to_settle_uix: np.ndarray | None = None,
    fx_under_to_settle_invert: np.ndarray | None = None,
    fx_strike_to_settle_invert: np.ndarray | None = None,
    under_discount_curve_uix: np.ndarray | None = None,
    strike_discount_curve_uix: np.ndarray | None = None,
) -> ForwardCurveBlock:
    sentinel_i32 = lambda v: np.full(n, v, dtype=np.int32)
    return ForwardCurveBlock(
        trade_uids=np.arange(n, dtype=np.int64),
        qty=qty,
        strike=strike,
        price_curve_uix=price_curve_uix,
        delivery_date_uix=delivery_date_uix,
        settlement_date_uix=(
            settlement_date_uix
            if settlement_date_uix is not None
            else sentinel_i32(NO_DATE_UIX)
        ),
        fx_under_to_settle_uix=(
            fx_under_to_settle_uix
            if fx_under_to_settle_uix is not None
            else sentinel_i32(NO_FX_RF_REQUIRED)
        ),
        fx_strike_to_settle_uix=(
            fx_strike_to_settle_uix
            if fx_strike_to_settle_uix is not None
            else sentinel_i32(NO_FX_RF_REQUIRED)
        ),
        fx_under_to_settle_invert=(
            fx_under_to_settle_invert
            if fx_under_to_settle_invert is not None
            else np.zeros(n, dtype=np.bool_)
        ),
        fx_strike_to_settle_invert=(
            fx_strike_to_settle_invert
            if fx_strike_to_settle_invert is not None
            else np.zeros(n, dtype=np.bool_)
        ),
        under_discount_curve_uix=(
            under_discount_curve_uix
            if under_discount_curve_uix is not None
            else sentinel_i32(NO_DISCOUNT_CURVE_ID)
        ),
        strike_discount_curve_uix=(
            strike_discount_curve_uix
            if strike_discount_curve_uix is not None
            else sentinel_i32(NO_DISCOUNT_CURVE_ID)
        ),
    )


def _make_market_data(
    *,
    price_curve_grid: np.ndarray,
    fx_spots: np.ndarray | None = None,
    discount_factor_grid: np.ndarray | None = None,
    n_scenarios: int = N_SCENARIOS,
) -> MarketDataMatrix:
    s = n_scenarios
    if fx_spots is None:
        fx_spots = np.empty((0, s), dtype=np.float64)
    if discount_factor_grid is None:
        discount_factor_grid = np.empty((0, 0, s), dtype=np.float64)
    return MarketDataMatrix(
        num_scenarios=s,
        fx_spots=fx_spots,
        prices=np.empty((0, s), dtype=np.float64),
        forward_nodes=np.empty((0, s), dtype=np.float64),
        vol_nodes=np.empty((0, s), dtype=np.float64),
        discount_factor_grid=discount_factor_grid,
        price_curve_grid=price_curve_grid,
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, s), dtype=np.float64),
    )


# =========================================================================
# TESTS — Basic PV = Qty * (CurveFwd - Strike)
# =========================================================================


def test_simple_forward_curve_no_fx_no_df() -> None:
    """PV = qty * (curve_fwd - strike) with no FX and no discounting."""
    # 1 curve, 2 dates, N_SCENARIOS
    pcg = np.zeros((1, 2, N_SCENARIOS))
    pcg[0, 0, :] = 50.0
    pcg[0, 1, :] = 110.0

    block = _make_forward_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array(
            [1], dtype=np.int32
        ),  # points to date index 1 => fwd=110
    )
    md = _make_market_data(price_curve_grid=pcg)

    result = price_forward_curve_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result, 10.0)


def test_multiple_delivery_dates() -> None:
    """Trades pointing to different delivery dates on same curve."""
    pcg = np.zeros((1, 3, N_SCENARIOS))
    pcg[0, 0, :] = 80.0
    pcg[0, 1, :] = 100.0
    pcg[0, 2, :] = 120.0

    block = _make_forward_curve_block(
        n=2,
        qty=np.array([1.0, 1.0]),
        strike=np.array([90.0, 110.0]),
        price_curve_uix=np.array([0, 0], dtype=np.int32),
        delivery_date_uix=np.array([0, 2], dtype=np.int32),
    )
    md = _make_market_data(price_curve_grid=pcg)

    result = price_forward_curve_block(block=block, market_data_mtx=md)

    # Trade 0: 1 * (80 - 90) = -10
    np.testing.assert_allclose(result[0, :], -10.0)
    # Trade 1: 1 * (120 - 110) = 10
    np.testing.assert_allclose(result[1, :], 10.0)


def test_per_scenario_curve_fwd() -> None:
    """Curve forward prices vary per scenario."""
    pcg = np.zeros((1, 1, 3))
    pcg[0, 0, :] = [90.0, 100.0, 110.0]

    block = _make_forward_curve_block(
        n=1,
        qty=np.array([2.0]),
        strike=np.array([100.0]),
        price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(price_curve_grid=pcg, n_scenarios=3)

    result = price_forward_curve_block(block=block, market_data_mtx=md)

    expected = np.array(
        [[2.0 * (90.0 - 100.0), 2.0 * (100.0 - 100.0), 2.0 * (110.0 - 100.0)]]
    )
    np.testing.assert_allclose(result, expected)


# =========================================================================
# TESTS — FX and Discounting
# =========================================================================


def test_underlying_fx_on_curve() -> None:
    """FX applied to the underlying curve leg."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    fx = np.full((1, N_SCENARIOS), 1.50)

    block = _make_forward_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([140.0]),
        price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        fx_under_to_settle_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(price_curve_grid=pcg, fx_spots=fx)

    result = price_forward_curve_block(block=block, market_data_mtx=md)

    # fwd leg = 100 * 1.5 = 150; strike = 140; pv = 10
    np.testing.assert_allclose(result, 10.0)


def test_both_legs_fx_and_df() -> None:
    """Full forward curve calculation with FX and discounting on both legs."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    fx = np.zeros((2, N_SCENARIOS))
    fx[0, :] = 1.20
    fx[1, :] = 1.10
    df = np.zeros((2, 1, N_SCENARIOS))
    df[0, 0, :] = 0.98
    df[1, 0, :] = 0.97

    block = _make_forward_curve_block(
        n=1,
        qty=np.array([5.0]),
        strike=np.array([90.0]),
        price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        fx_under_to_settle_uix=np.array([0], dtype=np.int32),
        fx_strike_to_settle_uix=np.array([1], dtype=np.int32),
        under_discount_curve_uix=np.array([0], dtype=np.int32),
        strike_discount_curve_uix=np.array([1], dtype=np.int32),
        settlement_date_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(
        price_curve_grid=pcg, fx_spots=fx, discount_factor_grid=df
    )

    result = price_forward_curve_block(block=block, market_data_mtx=md)

    expected_under = 100.0 * 1.20 * 0.98
    expected_strike = 90.0 * 1.10 * 0.97
    expected = 5.0 * (expected_under - expected_strike)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_fx_inversion_on_strike_leg() -> None:
    """Strike FX with inversion."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    fx = np.full((1, N_SCENARIOS), 4.0)

    block = _make_forward_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        fx_strike_to_settle_uix=np.array([0], dtype=np.int32),
        fx_strike_to_settle_invert=np.array([True]),
    )
    md = _make_market_data(price_curve_grid=pcg, fx_spots=fx)

    result = price_forward_curve_block(block=block, market_data_mtx=md)

    # fwd=100; strike leg = 100 * (1/4) = 25; pv = 100 - 25 = 75
    np.testing.assert_allclose(result, 75.0)


def test_output_shape() -> None:
    """Output shape matches (n_trades, n_scenarios)."""
    n, s = 4, 6
    pcg = np.ones((2, 3, s))
    block = _make_forward_curve_block(
        n=n,
        qty=np.ones(n),
        strike=np.zeros(n),
        price_curve_uix=np.array([0, 0, 1, 1], dtype=np.int32),
        delivery_date_uix=np.array([0, 1, 0, 2], dtype=np.int32),
    )
    md = _make_market_data(price_curve_grid=pcg, n_scenarios=s)

    result = price_forward_curve_block(block=block, market_data_mtx=md)

    assert result.shape == (n, s)
