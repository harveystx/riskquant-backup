# tests/valuation/pricers/test_forward_node.py
from __future__ import annotations

import numpy as np
import pytest

from riskquant.portfolio.blocks.forward_block import ForwardNodeBlock
from riskquant.market_data.state import MarketDataMatrix
from riskquant.valuation.pricers.forward_node import price_forward_node_block
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


def _make_forward_node_block(
    *,
    n: int,
    qty: np.ndarray,
    strike: np.ndarray,
    price_uix: np.ndarray,
    settlement_date_uix: np.ndarray | None = None,
    fx_under_to_settle_uix: np.ndarray | None = None,
    fx_strike_to_settle_uix: np.ndarray | None = None,
    fx_under_to_settle_invert: np.ndarray | None = None,
    fx_strike_to_settle_invert: np.ndarray | None = None,
    under_discount_curve_uix: np.ndarray | None = None,
    strike_discount_curve_uix: np.ndarray | None = None,
) -> ForwardNodeBlock:
    sentinel_i32 = lambda v: np.full(n, v, dtype=np.int32)
    return ForwardNodeBlock(
        trade_uids=np.arange(n, dtype=np.int64),
        qty=qty,
        strike=strike,
        price_uix=price_uix,
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
    forward_nodes: np.ndarray,
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
        forward_nodes=forward_nodes,
        vol_nodes=np.empty((0, s), dtype=np.float64),
        discount_factor_grid=discount_factor_grid,
        price_curve_grid=np.empty((0, 0, s), dtype=np.float64),
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, s), dtype=np.float64),
    )


# =========================================================================
# TESTS — Basic PV = Qty * (Fwd - Strike)
# =========================================================================


def test_simple_forward_no_fx_no_df() -> None:
    """PV = qty * (fwd - strike) with no FX and no discounting."""
    fwd_prices = np.full((1, N_SCENARIOS), 110.0)
    block = _make_forward_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        price_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd_prices)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result, 10.0)


def test_short_position() -> None:
    """Short forward: qty=-1 produces negative PV when Fwd > Strike."""
    fwd_prices = np.full((1, N_SCENARIOS), 110.0)
    block = _make_forward_node_block(
        n=1,
        qty=np.array([-1.0]),
        strike=np.array([100.0]),
        price_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd_prices)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result, -10.0)


def test_multiple_trades() -> None:
    """Multiple trades with distinct forwards and strikes."""
    fwd_prices = np.full((2, N_SCENARIOS), 0.0)
    fwd_prices[0, :] = 120.0
    fwd_prices[1, :] = 80.0

    block = _make_forward_node_block(
        n=2,
        qty=np.array([2.0, -3.0]),
        strike=np.array([100.0, 90.0]),
        price_uix=np.array([0, 1], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd_prices)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    # Trade 0: 2 * (120 - 100) = 40
    np.testing.assert_allclose(result[0, :], 40.0)
    # Trade 1: -3 * (80 - 90) = 30
    np.testing.assert_allclose(result[1, :], 30.0)


def test_per_scenario_fwd() -> None:
    """Forward prices differ per scenario."""
    fwd_prices = np.array([[100.0, 110.0, 120.0, 130.0]])
    block = _make_forward_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([105.0]),
        price_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd_prices)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    expected = np.array([[-5.0, 5.0, 15.0, 25.0]])
    np.testing.assert_allclose(result, expected)


# =========================================================================
# TESTS — FX Translation
# =========================================================================


def test_underlying_fx() -> None:
    """FX applied to the underlying leg: PV = qty * (fwd*fx - strike)."""
    fwd_prices = np.full((1, N_SCENARIOS), 100.0)
    fx = np.full((1, N_SCENARIOS), 1.50)

    block = _make_forward_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([140.0]),
        price_uix=np.array([0], dtype=np.int32),
        fx_under_to_settle_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd_prices, fx_spots=fx)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    # fwd leg = 100 * 1.5 = 150; strike leg = 140; pv = 1 * (150 - 140) = 10
    np.testing.assert_allclose(result, 10.0)


def test_strike_fx() -> None:
    """FX applied to the strike leg."""
    fwd_prices = np.full((1, N_SCENARIOS), 100.0)
    fx = np.full((1, N_SCENARIOS), 2.0)

    block = _make_forward_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([40.0]),
        price_uix=np.array([0], dtype=np.int32),
        fx_strike_to_settle_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd_prices, fx_spots=fx)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    # fwd leg = 100; strike leg = 40 * 2 = 80; pv = 1 * (100 - 80) = 20
    np.testing.assert_allclose(result, 20.0)


def test_fx_inversion_underlying() -> None:
    """Underlying FX with inversion."""
    fwd_prices = np.full((1, N_SCENARIOS), 100.0)
    fx = np.full((1, N_SCENARIOS), 2.0)

    block = _make_forward_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([0.0]),
        price_uix=np.array([0], dtype=np.int32),
        fx_under_to_settle_uix=np.array([0], dtype=np.int32),
        fx_under_to_settle_invert=np.array([True]),
    )
    md = _make_market_data(forward_nodes=fwd_prices, fx_spots=fx)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    # fwd leg = 100 * (1/2) = 50; strike leg = 0; pv = 50
    np.testing.assert_allclose(result, 50.0)


# =========================================================================
# TESTS — Discounting
# =========================================================================


def test_underlying_discounting() -> None:
    """Discount factor applied to the underlying leg."""
    fwd_prices = np.full((1, N_SCENARIOS), 100.0)
    # 1 curve, 1 date, N_SCENARIOS
    df_grid = np.full((1, 1, N_SCENARIOS), 0.95)

    block = _make_forward_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([0.0]),
        price_uix=np.array([0], dtype=np.int32),
        under_discount_curve_uix=np.array([0], dtype=np.int32),
        settlement_date_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(
        forward_nodes=fwd_prices, discount_factor_grid=df_grid
    )

    result = price_forward_node_block(block=block, market_data_mtx=md)

    # fwd leg = 100 * 0.95 = 95; strike leg = 0; pv = 95
    np.testing.assert_allclose(result, 95.0)


def test_both_legs_discounted_and_fx() -> None:
    """Full test: both legs with FX and discounting."""
    fwd_prices = np.full((1, N_SCENARIOS), 100.0)
    fx = np.full((2, N_SCENARIOS), 0.0)
    fx[0, :] = 1.20  # underlying FX
    fx[1, :] = 1.10  # strike FX
    df_grid = np.full((2, 1, N_SCENARIOS), 0.0)
    df_grid[0, 0, :] = 0.98  # underlying DF
    df_grid[1, 0, :] = 0.97  # strike DF

    block = _make_forward_node_block(
        n=1,
        qty=np.array([5.0]),
        strike=np.array([90.0]),
        price_uix=np.array([0], dtype=np.int32),
        fx_under_to_settle_uix=np.array([0], dtype=np.int32),
        fx_strike_to_settle_uix=np.array([1], dtype=np.int32),
        under_discount_curve_uix=np.array([0], dtype=np.int32),
        strike_discount_curve_uix=np.array([1], dtype=np.int32),
        settlement_date_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(
        forward_nodes=fwd_prices, fx_spots=fx, discount_factor_grid=df_grid
    )

    result = price_forward_node_block(block=block, market_data_mtx=md)

    # fwd leg = 100 * 1.20 * 0.98 = 117.6
    # strike leg = 90 * 1.10 * 0.97 = 95.997... wait: 90 * 1.10 = 99, * 0.97 = 96.03
    # pv = 5 * (117.6 - 96.03) = 5 * 21.57 = 107.85
    expected_under = 100.0 * 1.20 * 0.98
    expected_strike = 90.0 * 1.10 * 0.97
    expected = 5.0 * (expected_under - expected_strike)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_output_shape() -> None:
    """Output shape matches (n_trades, n_scenarios)."""
    n, s = 3, 7
    fwd_prices = np.ones((n, s))
    block = _make_forward_node_block(
        n=n,
        qty=np.ones(n),
        strike=np.zeros(n),
        price_uix=np.arange(n, dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd_prices, n_scenarios=s)

    result = price_forward_node_block(block=block, market_data_mtx=md)

    assert result.shape == (n, s)
