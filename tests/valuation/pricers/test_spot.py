# tests/valuation/pricers/test_spot.py
from __future__ import annotations

import numpy as np
import pytest

from riskquant.portfolio.blocks.spot_block import SpotBlock
from riskquant.market_data.state import MarketDataMatrix
from riskquant.valuation.pricers.spot import price_spot_block
from riskquant.constants import NO_FX_RF_REQUIRED

# =========================================================================
# CONSTANTS
# =========================================================================
N_SCENARIOS = 5


# =========================================================================
# HELPERS
# =========================================================================


def _make_spot_block(
    *,
    qty: np.ndarray,
    price_uix: np.ndarray,
    fx_uix: np.ndarray | None = None,
    fx_invert: np.ndarray | None = None,
) -> SpotBlock:
    n = len(qty)
    trade_uids = np.arange(n, dtype=np.int64)
    if fx_uix is None:
        fx_uix = np.full(n, NO_FX_RF_REQUIRED, dtype=np.int32)
    if fx_invert is None:
        fx_invert = np.zeros(n, dtype=np.bool_)
    return SpotBlock(
        trade_uids=trade_uids,
        qty=qty,
        price_uix=price_uix,
        fx_uix=fx_uix,
        fx_invert=fx_invert,
    )


def _make_market_data(
    *,
    prices: np.ndarray,
    fx_spots: np.ndarray | None = None,
) -> MarketDataMatrix:
    s = prices.shape[1]
    if fx_spots is None:
        fx_spots = np.empty((0, s), dtype=np.float64)
    return MarketDataMatrix(
        num_scenarios=s,
        fx_spots=fx_spots,
        prices=prices,
        forward_nodes=np.empty((0, s), dtype=np.float64),
        vol_nodes=np.empty((0, s), dtype=np.float64),
        discount_factor_grid=np.empty((0, 0, s), dtype=np.float64),
        price_curve_grid=np.empty((0, 0, s), dtype=np.float64),
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, s), dtype=np.float64),
    )


# =========================================================================
# TESTS — Basic PV = Qty * Price
# =========================================================================


def test_single_trade_no_fx() -> None:
    """PV = qty * price, no FX."""
    prices = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
    block = _make_spot_block(
        qty=np.array([2.0]),
        price_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(prices=prices)

    result = price_spot_block(block=block, market_data_mtx=md)

    expected = np.array([[20.0, 40.0, 60.0, 80.0, 100.0]])
    np.testing.assert_allclose(result, expected)


def test_multiple_trades_no_fx() -> None:
    """Multiple trades sharing and referencing different price factors."""
    prices = np.full((3, N_SCENARIOS), 100.0)
    prices[1, :] = 200.0
    prices[2, :] = 50.0

    block = _make_spot_block(
        qty=np.array([1.0, -2.0, 3.0]),
        price_uix=np.array([0, 1, 2], dtype=np.int32),
    )
    md = _make_market_data(prices=prices)

    result = price_spot_block(block=block, market_data_mtx=md)

    assert result.shape == (3, N_SCENARIOS)
    np.testing.assert_allclose(result[0, :], 100.0)
    np.testing.assert_allclose(result[1, :], -400.0)
    np.testing.assert_allclose(result[2, :], 150.0)


def test_negative_qty() -> None:
    """Short position produces negative PV."""
    prices = np.full((1, N_SCENARIOS), 50.0)
    block = _make_spot_block(
        qty=np.array([-10.0]),
        price_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(prices=prices)

    result = price_spot_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result, -500.0)


# =========================================================================
# TESTS — FX Translation
# =========================================================================


def test_single_trade_with_fx() -> None:
    """PV = qty * price * fx_rate."""
    prices = np.full((1, N_SCENARIOS), 100.0)
    fx_spots = np.full((1, N_SCENARIOS), 1.30)

    block = _make_spot_block(
        qty=np.array([5.0]),
        price_uix=np.array([0], dtype=np.int32),
        fx_uix=np.array([0], dtype=np.int32),
        fx_invert=np.array([False]),
    )
    md = _make_market_data(prices=prices, fx_spots=fx_spots)

    result = price_spot_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result, 5.0 * 100.0 * 1.30)


def test_fx_inversion() -> None:
    """When fx_invert=True, the FX rate is mathematically inverted (1/rate)."""
    prices = np.full((1, N_SCENARIOS), 100.0)
    fx_spots = np.full((1, N_SCENARIOS), 2.0)

    block = _make_spot_block(
        qty=np.array([1.0]),
        price_uix=np.array([0], dtype=np.int32),
        fx_uix=np.array([0], dtype=np.int32),
        fx_invert=np.array([True]),
    )
    md = _make_market_data(prices=prices, fx_spots=fx_spots)

    result = price_spot_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result, 100.0 * 0.5)


def test_mixed_fx_and_no_fx() -> None:
    """Mix of trades: some with FX, some without."""
    prices = np.full((2, N_SCENARIOS), 100.0)
    fx_spots = np.full((1, N_SCENARIOS), 1.50)

    block = _make_spot_block(
        qty=np.array([1.0, 1.0]),
        price_uix=np.array([0, 1], dtype=np.int32),
        fx_uix=np.array([NO_FX_RF_REQUIRED, 0], dtype=np.int32),
        fx_invert=np.array([False, False]),
    )
    md = _make_market_data(prices=prices, fx_spots=fx_spots)

    result = price_spot_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result[0, :], 100.0)
    np.testing.assert_allclose(result[1, :], 150.0)


def test_per_scenario_prices_and_fx() -> None:
    """Each scenario has distinct price and FX, verifying vectorized calc."""
    prices = np.array([[10.0, 20.0, 30.0]])
    fx_spots = np.array([[1.0, 1.5, 2.0]])

    block = _make_spot_block(
        qty=np.array([2.0]),
        price_uix=np.array([0], dtype=np.int32),
        fx_uix=np.array([0], dtype=np.int32),
        fx_invert=np.array([False]),
    )
    md = _make_market_data(prices=prices, fx_spots=fx_spots)

    result = price_spot_block(block=block, market_data_mtx=md)

    expected = np.array([[20.0, 60.0, 120.0]])
    np.testing.assert_allclose(result, expected)


def test_output_shape() -> None:
    """Output shape must be (n_trades, n_scenarios)."""
    n = 7
    s = 13
    prices = np.ones((n, s))
    block = _make_spot_block(
        qty=np.ones(n),
        price_uix=np.arange(n, dtype=np.int32),
    )
    md = _make_market_data(prices=prices)

    result = price_spot_block(block=block, market_data_mtx=md)

    assert result.shape == (n, s)


def test_mixed_fx_invert_flags() -> None:
    """Multiple trades with differing fx_invert flags."""
    prices = np.full((2, N_SCENARIOS), 100.0)
    fx_spots = np.full((2, N_SCENARIOS), 4.0)

    block = _make_spot_block(
        qty=np.array([1.0, 1.0]),
        price_uix=np.array([0, 1], dtype=np.int32),
        fx_uix=np.array([0, 1], dtype=np.int32),
        fx_invert=np.array([False, True]),
    )
    md = _make_market_data(prices=prices, fx_spots=fx_spots)

    result = price_spot_block(block=block, market_data_mtx=md)

    np.testing.assert_allclose(result[0, :], 400.0)
    np.testing.assert_allclose(result[1, :], 25.0)
