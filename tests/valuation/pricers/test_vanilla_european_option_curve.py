# tests/valuation/pricers/test_vanilla_european_option_curve.py
from __future__ import annotations

import numpy as np
import pytest

from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionCurveBlock,
)
from riskquant.market_data.state import MarketDataMatrix
from riskquant.valuation.pricers._deprecated.vanilla_european_option_curve_v1 import (
    price_vanilla_european_option_curve_block,
)
from riskquant.valuation.models.black_scholes import fv_black_scholes_vanilla_3d
from riskquant.constants import (
    NO_FX_RF_REQUIRED,
    NO_DISCOUNT_CURVE_ID,
    NO_DATE_UIX,
    MOCK_VOLATILITY,
)
from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_STRIKE,
    SURFACE_Y_AXIS_TYPE_LOG_STRIKE,
    SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD,
    SURFACE_Y_AXIS_TYPE_MONEYNESS_SPOT,
    SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD,
    SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_SPOT,
)

# =========================================================================
# CONSTANTS
# =========================================================================
N_SCENARIOS = 4


# =========================================================================
# HELPERS
# =========================================================================


def _make_option_curve_block(
    *,
    n: int,
    qty: np.ndarray,
    strike: np.ndarray,
    is_call: np.ndarray,
    underlying_price_curve_uix: np.ndarray,
    delivery_date_uix: np.ndarray,
    expiry_date_uix: np.ndarray,
    vol_surface_uix: np.ndarray,
    vol_surface_y_axis_type_id: np.ndarray,
    settlement_date_uix: np.ndarray | None = None,
    fx_under_to_settle_uix: np.ndarray | None = None,
    fx_strike_to_settle_uix: np.ndarray | None = None,
    fx_under_to_settle_invert: np.ndarray | None = None,
    fx_strike_to_settle_invert: np.ndarray | None = None,
    under_discount_curve_uix: np.ndarray | None = None,
    strike_discount_curve_uix: np.ndarray | None = None,
) -> VanillaEuropeanOptionCurveBlock:
    sentinel_i32 = lambda v: np.full(n, v, dtype=np.int32)
    return VanillaEuropeanOptionCurveBlock(
        trade_uids=np.arange(n, dtype=np.int64),
        qty=qty,
        strike=strike,
        is_call=is_call,
        underlying_price_curve_uix=underlying_price_curve_uix,
        delivery_date_uix=delivery_date_uix,
        expiry_date_uix=expiry_date_uix,
        vol_surface_uix=vol_surface_uix,
        vol_surface_y_axis_type_id=vol_surface_y_axis_type_id,
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
    tte_grid: np.ndarray,
    fx_spots: np.ndarray | None = None,
    discount_factor_grid: np.ndarray | None = None,
    vol_surface_strikes: np.ndarray | None = None,
    vol_surfaces: np.ndarray | None = None,
    n_scenarios: int = N_SCENARIOS,
) -> MarketDataMatrix:
    s = n_scenarios
    if fx_spots is None:
        fx_spots = np.empty((0, s), dtype=np.float64)
    if discount_factor_grid is None:
        discount_factor_grid = np.empty((0, 0, s), dtype=np.float64)
    if vol_surface_strikes is None:
        vol_surface_strikes = np.empty((0, 0), dtype=np.float64)
    if vol_surfaces is None:
        vol_surfaces = np.empty((0, 0, 0, 0), dtype=np.float64)
    return MarketDataMatrix(
        num_scenarios=s,
        fx_spots=fx_spots,
        prices=np.empty((0, s), dtype=np.float64),
        forward_nodes=np.empty((0, s), dtype=np.float64),
        vol_nodes=np.empty((0, s), dtype=np.float64),
        discount_factor_grid=discount_factor_grid,
        price_curve_grid=price_curve_grid,
        tte_grid=tte_grid,
        vol_surface_strikes=vol_surface_strikes,
        vol_surfaces=vol_surfaces,
    )


# =========================================================================
# TESTS — Mock Volatility fallback (empty vol_surfaces)
# =========================================================================


def test_mock_vol_atm_call() -> None:
    """With empty vol surfaces, pricer falls back to MOCK_VOLATILITY."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
        ),
    )
    md = _make_market_data(price_curve_grid=pcg, tte_grid=tte)

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    # Should match BS with MOCK_VOLATILITY = 0.50
    expected = fv_black_scholes_vanilla_3d(
        fwd=np.full((1, N_SCENARIOS), 100.0),
        strike=np.array([[100.0]]),
        vol=np.full((1, N_SCENARIOS), MOCK_VOLATILITY),
        tte=np.array([[1.0]]),
        is_call=np.array([[1]], dtype=np.int8),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_mock_vol_put() -> None:
    """Put option with mock vol fallback."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([0.5])

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([0], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
        ),
    )
    md = _make_market_data(price_curve_grid=pcg, tte_grid=tte)

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    assert np.all(result > 0)


def test_mock_vol_call_put_parity() -> None:
    """Put-call parity holds with mock vol (no FX/DF)."""
    pcg = np.full((1, 1, N_SCENARIOS), 110.0)
    tte = np.array([1.0])

    def _price(is_call_val: int):
        block = _make_option_curve_block(
            n=1,
            qty=np.array([1.0]),
            strike=np.array([100.0]),
            is_call=np.array([is_call_val], dtype=np.int8),
            underlying_price_curve_uix=np.array([0], dtype=np.int32),
            delivery_date_uix=np.array([0], dtype=np.int32),
            expiry_date_uix=np.array([0], dtype=np.int32),
            vol_surface_uix=np.array([0], dtype=np.int32),
            vol_surface_y_axis_type_id=np.array(
                [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
            ),
        )
        md = _make_market_data(price_curve_grid=pcg, tte_grid=tte)
        return price_vanilla_european_option_curve_block(
            block=block, market_data_mtx=md
        )

    call_pv = _price(1)
    put_pv = _price(0)

    np.testing.assert_allclose(call_pv - put_pv, 110.0 - 100.0, rtol=1e-10)


# =========================================================================
# TESTS — Quantity Scaling
# =========================================================================


def test_qty_scaling() -> None:
    """Quantity multiplier scales PV linearly."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])

    def _price_with_qty(q: float):
        block = _make_option_curve_block(
            n=1,
            qty=np.array([q]),
            strike=np.array([100.0]),
            is_call=np.array([1], dtype=np.int8),
            underlying_price_curve_uix=np.array([0], dtype=np.int32),
            delivery_date_uix=np.array([0], dtype=np.int32),
            expiry_date_uix=np.array([0], dtype=np.int32),
            vol_surface_uix=np.array([0], dtype=np.int32),
            vol_surface_y_axis_type_id=np.array(
                [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
            ),
        )
        md = _make_market_data(price_curve_grid=pcg, tte_grid=tte)
        return price_vanilla_european_option_curve_block(
            block=block, market_data_mtx=md
        )

    pv_1 = _price_with_qty(1.0)
    pv_10 = _price_with_qty(10.0)

    np.testing.assert_allclose(pv_10, 10.0 * pv_1, rtol=1e-12)


# =========================================================================
# TESTS — FX and Discounting
# =========================================================================


def test_with_underlying_fx() -> None:
    """FX on underlying leg changes the synthesized forward."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])
    fx = np.full((1, N_SCENARIOS), 1.50)

    block_no_fx = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
        ),
    )
    md_no_fx = _make_market_data(price_curve_grid=pcg, tte_grid=tte)
    pv_no_fx = price_vanilla_european_option_curve_block(
        block=block_no_fx, market_data_mtx=md_no_fx
    )

    block_fx = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
        ),
        fx_under_to_settle_uix=np.array([0], dtype=np.int32),
    )
    md_fx = _make_market_data(price_curve_grid=pcg, tte_grid=tte, fx_spots=fx)
    pv_fx = price_vanilla_european_option_curve_block(
        block=block_fx, market_data_mtx=md_fx
    )

    # With underlying FX > 1, the forward is higher, so call value should be higher
    assert np.all(pv_fx > pv_no_fx)


def test_with_strike_df_and_fx() -> None:
    """Strike FX and DF are applied correctly."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])
    fx = np.full((1, N_SCENARIOS), 1.20)
    df_grid = np.ones((1, 2, N_SCENARIOS))
    df_grid[0, 0, :] = 0.98  # expiry
    df_grid[0, 1, :] = 0.95  # settlement

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        settlement_date_uix=np.array([1], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
        ),
        fx_strike_to_settle_uix=np.array([0], dtype=np.int32),
        strike_discount_curve_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(
        price_curve_grid=pcg,
        tte_grid=tte,
        fx_spots=fx,
        discount_factor_grid=df_grid,
    )

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    assert result.shape == (1, N_SCENARIOS)
    assert np.all(result > 0)


# =========================================================================
# TESTS — Y-Axis Type Transforms
# =========================================================================


def test_y_axis_strike_type() -> None:
    """STRIKE y-axis type: y_target = K."""
    # With flat vol surface at 0.30, the result should use vol=0.30
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])

    # 1 surface, 1 date, 3 strikes, N_SCENARIOS
    vol_surface_strikes = np.array([[80.0, 100.0, 120.0]])
    vol_surfaces = np.full((1, 1, 3, N_SCENARIOS), 0.30)

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
        ),
    )
    md = _make_market_data(
        price_curve_grid=pcg,
        tte_grid=tte,
        vol_surface_strikes=vol_surface_strikes,
        vol_surfaces=vol_surfaces,
    )

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    # With flat vol=0.30, should match BS
    expected = fv_black_scholes_vanilla_3d(
        fwd=np.full((1, N_SCENARIOS), 100.0),
        strike=np.array([[100.0]]),
        vol=np.full((1, N_SCENARIOS), 0.30),
        tte=np.array([[1.0]]),
        is_call=np.array([[1]], dtype=np.int8),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_y_axis_log_strike_type() -> None:
    """LOG_STRIKE y-axis: y_target = ln(K). With flat vol, result matches BS."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])
    K = 100.0

    # Pillars in log(strike) space
    vol_surface_strikes = np.array(
        [[np.log(80.0), np.log(100.0), np.log(120.0)]]
    )
    vol_surfaces = np.full((1, 1, 3, N_SCENARIOS), 0.25)

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([K]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_LOG_STRIKE], dtype=np.int8
        ),
    )
    md = _make_market_data(
        price_curve_grid=pcg,
        tte_grid=tte,
        vol_surface_strikes=vol_surface_strikes,
        vol_surfaces=vol_surfaces,
    )

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    expected = fv_black_scholes_vanilla_3d(
        fwd=np.full((1, N_SCENARIOS), 100.0),
        strike=np.array([[K]]),
        vol=np.full((1, N_SCENARIOS), 0.25),
        tte=np.array([[1.0]]),
        is_call=np.array([[1]], dtype=np.int8),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_y_axis_moneyness_fwd_type() -> None:
    """MONEYNESS_FWD y-axis: y_target = K/F. With flat vol, matches BS."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])
    K = 100.0

    # ATM moneyness = K/F = 1.0
    vol_surface_strikes = np.array([[0.8, 1.0, 1.2]])
    vol_surfaces = np.full((1, 1, 3, N_SCENARIOS), 0.25)

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([K]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_MONEYNESS_FWD], dtype=np.int8
        ),
    )
    md = _make_market_data(
        price_curve_grid=pcg,
        tte_grid=tte,
        vol_surface_strikes=vol_surface_strikes,
        vol_surfaces=vol_surfaces,
    )

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    expected = fv_black_scholes_vanilla_3d(
        fwd=np.full((1, N_SCENARIOS), 100.0),
        strike=np.array([[K]]),
        vol=np.full((1, N_SCENARIOS), 0.25),
        tte=np.array([[1.0]]),
        is_call=np.array([[1]], dtype=np.int8),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_y_axis_log_moneyness_fwd_type() -> None:
    """LOG_MONEYNESS_FWD y-axis: y_target = ln(K/F). With flat vol at ATM, matches BS."""
    pcg = np.full((1, 1, N_SCENARIOS), 100.0)
    tte = np.array([1.0])
    K = 100.0

    # ATM log-moneyness = ln(1.0) = 0
    vol_surface_strikes = np.array([[-0.2, 0.0, 0.2]])
    vol_surfaces = np.full((1, 1, 3, N_SCENARIOS), 0.20)

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([K]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_LOG_MONEYNESS_FWD], dtype=np.int8
        ),
    )
    md = _make_market_data(
        price_curve_grid=pcg,
        tte_grid=tte,
        vol_surface_strikes=vol_surface_strikes,
        vol_surfaces=vol_surfaces,
    )

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    expected = fv_black_scholes_vanilla_3d(
        fwd=np.full((1, N_SCENARIOS), 100.0),
        strike=np.array([[K]]),
        vol=np.full((1, N_SCENARIOS), 0.20),
        tte=np.array([[1.0]]),
        is_call=np.array([[1]], dtype=np.int8),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


# =========================================================================
# TESTS — Multiple trades and output shape
# =========================================================================


def test_multiple_trades_different_dates() -> None:
    """Multiple trades referencing different delivery and expiry dates."""
    pcg = np.zeros((1, 2, N_SCENARIOS))
    pcg[0, 0, :] = 100.0
    pcg[0, 1, :] = 120.0
    tte = np.array([1.0, 0.5])

    block = _make_option_curve_block(
        n=2,
        qty=np.array([1.0, -2.0]),
        strike=np.array([100.0, 110.0]),
        is_call=np.array([1, 0], dtype=np.int8),
        underlying_price_curve_uix=np.array([0, 0], dtype=np.int32),
        delivery_date_uix=np.array([0, 1], dtype=np.int32),
        expiry_date_uix=np.array([0, 1], dtype=np.int32),
        vol_surface_uix=np.array([0, 0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE, SURFACE_Y_AXIS_TYPE_STRIKE],
            dtype=np.int8,
        ),
    )
    md = _make_market_data(price_curve_grid=pcg, tte_grid=tte)

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    assert result.shape == (2, N_SCENARIOS)
    # Trade 0: long call => positive
    assert np.all(result[0, :] > 0)
    # Trade 1: short put => negative (put is positive, * -2 makes it negative)
    assert np.all(result[1, :] < 0)


def test_output_shape() -> None:
    """Output shape matches (n_trades, n_scenarios)."""
    n, s = 5, 6
    pcg = np.full((2, 3, s), 100.0)
    tte = np.array([1.0, 0.5, 0.25])

    block = _make_option_curve_block(
        n=n,
        qty=np.ones(n),
        strike=np.full(n, 100.0),
        is_call=np.ones(n, dtype=np.int8),
        underlying_price_curve_uix=np.array([0, 0, 1, 1, 0], dtype=np.int32),
        delivery_date_uix=np.array([0, 1, 0, 2, 1], dtype=np.int32),
        expiry_date_uix=np.array([0, 0, 1, 1, 2], dtype=np.int32),
        vol_surface_uix=np.array([0, 0, 0, 0, 0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.full(
            n, SURFACE_Y_AXIS_TYPE_STRIKE, dtype=np.int8
        ),
    )
    md = _make_market_data(price_curve_grid=pcg, tte_grid=tte, n_scenarios=s)

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    assert result.shape == (n, s)


def test_per_scenario_fwd_curve() -> None:
    """Curve forwards vary per scenario, producing different PVs."""
    pcg = np.zeros((1, 1, 3))
    pcg[0, 0, :] = [90.0, 100.0, 110.0]
    tte = np.array([1.0])

    block = _make_option_curve_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_curve_uix=np.array([0], dtype=np.int32),
        delivery_date_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        vol_surface_uix=np.array([0], dtype=np.int32),
        vol_surface_y_axis_type_id=np.array(
            [SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8
        ),
    )
    md = _make_market_data(price_curve_grid=pcg, tte_grid=tte, n_scenarios=3)

    result = price_vanilla_european_option_curve_block(
        block=block, market_data_mtx=md
    )

    # Higher forward => higher call value
    assert result[0, 0] < result[0, 1] < result[0, 2]
