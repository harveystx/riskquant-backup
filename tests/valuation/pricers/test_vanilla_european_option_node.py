# tests/valuation/pricers/test_vanilla_european_option_node.py
from __future__ import annotations

import numpy as np
import pytest

from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionNodeBlock,
)
from riskquant.market_data.state import MarketDataMatrix
from riskquant.valuation.pricers._deprecated.vanilla_european_option_node_v1 import (
    price_vanilla_european_option_node_block,
)
from riskquant.valuation.models.black_scholes import fv_black_scholes_vanilla_3d
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


def _make_option_node_block(
    *,
    n: int,
    qty: np.ndarray,
    strike: np.ndarray,
    is_call: np.ndarray,
    underlying_price_uix: np.ndarray,
    vol_uix: np.ndarray,
    expiry_date_uix: np.ndarray,
    settlement_date_uix: np.ndarray | None = None,
    fx_under_to_settle_uix: np.ndarray | None = None,
    fx_strike_to_settle_uix: np.ndarray | None = None,
    fx_under_to_settle_invert: np.ndarray | None = None,
    fx_strike_to_settle_invert: np.ndarray | None = None,
    under_discount_curve_uix: np.ndarray | None = None,
    strike_discount_curve_uix: np.ndarray | None = None,
) -> VanillaEuropeanOptionNodeBlock:
    sentinel_i32 = lambda v: np.full(n, v, dtype=np.int32)
    return VanillaEuropeanOptionNodeBlock(
        trade_uids=np.arange(n, dtype=np.int64),
        qty=qty,
        strike=strike,
        is_call=is_call,
        underlying_price_uix=underlying_price_uix,
        vol_uix=vol_uix,
        expiry_date_uix=expiry_date_uix,
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
    vol_nodes: np.ndarray,
    tte_grid: np.ndarray,
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
        vol_nodes=vol_nodes,
        discount_factor_grid=discount_factor_grid,
        price_curve_grid=np.empty((0, 0, s), dtype=np.float64),
        tte_grid=tte_grid,
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, s), dtype=np.float64),
    )


# =========================================================================
# TESTS — Basic option pricing (no FX, no discounting)
# =========================================================================


def test_simple_atm_call_no_fx_no_df() -> None:
    """ATM call with no FX / DF should match raw BS FV * qty."""
    fwd = np.full((1, N_SCENARIOS), 100.0)
    vol = np.full((1, N_SCENARIOS), 0.20)
    tte = np.array([1.0])

    block = _make_option_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_uix=np.array([0], dtype=np.int32),
        vol_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd, vol_nodes=vol, tte_grid=tte)

    result = price_vanilla_european_option_node_block(
        block=block, market_data_mtx=md
    )

    # Manually compute: fwd_in_strike_ccy = fwd (no fx, no df adjustments)
    expected_fv = fv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=np.array([[100.0]]),
        vol=vol,
        tte=np.array([[1.0]]),
        is_call=np.array([[1]], dtype=np.int8),
    )
    np.testing.assert_allclose(result, expected_fv, rtol=1e-12)


def test_put_positive_value() -> None:
    """ATM put should have positive value."""
    fwd = np.full((1, N_SCENARIOS), 100.0)
    vol = np.full((1, N_SCENARIOS), 0.30)
    tte = np.array([0.5])

    block = _make_option_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([0], dtype=np.int8),
        underlying_price_uix=np.array([0], dtype=np.int32),
        vol_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd, vol_nodes=vol, tte_grid=tte)

    result = price_vanilla_european_option_node_block(
        block=block, market_data_mtx=md
    )

    assert np.all(result > 0)


def test_qty_scaling() -> None:
    """Quantity multiplier scales the PV linearly."""
    fwd = np.full((1, N_SCENARIOS), 100.0)
    vol = np.full((1, N_SCENARIOS), 0.20)
    tte = np.array([1.0])

    def _price_with_qty(q: float):
        block = _make_option_node_block(
            n=1,
            qty=np.array([q]),
            strike=np.array([100.0]),
            is_call=np.array([1], dtype=np.int8),
            underlying_price_uix=np.array([0], dtype=np.int32),
            vol_uix=np.array([0], dtype=np.int32),
            expiry_date_uix=np.array([0], dtype=np.int32),
        )
        md = _make_market_data(forward_nodes=fwd, vol_nodes=vol, tte_grid=tte)
        return price_vanilla_european_option_node_block(
            block=block, market_data_mtx=md
        )

    pv_1 = _price_with_qty(1.0)
    pv_5 = _price_with_qty(5.0)
    pv_neg = _price_with_qty(-3.0)

    np.testing.assert_allclose(pv_5, 5.0 * pv_1, rtol=1e-12)
    np.testing.assert_allclose(pv_neg, -3.0 * pv_1, rtol=1e-12)


def test_call_put_parity_no_fx_no_df() -> None:
    """Put-call parity: call - put = fwd - strike (when no DF or FX)."""
    fwd = np.full((1, N_SCENARIOS), 110.0)
    vol = np.full((1, N_SCENARIOS), 0.25)
    tte = np.array([1.0])

    def _price(is_call_val: int):
        block = _make_option_node_block(
            n=1,
            qty=np.array([1.0]),
            strike=np.array([100.0]),
            is_call=np.array([is_call_val], dtype=np.int8),
            underlying_price_uix=np.array([0], dtype=np.int32),
            vol_uix=np.array([0], dtype=np.int32),
            expiry_date_uix=np.array([0], dtype=np.int32),
        )
        md = _make_market_data(forward_nodes=fwd, vol_nodes=vol, tte_grid=tte)
        return price_vanilla_european_option_node_block(
            block=block, market_data_mtx=md
        )

    call_pv = _price(1)
    put_pv = _price(0)

    np.testing.assert_allclose(call_pv - put_pv, 110.0 - 100.0, rtol=1e-10)


def test_multiple_trades() -> None:
    """Multiple trades with different fwds, vols, TTE."""
    fwd = np.zeros((2, N_SCENARIOS))
    fwd[0, :] = 100.0
    fwd[1, :] = 120.0

    vol = np.zeros((2, N_SCENARIOS))
    vol[0, :] = 0.20
    vol[1, :] = 0.30

    tte = np.array([1.0, 0.5])

    block = _make_option_node_block(
        n=2,
        qty=np.array([1.0, 2.0]),
        strike=np.array([100.0, 110.0]),
        is_call=np.array([1, 0], dtype=np.int8),
        underlying_price_uix=np.array([0, 1], dtype=np.int32),
        vol_uix=np.array([0, 1], dtype=np.int32),
        expiry_date_uix=np.array([0, 1], dtype=np.int32),
    )
    md = _make_market_data(forward_nodes=fwd, vol_nodes=vol, tte_grid=tte)

    result = price_vanilla_european_option_node_block(
        block=block, market_data_mtx=md
    )

    assert result.shape == (2, N_SCENARIOS)
    # Both should be positive (ATM/near-ATM options)
    assert np.all(result > 0)


# =========================================================================
# TESTS — With FX and Discounting
# =========================================================================


def test_with_strike_df() -> None:
    """Discounting applied via the strike discount curve."""
    fwd = np.full((1, N_SCENARIOS), 100.0)
    vol = np.full((1, N_SCENARIOS), 0.20)
    tte = np.array([1.0])

    # 1 curve, 2 dates, N_SCENARIOS
    df_grid = np.ones((1, 2, N_SCENARIOS))
    df_grid[0, 0, :] = 0.99  # expiry date DF
    df_grid[0, 1, :] = 0.95  # settlement date DF

    block = _make_option_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_uix=np.array([0], dtype=np.int32),
        vol_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        settlement_date_uix=np.array([1], dtype=np.int32),
        strike_discount_curve_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(
        forward_nodes=fwd,
        vol_nodes=vol,
        tte_grid=tte,
        discount_factor_grid=df_grid,
    )

    result = price_vanilla_european_option_node_block(
        block=block, market_data_mtx=md
    )

    # The synthesized fwd divides by DF at expiry: fwd / 0.99
    # Then BS is computed, then PV is multiplied by DF at settlement: * 0.95
    # Result should differ from the no-DF case due to discounting effects
    assert result.shape == (1, N_SCENARIOS)
    assert np.all(result > 0)


def test_with_underlying_and_strike_fx() -> None:
    """FX on both underlying and strike legs."""
    fwd = np.full((1, N_SCENARIOS), 100.0)
    vol = np.full((1, N_SCENARIOS), 0.25)
    tte = np.array([1.0])
    fx = np.zeros((2, N_SCENARIOS))
    fx[0, :] = 1.50  # underlying FX
    fx[1, :] = 1.20  # strike FX

    block = _make_option_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_uix=np.array([0], dtype=np.int32),
        vol_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
        fx_under_to_settle_uix=np.array([0], dtype=np.int32),
        fx_strike_to_settle_uix=np.array([1], dtype=np.int32),
    )
    md = _make_market_data(
        forward_nodes=fwd,
        vol_nodes=vol,
        tte_grid=tte,
        fx_spots=fx,
    )

    result = price_vanilla_european_option_node_block(
        block=block, market_data_mtx=md
    )

    # The synthesized forward in strike ccy = fwd * fx_u / fx_k = 100*1.5/1.2 = 125
    # BS FV is computed on this forward, then translated back to settle ccy by * fx_k
    # Check manually:
    fwd_in_k = 100.0 * 1.50 / 1.20
    expected_fv = fv_black_scholes_vanilla_3d(
        fwd=np.full((1, N_SCENARIOS), fwd_in_k),
        strike=np.array([[100.0]]),
        vol=vol,
        tte=np.array([[1.0]]),
        is_call=np.array([[1]], dtype=np.int8),
    )
    # Translate to settle CCY
    expected = expected_fv * 1.20

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_per_scenario_vol() -> None:
    """Vols vary per scenario, producing different PVs per scenario."""
    fwd = np.full((1, 3), 100.0)
    vol = np.array([[0.10, 0.30, 0.50]])
    tte = np.array([1.0])

    block = _make_option_node_block(
        n=1,
        qty=np.array([1.0]),
        strike=np.array([100.0]),
        is_call=np.array([1], dtype=np.int8),
        underlying_price_uix=np.array([0], dtype=np.int32),
        vol_uix=np.array([0], dtype=np.int32),
        expiry_date_uix=np.array([0], dtype=np.int32),
    )
    md = _make_market_data(
        forward_nodes=fwd, vol_nodes=vol, tte_grid=tte, n_scenarios=3
    )

    result = price_vanilla_european_option_node_block(
        block=block, market_data_mtx=md
    )

    # Higher vol => higher ATM option value
    assert result[0, 0] < result[0, 1] < result[0, 2]


def test_output_shape() -> None:
    """Output shape matches (n_trades, n_scenarios)."""
    n, s = 3, 7
    fwd = np.full((n, s), 100.0)
    vol = np.full((n, s), 0.20)
    tte = np.array([1.0, 0.5, 0.25])

    block = _make_option_node_block(
        n=n,
        qty=np.ones(n),
        strike=np.full(n, 100.0),
        is_call=np.ones(n, dtype=np.int8),
        underlying_price_uix=np.arange(n, dtype=np.int32),
        vol_uix=np.arange(n, dtype=np.int32),
        expiry_date_uix=np.arange(n, dtype=np.int32),
    )
    md = _make_market_data(
        forward_nodes=fwd, vol_nodes=vol, tte_grid=tte, n_scenarios=s
    )

    result = price_vanilla_european_option_node_block(
        block=block, market_data_mtx=md
    )

    assert result.shape == (n, s)
