# tests/valuation/test_engine.py
from __future__ import annotations

import numpy as np
import pytest

from riskquant.constants import (
    NO_DATE_UIX,
    NO_DISCOUNT_CURVE_ID,
    NO_FX_RF_REQUIRED,
)
from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.blocks.spot_block import SpotBlock
from riskquant.portfolio.blocks.vanilla_forward_block import (
    VanillaForwardNodeBlock,
)
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.valuation.engine import run_valuation

# =========================================================================
# CONSTANTS
# =========================================================================
N_SCENARIOS = 4


# =========================================================================
# HELPERS
# =========================================================================


def _empty_md(s: int = N_SCENARIOS) -> MarketDataMatrix:
    """MarketDataMatrix with no data (for tests that only use spot prices)."""
    return MarketDataMatrix(
        num_scenarios=s,
        fx_spots=np.empty((0, s), dtype=np.float64),
        prices=np.empty((0, s), dtype=np.float64),
        forward_nodes=np.empty((0, s), dtype=np.float64),
        vol_nodes=np.empty((0, s), dtype=np.float64),
        discount_factor_grid=np.empty((0, 0, s), dtype=np.float64),
        price_curve_grid=np.empty((0, 0, s), dtype=np.float64),
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, s), dtype=np.float64),
    )


def _make_spot_block(n: int, price_offset: int = 0) -> SpotBlock:
    return SpotBlock(
        trade_uids=np.arange(n, dtype=np.int64),
        qty=np.ones(n, dtype=np.float64),
        price_uix=np.arange(price_offset, price_offset + n, dtype=np.int32),
        fx_uix=np.full(n, NO_FX_RF_REQUIRED, dtype=np.int32),
        fx_invert=np.zeros(n, dtype=np.bool_),
    )


def _make_fwd_node_block(n: int, price_offset: int = 0) -> VanillaForwardNodeBlock:
    return VanillaForwardNodeBlock(
        trade_uids=np.arange(100, 100 + n, dtype=np.int64),
        qty=np.ones(n, dtype=np.float64),
        strike=np.full(n, 50.0, dtype=np.float64),
        price_uix=np.arange(price_offset, price_offset + n, dtype=np.int32),
        settlement_date_uix=np.full(n, NO_DATE_UIX, dtype=np.int32),
        fx_under_to_settle_uix=np.full(n, NO_FX_RF_REQUIRED, dtype=np.int32),
        fx_strike_to_settle_uix=np.full(n, NO_FX_RF_REQUIRED, dtype=np.int32),
        fx_under_to_settle_invert=np.zeros(n, dtype=np.bool_),
        fx_strike_to_settle_invert=np.zeros(n, dtype=np.bool_),
        under_discount_curve_uix=np.full(n, NO_DISCOUNT_CURVE_ID, dtype=np.int32),
        strike_discount_curve_uix=np.full(n, NO_DISCOUNT_CURVE_ID, dtype=np.int32),
    )


# =========================================================================
# TESTS — Engine dispatch and output assembly
# =========================================================================


def test_single_spot_block() -> None:
    """Engine dispatches a single SpotBlock and returns correct shape and values."""
    n = 3
    prices = np.full((n, N_SCENARIOS), 10.0)

    spot = _make_spot_block(n)
    portfolio = CompiledPortfolio(spot=spot)

    md = MarketDataMatrix(
        num_scenarios=N_SCENARIOS,
        fx_spots=np.empty((0, N_SCENARIOS), dtype=np.float64),
        prices=prices,
        forward_nodes=np.empty((0, N_SCENARIOS), dtype=np.float64),
        vol_nodes=np.empty((0, N_SCENARIOS), dtype=np.float64),
        discount_factor_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        price_curve_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, N_SCENARIOS), dtype=np.float64),
    )

    result = run_valuation(portfolio=portfolio, market_data_mtx=md)

    assert result.shape == (n, N_SCENARIOS)
    # qty=1 * price=10 => PV=10 for all
    np.testing.assert_allclose(result, 10.0)


def test_single_forward_node_block() -> None:
    """Engine dispatches a single ForwardNodeBlock."""
    n = 2
    fwd = np.full((n, N_SCENARIOS), 100.0)

    fwd_block = _make_fwd_node_block(n)
    portfolio = CompiledPortfolio(vanilla_forward_node=fwd_block)

    md = MarketDataMatrix(
        num_scenarios=N_SCENARIOS,
        fx_spots=np.empty((0, N_SCENARIOS), dtype=np.float64),
        prices=np.empty((0, N_SCENARIOS), dtype=np.float64),
        forward_nodes=fwd,
        vol_nodes=np.empty((0, N_SCENARIOS), dtype=np.float64),
        discount_factor_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        price_curve_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, N_SCENARIOS), dtype=np.float64),
    )

    result = run_valuation(portfolio=portfolio, market_data_mtx=md)

    assert result.shape == (n, N_SCENARIOS)
    # qty=1, fwd=100, strike=50 => PV = 1 * (100 - 50) = 50
    np.testing.assert_allclose(result, 50.0)


def test_multiple_block_types_contiguous_output() -> None:
    """Engine assembles output from spot + forward blocks contiguously."""
    n_spot = 2
    n_fwd = 3

    spot_prices = np.full((n_spot, N_SCENARIOS), 25.0)
    fwd_prices = np.full((n_fwd, N_SCENARIOS), 80.0)

    spot = _make_spot_block(n_spot)
    fwd = _make_fwd_node_block(n_fwd)
    portfolio = CompiledPortfolio(spot=spot, vanilla_forward_node=fwd)

    md = MarketDataMatrix(
        num_scenarios=N_SCENARIOS,
        fx_spots=np.empty((0, N_SCENARIOS), dtype=np.float64),
        prices=spot_prices,
        forward_nodes=fwd_prices,
        vol_nodes=np.empty((0, N_SCENARIOS), dtype=np.float64),
        discount_factor_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        price_curve_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, N_SCENARIOS), dtype=np.float64),
    )

    result = run_valuation(portfolio=portfolio, market_data_mtx=md)

    assert result.shape == (n_spot + n_fwd, N_SCENARIOS)
    # Spot rows: qty=1 * 25 = 25
    np.testing.assert_allclose(result[:n_spot, :], 25.0)
    # Fwd rows: qty=1 * (80 - 50) = 30
    np.testing.assert_allclose(result[n_spot:, :], 30.0)


def test_empty_portfolio() -> None:
    """Empty portfolio (no blocks) returns shape (0, S)."""
    portfolio = CompiledPortfolio()
    md = _empty_md()

    result = run_valuation(portfolio=portfolio, market_data_mtx=md)

    assert result.shape == (0, N_SCENARIOS)


def test_unknown_block_type_raises() -> None:
    """A block type not in BLOCK_PRICER_MAP raises ValueError."""

    class FakeBlock:
        trade_uids = np.array([1], dtype=np.int64)

        @property
        def num_trades(self):
            return 1

    # Monkey-patch a portfolio to yield an unknown block
    portfolio = CompiledPortfolio()
    original_iter = portfolio.iter_blocks
    portfolio.__class__.iter_blocks = lambda self: (FakeBlock(),)

    md = _empty_md()

    with pytest.raises(ValueError, match="No pricer mapped"):
        run_valuation(portfolio=portfolio, market_data_mtx=md)

    # Restore
    portfolio.__class__.iter_blocks = original_iter


def test_total_trades_matches_output_rows() -> None:
    """Output row count equals portfolio.total_trades."""
    spot = _make_spot_block(4)
    fwd = _make_fwd_node_block(3)
    portfolio = CompiledPortfolio(spot=spot, vanilla_forward_node=fwd)

    spot_prices = np.ones((4, N_SCENARIOS))
    fwd_prices = np.full((3, N_SCENARIOS), 100.0)

    md = MarketDataMatrix(
        num_scenarios=N_SCENARIOS,
        fx_spots=np.empty((0, N_SCENARIOS), dtype=np.float64),
        prices=spot_prices,
        forward_nodes=fwd_prices,
        vol_nodes=np.empty((0, N_SCENARIOS), dtype=np.float64),
        discount_factor_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        price_curve_grid=np.empty((0, 0, N_SCENARIOS), dtype=np.float64),
        tte_grid=np.empty(0, dtype=np.float64),
        vol_surface_strikes=np.empty((0, 0), dtype=np.float64),
        vol_surfaces=np.empty((0, 0, 0, N_SCENARIOS), dtype=np.float64),
    )

    result = run_valuation(portfolio=portfolio, market_data_mtx=md)

    assert result.shape[0] == portfolio.total_trades
    assert result.shape[0] == 7
