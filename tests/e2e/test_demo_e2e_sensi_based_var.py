# tests/e2e/test_demo_e2e_sensi_based_var.py
"""
Tests for `scripts/demos/e2e/demo_e2e_sensi_based_var.py`.

Covers two increments on top of the demo's existing `RiskStaticConfig` /
`ProxyTopology` sections:

  1. positions (edge) -> IngestBatch -> CompiledPortfolio
  2. CompiledPortfolio + RiskFactorLevelSnapshot + RiskStaticConfig
     -> pricing (MTM) -> trade sensitivities, compiled into the canonical
     `solutions.sensitivity_pnl` book contract.

Historical/simulated shocks, proxy-adjusted shocks, sensitivity P&L, and
VaR/ES are not part of either increment and are not tested here.

The demo module is loaded directly from its file path -- `scripts/` ships
demo code, not an installed package, so there is no `scripts/__init__.py`
to import through (mirrors
`tests/solutions/sensitivity_pnl/test_canonical_usage_example_is_runnable.py`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import INVALID_UIX
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.universe import UixMapper
from riskquant.valuation.blocks.spot_block import SpotBlock


def _load_demo_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    demo_path = repo_root / "scripts" / "demos" / "e2e" / "demo_e2e_sensi_based_var.py"
    spec = importlib.util.spec_from_file_location("demo_e2e_sensi_based_var", demo_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DEMO = _load_demo_module()

# The demo's own three trades, by uid. Golden values below assume exactly
# these positions and the demo's own `_RISK_FACTOR_LEVELS`; if either
# changes, these tests and the demo's own `_run_golden_checks` must be
# updated together.
_AAPL_UID = 1_010
_BRENT_UID = 1_020
_WTI_UID = 1_030


@pytest.fixture(scope="module")
def rsc() -> RiskStaticConfig:
    """Compile the demo's own minimal RiskStaticConfig once per test module."""
    return _DEMO._run_risk_static_config()


@pytest.fixture(scope="module")
def portfolio_and_mapper(
    rsc: RiskStaticConfig,
) -> tuple[CompiledPortfolio, UixMapper]:
    """Compile the demo's own three-trade positions frame once per module."""
    ingest_batch = compile_raw_df_into_ingest_batch(
        raw_df=_DEMO._POSITIONS_DF, static_config=rsc
    )
    mapper = build_uix_mapper([ingest_batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=ingest_batch, uix_mapper=mapper
    )
    return portfolio, mapper


# =============================================================================
# 1. POSITIONS -> PORTFOLIO
# =============================================================================


def test_demo_positions_compile_to_expected_spot_block(
    rsc: RiskStaticConfig,
    portfolio_and_mapper: tuple[CompiledPortfolio, UixMapper],
) -> None:
    """
    The demo's own three-row positions frame, compiled end to end, must
    reproduce exact, hand-verifiable numbers: signed quantities in row
    order, and price uixs that resolve to AAPL_EQ / BRENT_CRUDE / WTI_CRUDE.
    """
    portfolio, mapper = portfolio_and_mapper
    spot = portfolio.spot
    assert spot is not None

    np.testing.assert_array_equal(spot.trade_uids, [_AAPL_UID, _BRENT_UID, _WTI_UID])
    np.testing.assert_allclose(spot.qty, [100.0, -50.0, 75.0])

    expected_rf_ids = [
        rsc.rel_name_to_rf_id["AAPL_EQ"],
        rsc.rel_name_to_rf_id["BRENT_CRUDE"],
        rsc.rel_name_to_rf_id["WTI_CRUDE"],
    ]
    for row, expected_rf_id in enumerate(expected_rf_ids):
        eff_uix = spot.effective_spot_price_uix[row]
        price_uix, _fx_eff_uix = mapper.spot_effective_pairs[eff_uix]
        resolved_rf_id = mapper.market_id_universe[price_uix]
        assert resolved_rf_id == expected_rf_id


def test_compiled_portfolio_structure_is_spot_only(
    portfolio_and_mapper: tuple[CompiledPortfolio, UixMapper],
) -> None:
    """
    Only `portfolio.compilers.registry.COMPILERS[SPOT_ID]` is registered
    today; a portfolio built from spot-only positions must populate exactly
    the spot block, leave the unregistered block fields `None`, and carry
    compact/typed arrays throughout.
    """
    portfolio, _mapper = portfolio_and_mapper

    assert isinstance(portfolio.spot, SpotBlock)
    assert portfolio.vanilla_forward is None
    assert portfolio.vanilla_european_option is None

    assert portfolio.total_trades == len(_DEMO._POSITIONS_DF)
    assert portfolio.total_blocks == 1

    spot = portfolio.spot
    assert spot.qty.dtype == np.float64
    assert spot.effective_spot_price_uix.dtype == np.int32
    assert spot.qty.shape[0] == spot.effective_spot_price_uix.shape[0] == 3


def test_brent_position_carries_an_fx_leg_others_do_not(
    portfolio_and_mapper: tuple[CompiledPortfolio, UixMapper],
) -> None:
    """
    Structural precondition for the cross-currency sensitivity below: the
    EUR-settled Brent position must resolve to a real FX-effective pair
    (`fx_eff_uix != INVALID_UIX`); the USD-settled AAPL/WTI positions must
    resolve to the "no FX leg" sentinel.
    """
    portfolio, mapper = portfolio_and_mapper
    spot = portfolio.spot

    fx_eff_uix_by_uid = {}
    for i, uid in enumerate(spot.trade_uids.tolist()):
        eff_uix = spot.effective_spot_price_uix[i]
        _price_uix, fx_eff_uix = mapper.spot_effective_pairs[eff_uix]
        fx_eff_uix_by_uid[uid] = fx_eff_uix

    assert fx_eff_uix_by_uid[_AAPL_UID] == INVALID_UIX
    assert fx_eff_uix_by_uid[_WTI_UID] == INVALID_UIX
    assert fx_eff_uix_by_uid[_BRENT_UID] != INVALID_UIX


def test_unregistered_instrument_fails_loud_not_silent(
    rsc: RiskStaticConfig,
) -> None:
    """
    A FORWARD-family row must never be silently dropped from the compiled
    portfolio. `VANILLA_FORWARD_NODE` is a structurally valid instrument
    token -- ingest accepts it -- but no block compiler is registered for
    it yet, so `compile_portfolio_from_ingest_batch` must raise rather than
    quietly compile a portfolio missing that trade.
    """
    raw = pd.DataFrame(
        {
            "uid": [2001],
            "instrument": ["VANILLA_FORWARD_NODE"],
            "quantity": [10.0],
            "underlying": ["AAPL_EQ"],
            "settlement_ccy": ["USD"],
            "strike_price": [100.0],
            "strike_ccy": ["USD"],
        }
    )

    ingest_batch = compile_raw_df_into_ingest_batch(raw_df=raw, static_config=rsc)
    mapper = build_uix_mapper([ingest_batch])

    with pytest.raises(RuntimeError, match="No compiler registered"):
        compile_portfolio_from_ingest_batch(
            ingest_batch=ingest_batch, uix_mapper=mapper
        )


def test_demo_positions_to_portfolio_section_runs(
    capsys: pytest.CaptureFixture[str], rsc: RiskStaticConfig
) -> None:
    """
    `_run_portfolio` is the exact function the demo's `main()` calls for
    this increment; running it directly keeps this test fast while still
    exercising the real demo code path and its printed summary.
    """
    portfolio, mapper = _DEMO._run_portfolio(rsc=rsc)

    assert portfolio.total_trades == 3
    assert portfolio.total_blocks == 1
    assert isinstance(mapper, UixMapper)

    captured = capsys.readouterr()
    assert "COMPILED PORTFOLIO SUMMARY" in captured.out
    assert "SpotBlock" in captured.out


# =============================================================================
# 2. PRICING
# =============================================================================


@pytest.fixture(scope="module")
def pricing_and_sensitivities(
    rsc: RiskStaticConfig,
    portfolio_and_mapper: tuple[CompiledPortfolio, UixMapper],
):
    """Run the demo's own pricing + sensitivity derivation once per module."""
    portfolio, mapper = portfolio_and_mapper
    snapshot = _DEMO.build_risk_factor_level_snapshot_from_dict(
        as_of_date=_DEMO._SNAPSHOT_AS_OF_DATE,
        levels_dict=_DEMO._RISK_FACTOR_LEVELS,
        name_to_id_map=rsc.rel_name_to_rf_id,
    )
    market_data = _DEMO._run_market_data(rsc=rsc, uix_mapper=mapper, snapshot=snapshot)
    pricing = _DEMO._run_pricing(
        portfolio=portfolio, uix_mapper=mapper, market_data=market_data
    )
    calculator = _DEMO._run_sensitivities(
        rsc=rsc, portfolio=portfolio, uix_mapper=mapper, market_data=market_data
    )
    return pricing, calculator


def _pv(pricing, uid: int) -> float:
    return float(pricing.get_trade_values(uid)[0])


def _first_order(calculator, position_id: int, factor_id: int) -> float:
    book = calculator.book
    m = int(np.searchsorted(book.position_uids, position_id))
    start, end = book.position_offsets[m], book.position_offsets[m + 1]
    for local, active_uix in enumerate(book.factor_uix[start:end]):
        if int(book.active_factor_ids[active_uix]) == factor_id:
            return float(book.first_order[start:end][local])
    raise KeyError((position_id, factor_id))


def test_portfolio_valuation_matches_golden_mtm(rsc, pricing_and_sensitivities) -> None:
    """
    AAPL and WTI are USD-priced/settled: PV = qty * price, no FX factor.
    Brent is USD-priced, EUR-settled: PV = qty * price_usd * USDEUR (the
    quanto / cross-currency case), matching the golden value in the task
    spec exactly: -50 * 82.30 * 1.0800 = -4444.20 EUR.
    """
    pricing, _calculator = pricing_and_sensitivities

    assert _pv(pricing, _AAPL_UID) == pytest.approx(100.0 * 200.00, abs=1e-2)
    assert _pv(pricing, _WTI_UID) == pytest.approx(75.0 * 70.00, abs=1e-2)
    assert _pv(pricing, _BRENT_UID) == pytest.approx(-4444.20, abs=1e-2)


def test_pricing_result_shape_and_dtype(pricing_and_sensitivities) -> None:
    pricing, _calculator = pricing_and_sensitivities
    assert pricing.num_trades == 3
    assert pricing.num_scenarios == 1
    assert pricing.trade_uids.dtype == np.int64
    assert pricing.values.dtype == np.float64


# =============================================================================
# 3. TRADE SENSITIVITIES
# =============================================================================


def test_aapl_sensitivity_is_quantity_only(rsc, pricing_and_sensitivities) -> None:
    """AAPL has no FX leg: exactly one sensitivity row, dV/dAAPL_EQ = qty."""
    _pricing, calculator = pricing_and_sensitivities
    aapl_rf = int(rsc.rel_name_to_rf_id["AAPL_EQ"])
    assert _first_order(calculator, _AAPL_UID, aapl_rf) == pytest.approx(
        100.0, abs=1e-2
    )


def test_wti_sensitivity_is_quantity_only(rsc, pricing_and_sensitivities) -> None:
    """WTI has no FX leg: exactly one sensitivity row, dV/dWTI_CRUDE = qty."""
    _pricing, calculator = pricing_and_sensitivities
    wti_rf = int(rsc.rel_name_to_rf_id["WTI_CRUDE"])
    assert _first_order(calculator, _WTI_UID, wti_rf) == pytest.approx(75.0, abs=1e-2)


def test_brent_produces_both_commodity_and_fx_sensitivity(
    rsc, pricing_and_sensitivities
) -> None:
    """
    The EUR-settled Brent position must NOT be treated as a single-factor
    commodity exposure: it must carry both a BRENT_CRUDE sensitivity
    (dV/dBrent = qty * USDEUR = -54.00 EUR/USD) and a USDEUR sensitivity
    (dV/dUSDEUR = qty * Brent_USD = -4115.00 EUR per unit FX) -- the exact
    golden values given in the task spec.
    """
    _pricing, calculator = pricing_and_sensitivities
    brent_rf = int(rsc.rel_name_to_rf_id["BRENT_CRUDE"])
    usdeur_rf = int(rsc.rel_name_to_rf_id["USDEUR"])

    assert _first_order(calculator, _BRENT_UID, brent_rf) == pytest.approx(
        -54.00, abs=1e-2
    )
    assert _first_order(calculator, _BRENT_UID, usdeur_rf) == pytest.approx(
        -4115.00, abs=1e-2
    )


def test_no_unexpected_risk_factors_in_the_book(rsc, pricing_and_sensitivities) -> None:
    """The compiled book's active factors are exactly the four expected rf_ids."""
    _pricing, calculator = pricing_and_sensitivities
    expected = {
        int(rsc.rel_name_to_rf_id[name])
        for name in ("AAPL_EQ", "BRENT_CRUDE", "USDEUR", "WTI_CRUDE")
    }
    actual = {int(rf) for rf in calculator.book.active_factor_ids}
    assert actual == expected


def test_sensitivity_book_dtypes_and_shape(pricing_and_sensitivities) -> None:
    """
    Canonical `CompiledSensitivityBook` dtype contract: position_uids int64,
    active_factor_ids int32, first_order float64. 3 positions, 4 active
    factors, 4 diagonal entries (1 AAPL + 2 Brent + 1 WTI), 1 cross entry.
    """
    _pricing, calculator = pricing_and_sensitivities
    book = calculator.book

    assert book.position_uids.dtype == np.int64
    assert book.active_factor_ids.dtype == np.int32
    assert book.first_order.dtype == np.float64

    assert book.position_uids.shape[0] == 3
    assert book.active_factor_ids.shape[0] == 4
    assert book.first_order.shape[0] == 4
    assert book.cross_second_order.shape[0] == 1


def test_demo_contains_no_local_sensitivity_mathematics() -> None:
    """
    The demo must consume the production sensitivity layer, not reimplement
    it. The old demo-local `_spot_block_sensitivities` is gone, and the
    orchestration function delegates to `run_sensitivities`.
    """
    import inspect

    assert not hasattr(_DEMO, "_spot_block_sensitivities")

    source = inspect.getsource(_DEMO._run_sensitivities)
    assert "run_sensitivities(" in source
    assert "build_sensitivity_book_spec(" in source


def test_production_sensitivity_core_sees_no_risk_factor_names() -> None:
    """
    Names must not enter the numerical sensitivity core. Assert against the
    production modules now that the mathematics lives there.
    """
    import inspect

    import riskquant.sensitivities.engine as engine_mod
    import riskquant.sensitivities.sensitizer_kernels.spot as kernel_mod
    import riskquant.sensitivities.sensitizers.spot as sensitizer_mod

    for mod in (engine_mod, sensitizer_mod, kernel_mod):
        source = inspect.getsource(mod)
        for name in ("AAPL_EQ", "BRENT_CRUDE", "USDEUR", "WTI_CRUDE"):
            assert name not in source, f"{mod.__name__} leaks factor name {name}"
