# tests/e2e/test_demo_e2e_sensi_based_var.py
"""
Tests for `scripts/demos/e2e/demo_e2e_sensi_based_var.py` and its shared
`_fixtures.py`.

Covers:

  1. positions (edge) -> IngestBatch -> CompiledPortfolio
  2. CompiledPortfolio + RiskFactorLevelSnapshot + RiskStaticConfig
     -> pricing (MTM) -> trade sensitivities, compiled into the canonical
     `solutions.sensitivity_pnl` book contract.
  3. A source-inspection guard: the demo's production-call section must go
     through `riskquant.solutions.risk_run.api.run_scenario_var_es`, never
     reconstruct scenario generation, shock application, reporting-currency
     conversion, or VaR/ES by hand.
  4. An integration test: the demo's own fixture (portfolio, sensitivity
     book, proxy-resolved shock-universe axis, ReportingSpec) actually
     drives `run_scenario_var_es` to a valid `RiskRun`, reconciled against
     the D019 golden fixture's pinned VaR/ES.

Exhaustive production-level testing of `run_scenario_var_es` itself (identity
tamper-checks, stale-input refusal, reproducibility, batch-size invariance,
seed policy) lives in `tests/solutions/risk_run` and is NOT duplicated here;
this file only tests THIS demo's wiring into that production entry point.

Demo modules live under `scripts/`, which ships no `__init__.py` package
markers at all, so they are loaded directly from their file path -- mirrors
`tests/solutions/sensitivity_pnl/test_canonical_usage_example_is_runnable.py`.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
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
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig
from riskquant.universe import UixMapper
from riskquant.valuation.blocks.spot_block import SpotBlock

_DEMO_DIR = Path(__file__).resolve().parents[2] / "scripts" / "demos" / "e2e"


def _load_module(name: str, filename: str) -> ModuleType:
    """`demo_e2e_sensi_based_var.py` itself does `import _fixtures`, so the
    demo directory must be on `sys.path` before either module is loaded.
    Registering the module in `sys.modules` before `exec_module` is required
    for dataclass field-type resolution (`@dataclass` in `_fixtures.py`
    looks up `sys.modules[cls.__module__]`)."""
    if str(_DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(_DEMO_DIR))
    spec = importlib.util.spec_from_file_location(name, _DEMO_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FIXTURES = _load_module("_fixtures", "_fixtures.py")
_DEMO = _load_module("demo_e2e_sensi_based_var", "demo_e2e_sensi_based_var.py")

# The demo's own three trades, by uid. Golden values below assume exactly
# these positions and `_fixtures._RISK_FACTOR_LEVELS`; if either changes,
# these tests and `_fixtures.run_fixture_golden_checks` must be updated
# together (D019).
_AAPL_UID = 1_010
_BRENT_UID = 1_020
_WTI_UID = 1_030


@pytest.fixture(scope="module")
def rsc() -> RiskStaticConfig:
    """Compile the fixture's own minimal RiskStaticConfig once per module."""
    return _FIXTURES._run_risk_static_config(verbose=False)


@pytest.fixture(scope="module")
def portfolio_and_mapper(
    rsc: RiskStaticConfig,
) -> tuple[CompiledPortfolio, UixMapper]:
    """Compile the fixture's own three-trade positions frame once per module."""
    ingest_batch = compile_raw_df_into_ingest_batch(
        raw_df=_FIXTURES._POSITIONS_DF, static_config=rsc
    )
    mapper = build_uix_mapper(
        [ingest_batch],
        as_of_ord32=int(_FIXTURES.date_to_ord32(_FIXTURES._SNAPSHOT_AS_OF_DATE)),
    )
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
    The fixture's own three-row positions frame, compiled end to end, must
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

    assert portfolio.total_trades == len(_FIXTURES._POSITIONS_DF)
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
    raw = pd.DataFrame({
        "uid": [2001],
        "instrument": ["VANILLA_FORWARD_NODE"],
        "quantity": [10.0],
        "underlying": ["AAPL_EQ"],
        "settlement_ccy": ["USD"],
        "strike_price": [100.0],
        "strike_ccy": ["USD"],
        # Added as test was failing because of missing columns
        "delivery_date": ["2024-12-31"],
        "settlement_date": ["2025-01-02"],
        "under_discount_curve": ["USD"],
        "strike_discount_curve": ["USD"],
    })

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
    `_run_portfolio` is the exact function `build_fixture` calls for this
    step; running it directly keeps this test fast while still exercising
    the real fixture code path and its printed summary.
    """
    portfolio, mapper = _FIXTURES._run_portfolio(rsc, verbose=True)

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
    """Run the fixture's own pricing + sensitivity derivation once per module."""
    portfolio, mapper = portfolio_and_mapper
    snapshot = _FIXTURES.build_risk_factor_level_snapshot_from_dict(
        as_of_date=_FIXTURES._SNAPSHOT_AS_OF_DATE,
        levels_dict=_FIXTURES._RISK_FACTOR_LEVELS,
        static_config=rsc,
    )
    market_data = _FIXTURES._run_market_data(
        rsc=rsc, uix_mapper=mapper, snapshot=snapshot
    )
    pricing = _FIXTURES._run_pricing(
        portfolio=portfolio, uix_mapper=mapper, market_data=market_data, verbose=False
    )
    calculator = _FIXTURES._run_sensitivities(
        rsc=rsc,
        portfolio=portfolio,
        uix_mapper=mapper,
        market_data=market_data,
        verbose=False,
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
    quanto / cross-currency case), matching the golden value exactly:
    -50 * 82.30 * 1.0800 = -4444.20 EUR.
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
    (dV/dUSDEUR = qty * Brent_USD = -4115.00 EUR per unit FX).
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


def test_fixture_contains_no_local_sensitivity_mathematics() -> None:
    """
    The fixture must consume the production sensitivity layer, not
    reimplement it: `_run_sensitivities` delegates to `run_sensitivities`.
    """
    source = inspect.getsource(_FIXTURES._run_sensitivities)
    assert "run_sensitivities(" in source
    assert "build_sensitivity_book_spec(" in source


def test_production_sensitivity_core_sees_no_risk_factor_names() -> None:
    """
    Names must not enter the numerical sensitivity core. Assert against the
    production modules that hold the mathematics.
    """
    import riskquant.sensitivities.engine as engine_mod
    import riskquant.sensitivities.sensitizer_kernels.spot as kernel_mod
    import riskquant.sensitivities.sensitizers.spot as sensitizer_mod

    for mod in (engine_mod, sensitizer_mod, kernel_mod):
        source = inspect.getsource(mod)
        for name in ("AAPL_EQ", "BRENT_CRUDE", "USDEUR", "WTI_CRUDE"):
            assert name not in source, f"{mod.__name__} leaks factor name {name}"


# =============================================================================
# 4. PRODUCTION SCENARIO-VAR ENTRY POINT
# =============================================================================


def test_canonical_demo_uses_the_production_risk_run_api() -> None:
    """
    The canonical demo's production-call section must go through
    `run_scenario_var_es`, never reconstruct scenario generation, shock
    application, reporting-currency conversion, or VaR/ES by hand -- that
    would silently re-introduce the manual orchestration this refactor
    removed (see DECISIONS.md D042-D044). The demo's separate, explicitly
    labelled `_run_validation` function is exempt: it deliberately drives
    the low-level building blocks directly, on a hand-authored 3-scenario
    case no production `ScenarioModelConfig` represents, to hand-verify the
    wiring -- not to demonstrate production usage.
    """
    source = inspect.getsource(_DEMO._run_canonical)
    assert "run_scenario_var_es(" in source
    for forbidden in (
        "apply_shocks_to_base_levels(",
        "convert_scenario_pnl_to_reporting_currency(",
        "compute_var_es(",
        "build_scenario_generator(",
    ):
        assert forbidden not in source, (
            f"_run_canonical must not reimplement {forbidden!r}; that is "
            "run_scenario_var_es's job"
        )


@pytest.fixture(scope="module")
def canonical_risk_run():
    """The demo's own fixture, driven through the production entry point
    with the same model/horizon/confidence level the canonical demo uses."""
    fixture = _FIXTURES.build_fixture(verbose=False)
    su = fixture.shock_universe
    run = run_scenario_var_es(
        scenario_config=HistoricalGeneratorConfig(model="historical", horizon=1),
        data=fixture.historical_matrix,
        required_factor_ids=su.required_factor_ids,
        required_shift_type_id=su.required_shift_type_id,
        source_factor_ids=su.source_factor_ids,
        shock_scale=su.shock_scale,
        base_levels=su.base_levels,
        methodology=fixture.methodology,
        taylor_derived=fixture.taylor_derived,
        reporting=fixture.reporting,
        confidence_level=2.0 / 3.0,
        batch_size=4,
    )
    return run, fixture


def test_demo_fixture_drives_the_production_entry_point(canonical_risk_run) -> None:
    """The demo's fixture (proxy-resolved shock-universe axis, compiled
    book, ReportingSpec) is well-formed input to `run_scenario_var_es`: a
    complete, self-identifying `RiskRun` comes back."""
    run, fixture = canonical_risk_run
    assert run.reporting_ccy == "USD"
    assert run.reproducible is True
    assert run.identity == run.manifest.identity()

    n = fixture.calculator.book.position_uids.shape[0] + 1
    assert run.var.shape == (n,)
    assert run.es.shape == (n,)
    assert np.all(np.isfinite(run.var))
    assert np.all(np.isfinite(run.es))


def _independent_reporting_pnl(fixture) -> np.ndarray:
    """
    L0 oracle for the D019 book: settlement Taylor P&L converted to reporting
    currency by the ordinary-MTM identity dW = dV*q_s + V0*(q_s - q0) (D081 §2),
    written as a plain per-position loop that never calls
    `riskquant.reporting`. USDEUR quotes EUR per USD, so EUR->USD is 1/USDEUR.
    """
    import polars as pl

    from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
    from riskquant.solutions.scenario_models.api import build_scenario_generator
    from riskquant.solutions.sensitivity_pnl.api import FactorAxis

    su = fixture.shock_universe
    gen = build_scenario_generator(
        HistoricalGeneratorConfig(model="historical", horizon=1),
        data=fixture.historical_matrix,
    )
    k, s_total = su.required_factor_ids.shape[0], gen.total_scenarios
    shocked = np.empty((k, s_total), dtype=np.float64)
    for batch in gen.iter_batches(
        required_shift_type_id=su.required_shift_type_id,
        source_factor_ids=su.source_factor_ids,
        shock_scale=su.shock_scale,
        batch_size=s_total,
    ):
        out = np.empty((k, batch.shocks.shape[0]), dtype=np.float64)
        apply_shocks_to_base_levels(base_levels=su.base_levels, batch=batch, out=out)
        shocked[:, batch.scenario_offset : batch.scenario_offset + out.shape[1]] = out
    axis = FactorAxis.from_keys(
        pl.Series("factor_id", su.required_factor_ids.tolist(), dtype=pl.Int32)
    )
    dv = np.asarray(
        fixture.calculator.run(
            axis.levels(base_levels=su.base_levels, shocked_levels=shocked)
        ).pnl,
        dtype=np.float64,
    )
    rep = fixture.reporting
    usdeur_s = shocked[rep.fx_factor_rows][0]
    usdeur_0 = float(su.base_levels[rep.fx_factor_rows][0])
    out = np.empty_like(dv)
    for i, ccy in enumerate(rep.settlement_ccys):
        if ccy == rep.reporting_ccy:
            rate_s, rate_0 = np.ones(s_total), 1.0
        else:
            rate_s, rate_0 = 1.0 / usdeur_s, 1.0 / usdeur_0
        out[i] = dv[i] * rate_s + float(fixture.taylor_derived.base_settlement_values[i]) * (
            rate_s - rate_0
        )
    return out


def test_demo_fixture_reconciles_with_pinned_golden_var_es(canonical_risk_run) -> None:
    """
    D019 golden fixture under D081: this exact universe, 10-day history, proxy
    scale and confidence level pin the portfolio VaR/ES.

    History of the pin. Under D014 (`dW = dV*q_s`, superseded) the portfolio VaR
    was -159.4309 / ES -257.1254. Under D081 ordinary MTM
    (`dW = V_s*q_s - V_0*q_0 = dV*q_s + V_0*(q_s - q_0)`) the EUR-settled Brent
    position (V_0 = -4444.20 EUR) carries base-value translation at each
    scenario's USDEUR, which moves the pin to -172.8161 / -269.8449. Both sets
    were reproduced by the independent per-position loop below (not by
    `riskquant.reporting`), so the change is the contract, not a defect.
    """
    from riskquant.scenario_metrics.api.var_es import compute_var_es

    run, fixture = canonical_risk_run
    assert run.portfolio_var == pytest.approx(-172.8161, abs=5e-3)
    assert run.portfolio_es == pytest.approx(-269.8449, abs=5e-3)

    # independent recomputation from the same inputs
    rows = _independent_reporting_pnl(fixture)
    groups = np.ascontiguousarray(np.vstack([rows, rows.sum(axis=0)[None, :]]))
    ref = compute_var_es(groups, confidence_level=2.0 / 3.0)
    assert run.portfolio_var == pytest.approx(float(ref.var[-1]), abs=1e-9)
    assert run.portfolio_es == pytest.approx(float(ref.es[-1]), abs=1e-9)