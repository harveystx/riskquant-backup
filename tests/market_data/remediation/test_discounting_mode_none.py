# tests/market_data/remediation/test_discounting_mode_none.py
"""
D061 `discounting_mode = NONE` — explicit no-discounting, end to end.

    NONE  -> discount_factor_grid == 1.0 -> valuation -> pricer
    CURVE -> NotImplementedError(D061), before any curve machinery runs

Everything here drives the real path: canonical static config, real ingest,
real UixMapper, real `build_market_data_matrix_from_snapshot`, the real fold and
the real pricers. Expected values come from independent arithmetic (and SciPy's
normal CDF for the option), never from a production helper.
"""

from __future__ import annotations

from datetime import date

import hostile_fixture as hf
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import riskquant.market_data.builder as builder_mod
import riskquant.market_data.curves.interp_defs as interp_defs
from riskquant.config.enums.curve_input_types import CurveInputType
from riskquant.config.enums.discounting_modes import (
    VALID_DISCOUNTING_MODES,
    DiscountingMode,
)
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.constants import NO_ID, TTE_FLOOR
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

AS_OF = hf.AS_OF
PILLAR = "BRENT_FWD[M+0]"
CURVE = "USD_OIS"
VOL = "VOL_EQ_USD"
DELIVERY = date(2027, 2, 15)
SETTLEMENT = date(2027, 2, 17)
EXPIRY = date(2027, 2, 16)
CDF_RTOL = 5e-5  # D046: production uses the Abramowitz-Stegun CDF, the oracle SciPy


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())


def _forward(**overrides) -> pd.DataFrame:
    row = dict(
        uid=1,
        instrument="VANILLA_FORWARD_NODE",
        quantity=1_000.0,
        underlying=PILLAR,
        settlement_ccy="USD",
        strike_price=75.0,
        strike_ccy="USD",
        delivery_date=DELIVERY,
        settlement_date=SETTLEMENT,
        under_discount_curve=CURVE,
        strike_discount_curve=CURVE,
    )
    row.update(overrides)
    return pd.DataFrame([row])


def _option(**overrides) -> pd.DataFrame:
    row = dict(
        uid=2,
        instrument="VANILLA_EUROPEAN_OPTION_NODE",
        quantity=500.0,
        underlying=PILLAR,
        settlement_ccy="USD",
        strike_price=75.0,
        strike_ccy="USD",
        is_call=1.0,
        expiry_date=EXPIRY,
        settlement_date=SETTLEMENT,
        volatility_container_name=VOL,
        under_discount_curve=CURVE,
        strike_discount_curve=CURVE,
    )
    row.update(overrides)
    return pd.DataFrame([row])


def _run(rsc, frame, levels, mode=DiscountingMode.NONE, as_of=AS_OF):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=frame, static_config=rsc, copy=True, as_of_date=as_of
    )
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    um = build_universe_mapping(
        discounting_mode=mode, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=as_of,
        levels_dict={
            names[int(i)]: levels[names[int(i)]] for i in um.required_factor_ids
        },
        static_config=rsc,
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=mode,
        as_of_date=as_of,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return matrix, state, float(out[0, 0])


@pytest.fixture
def spy(monkeypatch):
    """Records any entry into curve construction."""
    calls: list[str] = []
    for module, name in (
        (builder_mod, "build_zero_curve_discount_factor_grid"),
        (builder_mod, "_build_price_curve_grid"),
        (builder_mod, "_scatter_price_curve_grid_to_universe"),
        (interp_defs, "interp_linear_on_log_values_linear_extrap_3d_numba_kernel"),
    ):
        real = getattr(module, name)

        def wrapped(*a, _real=real, _name=name, **k):
            calls.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(module, name, wrapped)
    return calls


# ----------------------------------------------------------------- enum contract
def test_discounting_mode_enum_has_exactly_two_members():
    assert [m.value for m in DiscountingMode] == ["NONE", "CURVE"]
    assert {"NONE", "CURVE"} == VALID_DISCOUNTING_MODES
    assert DiscountingMode.NONE == "NONE"


def test_the_mode_has_no_default_at_the_producer_boundary():
    import inspect

    for func in (
        builder_mod.build_market_data_matrix,
        __import__(
            "riskquant.market_data.assemble", fromlist=["x"]
        ).build_market_data_matrix_from_snapshot,
    ):
        param = inspect.signature(func).parameters["discounting_mode"]
        assert param.default is inspect.Parameter.empty


# --------------------------------------------------------------- A. NONE forward
def test_none_mode_forward_prices_without_discounting(rsc):
    levels = {PILLAR: 82.30, f"{CURVE}[3M]": 4.25}
    _, _, pv = _run(rsc, _forward(), levels)
    # independent oracle: DF = 1 on both legs, single currency
    assert pv == pytest.approx(1_000.0 * (82.30 - 75.0), rel=1e-12)


# ---------------------------------------------------------------- B. NONE option
def test_none_mode_option_prices_undiscounted_black76(rsc):
    levels = {PILLAR: 82.30, VOL: 31.0, f"{CURVE}[3M]": 4.25}
    matrix, _, pv = _run(rsc, _option(), levels)

    f, k, sigma = 82.30, 75.0, 0.31  # VOL_EQ_USD is PERCENT -> decimal (D069)
    tte = max((date_to_ord32(EXPIRY) - date_to_ord32(AS_OF)) / 365.25, float(TTE_FLOOR))
    d1 = (np.log(f / k) + 0.5 * sigma * sigma * tte) / (sigma * np.sqrt(tte))
    d2 = d1 - sigma * np.sqrt(tte)
    expected = 500.0 * (f * norm.cdf(d1) - k * norm.cdf(d2))  # no discount factor
    assert pv == pytest.approx(expected, rel=CDF_RTOL)
    assert np.all(matrix.discount_factor_grid == 1.0)


# ------------------------------------------------------------------ C. NONE grid
def test_none_mode_grid_is_ones_over_the_run_universes(rsc):
    levels = {PILLAR: 82.30, f"{CURVE}[3M]": 4.25}
    matrix, state, _ = _run(rsc, _forward(), levels)
    n_curves = 1  # USD_OIS
    n_dates = 2  # delivery and settlement
    assert matrix.discount_factor_grid.shape == (n_curves, n_dates, 1)
    assert np.all(matrix.discount_factor_grid == 1.0)
    assert matrix.discount_factor_grid.dtype == np.float64
    # the fold carries the ones through unchanged
    assert state.effective_discount_factors.shape[0] > 0
    assert np.all(state.effective_discount_factors == 1.0)


def test_a_run_without_discount_curves_keeps_the_empty_grid(rsc):
    spot = pd.DataFrame([
        dict(
            uid=3,
            instrument="SPOT",
            quantity=10.0,
            underlying="EQ_USD",
            settlement_ccy="USD",
        )
    ])
    matrix, state, pv = _run(rsc, spot, {"EQ_USD": 100.0})
    assert matrix.discount_factor_grid.shape == (0, 0, 1)
    assert state.effective_discount_factors.shape[0] == 0
    assert pv == pytest.approx(1_000.0, rel=1e-12)


# ------------------------------------------------------------- D. date validation
def test_target_date_before_the_as_of_fails(rsc):
    stale = _forward(delivery_date=date(2026, 2, 15), settlement_date=date(2026, 2, 17))
    with pytest.raises(Exception) as excinfo:
        _run(rsc, stale, {PILLAR: 82.30, f"{CURVE}[3M]": 4.25})
    # rejected by the dated-contract gate at ingest or by the producer's
    # pre-as-of rule; either way never silently clipped to DF = 1
    assert "D065" in str(excinfo.value) or "precede the run as-of" in str(excinfo.value)


def test_pre_as_of_date_is_rejected_at_the_producer(rsc):
    """Direct producer call: a target ordinal before the as-of is not supported."""
    import dataclasses

    batch = compile_raw_df_into_ingest_batch(
        raw_df=_forward(), static_config=rsc, copy=True, as_of_date=AS_OF
    )
    mapper = build_uix_mapper([batch])
    stale = np.sort(
        np.append(mapper.date_ord32_universe, np.int32(int(date_to_ord32(AS_OF)) - 10))
    ).astype(np.int32)
    poisoned = dataclasses.replace(mapper, date_ord32_universe=stale)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=poisoned
    )
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=AS_OF,
        levels_dict={PILLAR: 82.30, f"{CURVE}[3M]": 4.25},
        static_config=rsc,
    )
    with pytest.raises(ValueError, match="precede the run as-of"):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=AS_OF,
            uix_mapper=poisoned,
            universe_map=um,
            snapshot=snapshot,
        )


def test_target_date_equal_to_the_as_of_is_one(rsc):
    same_day = _option(expiry_date=AS_OF, settlement_date=AS_OF)
    matrix, state, _ = _run(
        rsc, same_day, {PILLAR: 82.30, VOL: 31.0, f"{CURVE}[3M]": 4.25}
    )
    as_of_row = int(np.searchsorted(matrix.tte_grid, 0.0))
    assert np.all(matrix.discount_factor_grid == 1.0)
    assert matrix.discount_factor_grid[0, as_of_row, 0] == 1.0
    assert np.all(state.effective_discount_factors == 1.0)


# --------------------------------------------------------------- E. CURVE closed
@pytest.mark.parametrize(
    "curve_input_type",
    [None, *(m.value for m in CurveInputType)],
)
def test_curve_mode_fails_closed_for_every_curve_input_type(rsc, spy, curve_input_type):
    """
    Declaring an input type on every discount curve of the HOSTILE fixture does
    not unlock CURVE: none of its curves carries D080 maturity metadata, so the
    layout plan fails the whole run before any level is gathered and before any
    curve function is entered.
    """
    import polars as pl

    dfs = hf.static_config_dfs()
    if curve_input_type is not None:
        cm = dfs["curve_meta"]
        dfs["curve_meta"] = cm.with_columns(
            pl.Series(
                "curve_input_type",
                [
                    curve_input_type if b != "NOT_A_DISCOUNT_CURVE" else None
                    for b in cm["discount_basis"].to_list()
                ],
                dtype=pl.String,
            )
        )
    declared = compile_risk_static_config_from_dict_of_dfs(dfs)
    with pytest.raises(NotImplementedError, match="D061"):
        _run(
            declared,
            _forward(),
            {PILLAR: 82.30, f"{CURVE}[3M]": 4.25},
            mode=DiscountingMode.CURVE,
        )
    assert spy == []  # no curve construction of any kind was entered


# ------------------------------------------------ F. curve levels are irrelevant
def test_none_mode_pv_is_invariant_to_the_curve_levels(rsc):
    base = {PILLAR: 82.30, f"{CURVE}[3M]": 4.25}
    shocked = {PILLAR: 82.30, f"{CURVE}[3M]": 40.00}  # materially different rate
    matrix_a, _, pv_a = _run(rsc, _forward(), base)
    matrix_b, _, pv_b = _run(rsc, _forward(), shocked)
    assert base[f"{CURVE}[3M]"] != shocked[f"{CURVE}[3M]"]
    assert pv_a == pv_b == pytest.approx(1_000.0 * (82.30 - 75.0), rel=1e-12)
    np.testing.assert_array_equal(
        matrix_a.discount_factor_grid, matrix_b.discount_factor_grid
    )


# ------------------------------------------------------ G. invalid state still fails
def test_none_mode_does_not_rescue_a_missing_curve_identity(rsc):
    import dataclasses

    batch = compile_raw_df_into_ingest_batch(
        raw_df=_forward(), static_config=rsc, copy=True, as_of_date=AS_OF
    )
    stripped = dataclasses.replace(
        batch,
        under_discount_curve_ids=np.full(1, NO_ID, np.int32),
        strike_discount_curve_ids=np.full(1, NO_ID, np.int32),
    )
    mapper = build_uix_mapper([stripped])
    with pytest.raises(ValueError, match="sentinel coordinate cannot be packed"):
        compile_portfolio_from_ingest_batch(ingest_batch=stripped, uix_mapper=mapper)


def test_none_mode_keeps_the_d062_and_d066_linkage_requirements(rsc):
    with pytest.raises(ValueError, match="Missing required columns"):
        compile_raw_df_into_ingest_batch(
            raw_df=_forward().drop(columns=["under_discount_curve"]),
            static_config=rsc,
            copy=True,
            as_of_date=AS_OF,
        )
    with pytest.raises(Exception, match="curve currency must equal leg"):
        compile_raw_df_into_ingest_batch(
            raw_df=_forward(strike_ccy="EUR"),
            static_config=rsc,
            copy=True,
            as_of_date=AS_OF,
        )


def test_none_mode_still_rejects_non_finite_levels(rsc):
    """NONE does not bypass the producer's level-domain gate (AUD-05)."""
    with pytest.raises(ValueError, match="non-finite"):
        _run(rsc, _forward(), {PILLAR: float("nan"), f"{CURVE}[3M]": 4.25})


def test_discount_curve_levels_are_not_even_required_under_none(rsc):
    """
    A LABEL discount curve contributes no pillars to the universe (the known
    D061 LABEL gap), so under NONE its levels are neither required nor read.
    The curve's IDENTITY is still required by D062/D066 - only its levels are
    unused.
    """
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_forward(), static_config=rsc, copy=True, as_of_date=AS_OF
    )
    mapper = build_uix_mapper([batch])
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    required = [names[int(i)] for i in um.required_factor_ids]
    assert required == [PILLAR]  # no USD_OIS pillar
    assert mapper.discount_curve_id_universe.shape[0] == 1  # identity still present
