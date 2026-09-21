# tests/market_data/remediation/test_d080_zero_curve_constructor.py
"""
D080 Phase 3: the live USD_SOFR_OIS constructor, certified against `d080_oracle`.

Everything runs the REAL path — canonical config, real ingest, real universe
plan, real producer, real fold, real pricers — and every expected value is
produced by the oracle, which imports stdlib and numpy only.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import math

import d080_oracle as oracle
import numpy as np
import pandas as pd
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.constants import DAYS_PER_YR_ACT_365F
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.ingest_fixtures.market_data.canonical_market_data_snapshot import (
    CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.curves.zero_curve import (
    build_zero_curve_discount_factor_grid,
)
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

A = CANONICAL_AS_OF_DATE
CURVE = "USD_SOFR_OIS"
LABELS = ("ON", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y")
PERCENT = {
    label: CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT[f"{CURVE}[{label}]"]
    for label in LABELS
}
TOL = 1e-13  # linear-in-lnDF + exp on float64; reported max is far tighter


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )


@pytest.fixture(scope="module")
def us_holidays() -> frozenset[dt.date]:
    rows = CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS["holiday_calendars"]
    return frozenset(
        dt.date.fromisoformat(str(r["holiday_date"]))
        for r in rows.iter_rows(named=True)
        if r["holiday_calendar_name"] == "US"
    )


@pytest.fixture(scope="module")
def oracle_curve(us_holidays):
    return oracle.curve(
        labels=LABELS, percent_levels=PERCENT, as_of=A, holidays=us_holidays
    )


def _book(dates: list[dt.date]) -> pd.DataFrame:
    """Forwards whose delivery/settlement dates pin the run's date universe."""
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            dict(
                uid=i + 1,
                instrument="VANILLA_FORWARD_NODE",
                quantity=1.0,
                underlying="BRENT_CRUDE_FWD[M+0]",
                settlement_ccy="USD",
                strike_price=70.0,
                strike_ccy="USD",
                delivery_date=dt.date(2026, 2, 15),
                settlement_date=d,
                under_discount_curve=CURVE,
                strike_discount_curve=CURVE,
            )
        )
    return pd.DataFrame(rows)


def _plan(rsc, dates, as_of=A):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_book(dates), static_config=rsc, copy=True, as_of_date=as_of
    )
    mapper = build_uix_mapper([batch])
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    return batch, mapper, um


def _levels_for(rsc, um, percent=PERCENT, other=80.0):
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    out = {}
    for rf in um.required_factor_ids:
        name = names[int(rf)]
        label = name.split("[")[-1].rstrip("]")
        out[name] = percent.get(label, other) if name.startswith(CURVE) else other
    return out


def _grid(rsc, dates, percent=PERCENT, as_of=A):
    _, mapper, um = _plan(rsc, dates, as_of)
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=as_of, levels_dict=_levels_for(rsc, um, percent), static_config=rsc
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.CURVE,
        as_of_date=as_of,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    return matrix, mapper, um


def _df_at(matrix, mapper, date: dt.date, s=0) -> float:
    row = int(
        np.searchsorted(mapper.date_ord32_universe, np.int32(date_to_ord32(date)))
    )
    assert mapper.date_ord32_universe[row] == date_to_ord32(date)
    return float(matrix.discount_factor_grid[0, row, s])


def _oracle_df(oracle_curve, date: dt.date) -> float:
    t, ln = oracle_curve
    return oracle.discount_factor(
        t_target=oracle.act365f(A, date), pillar_t=t, pillar_ln=ln
    )


# ------------------------------------------------------------- 31/32 pillars
def test_every_pillar_is_reproduced_exactly(rsc, us_holidays, oracle_curve):
    maturities = [
        oracle.maturity(label=lab, as_of=A, holidays=us_holidays) for lab in LABELS
    ]
    matrix, mapper, _ = _grid(rsc, maturities)
    worst = 0.0
    for label, mat in zip(LABELS, maturities, strict=True):
        produced = _df_at(matrix, mapper, mat)
        expected = math.exp(-PERCENT[label] * 0.01 * oracle.act365f(A, mat))
        worst = max(worst, abs(produced - expected))
        assert produced == pytest.approx(expected, rel=TOL, abs=0), label
    assert worst < 1e-15  # report: float64 linear-in-lnDF hits pillars to ~1 ulp
    ten_y = _df_at(matrix, mapper, maturities[-1])
    assert ten_y == pytest.approx(0.669733, abs=5e-7)


# --------------------------------------------------------------- 33 anchor
def test_the_as_of_row_is_exactly_one_for_every_scenario(rsc):
    dates = [A, dt.date(2026, 3, 2)]
    matrix, mapper, _ = _grid(rsc, dates)
    assert _df_at(matrix, mapper, A) == 1.0
    shocked, mapper2, _ = _grid(rsc, dates, {k: v + 3.0 for k, v in PERCENT.items()})
    assert (
        _df_at(shocked, mapper2, A) == 1.0
    )  # a 300 bp shock does not touch the anchor


# ----------------------------------------------------------- 34 interior x8
def test_interior_targets_between_every_adjacent_pillar_pair(
    rsc, us_holidays, oracle_curve
):
    maturities = [
        oracle.maturity(label=lab, as_of=A, holidays=us_holidays) for lab in LABELS
    ]
    targets = []
    for lo, hi in itertools.pairwise(maturities):
        gap = (hi - lo).days
        assert gap >= 2, (lo, hi)
        targets.append(lo + dt.timedelta(days=max(1, gap // 3)))  # non-midpoint
    matrix, mapper, _ = _grid(rsc, targets)
    for target in targets:
        assert _df_at(matrix, mapper, target) == pytest.approx(
            _oracle_df(oracle_curve, target), rel=TOL, abs=0
        )


def test_interpolation_is_affine_in_ln_df_not_in_rate_or_df(
    rsc, us_holidays, oracle_curve
):
    five_y = oracle.maturity(label="5Y", as_of=A, holidays=us_holidays)
    ten_y = oracle.maturity(label="10Y", as_of=A, holidays=us_holidays)
    mid = five_y + dt.timedelta(days=(ten_y - five_y).days // 2)
    matrix, mapper, _ = _grid(rsc, [five_y, mid, ten_y])
    ln5, ln10 = (
        math.log(_df_at(matrix, mapper, five_y)),
        math.log(_df_at(matrix, mapper, ten_y)),
    )
    t5, tm, t10 = (oracle.act365f(A, d) for d in (five_y, mid, ten_y))
    w = (tm - t5) / (t10 - t5)
    assert math.log(_df_at(matrix, mapper, mid)) == pytest.approx(
        ln5 + w * (ln10 - ln5), rel=TOL
    )
    # the two rejected alternatives are measurably different here
    linear_df = math.exp(ln5) + w * (math.exp(ln10) - math.exp(ln5))
    linear_zero = math.exp(-((-ln5 / t5) + w * ((-ln10 / t10) - (-ln5 / t5))) * tm)
    assert abs(_df_at(matrix, mapper, mid) - linear_df) > 1e-4
    assert abs(_df_at(matrix, mapper, mid) - linear_zero) > 1e-4


# ------------------------------------------------------------ 35 right tail
def test_right_tail_extends_the_final_ln_df_slope(rsc, us_holidays, oracle_curve):
    ten_y = oracle.maturity(label="10Y", as_of=A, holidays=us_holidays)
    targets = [
        ten_y,
        ten_y + dt.timedelta(days=1),
        ten_y + dt.timedelta(days=5 * 365),
        ten_y + dt.timedelta(days=10 * 365),
    ]
    matrix, mapper, _ = _grid(rsc, targets)
    t, ln = oracle_curve
    s_last = (ln[-1] - ln[-2]) / (t[-1] - t[-2])
    assert s_last == pytest.approx(-0.0389978106, abs=1e-9)
    for target in targets:
        assert _df_at(matrix, mapper, target) == pytest.approx(
            _oracle_df(oracle_curve, target), rel=TOL, abs=0
        )
    df_10, df_far = (
        _df_at(matrix, mapper, targets[0]),
        _df_at(matrix, mapper, targets[-1]),
    )
    assert df_far < df_10  # not flat DF beyond 10Y
    slope = (math.log(df_far) - math.log(df_10)) / (
        oracle.act365f(A, targets[-1]) - t[-1]
    )
    assert slope == pytest.approx(s_last, rel=1e-12)
    assert -slope > 0.0  # forward_last = -s_last is a positive forward rate


# ----------------------------------------------------------------- 36 shocks
def test_scenario_shocks_are_additive_on_the_decimal_rate(rsc, us_holidays):
    dates = [
        oracle.maturity(label=lab, as_of=A, holidays=us_holidays)
        for lab in ("1Y", "5Y", "10Y")
    ]
    _, mapper, um = _plan(rsc, dates)
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict=_levels_for(rsc, um), static_config=rsc
    )
    base = extract_levels_from_snapshot(
        required_ids=um.required_factor_ids, snapshot=snapshot
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    required = [names[int(rf)] for rf in um.required_factor_ids]
    # scenarios: 0 unshocked, +100bp parallel, -100bp parallel, non-parallel, per-scenario mixed
    shocks = {
        0: {},
        1: dict.fromkeys(LABELS, 0.01),
        2: dict.fromkeys(LABELS, -0.01),
        3: {"1Y": 0.0025, "5Y": -0.0050, "10Y": 0.0100},
        4: {"ON": 0.02, "10Y": -0.03},
    }
    flat = np.repeat(
        np.asarray(base, dtype=np.float64).reshape(-1, 1), len(shocks), axis=1
    )
    for s, table in shocks.items():
        for i, name in enumerate(required):
            if name.startswith(CURVE):
                flat[i, s] += table.get(name.split("[")[-1].rstrip("]"), 0.0)
    matrix = build_market_data_matrix_from_flat_levels(
        discounting_mode=DiscountingMode.CURVE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        flat_levels=np.ascontiguousarray(flat),
        num_scenarios=len(shocks),
    )
    for s, table in shocks.items():
        t, ln = oracle.curve(
            labels=LABELS,
            percent_levels=PERCENT,
            as_of=A,
            holidays=us_holidays,
            shocks=table,
        )
        for d in dates:
            expected = oracle.discount_factor(
                t_target=oracle.act365f(A, d), pillar_t=t, pillar_ln=ln
            )
            assert _df_at(matrix, mapper, d, s) == pytest.approx(
                expected, rel=TOL, abs=0
            ), (s, d)
    # direction, as a by-product: +100bp lowers, -100bp raises the 10Y DF
    assert (
        _df_at(matrix, mapper, dates[-1], 1)
        < _df_at(matrix, mapper, dates[-1], 0)
        < _df_at(matrix, mapper, dates[-1], 2)
    )


# ------------------------------------------------------- 37 zero / negative
@pytest.mark.parametrize(
    "percent, expectation",
    [(0.0, "one"), (-1.25, "above_one"), (4.80, "below_one")],
)
def test_zero_and_negative_rates_are_not_clipped(
    rsc, us_holidays, percent, expectation
):
    dates = [
        oracle.maturity(label="5Y", as_of=A, holidays=us_holidays),
        dt.date(2040, 1, 15),
    ]
    matrix, mapper, _ = _grid(rsc, dates, dict.fromkeys(LABELS, percent))
    values = [_df_at(matrix, mapper, d) for d in dates]
    if expectation == "one":
        assert values == [1.0, 1.0]
    elif expectation == "above_one":
        assert all(v > 1.0 for v in values)
    else:
        assert all(0.0 < v < 1.0 for v in values)


def test_mixed_sign_rates_match_the_oracle(rsc, us_holidays):
    mixed = dict(
        zip(LABELS, (-0.5, -0.2, 0.0, 0.3, 0.8, 1.5, 2.2, 3.1, 3.4), strict=True)
    )
    dates = [
        oracle.maturity(label=lab, as_of=A, holidays=us_holidays) for lab in LABELS
    ]
    dates.append(dt.date(2026, 4, 1))
    matrix, mapper, _ = _grid(rsc, dates, mixed)
    t, ln = oracle.curve(
        labels=LABELS, percent_levels=mixed, as_of=A, holidays=us_holidays
    )
    for d in dates:
        expected = oracle.discount_factor(
            t_target=oracle.act365f(A, d), pillar_t=t, pillar_ln=ln
        )
        assert _df_at(matrix, mapper, d) == pytest.approx(expected, rel=TOL, abs=0)


# ---------------------------------------------------------- 38 single pillar
def test_anchor_plus_one_market_pillar_constructs_and_extrapolates():
    pillar_t = np.array([[1.0]])
    rates = np.array([[[0.04, -0.01]]])  # two scenarios
    targets = np.array([0.0, 0.5, 1.0, 3.0])
    grid = build_zero_curve_discount_factor_grid(
        pillar_t=pillar_t, pillar_rates=rates, target_t=targets
    )
    assert grid.shape == (1, 4, 2)
    for s, z in enumerate((0.04, -0.01)):
        assert grid[0, 0, s] == 1.0
        assert grid[0, 1, s] == pytest.approx(math.exp(-z * 0.5), rel=TOL)
        assert grid[0, 2, s] == pytest.approx(math.exp(-z * 1.0), rel=TOL)
        assert grid[0, 3, s] == pytest.approx(math.exp(-z * 3.0), rel=TOL)  # slope -z


# --------------------------------------------------------------- 39 validation
def test_constructor_rejects_invalid_numeric_inputs():
    good_t, good_r, good_x = (
        np.array([[0.5, 1.0]]),
        np.array([[[0.04], [0.05]]]),
        np.array([0.0, 2.0]),
    )
    with pytest.raises(ValueError, match="at least one market pillar"):
        build_zero_curve_discount_factor_grid(
            pillar_t=np.empty((1, 0)), pillar_rates=np.empty((1, 0, 1)), target_t=good_x
        )
    with pytest.raises(ValueError, match="mature after the as-of"):
        build_zero_curve_discount_factor_grid(
            pillar_t=np.array([[0.0, 1.0]]), pillar_rates=good_r, target_t=good_x
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_zero_curve_discount_factor_grid(
            pillar_t=np.array([[1.0, 0.5]]), pillar_rates=good_r, target_t=good_x
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_zero_curve_discount_factor_grid(
            pillar_t=np.array([[1.0, 1.0]]), pillar_rates=good_r, target_t=good_x
        )
    with pytest.raises(ValueError, match="non-finite zero rate"):
        build_zero_curve_discount_factor_grid(
            pillar_t=good_t,
            pillar_rates=np.array([[[np.nan], [0.05]]]),
            target_t=good_x,
        )
    with pytest.raises(ValueError, match="pre-as-of"):
        build_zero_curve_discount_factor_grid(
            pillar_t=good_t, pillar_rates=good_r, target_t=np.array([-0.1, 1.0])
        )


def test_unsupported_curves_in_the_same_run_fail_the_whole_run(rsc):
    import polars as pl

    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm = dfs["curve_meta"]
    dfs["curve_meta"] = cm.with_columns(
        pl.Series(
            "curve_compounding",
            ["ANNUAL" if n == CURVE else None for n in cm["curve_name"].to_list()],
            dtype=pl.String,
        )
    )
    annual = compile_risk_static_config_from_dict_of_dfs(dfs)
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_book([dt.date(2026, 3, 2)]),
        static_config=annual,
        copy=True,
        as_of_date=A,
    )
    mapper = build_uix_mapper([batch])
    with pytest.raises(NotImplementedError, match="curve_compounding=ANNUAL"):
        build_universe_mapping(
            discounting_mode=DiscountingMode.CURVE,
            static_config=annual,
            uix_mapper=mapper,
        )


def test_pre_as_of_target_fails_before_construction(rsc):
    _, mapper, _um = _plan(rsc, [dt.date(2026, 3, 2)])
    stale = np.sort(
        np.append(mapper.date_ord32_universe, np.int32(date_to_ord32(A) - 5))
    ).astype(np.int32)
    poisoned = dataclasses.replace(mapper, date_ord32_universe=stale)
    um2 = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=poisoned
    )
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict=_levels_for(rsc, um2), static_config=rsc
    )
    with pytest.raises(ValueError, match="precede the run as-of"):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.CURVE,
            as_of_date=A,
            uix_mapper=poisoned,
            universe_map=um2,
            snapshot=snapshot,
        )


# ------------------------------------------------------------------- 40 order
def test_source_row_order_has_no_effect_on_the_grid(rsc):
    dates = [dt.date(2026, 3, 2), dt.date(2030, 6, 1)]
    _, mapper, um = _plan(rsc, dates)
    levels = _levels_for(rsc, um)
    a = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.CURVE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=compile_risk_factor_level_snapshot_from_levels_dict(
            as_of_date=A, levels_dict=levels, static_config=rsc
        ),
    )
    shuffled = dict(reversed(list(levels.items())))
    b = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.CURVE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=compile_risk_factor_level_snapshot_from_levels_dict(
            as_of_date=A, levels_dict=shuffled, static_config=rsc
        ),
    )
    np.testing.assert_array_equal(a.discount_factor_grid, b.discount_factor_grid)


# ----------------------------------------------------------------- 41 scaling
def test_double_normalisation_would_be_caught(rsc, us_holidays):
    ten_y = oracle.maturity(label="10Y", as_of=A, holidays=us_holidays)
    matrix, mapper, _ = _grid(rsc, [ten_y])
    produced = _df_at(matrix, mapper, ten_y)
    once = math.exp(-0.0400 * oracle.act365f(A, ten_y))  # 4.00 PERCENT -> 0.0400
    twice = math.exp(
        -0.000400 * oracle.act365f(A, ten_y)
    )  # what a second x0.01 would give
    assert produced == pytest.approx(once, rel=TOL)
    assert abs(produced - twice) > 0.3


# --------------------------------------------------------------- 42 calendar
def test_fixture_true_spot_and_eom_are_what_the_constructor_sees(rsc, us_holidays):
    assert oracle.bd_add(A, 2, us_holidays) == dt.date(2026, 1, 19)
    assert oracle.maturity(
        label="1M", as_of=dt.date(2026, 1, 28), holidays=us_holidays
    ) == dt.date(2026, 2, 27)
    assert oracle.maturity(label="1M", as_of=A, holidays=us_holidays) == dt.date(
        2026, 2, 19
    )  # no EOM


# ------------------------------------------------------- 29 valuation integration
def test_node_forward_prices_with_constructed_discount_factors(rsc, oracle_curve):
    delivery, settlement = dt.date(2026, 2, 15), dt.date(2026, 2, 17)
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_book([settlement]).assign(quantity=1_000.0, strike_price=75.0),
        static_config=rsc,
        copy=True,
        as_of_date=A,
    )
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    levels = _levels_for(rsc, um, other=82.30)
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict=levels, static_config=rsc
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.CURVE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((1, 1))
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    df_del, df_set = (
        _oracle_df(oracle_curve, delivery),
        _oracle_df(oracle_curve, settlement),
    )
    expected = 1_000.0 * (82.30 * df_del - 75.0 * df_set)
    assert out[0, 0] == pytest.approx(expected, rel=1e-12)
    assert np.all(state.effective_discount_factors < 1.0)


def test_none_mode_is_untouched_by_the_live_constructor(rsc):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_book([dt.date(2026, 3, 2)]), static_config=rsc, copy=True, as_of_date=A
    )
    mapper = build_uix_mapper([batch])
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    assert [names[int(i)] for i in um.required_factor_ids] == ["BRENT_CRUDE_FWD[M+0]"]
    assert um.shape_discount_curves == (1, 0) and um.raw_discount_tenors.shape == (1, 0)
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict={"BRENT_CRUDE_FWD[M+0]": 80.0}, static_config=rsc
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    assert np.all(matrix.discount_factor_grid == 1.0)
    assert DAYS_PER_YR_ACT_365F == 365.0


# ------------------------------------------------- D080.1 identity, not metadata
def test_syntactically_complete_metadata_on_another_curve_does_not_construct(rsc):
    """
    Declaring the full D080 metadata on EUR_ESTR_OIS must NOT make it
    constructible: construction is tied to curve identity (D080.1), and only
    USD_SOFR_OIS / 250000000 is supported.
    """
    import polars as pl

    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm = dfs["curve_meta"]
    names = cm["curve_name"].to_list()
    decl = {"EUR_ESTR_OIS", CURVE}
    dfs["curve_meta"] = cm.with_columns(
        pl.Series(
            "curve_input_type",
            ["ZERO_RATE" if n in decl else None for n in names],
            dtype=pl.String,
        ),
        pl.Series(
            "curve_compounding",
            ["CONTINUOUS" if n in decl else None for n in names],
            dtype=pl.String,
        ),
        pl.Series(
            "maturity_calendar",
            [
                "EUR" if n == "EUR_ESTR_OIS" else ("US" if n == CURVE else None)
                for n in names
            ],
            dtype=pl.String,
        ),
        pl.Series(
            "spot_lag_days", [2 if n in decl else None for n in names], dtype=pl.Int64
        ),
        pl.Series(
            "business_day_convention",
            ["MOD_FOLLOWING" if n in decl else None for n in names],
            dtype=pl.String,
        ),
        pl.Series(
            "end_of_month",
            [True if n in decl else None for n in names],
            dtype=pl.Boolean,
        ),
    )
    declared = compile_risk_static_config_from_dict_of_dfs(dfs)
    # a USD-settled BRENT forward whose STRIKE leg is EUR: the EUR discount
    # curve enters the run universe alongside USD_SOFR_OIS
    fwd = _book([dt.date(2026, 2, 17)]).assign(
        strike_ccy="EUR", strike_discount_curve="EUR_ESTR_OIS"
    )
    batch = compile_raw_df_into_ingest_batch(
        raw_df=fwd, static_config=declared, copy=True, as_of_date=A
    )
    mapper = build_uix_mapper([batch])
    with pytest.raises(
        NotImplementedError, match="not a D080-supported curve identity"
    ):
        build_universe_mapping(
            discounting_mode=DiscountingMode.CURVE,
            static_config=declared,
            uix_mapper=mapper,
        )


def test_node_option_prices_with_constructed_discount_factors(rsc, oracle_curve):
    """Black-76 on the pillar forward, discounted at settlement by the D080 DF."""
    from scipy.stats import norm

    from riskquant.constants import TTE_FLOOR

    settle = dt.date(2026, 7, 22)
    expiry = settle - dt.timedelta(days=2)
    row = dict(
        uid=1,
        instrument="VANILLA_EUROPEAN_OPTION_NODE",
        quantity=500.0,
        underlying="BRENT_CRUDE_FWD[M+0]",
        settlement_ccy="USD",
        strike_price=75.0,
        strike_ccy="USD",
        expiry_date=expiry,
        settlement_date=settle,
        is_call=1.0,
        volatility_container_name="BRENT_CRUDE_ATM_VOL",  # VOL node, DECIMAL units
        under_discount_curve=CURVE,
        strike_discount_curve=CURVE,
    )
    batch = compile_raw_df_into_ingest_batch(
        raw_df=pd.DataFrame([row]), static_config=rsc, copy=True, as_of_date=A
    )
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    levels = {}
    for rf in um.required_factor_ids:
        name = names[int(rf)]
        if name.startswith(CURVE):
            levels[name] = PERCENT[name.split("[")[-1].rstrip("]")]
        elif "VOL" in name:
            levels[name] = 0.31
        else:
            levels[name] = 82.30
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict=levels, static_config=rsc
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.CURVE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((1, 1))
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)

    df_settle = _oracle_df(oracle_curve, settle)
    f, k, sigma = 82.30, 75.0, 0.31
    tte = max((date_to_ord32(expiry) - date_to_ord32(A)) / 365.25, float(TTE_FLOOR))
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * tte) / (sigma * math.sqrt(tte))
    d2 = d1 - sigma * math.sqrt(tte)
    expected = 500.0 * (f * norm.cdf(d1) - k * norm.cdf(d2)) * df_settle
    assert out[0, 0] == pytest.approx(expected, rel=5e-5)  # D046 CDF band
    assert df_settle < 1.0
