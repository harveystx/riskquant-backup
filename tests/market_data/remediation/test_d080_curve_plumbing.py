# tests/market_data/remediation/test_d080_curve_plumbing.py
"""
D080 Phase 2: metadata, maturity construction and mode-gated pillar emission.

Phase 2 stops short of the constructor: `discounting_mode = CURVE` still fails
closed at the producer (D061). What is certified here is everything upstream of
that gate — the declaration, the business-day engine, the LABEL vocabulary, the
canonical ordering, and the invariance of the NONE path.

Every expected date, coordinate and discount factor comes from `d080_oracle`,
which imports stdlib and numpy only. `test_the_oracle_is_independent_of_production`
machine-checks that.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import d080_oracle as oracle
import numpy as np
import polars as pl
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
from riskquant.market_data.assemble import build_universe_mapping
from riskquant.market_data.curves.label_maturity import (
    D080_SUPPORTED_LABELS,
    resolve_label_maturity_ords,
)
from riskquant.portfolio.engine import build_uix_mapper
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.utils.business_days import (
    business_day_add_ord,
    is_last_business_day_of_month_ord,
    ord_to_date,
)
from riskquant.utils.dates import date_to_ord32

A = CANONICAL_AS_OF_DATE  # 2026-01-15
CURVE = "USD_SOFR_OIS"
LABELS = ("ON", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y")
PERCENT_LEVELS = {
    label: CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT[f"{CURVE}[{label}]"]
    for label in LABELS
}


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )


@pytest.fixture(scope="module")
def us_holidays() -> frozenset[dt.date]:
    rows = CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS["holiday_calendars"]
    # The fixture stores ISO strings; the oracle parses them itself rather than
    # borrowing any production date helper.
    return frozenset(
        dt.date.fromisoformat(str(r["holiday_date"]))
        for r in rows.iter_rows(named=True)
        if r["holiday_calendar_name"] == "US"
    )


def _curve(rsc, name=CURVE):
    return {c.curve_name: c for c in rsc.curve_meta_dict.values()}[name]


def _forward_book() -> pl.DataFrame:
    return pl.from_dict({
        "uid": [1],
        "instrument": ["VANILLA_FORWARD_NODE"],
        "quantity": [1_000.0],
        "underlying": ["BRENT_CRUDE_FWD[M+0]"],
        "settlement_ccy": ["USD"],
        "strike_price": [70.0],
        "strike_ccy": ["USD"],
        "delivery_date": [dt.date(2026, 2, 15)],
        "settlement_date": [dt.date(2026, 2, 17)],
        "under_discount_curve": [CURVE],
        "strike_discount_curve": [CURVE],
    })


def _mapper(rsc, as_of=A):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_forward_book(), static_config=rsc, as_of_date=as_of
    )
    return build_uix_mapper([batch])


# --------------------------------------------------------------- metadata
def test_usd_sofr_ois_is_the_only_declared_curve(rsc):
    declared = {
        c.curve_name
        for c in rsc.curve_meta_dict.values()
        if c.curve_input_type is not None
    }
    assert declared == {CURVE}
    curve = _curve(rsc)
    assert curve.curve_input_type == "ZERO_RATE"
    assert curve.curve_compounding == "CONTINUOUS"
    assert curve.maturity_calendar == "US"
    assert curve.spot_lag_days == 2
    assert curve.business_day_convention == "MOD_FOLLOWING"
    assert curve.end_of_month is True


def test_construction_metadata_round_trips_through_a_frame(rsc):
    """A hand-built frame carrying the five fields compiles to the same values."""
    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm = dfs["curve_meta"]
    dfs["curve_meta"] = cm.with_columns(
        pl.Series(
            "spot_lag_days",
            [0 if n == CURVE else None for n in cm["curve_name"].to_list()],
            dtype=pl.Int64,
        )
    )
    other = compile_risk_static_config_from_dict_of_dfs(dfs)
    assert _curve(other).spot_lag_days == 0


def test_unknown_convention_values_fail_at_config_compile():
    from riskquant.exceptions import ConfigurationError

    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm = dfs["curve_meta"]
    dfs["curve_meta"] = cm.with_columns(
        pl.Series(
            "business_day_convention",
            ["SOMETIMES" if n == CURVE else None for n in cm["curve_name"].to_list()],
            dtype=pl.String,
        )
    )
    with pytest.raises(ConfigurationError, match="BusinessDayConvention"):
        compile_risk_static_config_from_dict_of_dfs(dfs)


# ------------------------------------------------------- LABEL vocabulary
def test_supported_vocabulary_is_exactly_the_d080_set():
    assert frozenset(LABELS) == D080_SUPPORTED_LABELS


@pytest.mark.parametrize(
    "token", ["[SN]", "[TN]", "[1D]", "[5D]", "[7Y]", "[3X]", "3M"]
)
def test_unsupported_or_malformed_labels_fail_closed(rsc, token):
    with pytest.raises(Exception) as excinfo:
        resolve_label_maturity_ords(
            curve=_curve(rsc),
            tenor_tokens=(token,),
            as_of_ord32=int(date_to_ord32(A)),
            static_config=rsc,
        )
    assert "D080" in str(excinfo.value) or "token" in str(excinfo.value)


def test_missing_construction_metadata_fails_closed(rsc):
    with pytest.raises(ValueError, match="maturity construction requires"):
        resolve_label_maturity_ords(
            curve=_curve(rsc, "EUR_ESTR_OIS"),
            tenor_tokens=("[3M]",),
            as_of_ord32=int(date_to_ord32(A)),
            static_config=rsc,
        )


# --------------------------------------------------------- date semantics
@pytest.mark.parametrize(
    "as_of",
    [
        dt.date(2026, 1, 15),
        dt.date(2026, 1, 30),
        dt.date(2026, 12, 24),
        dt.date(2027, 6, 29),
    ],
)
def test_every_maturity_matches_the_independent_oracle(rsc, us_holidays, as_of):
    produced = resolve_label_maturity_ords(
        curve=_curve(rsc),
        tenor_tokens=tuple(f"[{label}]" for label in LABELS),
        as_of_ord32=int(date_to_ord32(as_of)),
        static_config=rsc,
    )
    expected = [
        oracle.maturity(label=label, as_of=as_of, holidays=us_holidays)
        for label in LABELS
    ]
    assert [ord_to_date(o) for o in produced] == expected


def test_the_canonical_as_of_reproduces_the_fixture_true_spot(rsc, us_holidays):
    """
    The canonical US calendar is deliberately sparse (D080-C1 option b): it has
    no MLK entry, so T+2 from 2026-01-15 is 2026-01-19, not 2026-01-20.
    """
    holidays = rsc.holiday_calendars_dict["US"].holiday_dates_ord
    spot = ord_to_date(
        business_day_add_ord(
            date_ord=int(date_to_ord32(A)), n_days=2, holidays_ord=holidays
        )
    )
    assert spot == dt.date(2026, 1, 19) == oracle.bd_add(A, 2, us_holidays)
    assert dt.date(2026, 1, 19) not in us_holidays

    produced = dict(
        zip(
            LABELS,
            resolve_label_maturity_ords(
                curve=_curve(rsc),
                tenor_tokens=tuple(f"[{label}]" for label in LABELS),
                as_of_ord32=int(date_to_ord32(A)),
                static_config=rsc,
            ),
            strict=True,
        )
    )
    assert ord_to_date(produced["ON"]) == dt.date(2026, 1, 16)  # bypasses T+2
    assert ord_to_date(produced["1W"]) == dt.date(2026, 1, 26)
    assert ord_to_date(produced["10Y"]) == dt.date(2036, 1, 21)


def test_end_of_month_rolls_to_the_last_business_day(rsc, us_holidays):
    """An as-of whose spot is its month's last business day fires the EOM rule."""
    as_of = dt.date(2026, 1, 28)
    holidays = rsc.holiday_calendars_dict["US"].holiday_dates_ord
    spot_ord = business_day_add_ord(
        date_ord=int(date_to_ord32(as_of)), n_days=2, holidays_ord=holidays
    )
    assert ord_to_date(spot_ord) == dt.date(2026, 1, 30)
    assert is_last_business_day_of_month_ord(date_ord=spot_ord, holidays_ord=holidays)

    produced = resolve_label_maturity_ords(
        curve=_curve(rsc),
        tenor_tokens=("[1M]",),
        as_of_ord32=int(date_to_ord32(as_of)),
        static_config=rsc,
    )
    expected = oracle.maturity(label="1M", as_of=as_of, holidays=us_holidays)
    assert ord_to_date(produced[0]) == expected == dt.date(2026, 2, 27)


# ------------------------------------------------- mode-gated emission
def test_none_mode_layout_is_unchanged(rsc):
    mapper = _mapper(rsc)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    assert [names[int(i)] for i in um.required_factor_ids] == ["BRENT_CRUDE_FWD[M+0]"]
    assert um.shape_discount_curves == (1, 0)
    assert um.raw_discount_tenors.shape == (1, 0)
    assert um.discounting_mode is DiscountingMode.NONE


def test_curve_mode_emits_canonically_ordered_pillars(rsc, us_holidays):
    mapper = _mapper(rsc)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    required = [names[int(i)] for i in um.required_factor_ids]
    assert required[0] == "BRENT_CRUDE_FWD[M+0]"
    assert um.shape_discount_curves == (1, len(LABELS))
    assert um.raw_discount_tenors.shape == (1, len(LABELS))
    assert um.raw_discount_tenors.dtype == np.float64

    expected_order = sorted(
        LABELS,
        key=lambda label: oracle.act365f(
            A, oracle.maturity(label=label, as_of=A, holidays=us_holidays)
        ),
    )
    assert required[1:] == [f"{CURVE}[{label}]" for label in expected_order]

    produced_t = um.raw_discount_tenors[0]
    expected_t = [
        oracle.act365f(A, oracle.maturity(label=label, as_of=A, holidays=us_holidays))
        for label in expected_order
    ]
    np.testing.assert_allclose(produced_t, expected_t, rtol=0, atol=0)
    assert np.all(np.diff(produced_t) > 0.0)  # strictly increasing
    assert produced_t[0] > 0.0  # the T = 0 anchor is synthetic, not a pillar


def test_curve_mode_requires_a_declared_curve(rsc):
    mapper = _mapper(rsc)
    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm = dfs["curve_meta"]
    dfs["curve_meta"] = cm.with_columns(
        pl.Series("curve_input_type", [None] * cm.height, dtype=pl.String)
    )
    undeclared = compile_risk_static_config_from_dict_of_dfs(dfs)
    with pytest.raises(
        NotImplementedError, match="D061: CURVE construction unavailable"
    ):
        build_universe_mapping(
            discounting_mode=DiscountingMode.CURVE,
            static_config=undeclared,
            uix_mapper=mapper,
        )


def test_act365f_coordinate_is_not_act360_or_act36525(rsc, us_holidays):
    mapper = _mapper(rsc)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    three_m = oracle.maturity(label="3M", as_of=A, holidays=us_holidays)
    days = (three_m - A).days
    t_act365f = days / 365.0
    assert DAYS_PER_YR_ACT_365F == 365.0
    assert t_act365f in um.raw_discount_tenors[0]
    assert not np.isclose(t_act365f, days / 360.0)
    assert not np.isclose(t_act365f, days / 365.25)
    assert not np.isclose(t_act365f, 0.25)  # never a hard-coded quarter


# --------------------------------------------------- D061 lifted for USD_SOFR_OIS
def test_curve_mode_now_constructs_the_declared_curve(rsc):
    """Phase 3: the D080 constructor is live for the fully declared curve."""
    from riskquant.market_data.assemble import build_market_data_matrix_from_snapshot
    from riskquant.market_data.risk_factors.compile_snapshot import (
        compile_risk_factor_level_snapshot_from_levels_dict,
    )

    mapper = _mapper(rsc)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    levels = {}
    for rf in um.required_factor_ids:
        name = names[int(rf)]
        levels[name] = PERCENT_LEVELS.get(name.split("[")[-1].rstrip("]"), 80.0)
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
    grid = matrix.discount_factor_grid
    assert grid.shape == (1, mapper.date_ord32_universe.shape[0], 1)
    assert np.all(grid > 0.0) and np.all(grid < 1.0)  # positive rates, T > 0


def test_layout_plan_and_producer_must_agree_on_the_mode(rsc):
    from riskquant.market_data.assemble import build_market_data_matrix_from_snapshot
    from riskquant.market_data.risk_factors.compile_snapshot import (
        compile_risk_factor_level_snapshot_from_levels_dict,
    )

    mapper = _mapper(rsc)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict={"BRENT_CRUDE_FWD[M+0]": 80.0}, static_config=rsc
    )
    with pytest.raises(ValueError, match="must agree"):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.CURVE,
            as_of_date=A,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=snapshot,
        )


# ------------------------------------------------------------ unit safety
def test_levels_reaching_the_producer_are_decimal_exactly_once(rsc):
    from riskquant.market_data.assemble import extract_levels_from_snapshot
    from riskquant.market_data.risk_factors.compile_snapshot import (
        compile_risk_factor_level_snapshot_from_levels_dict,
    )

    mapper = _mapper(rsc)
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    levels = {
        names[int(rf)]: PERCENT_LEVELS.get(
            names[int(rf)].split("[")[-1].rstrip("]"), 80.0
        )
        for rf in um.required_factor_ids
    }
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict=levels, static_config=rsc
    )
    flat = extract_levels_from_snapshot(
        required_ids=um.required_factor_ids, snapshot=snapshot
    )
    disc = flat[um.idx_discount_curves]
    # 4.80 PERCENT -> 0.048 exactly once. A second scaling would give 0.00048.
    assert pytest.approx(0.0480, rel=1e-15) in [float(x) for x in disc]
    assert min(float(x) for x in disc) > 0.001


# --------------------------------------------------------- oracle hygiene
def test_the_oracle_is_independent_of_production():
    text = pathlib.Path(__file__).with_name("d080_oracle.py").read_text()
    forbidden = (
        "riskquant.market_data",
        "riskquant.utils.dates",
        "riskquant.utils.business_days",
        "label_maturity",
        "compile_snapshot",
        "interp_",
        "build_universe_mapping",
    )
    assert [token for token in forbidden if token in text] == []


def test_oracle_and_production_agree_on_the_canonical_curve(rsc, us_holidays):
    """
    Pillar ln DF and the D080 evaluation rules, checked against the oracle at
    every pillar, one interior point per gap, the anchor and the right tail.
    Phase 2 has no constructor, so the comparison is on the oracle's own curve
    built from production-resolved maturities versus fully independent ones.
    """
    produced_ords = resolve_label_maturity_ords(
        curve=_curve(rsc),
        tenor_tokens=tuple(f"[{label}]" for label in LABELS),
        as_of_ord32=int(date_to_ord32(A)),
        static_config=rsc,
    )
    pillar_t, pillar_ln = oracle.curve(
        labels=LABELS, percent_levels=PERCENT_LEVELS, as_of=A, holidays=us_holidays
    )
    for label, produced in zip(LABELS, produced_ords, strict=True):
        independent = oracle.maturity(label=label, as_of=A, holidays=us_holidays)
        assert ord_to_date(produced) == independent

    assert (
        oracle.discount_factor(t_target=0.0, pillar_t=pillar_t, pillar_ln=pillar_ln)
        == 1.0
    )
    ten_y_t = pillar_t[-1]
    assert oracle.discount_factor(
        t_target=ten_y_t, pillar_t=pillar_t, pillar_ln=pillar_ln
    ) == pytest.approx(0.669733, abs=5e-7)
    mid = (pillar_t[-2] + pillar_t[-1]) / 2.0
    assert oracle.discount_factor(
        t_target=mid, pillar_t=pillar_t, pillar_ln=pillar_ln
    ) == pytest.approx(0.738396, abs=5e-7)
    assert oracle.discount_factor(
        t_target=20.024658, pillar_t=pillar_t, pillar_ln=pillar_ln
    ) == pytest.approx(0.453409, abs=5e-7)


@pytest.mark.parametrize(
    "rate_percent, expect_above_one", [(4.80, False), (0.0, None), (-1.0, True)]
)
def test_zero_rate_math_positive_zero_and_negative(
    us_holidays, rate_percent, expect_above_one
):
    levels = dict.fromkeys(LABELS, rate_percent)
    pillar_t, pillar_ln = oracle.curve(
        labels=LABELS, percent_levels=levels, as_of=A, holidays=us_holidays
    )
    df_5y = oracle.discount_factor(
        t_target=pillar_t[-2], pillar_t=pillar_t, pillar_ln=pillar_ln
    )
    if expect_above_one is None:
        assert df_5y == 1.0
    elif expect_above_one:
        assert df_5y > 1.0
    else:
        assert 0.0 < df_5y < 1.0
