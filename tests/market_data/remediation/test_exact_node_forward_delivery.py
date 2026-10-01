# tests/market_data/remediation/test_exact_node_forward_delivery.py
"""
D101 / D065 / D062: NODE-forward delivery semantics on the REAL production path.

    raw polars frame
        -> compile_raw_df_into_ingest_batch      (dispatcher + identity gate)
        -> build_uix_mapper                      (EFP quads, EDF triples)
        -> compile_portfolio_from_ingest_batch   (VanillaForwardBlock)
        -> build_universe_mapping (CURVE)        (D080 USD_SOFR_OIS discounting)
        -> build_market_data_matrix_from_snapshot
        -> assemble_valuation_state              (the real fold)
        -> run_valuation                         (the real kernel)

Nothing below hand-builds an `IngestBatch`, a `MarketDataMatrix` or a
`ValuationState`, except the one post-ingest bypass that probes the D062
packing guard (`test_post_ingest_missing_settlement_stops_at_the_d062_guard`).

Test-owned configuration
------------------------
The canonical config has no exact-NODE forward underlying: every PRICE curve
is an OFFSET curve and every LABEL curve is a RATE curve. The tests therefore
extend the canonical tables with ONE LABEL + PRICE curve, `TEST_EXACT_FWD`,
with two points. Static-config validation admits it unchanged; nothing else
is added.

Independent oracle
------------------
`_oracle_pv` reads the raw trade row, the configured level of the EXACT node
by its name, the FX level by its configured orientation (USDEUR = EUR per USD,
D078.4) and discount factors from `d080_oracle` (stdlib + numpy only). It
derives the asset-leg date from the raw row: delivery when supplied, settlement
otherwise. It calls no production compiler, mapper, fold or pricer.

Tolerance
---------
`d080_oracle` reproduces production DFs to 1e-13 relative (D080 TOL). The book
keeps |F*DF_a - K*DF_k| >= 1/5 of F*DF_a, so leg cancellation amplifies DF error
at most 5x: RTOL = 1e-12.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import d080_oracle as oracle
import numpy as np
import polars as pl
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.constants import NO_DATE_ORD32
from riskquant.exceptions import PrettyKeyError
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.ingest_fixtures.market_data.canonical_market_data_snapshot import (
    CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT,
)
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

A = CANONICAL_AS_OF_DATE  # 2026-01-15
DISCOUNT = "USD_SOFR_OIS"  # the only constructible curve (D080.1)
SOFR_LABELS = ("ON", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y")
SOFR_PERCENT = {
    lab: CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT[f"{DISCOUNT}[{lab}]"]
    for lab in SOFR_LABELS
}
RTOL = 1e-12

# ---- test-owned exact-NODE curve -------------------------------------------
EXACT_CURVE = "TEST_EXACT_FWD"
EXACT_CURVE_ID = 250_013_000  # next free CURVES-range id after the canonical 13
RF_1M = 14_000  # next free curve-point block; consecutive on purpose (mutation)
RF_3M = 14_001
NODE_A = f"{EXACT_CURVE}[1M]"
NODE_B = f"{EXACT_CURVE}[3M]"

# Deliberately inconsistent neighbouring exact nodes: any delivery-driven
# selection or interpolation produces a value that is not LEVEL[NODE_A].
LEVEL = {NODE_A: 100.0, NODE_B: 150.0, "USDEUR": 0.9137}

D1 = dt.date(2026, 3, 16)
D2 = dt.date(2026, 5, 15)
S = dt.date(2026, 6, 17)


# =========================================================================
# CONFIGURATION
# =========================================================================


def _extended_dfs() -> dict[str, pl.DataFrame]:
    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    cm = dfs["curve_meta"]
    row: dict[str, object] = {c: None for c in cm.columns}
    row.update(
        curve_id=EXACT_CURVE_ID,
        curve_name=EXACT_CURVE,
        quote_ccy="USD",
        asset_class="COMMODITY",
        asset_subclass="CM_OIL",
        quote_type="PRICE",
        unit_scale="DECIMAL",
        shift_type="LOG_RETURN",
        x_axis_kind="LABEL",
        x_interp_method="LINEAR",
        x_extrap_method="FLAT",
        discount_basis="NOT_A_DISCOUNT_CURVE",
        is_default_discount_for_ccy=False,
    )
    dfs["curve_meta"] = pl.concat([cm, pl.DataFrame([row], schema=cm.schema)])
    cp = dfs["curve_point_meta"]
    points = pl.DataFrame(
        {
            "rf_id": [RF_1M, RF_3M],
            "curve_id": [EXACT_CURVE_ID, EXACT_CURVE_ID],
            "tenor_token": ["[1M]", "[3M]"],
        },
        schema=cp.schema,
    )
    dfs["curve_point_meta"] = pl.concat([cp, points])
    return dfs


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(_extended_dfs())


@pytest.fixture(scope="module")
def oracle_curve():
    rows = CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS["holiday_calendars"]
    holidays = frozenset(
        dt.date.fromisoformat(str(r["holiday_date"]))
        for r in rows.iter_rows(named=True)
        if r["holiday_calendar_name"] == "US"
    )
    return oracle.curve(
        labels=SOFR_LABELS, percent_levels=SOFR_PERCENT, as_of=A, holidays=holidays
    )


# =========================================================================
# BOOK
# =========================================================================


def _fwd(uid: int, underlying: str, **kw: object) -> dict[str, object]:
    row: dict[str, object] = dict(
        uid=uid,
        instrument="VANILLA_FORWARD_NODE",
        quantity=1_000.0,
        underlying=underlying,
        settlement_ccy="USD",
        strike_price=80.0,
        strike_ccy="USD",
        delivery_date=None,
        settlement_date=S,
        under_discount_curve=DISCOUNT,
        strike_discount_curve=DISCOUNT,
    )
    row.update(kw)
    return row


def _book() -> list[dict[str, object]]:
    """Hostile uid order; one block mixing delivery-present and -absent rows."""
    return [
        _fwd(9003, NODE_A, delivery_date=D1),
        _fwd(41, NODE_A, delivery_date=D2),
        _fwd(7, NODE_A),  # delivery absent -> settlement
        _fwd(120, NODE_B, delivery_date=D1, quantity=-250.0, strike_price=120.0),
        _fwd(5, NODE_A, settlement_ccy="EUR", quantity=-400.0),  # direct FX
    ]


def _ingest(rsc, rows, as_of=A):
    return compile_raw_df_into_ingest_batch(
        raw_df=pl.DataFrame(rows), static_config=rsc, as_of_date=as_of
    )


def _levels(rsc, um) -> dict[str, float]:
    names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    out: dict[str, float] = {}
    for rf in um.required_factor_ids:
        name = names[int(rf)]
        if name.startswith(f"{DISCOUNT}["):
            out[name] = SOFR_PERCENT[name[len(DISCOUNT) + 1 : -1]]
        else:
            out[name] = LEVEL[name]  # KeyError = unexpected factor: fail fast
    return out


def _run(rsc, rows):
    batch = _ingest(rsc, rows)
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.CURVE, static_config=rsc, uix_mapper=mapper
    )
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict=_levels(rsc, um), static_config=rsc
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.CURVE,
        as_of_date=A,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((portfolio.total_trades, state.batch_size), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return batch, mapper, portfolio, matrix, state, out


# =========================================================================
# INDEPENDENT ORACLE
# =========================================================================


def _df(oracle_curve, d: dt.date) -> float:
    t, ln = oracle_curve
    return oracle.discount_factor(
        t_target=oracle.act365f(A, d), pillar_t=t, pillar_ln=ln
    )


def _fx_usd_to(ccy: str) -> float:
    return 1.0 if ccy == "USD" else {"EUR": LEVEL["USDEUR"]}[ccy]


def _asset_date(row) -> dt.date:
    return (
        row["delivery_date"]
        if row["delivery_date"] is not None
        else row["settlement_date"]
    )


def _oracle_efp(row, oracle_curve) -> float:
    """F_node(exact name) * DF_USD(T_asset) * FX(USD -> settle)."""
    return (
        LEVEL[row["underlying"]]
        * _df(oracle_curve, _asset_date(row))
        * _fx_usd_to(row["settlement_ccy"])
    )


def _oracle_pv(row, oracle_curve) -> float:
    strike_leg = (
        row["strike_price"]
        * _df(oracle_curve, row["settlement_date"])
        * _fx_usd_to(row["settlement_ccy"])
    )
    return row["quantity"] * (_oracle_efp(row, oracle_curve) - strike_leg)


def _row_of(portfolio, uid: int) -> int:
    hits = np.flatnonzero(portfolio.trade_uids == uid)
    assert hits.size == 1, f"uid {uid} must own exactly one row"
    return int(hits[0])


# =========================================================================
# EXACT NODE: POSITIVE PRODUCTION PATH
# =========================================================================


def test_fixture_discriminates_every_candidate_asset_date(oracle_curve):
    """DF(D1), DF(D2) and DF(S) differ, so a wrong date choice is observable."""
    dfs = [_df(oracle_curve, d) for d in (D1, D2, S)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(dfs[i] - dfs[j]) / dfs[i] > 1e-4


def test_exact_node_ingests_with_and_without_delivery(rsc):
    batch = _ingest(rsc, _book())
    by_uid = {int(u): i for i, u in enumerate(batch.uids.tolist())}
    rf_a = int(rsc.rel_name_to_rf_id[NODE_A])
    for uid in (9003, 41, 7, 5):
        assert int(batch.underlying_ids[by_uid[uid]]) == rf_a == RF_1M
    assert int(batch.underlying_ids[by_uid[120]]) == RF_3M
    assert int(batch.delivery_ord[by_uid[9003]]) == int(date_to_ord32(D1))
    assert int(batch.delivery_ord[by_uid[41]]) == int(date_to_ord32(D2))
    assert int(batch.delivery_ord[by_uid[7]]) == int(NO_DATE_ORD32)
    assert int(batch.delivery_ord[by_uid[5]]) == int(NO_DATE_ORD32)


def test_delivery_never_selects_the_exact_node(rsc):
    """Every NODE_A row reads one market row, whatever its delivery."""
    _, mapper, portfolio, _, _, _ = _run(rsc, _book())
    block = portfolio.vanilla_forward
    assert block is not None
    quads = mapper.efp_effective_quads
    row_a = int(np.searchsorted(mapper.market_id_universe, RF_1M))
    row_b = int(np.searchsorted(mapper.market_id_universe, RF_3M))
    assert int(mapper.market_id_universe[row_a]) == RF_1M
    assert int(mapper.market_id_universe[row_b]) == RF_3M
    for uid in (9003, 41, 7, 5):
        r = _row_of(portfolio, uid)
        assert int(quads[block.effective_forward_price_uix[r], 0]) == row_a
    r = _row_of(portfolio, 120)
    assert int(quads[block.effective_forward_price_uix[r], 0]) == row_b


def test_asset_leg_date_is_delivery_or_settlement(rsc):
    _, mapper, portfolio, _, _, _ = _run(rsc, _book())
    block = portfolio.vanilla_forward
    assert block is not None
    quads = mapper.efp_effective_quads
    dates = mapper.date_ord32_universe
    expected = {9003: D1, 41: D2, 7: S, 5: S, 120: D1}
    for uid, d in expected.items():
        date_uix = int(
            quads[block.effective_forward_price_uix[_row_of(portfolio, uid)], 2]
        )
        assert int(dates[date_uix]) == int(date_to_ord32(d)), uid


def test_neighbouring_exact_node_is_in_the_run_and_differs(rsc):
    """The anti-selection check is not vacuous: NODE_B is a live market row."""
    _, mapper, _, matrix, _, _ = _run(rsc, _book())
    row_a = int(np.searchsorted(mapper.market_id_universe, RF_1M))
    row_b = int(np.searchsorted(mapper.market_id_universe, RF_3M))
    assert float(matrix.forward_nodes[row_a, 0]) == LEVEL[NODE_A]
    assert float(matrix.forward_nodes[row_b, 0]) == LEVEL[NODE_B]


def test_effective_forward_factor_is_exact_node_times_asset_df_times_fx(
    rsc, oracle_curve
):
    _, _, portfolio, _, state, _ = _run(rsc, _book())
    block = portfolio.vanilla_forward
    assert block is not None
    for row in _book():
        r = _row_of(portfolio, row["uid"])
        produced = float(
            state.effective_forward_prices[block.effective_forward_price_uix[r], 0]
        )
        np.testing.assert_allclose(
            produced,
            _oracle_efp(row, oracle_curve),
            rtol=RTOL,
            atol=0.0,
            err_msg=f"uid {row['uid']}",
        )


def test_production_pv_matches_the_independent_oracle(rsc, oracle_curve):
    _, _, portfolio, _, _, out = _run(rsc, _book())
    for row in _book():
        np.testing.assert_allclose(
            out[_row_of(portfolio, row["uid"]), 0],
            _oracle_pv(row, oracle_curve),
            rtol=RTOL,
            atol=0.0,
            err_msg=f"uid {row['uid']}",
        )


def test_omitted_delivery_values_exactly_as_delivery_equal_to_settlement(rsc):
    """Absent delivery and delivery == settlement are the same economic trade."""
    absent = [_fwd(1, NODE_A)]
    explicit = [_fwd(1, NODE_A, delivery_date=S)]
    _, _, _, _, _, out_absent = _run(rsc, absent)
    _, _, _, _, _, out_explicit = _run(rsc, explicit)
    np.testing.assert_array_equal(out_absent, out_explicit)


def test_production_path_is_bitwise_deterministic(rsc):
    first = _run(rsc, _book())
    second = _run(rsc, _book())
    assert first[1].fingerprint == second[1].fingerprint
    b1, b2 = first[2].vanilla_forward, second[2].vanilla_forward
    assert b1 is not None and b2 is not None
    for field in (
        "trade_uids",
        "qty",
        "strike",
        "effective_forward_price_uix",
        "effective_discount_factor_uix",
    ):
        np.testing.assert_array_equal(getattr(b1, field), getattr(b2, field))
    np.testing.assert_array_equal(
        first[4].effective_forward_prices, second[4].effective_forward_prices
    )
    np.testing.assert_array_equal(first[5], second[5])


# =========================================================================
# RELATIVE NODE / UNSUPPORTED ROLL / CURVE: FAIL CLOSED AT INGEST
# =========================================================================


def test_relative_node_without_delivery_fails_at_ingest(rsc):
    with pytest.raises(
        PrettyKeyError, match=r"requires delivery_date for relative NODE"
    ):
        _ingest(rsc, [_fwd(1, "BRENT_CRUDE_FWD[M+0]")])


@pytest.mark.parametrize(
    ("underlying", "ccy", "curve", "delivery"),
    [
        ("CO2_EUA_ANNUAL_FWD[Y+0]", "EUR", "EUR_ESTR_OIS", None),
        ("CO2_EUA_ANNUAL_FWD[Y+0]", "EUR", "EUR_ESTR_OIS", dt.date(2026, 12, 15)),
        ("POWER_CH_BASELOAD_HY[HY+0]", "CHF", "CHF_SARON_OIS", None),
        ("POWER_CH_BASELOAD_HY[HY+0]", "CHF", "CHF_SARON_OIS", dt.date(2026, 9, 15)),
    ],
)
def test_unsupported_relative_roll_fails_closed(rsc, underlying, ccy, curve, delivery):
    """ROLL_ON_MMDD / HALF_YEAR_BY_LABEL are relative: never exact (D065)."""
    row = _fwd(
        1,
        underlying,
        settlement_ccy=ccy,
        strike_ccy=ccy,
        under_discount_curve=curve,
        strike_discount_curve=curve,
        delivery_date=delivery,
    )
    with pytest.raises(PrettyKeyError, match=r"only defined for CALENDAR_PERIOD"):
        _ingest(rsc, [row])


def _curve_fwd(**kw: object) -> dict[str, object]:
    row = _fwd(1, "BRENT_CRUDE_FWD", instrument="VANILLA_FORWARD_CURVE")
    row.update(kw)
    return row


def test_curve_forward_without_delivery_fails_at_ingest(rsc):
    with pytest.raises(ValueError, match=r"Forward missing 'delivery_date'"):
        _ingest(rsc, [_curve_fwd()])
    no_column = {k: v for k, v in _curve_fwd().items() if k != "delivery_date"}
    with pytest.raises(ValueError, match=r"Missing required columns"):
        _ingest(rsc, [no_column])


def test_curve_forward_with_delivery_passes_ingest_and_stays_unregistered(rsc):
    """Control: the CURVE rejection above is the delivery rule, not D050."""
    batch = _ingest(rsc, [_curve_fwd(delivery_date=D1)])
    mapper = build_uix_mapper([batch])
    with pytest.raises(RuntimeError, match=r"No compiler registered"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=mapper)


def test_exact_node_without_settlement_fails_at_ingest(rsc):
    with pytest.raises(ValueError, match=r"Forward missing 'settlement_date'"):
        _ingest(rsc, [_fwd(1, NODE_A, settlement_date=None)])


def test_unknown_exact_node_fails_at_ingest(rsc):
    with pytest.raises(PrettyKeyError, match=r"NEITHER"):
        _ingest(rsc, [_fwd(1, f"{EXACT_CURVE}[2M]")])


def test_post_ingest_missing_settlement_stops_at_the_d062_guard(rsc):
    """
    Bypass ingest (which rejects a missing settlement) on an exact-NODE row
    without delivery. Its asset-leg date IS settlement (D101), and the asset
    leg is resolved first, so the earliest authoritative stop is the D062
    packing guard on `effective_forward_prices`, before any gather or kernel.
    """
    good = _ingest(rsc, [_fwd(1, NODE_A), _fwd(2, NODE_A, settlement_date=D2)])
    settle = good.settle_ord.copy()
    settle[0] = NO_DATE_ORD32
    bypass = dataclasses.replace(good, settle_ord=np.ascontiguousarray(settle))
    mapper = build_uix_mapper([bypass])
    with pytest.raises(
        ValueError,
        match=r"effective_forward_prices: packed column 2 holds a coordinate outside",
    ):
        compile_portfolio_from_ingest_batch(ingest_batch=bypass, uix_mapper=mapper)


# =========================================================================
# FALLBACK INVARIANT: absent delivery reaches compilation only for EXACT rows
# =========================================================================


def test_absent_delivery_is_admitted_only_for_label_price_points(rsc):
    """
    Exhaustive over every PRICE curve point of the run config: a NODE forward
    without delivery passes ingest iff the parent x-axis kind is LABEL. This is
    the precondition of the settlement fallback in build_uix_mapper and the
    forward compiler.
    """
    dfs = _extended_dfs()
    default_curve = {
        r["quote_ccy"]: r["curve_name"]
        for r in dfs["curve_meta"].iter_rows(named=True)
        if r["is_default_discount_for_ccy"]
    }
    admitted: set[str] = set()
    checked = 0
    for name, rf in rsc.rel_name_to_rf_id.items():
        parent = int(rsc.rf_id_to_parent_id[int(rf)])
        curve = rsc.curve_meta_dict.get(parent)
        if (
            curve is None
            or curve.quote_type != "PRICE"
            or not name.startswith(curve.curve_name)
        ):
            continue
        ccy = rsc.ccy_id_to_name[int(rsc.container_ccy_id[rsc.container_slot(parent)])]
        row = _fwd(
            1,
            name,
            settlement_ccy=ccy,
            strike_ccy=ccy,
            under_discount_curve=default_curve[ccy],
            strike_discount_curve=default_curve[ccy],
        )
        checked += 1
        try:
            _ingest(rsc, [row])
        except (PrettyKeyError, ValueError):
            continue
        admitted.add(name)
    assert checked >= 20  # every canonical PRICE point plus the two LABEL points
    assert admitted == {NODE_A, NODE_B}
