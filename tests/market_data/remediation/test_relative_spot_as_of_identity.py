# tests/market_data/remediation/test_relative_spot_as_of_identity.py
"""
AUD-04: relative SPOT as-of and contract identity.

Contract (D063, D067, D064, D074):

* A relative curve-point name (`BRENT_CRUDE_FWD[M+0]`) resolves to a PILLAR
  rf_id through the static `rel_name_to_rf_id` table. The pillar is the same at
  every as-of.
* The dated contract a pillar denotes is a function of (pillar, run as-of). The
  run as-of is fixed once on the `UixMapper` (D067), enters its fingerprint
  (D064) and binds the compiled portfolio and valuation state (D074).
* Missing run as-of fails before any market-data state exists. The
  NO_DATE_ORD32 sentinel is never a run as-of.

The expected dated contract comes from an independent calendar oracle built
from curve metadata; identity is asserted on rf_ids, never on market levels.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.rf_resolution import (
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.constants import NO_DATE_ORD32
from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.ingest_fixtures.market_data.canonical_market_data_snapshot import (
    CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT,
)
from riskquant.ingest_fixtures.portfolio.exhaustive import (
    CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA,
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
MONTH_END = date(2026, 1, 31)
NEXT_MONTH = date(2026, 2, 1)  # CALENDAR_PERIOD roll boundary
SENTINEL_DATE = (
    date(1970, 1, 1) + pd.Timedelta(days=int(NO_DATE_ORD32)).to_pytimedelta()
)
DATED_PILLARS = ("BRENT_CRUDE_FWD[M+0]", "WTI_CRUDE_FWD[M+0]", "HENRY_HUB_GAS_FWD[M+1]")


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS
    )


# --------------------------------------------------------------- oracle
def _oracle_contract(name: str, as_of: date) -> str:
    """Independent CALENDAR_PERIOD / LABEL identity from curve metadata."""
    meta = {
        r["curve_name"]: r
        for r in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS["curve_meta"].iter_rows(
            named=True
        )
    }
    curve, token = name.split("[", 1)
    token = token.rstrip("]")
    row = meta[curve]
    if row["x_axis_kind"] == "LABEL":
        return f"{curve}_{token}"
    assert (row["x_axis_kind"], row["roll_method"]) == ("OFFSET_M", "CALENDAR_PERIOD")
    months = as_of.year * 12 + (as_of.month - 1) + int(row["anchor_offset"])
    months += int(token.split("+")[1])
    return f"{curve}_{months // 12:04d}{months % 12 + 1:02d}"


def _production_contract(rsc, rf_id: int, as_of: date) -> str:
    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=rsc, as_of_date=as_of
    )
    return {int(v): k for k, v in mapping.abs_to_rf_id.items()}[int(rf_id)]


# ------------------------------------------------------------ production path
def _book(*rows):
    return pd.DataFrame([
        dict(
            uid=i + 1, instrument="SPOT", quantity=q, underlying=u, settlement_ccy="USD"
        )
        for i, (u, q) in enumerate(rows)
    ])


def _run(rsc, book, *, as_of, levels, ingest_as_of=None, portfolio=None):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=book, static_config=rsc, copy=True, as_of_date=ingest_as_of
    )
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(as_of)))
    portfolio = portfolio or compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=as_of, levels_dict=levels, static_config=rsc
    )
    matrix = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=as_of,
        uix_mapper=mapper,
        universe_map=um,
        snapshot=snapshot,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((portfolio.total_trades, 1))
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    rows = [
        int(mapper.market_id_universe[mapper.spot_effective_pairs[int(e), 0]])
        for e in portfolio.spot.effective_spot_price_uix
    ]
    return batch, mapper, portfolio, matrix, out[:, 0], rows


# --------------------------------------------------------------- Test 1
@pytest.mark.parametrize("name", DATED_PILLARS)
@pytest.mark.parametrize("ingest_as_of", [None, A])
def test_valid_relative_spot_resolves_to_its_pillar_and_dated_contract(
    rsc, name, ingest_as_of
):
    batch, mapper, portfolio, _, pv, rows = _run(
        rsc,
        _book((name, 100.0)),
        as_of=A,
        levels={name: 80.0},
        ingest_as_of=ingest_as_of,
    )
    pillar = int(rsc.rel_name_to_rf_id[name])
    assert int(batch.underlying_ids[0]) == pillar
    assert rows == [pillar]  # matrix row is the pillar's row, by rf identity
    assert mapper.as_of_ord32 == int(date_to_ord32(A))
    assert portfolio.uix_fingerprint == mapper.fingerprint
    assert _production_contract(rsc, pillar, A) == _oracle_contract(name, A)
    assert pv[0] == pytest.approx(8000.0, rel=1e-15)


# --------------------------------------------------------------- Test 2
@pytest.mark.parametrize("name", DATED_PILLARS)
def test_missing_run_as_of_fails_before_market_data_state(rsc, name):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_book((name, 1.0)), static_config=rsc, copy=True
    )
    assert int(batch.as_of_ord32) == int(NO_DATE_ORD32)
    mapper = build_uix_mapper([batch])
    assert mapper.as_of_ord32 is None
    um = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=A, levels_dict={name: 80.0}, static_config=rsc
    )
    with pytest.raises(ValueError, match=r"UixMapper.as_of_ord32 None \(D067\)"):
        build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=A,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=snapshot,
        )


# --------------------------------------------------------------- Test 3
@pytest.mark.parametrize("name", DATED_PILLARS)
def test_different_as_of_keeps_the_pillar_and_moves_the_contract(rsc, name):
    pillar = int(rsc.rel_name_to_rf_id[name])
    _, mapper_a, portfolio_a, _, _, rows_a = _run(
        rsc, _book((name, 1.0)), as_of=MONTH_END, levels={name: 80.0}
    )
    _, mapper_b, _, _, _, rows_b = _run(
        rsc, _book((name, 1.0)), as_of=NEXT_MONTH, levels={name: 80.0}
    )
    assert rows_a == rows_b == [pillar]
    assert _production_contract(rsc, pillar, MONTH_END) == _oracle_contract(
        name, MONTH_END
    )
    assert _production_contract(rsc, pillar, NEXT_MONTH) == _oracle_contract(
        name, NEXT_MONTH
    )
    assert _oracle_contract(name, MONTH_END) != _oracle_contract(name, NEXT_MONTH)
    assert mapper_a.fingerprint != mapper_b.fingerprint  # as-of is identity

    with pytest.raises(ValueError, match="different UixMappers"):
        _run(
            rsc,
            _book((name, 1.0)),
            as_of=NEXT_MONTH,
            levels={name: 80.0},
            portfolio=portfolio_a,
        )
    with pytest.raises(ValueError, match="D067"):
        build_uix_mapper(
            [
                compile_raw_df_into_ingest_batch(
                    raw_df=_book((name, 1.0)),
                    static_config=rsc,
                    copy=True,
                    as_of_date=MONTH_END,
                )
            ],
            as_of_ord32=int(date_to_ord32(NEXT_MONTH)),
        )


# --------------------------------------------------------------- Test 4


# --------------------------------------------------------------- Test 5
def test_the_no_date_sentinel_is_never_a_run_as_of(rsc):
    batch = compile_raw_df_into_ingest_batch(
        raw_df=_book(("BRENT_CRUDE_FWD[M+0]", 1.0)), static_config=rsc, copy=True
    )
    with pytest.raises(ValueError, match="NO_DATE_ORD32 sentinel"):
        build_uix_mapper([batch], as_of_ord32=int(NO_DATE_ORD32))
    assert int(date_to_ord32(SENTINEL_DATE)) == int(NO_DATE_ORD32)
    with pytest.raises(ValueError, match="NO_DATE_ORD32 sentinel"):
        compile_raw_df_into_ingest_batch(
            raw_df=_book(("BRENT_CRUDE_FWD[M+0]", 1.0)),
            static_config=rsc,
            copy=True,
            as_of_date=SENTINEL_DATE,
        )


# --------------------------------------------------------------- Test 6
def test_identity_is_not_value_derived(rsc):
    m0, m1 = "BRENT_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+1]"
    book = _book((m0, 10.0), (m1, 1.0))
    equal = {m0: 80.0, m1: 80.0}
    batch, _, _, _, pv_equal, rows = _run(rsc, book, as_of=A, levels=equal)
    assert rows == [int(rsc.rel_name_to_rf_id[m0]), int(rsc.rel_name_to_rf_id[m1])]

    _, _, _, _, pv_m1_moved, _ = _run(rsc, book, as_of=A, levels={m0: 80.0, m1: 81.0})
    assert pv_m1_moved[0] == pv_equal[0] and pv_m1_moved[1] != pv_equal[1]

    # neighbouring-contract mutation: same book, M+0 row forced to the M+1 pillar
    wrong = batch.underlying_ids.copy()
    wrong[0] = rsc.rel_name_to_rf_id[m1]
    bad = dataclasses.replace(batch, underlying_ids=np.ascontiguousarray(wrong))
    bad_mapper = build_uix_mapper([bad], as_of_ord32=int(date_to_ord32(A)))
    bad_portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=bad, uix_mapper=bad_mapper
    )
    bad_rows = [
        int(bad_mapper.market_id_universe[bad_mapper.spot_effective_pairs[int(e), 0]])
        for e in bad_portfolio.spot.effective_spot_price_uix
    ]
    assert bad_rows[0] != rows[0]  # detected by rf identity although levels are equal


# --------------------------------------------------- canonical fixture + determinism
def test_canonical_relative_spot_rows_resolve_at_the_canonical_as_of(rsc):
    frame = pd.DataFrame(CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA)
    spot = frame[(frame.instrument == "SPOT") & frame.underlying.str.contains(r"\[")]
    cols = ["uid", "instrument", "quantity", "underlying", "settlement_ccy"]
    rate_rows = spot[spot.underlying == "USD_LIBOR_3M[10Y]"]
    assert rate_rows.uid.tolist() == [11]
    with pytest.raises(Exception, match="D063"):
        compile_raw_df_into_ingest_batch(
            raw_df=rate_rows[cols], static_config=rsc, copy=True, as_of_date=A
        )

    priced = spot[spot.underlying != "USD_LIBOR_3M[10Y]"][cols].reset_index(drop=True)
    assert priced.uid.tolist() == [7, 8, 9, 10]

    def resolve():
        batch = compile_raw_df_into_ingest_batch(
            raw_df=priced, static_config=rsc, copy=True, as_of_date=A
        )
        mapper = build_uix_mapper([batch])
        portfolio = compile_portfolio_from_ingest_batch(
            ingest_batch=batch, uix_mapper=mapper
        )
        um = build_universe_mapping(
            discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
        )
        names = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
        snapshot = compile_risk_factor_level_snapshot_from_levels_dict(
            as_of_date=A,
            levels_dict={
                names[int(i)]: CANONICAL_MARKET_DATA_SNAPSHOT_LEVELS_DICT[names[int(i)]]
                for i in um.required_factor_ids
            },
            static_config=rsc,
        )
        matrix = build_market_data_matrix_from_snapshot(
            discounting_mode=DiscountingMode.NONE,
            as_of_date=A,
            uix_mapper=mapper,
            universe_map=um,
            snapshot=snapshot,
        )
        assert matrix.uix_fingerprint == mapper.fingerprint
        by_uid = dict(
            zip(batch.uids.tolist(), batch.underlying_ids.tolist(), strict=True)
        )
        rows = {
            int(uid): int(
                mapper.market_id_universe[mapper.spot_effective_pairs[int(e), 0]]
            )
            for uid, e in zip(
                portfolio.spot.trade_uids.tolist(),
                portfolio.spot.effective_spot_price_uix,
                strict=True,
            )
        }
        return by_uid, rows, mapper.market_id_universe.tolist(), mapper.fingerprint

    first = resolve()
    for _, r in priced.iterrows():
        pillar = int(rsc.rel_name_to_rf_id[r.underlying])
        assert first[0][int(r.uid)] == pillar
        assert first[1][int(r.uid)] == pillar
        assert _production_contract(rsc, pillar, A) == _oracle_contract(r.underlying, A)
    assert resolve() == first == resolve()  # deterministic
