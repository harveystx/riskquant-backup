# tests/market_data/remediation/test_portfolio_mapper_identity.py
"""
AUD-02: portfolio <-> UixMapper identity.

Invariant A (membership): every non-sentinel global id a compiler translates
must be an element of the mapper universe. An absent id raises inside the
translation; no insertion-position UIX is ever returned.

Invariant B (object identity): a compiled portfolio carries the fingerprint of
the mapper that stamped it. `run_valuation` and `run_sensitivities` refuse
state, matrices or mappers with any other fingerprint.

Expected values come from `md_oracle` (economic names and quotes only).
Every mutation first proves it moves a consumed production output (D059).
"""

from __future__ import annotations

from datetime import date

import hostile_fixture as hf
import md_oracle as oracle
import numpy as np
import pandas as pd
import pytest

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.constants import INVALID_UIX
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.universe import translate_ids_to_uix_or_raise
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

AS_OF_ORD = int(date_to_ord32(hf.AS_OF))
RTOL = 1e-13


@pytest.fixture(scope="module")
def rsc():
    return compile_risk_static_config_from_dict_of_dfs(hf.static_config_dfs())


def _book(*rows):
    return pd.DataFrame([
        dict(uid=i + 1, instrument="SPOT", quantity=q, underlying=u, settlement_ccy=s)
        for i, (u, s, q) in enumerate(rows)
    ])


def _batch(rsc, book):
    return compile_raw_df_into_ingest_batch(raw_df=book, static_config=rsc, copy=True)


def _market(rsc, batch, levels):
    mapper = build_uix_mapper([batch], as_of_ord32=AS_OF_ORD)
    um = build_universe_mapping(static_config=rsc, uix_mapper=mapper)
    matrix = build_market_data_matrix_from_flat_levels(
        as_of_date=hf.AS_OF,
        uix_mapper=mapper,
        universe_map=um,
        flat_levels=hf.flat_levels_for(um, rsc, levels),
        num_scenarios=hf.S,
    )
    return mapper, matrix


def _value(portfolio, mapper, matrix):
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((portfolio.total_trades, hf.S), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return out


def _oracle_pv(book, levels):
    mkt = oracle.EconomicMarket(
        levels=levels,
        factor_ccy=hf.FACTOR_CCY,
        fx_quotes={pair: levels[name] for pair, name in hf.FX_QUOTES.items()},
    )
    return [
        oracle.spot_pv(
            mkt,
            qty=r.quantity,
            underlying=r.underlying,
            settle=r.settlement_ccy,
            n=hf.S,
        )
        for r in book.itertuples(index=False)
    ]


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


# ----------------------------------------------------------------- positive path
def test_portfolio_with_its_own_mapper_and_matrix_reconciles(rsc):
    book = _book(("EQ_GBP", "EUR", -3000.0), ("EQ_USD", "EUR", 120.0))
    batch = _batch(rsc, book)
    mapper, matrix = _market(rsc, batch, hf.LEVELS)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    assert portfolio.uix_fingerprint == mapper.fingerprint
    out = _value(portfolio, mapper, matrix)
    expected = _oracle_pv(book, hf.LEVELS)
    for row, uid in enumerate(portfolio.trade_uids.tolist()):
        assert _rel(out[row], expected[uid - 1]) <= RTOL


# ------------------------------------------------ Invariant A: membership proof
@pytest.mark.parametrize(
    ("label", "portfolio_rows", "mapper_rows", "absent"),
    [
        # universe [7, 31, 402]; 58 would insert at the row of 402 (EURGBP)
        (
            "between",
            [("EQ_GBP", "GBP", -3000.0)],
            [("EQ_USD", "EUR", 1.0), ("EURGBP", "GBP", 1.0)],
            58,
        ),
        # universe [31]; 7 would insert at the row of 31 (USDEUR)
        ("below_min", [("EQ_USD", "USD", 100.0)], [("USDEUR", "EUR", 1.0)], 7),
        # universe [7]; 90001 would insert one past the end
        ("above_max", [("EQ_EUR", "EUR", 3.0)], [("EQ_USD", "USD", 1.0)], 90001),
        # underlying 7 present, FX rf 31 absent from universe [7, 402]
        (
            "fx_leg",
            [("EQ_USD", "EUR", 100.0)],
            [("EQ_USD", "USD", 1.0), ("EURGBP", "GBP", 1.0)],
            31,
        ),
    ],
)
def test_absent_id_never_becomes_an_insertion_point_uix(
    rsc, label, portfolio_rows, mapper_rows, absent
):
    portfolio_batch = _batch(rsc, _book(*portfolio_rows))
    mapper, _ = _market(rsc, _batch(rsc, _book(*mapper_rows)), hf.LEVELS)
    assert absent not in mapper.market_id_universe.tolist()
    with pytest.raises(
        ValueError,
        match=rf"absent from the UixMapper universe \(AUD-02\): \[{absent}\]",
    ):
        compile_portfolio_from_ingest_batch(
            ingest_batch=portfolio_batch, uix_mapper=mapper
        )


def test_translation_primitive_returns_no_coordinate_for_an_absent_id(rsc):
    mapper, _ = _market(
        rsc,
        _batch(rsc, _book(("EQ_USD", "EUR", 1.0), ("EURGBP", "GBP", 1.0))),
        hf.LEVELS,
    )
    assert mapper.market_id_universe.tolist() == [7, 31, 402]
    # the pre-remediation unsafe join mapped 58 to row 2 (402)
    assert int(np.searchsorted(mapper.market_id_universe, 58)) == 2
    with pytest.raises(ValueError, match="absent"):
        mapper.map_market_ids_to_uix(np.array([7, 58], dtype=np.int32))
    with pytest.raises(ValueError, match="absent"):
        mapper.map_discount_ids_to_uix(np.array([250_001_000], dtype=np.int32))
    with pytest.raises(ValueError, match="absent"):
        mapper.map_dates_to_uix(np.array([AS_OF_ORD], dtype=np.int32))


def test_translation_primitive_is_exact_on_sparse_interleaved_ids_with_sentinels():
    universe = np.array([7, 31, 58, 402, 1200, 5003, 90001], dtype=np.int32)
    raw = np.array([90001, -1, 402, 7, -999_999, 5003, 31], dtype=np.int32)
    out = translate_ids_to_uix_or_raise(universe=universe, raw=raw, ctx="probe")
    assert out.dtype == np.int32
    assert out.tolist() == [6, INVALID_UIX, 3, 0, INVALID_UIX, 5, 1]


def test_missing_date_in_a_foreign_mapper_is_rejected(rsc):
    fwd = dict(
        uid=1,
        instrument="VANILLA_FORWARD_NODE",
        quantity=10.0,
        underlying="BRENT_FWD[M+0]",
        settlement_ccy="USD",
        strike_price=80.0,
        strike_ccy="USD",
        settlement_date=date(2027, 2, 17),
        under_discount_curve="USD_OIS",
        strike_discount_curve="USD_OIS",
    )
    batch_a = compile_raw_df_into_ingest_batch(
        raw_df=pd.DataFrame([dict(fwd, delivery_date=date(2027, 2, 15))]),
        static_config=rsc,
        copy=True,
        as_of_date=hf.AS_OF,
    )
    batch_b = compile_raw_df_into_ingest_batch(
        raw_df=pd.DataFrame([dict(fwd, delivery_date=date(2027, 2, 20))]),
        static_config=rsc,
        copy=True,
        as_of_date=hf.AS_OF,
    )
    mapper_b = build_uix_mapper([batch_b])
    with pytest.raises(ValueError, match=r"date ordinals.*absent"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch_a, uix_mapper=mapper_b)


# ------------------------------------------ Invariant B: portfolio object identity
def test_same_shape_foreign_mapper_with_overlapping_uix_is_rejected(rsc):
    book_a = _book(("EQ_USD", "USD", 100.0))
    book_b = _book(("EQ_EUR", "EUR", 100.0))
    batch_a, batch_b = _batch(rsc, book_a), _batch(rsc, book_b)
    mapper_a, matrix_a = _market(rsc, batch_a, hf.LEVELS)
    mapper_b, matrix_b = _market(rsc, batch_b, hf.LEVELS)
    portfolio_a = compile_portfolio_from_ingest_batch(
        ingest_batch=batch_a, uix_mapper=mapper_a
    )
    portfolio_b = compile_portfolio_from_ingest_batch(
        ingest_batch=batch_b, uix_mapper=mapper_b
    )

    # same dimensions, different identities, numerically identical stamped uix
    assert mapper_a.market_id_universe.shape == mapper_b.market_id_universe.shape
    assert mapper_a.market_id_universe.tolist() != mapper_b.market_id_universe.tolist()
    assert (
        portfolio_a.spot.effective_spot_price_uix.tolist()
        == portfolio_b.spot.effective_spot_price_uix.tolist()
    )

    # the foreign state is economically different for this trade
    pv_a = _value(portfolio_a, mapper_a, matrix_a)[0]
    assert _rel(pv_a, _oracle_pv(book_a, hf.LEVELS)[0]) <= RTOL
    assert not np.allclose(
        _oracle_pv(book_a, hf.LEVELS)[0], _oracle_pv(book_b, hf.LEVELS)[0]
    )

    state_b = assemble_valuation_state(matrix=matrix_b, uix_mapper=mapper_b)
    out = np.full((1, hf.S), np.nan)
    with pytest.raises(ValueError, match="different UixMappers"):
        run_valuation(portfolio=portfolio_a, valuation_state=state_b, out=out)
    assert np.isnan(out).all()  # rejected before any kernel wrote a PV

    with pytest.raises(ValueError, match="different UixMapper"):
        run_sensitivities(
            portfolio=portfolio_a, uix_mapper=mapper_b, market_data=matrix_b
        )
    with pytest.raises(ValueError, match="different UixMapper"):
        run_sensitivities(
            portfolio=portfolio_a, uix_mapper=mapper_a, market_data=matrix_b
        )
    # existing D064 contract for mapper A + matrix B stays in force
    with pytest.raises(ValueError, match="different UixMapper"):
        assemble_valuation_state(matrix=matrix_b, uix_mapper=mapper_a)


# -------------------------------------------------------------- D059 mutations
def _perturb(name, factor):
    levels = {k: v.copy() for k, v in hf.LEVELS.items()}
    levels[name] = levels[name] * factor
    return levels


def test_insertion_target_is_consumed_and_economically_distinct(rsc):
    """
    In the `between` trap, EURGBP (402) is the row EQ_GBP (58) used to alias.
    1. EURGBP is consumed by the mapper's own portfolio (mutation moves its PV).
    2. The EQ_GBP oracle does not depend on EURGBP when settled in GBP, so the
       aliased price was economically wrong, not a rounding variant.
    3. The hostile pairing is rejected (see Invariant A test).
    """
    mapper_book = _book(("EQ_USD", "EUR", 1.0), ("EURGBP", "GBP", 1.0))
    batch = _batch(rsc, mapper_book)
    mapper, matrix = _market(rsc, batch, hf.LEVELS)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    base = _value(portfolio, mapper, matrix)

    moved_levels = _perturb("EURGBP", 1.07)
    mapper_m, matrix_m = _market(rsc, batch, moved_levels)
    portfolio_m = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper_m
    )
    moved = _value(portfolio_m, mapper_m, matrix_m)
    eurgbp_row = portfolio.trade_uids.tolist().index(2)
    assert not np.array_equal(base[eurgbp_row], moved[eurgbp_row])

    trap = _book(("EQ_GBP", "GBP", -3000.0))
    assert np.array_equal(
        _oracle_pv(trap, hf.LEVELS)[0], _oracle_pv(trap, moved_levels)[0]
    )
    aliased_value = -3000.0 * hf.LEVELS["EURGBP"]
    assert not np.allclose(_oracle_pv(trap, hf.LEVELS)[0], aliased_value)


def test_consumed_level_mutation_moves_production_and_oracle_together(rsc):
    book = _book(("EQ_GBP", "EUR", -3000.0))
    batch = _batch(rsc, book)
    mapper, matrix = _market(rsc, batch, hf.LEVELS)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    base = _value(portfolio, mapper, matrix)[0]

    levels = _perturb("EQ_GBP", 0.95)
    mapper_m, matrix_m = _market(rsc, batch, levels)
    portfolio_m = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper_m
    )
    moved = _value(portfolio_m, mapper_m, matrix_m)[0]

    assert not np.array_equal(base, moved)
    assert not np.allclose(_oracle_pv(book, hf.LEVELS)[0], _oracle_pv(book, levels)[0])
    assert _rel(moved, _oracle_pv(book, levels)[0]) <= RTOL
    # same economic universe, different levels: identical mapper identity
    assert mapper_m.fingerprint == mapper.fingerprint
