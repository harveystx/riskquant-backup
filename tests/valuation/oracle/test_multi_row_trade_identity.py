# tests/valuation/oracle/test_multi_row_trade_identity.py
"""
Row/trade cardinality on the PRODUCTION path.

    one economic trade -> many valuation rows -> trade PV = sum(row PVs)

Phase 6 set out to prove this with cashflows. Cashflow ingest does not exist
(see the Phase-6 report), so the same cardinality model is exercised with the
block types that ARE production-reachable: one economic trade whose uid appears
in BOTH the forward block and the option block.

That is not a workaround. It is the identical invariant -- `trade_uids` is not
unique, rows for one trade are not contiguous (they sit in different blocks),
and the result container must aggregate rather than index -- and it is the
invariant `ValuationResult.get_trade_pv` was written for in D045/VAL-05.

Hostile by construction: uids `[9007, 41, 120003, 9007]`, non-ascending, with
9007 repeated ACROSS two block types and therefore across a block-offset
boundary.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from oracle import economic_oracle as oracle
from oracle import option_fixture as of

from riskquant.market_data.compile import assemble_valuation_state
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.valuation.engine import run_valuation
from riskquant.valuation.result import ValuationResult

RTOL_CDF = 5e-5
RTOL_LINEAR = 1e-13

# uid 9007 owns BOTH the forward row and the last option row.
MULTI_ROW_UID = 9007
UIDS = [9007, 41, 120003, 9007]


def _multi_row_batch():
    """`of.ingest_batch()` with the last option re-tagged to the forward's uid."""
    batch = of.ingest_batch()
    return dataclasses.replace(batch, uids=np.ascontiguousarray(UIDS, dtype=np.int64))


def _compile():
    batch = _multi_row_batch()
    mapper = build_uix_mapper([batch], as_of_ord32=of.AS_OF_ORD)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    state = assemble_valuation_state(
        matrix=of.build_market_matrix(mapper), uix_mapper=mapper
    )
    return portfolio, state


def _result() -> ValuationResult:
    portfolio, state = _compile()
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return ValuationResult(trade_uids=portfolio.trade_uids, values=out)


def _oracle_rows() -> list[tuple[int, np.ndarray]]:
    """
    `(uid, pv)` per ECONOMIC row, in block order, with the uid re-tagged exactly
    as the batch re-tags it. The fixture -- not production -- knows which rows
    belong to which trade.
    """
    m = of.economic_market()
    rows: list[tuple[int, np.ndarray]] = []
    for uid, t in zip(UIDS, of.economic_book(), strict=True):
        if isinstance(t, oracle.ForwardTrade):
            rows.append((uid, oracle.pv_forward(t, m, of.SETTLE_CCY)))
        elif isinstance(t, oracle.CrossCurrencyOptionTrade):
            rows.append((uid, oracle.pv_cross_currency_option(t, m, of.SETTLE_CCY)))
        else:  # pragma: no cover
            raise TypeError(f"unknown economic trade type: {type(t)!r}")
    return rows


def _tol(index: int) -> float:
    return RTOL_LINEAR if index == 0 else RTOL_CDF


# =========================================================================
# CARDINALITY
# =========================================================================


def test_one_trade_produces_rows_in_two_different_blocks():
    portfolio, _ = _compile()
    assert portfolio.trade_uids.tolist() == UIDS
    assert portfolio.vanilla_forward.trade_uids.tolist() == [MULTI_ROW_UID]
    assert MULTI_ROW_UID in portfolio.vanilla_european_option.trade_uids.tolist()
    # The two rows for uid 9007 straddle a block-offset boundary.
    positions = [i for i, u in enumerate(UIDS) if u == MULTI_ROW_UID]
    assert positions == [0, 3]
    assert portfolio.block_offsets.tolist() == [0, 1]


def test_trade_uids_are_neither_unique_nor_sorted():
    portfolio, _ = _compile()
    uids = portfolio.trade_uids.tolist()
    assert len(set(uids)) < len(uids)
    assert uids != sorted(uids)


def test_num_rows_and_num_trades_differ():
    r = _result()
    assert r.num_rows == 4
    assert r.num_trades == 3


# =========================================================================
# ROW-LEVEL ACCEPTANCE
# =========================================================================


def test_every_row_matches_the_independent_oracle():
    r = _result()
    for i, (uid, want) in enumerate(_oracle_rows()):
        assert int(r.trade_uids[i]) == uid
        np.testing.assert_allclose(
            r.values[i], want, rtol=_tol(i), atol=0.0, err_msg=f"row {i} / uid {uid}"
        )


# =========================================================================
# TRADE AGGREGATION -- the Phase-6 invariant
# =========================================================================


def test_get_trade_values_returns_every_row_the_trade_produced():
    r = _result()
    got = r.get_trade_values(MULTI_ROW_UID)
    assert got.shape == (2, of.N_SCENARIOS)


def test_trade_pv_equals_the_independently_summed_row_pvs():
    """
    PV_trade = sum over the rows the FIXTURE says belong to the trade -- not
    over whatever rows production chose to group.
    """
    r = _result()
    rows = _oracle_rows()

    for uid in sorted(set(UIDS)):
        want = sum(
            (pv for row_uid, pv in rows if row_uid == uid),
            start=np.zeros(of.N_SCENARIOS),
        )
        np.testing.assert_allclose(
            r.get_trade_pv(uid), want, rtol=RTOL_CDF, atol=0.0, err_msg=f"uid {uid}"
        )


def test_multi_row_trade_pv_is_not_either_row_alone():
    """The aggregation must be a real sum, not a first-match lookup."""
    r = _result()
    total = r.get_trade_pv(MULTI_ROW_UID)
    rows = r.get_trade_values(MULTI_ROW_UID)
    assert not np.allclose(total, rows[0])
    assert not np.allclose(total, rows[1])
    np.testing.assert_allclose(total, rows[0] + rows[1], rtol=1e-15)


def test_portfolio_total_is_the_sum_over_trades_and_over_rows():
    """Both groupings must reach the same portfolio PV."""
    r = _result()
    by_row = r.values.sum(axis=0)
    by_trade = sum(
        (r.get_trade_pv(uid) for uid in sorted(set(UIDS))),
        start=np.zeros(of.N_SCENARIOS),
    )
    np.testing.assert_allclose(by_row, by_trade, rtol=1e-14)

    oracle_total = sum(
        (pv for _uid, pv in _oracle_rows()), start=np.zeros(of.N_SCENARIOS)
    )
    np.testing.assert_allclose(by_row, oracle_total, rtol=RTOL_CDF)


# =========================================================================
# MUTATION
# =========================================================================


def test_wrong_trade_grouping_is_detected():
    """
    Re-tag the second 9007 row to a different uid. Production will then report a
    3-row/4-trade book whose 9007 PV is one row, and the oracle -- which groups
    from the fixture -- must disagree.
    """
    r = _result()
    mis_tagged = ValuationResult(
        trade_uids=np.ascontiguousarray([9007, 41, 120003, 777], dtype=np.int64),
        values=r.values,
    )
    rows = _oracle_rows()
    want = sum(
        (pv for row_uid, pv in rows if row_uid == MULTI_ROW_UID),
        start=np.zeros(of.N_SCENARIOS),
    )
    assert not np.allclose(
        mis_tagged.get_trade_pv(MULTI_ROW_UID), want, rtol=1e-6, atol=1e-10
    )


def test_block_row_permutation_breaks_the_uid_binding():
    portfolio, state = _compile()
    o = portfolio.vanilla_european_option
    idx = np.array([2, 0, 1])
    permuted = dataclasses.replace(
        portfolio,
        vanilla_european_option=dataclasses.replace(
            o,
            trade_uids=np.ascontiguousarray(o.trade_uids[idx]),
            qty=np.ascontiguousarray(o.qty[idx]),
            strike=np.ascontiguousarray(o.strike[idx]),
            log_strike=np.ascontiguousarray(o.log_strike[idx]),
            cp_mult=np.ascontiguousarray(o.cp_mult[idx]),
            sqrt_tte=np.ascontiguousarray(o.sqrt_tte[idx]),
            half_tte=np.ascontiguousarray(o.half_tte[idx]),
            forward_price_strike_ccy_uix=np.ascontiguousarray(
                o.forward_price_strike_ccy_uix[idx]
            ),
            option_log_forward_price_strike_ccy_uix=np.ascontiguousarray(
                o.option_log_forward_price_strike_ccy_uix[idx]
            ),
            vol_uix=np.ascontiguousarray(o.vol_uix[idx]),
            effective_discount_factor_uix=np.ascontiguousarray(
                o.effective_discount_factor_uix[idx]
            ),
        ),
    )
    out = np.empty(
        (permuted.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=permuted, valuation_state=state, out=out)

    expected = np.stack([pv for _uid, pv in _oracle_rows()])
    assert not np.allclose(out, expected, rtol=1e-4, atol=1e-10)


# =========================================================================
# CASHFLOW BLOCKS REMAIN UNREACHABLE -- pinned, not assumed
# =========================================================================


def test_cashflow_blocks_are_still_not_production_reachable():
    """
    Phase 6 could not wire these: `IngestBatch` carries no notional, accrual
    factor, rate, spread, principal amount, pay/receive direction or reset date,
    and `MarketDataMatrix` has no forward-rate producer. Delete this test only
    when those inputs exist.
    """
    from riskquant.portfolio.compiled import CompiledPortfolio
    from riskquant.portfolio.ingest.batch import IngestBatch
    from riskquant.portfolio.ingest.schema import ALL_ALLOWED_COLS_SET

    slots = {f.name for f in dataclasses.fields(CompiledPortfolio) if f.init}
    assert "fixed_cashflow" not in slots
    assert "floating_cashflow" not in slots
    assert "principal_cashflow" not in slots

    ingest_fields = {f.name for f in dataclasses.fields(IngestBatch)}
    for required in ("notional", "accrual_factor", "fixed_rate", "spread", "amount"):
        assert required not in ingest_fields, required
    for column in ("notional", "accrual_factor", "fixed_rate", "spread"):
        assert column not in ALL_ALLOWED_COLS_SET, column


def test_forward_rate_state_field_has_no_producer():
    """`ValuationState.forward_rates` exists; nothing in the raw tier fills it."""
    from riskquant.market_data.state import MarketDataMatrix

    raw_fields = {f.name for f in dataclasses.fields(MarketDataMatrix)}
    assert not any("forward_rate" in f for f in raw_fields)

    _, state = _compile()
    assert state.forward_rates.size == 0


@pytest.mark.parametrize("instrument_id", [5])
def test_unregistered_instrument_fails_loud(instrument_id):
    """VANILLA_IRS (5) must raise, not silently drop."""
    from riskquant.portfolio.compilers.registry import COMPILERS

    assert COMPILERS[instrument_id] is None
