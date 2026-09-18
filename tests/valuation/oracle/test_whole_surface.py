# tests/valuation/oracle/test_whole_surface.py
"""
Phase 6 whole-surface certification.

One production path, all three supported block types, reconciled at row, trade
and portfolio level against the independent economic oracle, then attacked with
a mutation matrix covering every semantic binding the supported surface has.

    IngestBatch -> build_uix_mapper -> compile_portfolio_from_ingest_batch
                -> MarketDataMatrix -> assemble_valuation_state
                -> run_valuation   -> ValuationResult

Nothing here hand-builds a block. The oracle imports no riskquant symbol and
groups rows from the FIXTURE, never from production's lookup.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from oracle import economic_oracle as oracle
from oracle import whole_surface_fixture as ws

from riskquant.market_data.compile import assemble_valuation_state
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.valuation.engine import run_valuation
from riskquant.valuation.result import ValuationResult

# D046 envelope. Spot and forward carry no CDF and are held to fp noise; the
# options in this fixture span |d2| <= 2.32, whose D046 band is 5e-5.
RTOL_LINEAR = 1e-13
RTOL_CDF = 5e-5

# Row index -> tolerance. Rows 0 (spot) and 1 (forward) are CDF-free.
_LINEAR_ROWS = frozenset({0, 1})


def _tol(row: int) -> float:
    return RTOL_LINEAR if row in _LINEAR_ROWS else RTOL_CDF


def _compile(batch=None):
    batch = batch if batch is not None else ws.ingest_batch()
    mapper = build_uix_mapper([batch], as_of_ord32=ws.AS_OF_ORD)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    state = assemble_valuation_state(
        matrix=ws.build_market_matrix(mapper), uix_mapper=mapper
    )
    return portfolio, state, mapper


def _run(portfolio, state) -> np.ndarray:
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return out


def _result() -> ValuationResult:
    portfolio, state, _ = _compile()
    return ValuationResult(
        trade_uids=portfolio.trade_uids, values=_run(portfolio, state)
    )


def _oracle_rows() -> list[tuple[int, np.ndarray]]:
    """`(uid, pv)` per economic row, in block order, grouped by the fixture."""
    m = ws.economic_market()
    rows: list[tuple[int, np.ndarray]] = []
    for t in ws.economic_book():
        if isinstance(t, oracle.SpotTrade):
            pv = oracle.pv_spot(t, m, ws.SETTLE_CCY)
        elif isinstance(t, oracle.ForwardTrade):
            pv = oracle.pv_forward(t, m, ws.SETTLE_CCY)
        elif isinstance(t, oracle.CrossCurrencyOptionTrade):
            pv = oracle.pv_cross_currency_option(t, m, ws.SETTLE_CCY)
        else:  # pragma: no cover
            raise TypeError(f"unknown economic trade type: {type(t)!r}")
        rows.append((t.trade_uid, pv))
    return rows


def _expected_matrix() -> np.ndarray:
    return np.stack([pv for _uid, pv in _oracle_rows()])


# =========================================================================
# SURFACE CLOSURE
# =========================================================================


def test_all_three_supported_blocks_are_present():
    portfolio, _, _ = _compile()
    assert [type(b).__name__ for b in portfolio.blocks] == [
        "SpotBlock",
        "VanillaForwardBlock",
        "VanillaEuropeanOptionBlock",
    ]
    assert portfolio.total_trades == 5
    assert portfolio.block_offsets.tolist() == [0, 1, 2]


def test_every_state_field_a_supported_block_reads_is_populated():
    """
    The closure invariant: a production-supported block must never consume an
    unpopulated universe.
    """
    _, state, _ = _compile()
    for field in (
        "effective_spot_prices",
        "effective_forward_prices",
        "effective_discount_factors",
        "vols",
        "forward_prices_strike_ccy",
        "option_log_forward_prices_strike_ccy",
    ):
        arr = getattr(state, field)
        assert arr.shape[0] > 0, field
        assert arr.shape[1] == ws.N_SCENARIOS, field
        assert arr.dtype == np.float64, field
        assert arr.flags.c_contiguous, field


def test_forward_rates_is_dormant_not_reachable():
    """
    `forward_rates` has no producer and no supported consumer. It is dormant
    future state for floating cashflows (D054), not a reachable empty universe.
    """
    _, state, _ = _compile()
    assert state.forward_rates.size == 0
    from riskquant.portfolio.compiled import CompiledPortfolio

    slots = {f.name for f in dataclasses.fields(CompiledPortfolio) if f.init}
    assert "floating_cashflow" not in slots


def test_every_stamped_uix_is_in_range_for_its_own_universe():
    portfolio, state, _ = _compile()
    checks = [
        (portfolio.spot.effective_spot_price_uix, state.effective_spot_prices),
        (
            portfolio.vanilla_forward.effective_forward_price_uix,
            state.effective_forward_prices,
        ),
        (
            portfolio.vanilla_forward.effective_discount_factor_uix,
            state.effective_discount_factors,
        ),
        (
            portfolio.vanilla_european_option.forward_price_strike_ccy_uix,
            state.forward_prices_strike_ccy,
        ),
        (
            portfolio.vanilla_european_option.option_log_forward_price_strike_ccy_uix,
            state.option_log_forward_prices_strike_ccy,
        ),
        (portfolio.vanilla_european_option.vol_uix, state.vols),
        (
            portfolio.vanilla_european_option.effective_discount_factor_uix,
            state.effective_discount_factors,
        ),
    ]
    for uix, table in checks:
        assert uix.dtype == np.int32
        assert int(uix.min()) >= 0
        assert int(uix.max()) < table.shape[0]


# =========================================================================
# ROW / TRADE / PORTFOLIO RECONCILIATION
# =========================================================================


def test_every_row_matches_the_independent_oracle():
    r = _result()
    for i, (uid, want) in enumerate(_oracle_rows()):
        assert int(r.trade_uids[i]) == uid
        np.testing.assert_allclose(
            r.values[i], want, rtol=_tol(i), atol=0.0, err_msg=f"row {i} / uid {uid}"
        )


def test_trade_pv_matches_independent_fixture_grouping():
    r = _result()
    rows = _oracle_rows()
    for uid in sorted(set(ws.UIDS)):
        want = sum(
            (pv for row_uid, pv in rows if row_uid == uid),
            start=np.zeros(ws.N_SCENARIOS),
        )
        np.testing.assert_allclose(
            r.get_trade_pv(uid), want, rtol=RTOL_CDF, atol=0.0, err_msg=f"uid {uid}"
        )


def test_multi_row_uid_spans_spot_and_option_blocks():
    """uid 9007 owns row 0 (spot) and row 3 (option) -- two block boundaries apart."""
    r = _result()
    assert r.num_rows == 5
    assert r.num_trades == 4
    assert [i for i, u in enumerate(ws.UIDS) if u == ws.MULTI_ROW_UID] == [0, 3]
    got = r.get_trade_values(ws.MULTI_ROW_UID)
    assert got.shape == (2, ws.N_SCENARIOS)
    np.testing.assert_allclose(
        r.get_trade_pv(ws.MULTI_ROW_UID), got[0] + got[1], rtol=1e-15
    )


def test_portfolio_total_reconciles_three_ways():
    r = _result()
    by_row = r.values.sum(axis=0)
    by_trade = sum(
        (r.get_trade_pv(uid) for uid in sorted(set(ws.UIDS))),
        start=np.zeros(ws.N_SCENARIOS),
    )
    by_oracle = _expected_matrix().sum(axis=0)
    np.testing.assert_allclose(by_row, by_trade, rtol=1e-14)
    np.testing.assert_allclose(by_row, by_oracle, rtol=RTOL_CDF)


# =========================================================================
# MUTATION MATRIX -- every semantic binding on the supported surface
# =========================================================================


def _swap(a: np.ndarray, i: int, j: int) -> np.ndarray:
    b = a.copy()
    b[[i, j]] = b[[j, i]]
    return np.ascontiguousarray(b)


def _rep(state, **kw):
    return dataclasses.replace(state, **kw)


STATE_MUTATIONS = {
    # --- SPOT ---
    # Rows 0 and 1 of `effective_spot_prices`, because the spot block consumes
    # row 1. Swapping two rows the block never reads is a NON-IDENTIFYING
    # mutation and would pass while proving nothing -- see
    # `test_every_state_mutation_touches_a_consumed_row`.
    "spot: wrong price row": lambda s: _rep(
        s, effective_spot_prices=_swap(s.effective_spot_prices, 0, 1)
    ),
    "spot: FX direction inverted": lambda s: _rep(
        s,
        effective_spot_prices=np.ascontiguousarray(
            s.effective_spot_prices / (0.91370**2)
        ),
    ),
    # --- FORWARD ---
    "forward: wrong effective forward leg": lambda s: _rep(
        s, effective_forward_prices=_swap(s.effective_forward_prices, 0, 1)
    ),
    "forward: wrong strike discount row": lambda s: _rep(
        s, effective_discount_factors=_swap(s.effective_discount_factors, 0, 3)
    ),
    "forward/option: strike-leg FX inversion dropped": lambda s: _rep(
        s,
        effective_discount_factors=np.ascontiguousarray(
            s.effective_discount_factors * (0.84655**2)
        ),
    ),
    # --- OPTION ---
    "option: wrong strike-ccy forward": lambda s: _rep(
        s, forward_prices_strike_ccy=_swap(s.forward_prices_strike_ccy, 0, 1)
    ),
    "option: wrong log-forward": lambda s: _rep(
        s,
        option_log_forward_prices_strike_ccy=_swap(
            s.option_log_forward_prices_strike_ccy, 0, 1
        ),
    ),
    "option: wrong volatility": lambda s: _rep(s, vols=_swap(s.vols, 0, 1)),
    "option: asset-leg FX substituted": lambda s: _rep(
        s,
        forward_prices_strike_ccy=np.ascontiguousarray(
            s.forward_prices_strike_ccy * 1.17
        ),
    ),
}


def test_every_state_mutation_touches_a_consumed_row():
    """
    Guard the guards: a mutation that permutes rows no block reads changes
    nothing and would pass vacuously. Assert each mutation actually moves the
    production output before asking whether the oracle rejects it.
    """
    portfolio, state, _ = _compile()
    base = _run(portfolio, state)
    for name, fn in STATE_MUTATIONS.items():
        moved = _run(portfolio, fn(state))
        assert not np.array_equal(base, moved), (
            f"mutation '{name}' left the output bit-identical -- it permutes "
            "rows no supported block consumes and proves nothing"
        )


@pytest.mark.parametrize("name", list(STATE_MUTATIONS))
def test_state_mutation_disagrees_with_the_independent_oracle(name):
    portfolio, state, _ = _compile()
    out = _run(portfolio, STATE_MUTATIONS[name](state))
    assert not np.allclose(out, _expected_matrix(), rtol=1e-4, atol=1e-10), (
        f"mutation '{name}' was NOT detected against the independent oracle"
    )


def _permute_option(portfolio, idx):
    o = portfolio.vanilla_european_option
    return dataclasses.replace(
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


def test_option_block_row_permutation_is_detected():
    portfolio, state, _ = _compile()
    out = _run(_permute_option(portfolio, np.array([2, 0, 1])), state)
    assert not np.allclose(out, _expected_matrix(), rtol=1e-4, atol=1e-10)


def test_cross_universe_uix_swap_is_detected_by_the_oracle():
    """
    §8 recurrence probe. `vol_uix` and `effective_discount_factor_uix` are both
    dense int32 into DIFFERENT tables, and every stamped vol row is also a valid
    EDF row -- so a range check cannot see the swap. Only the oracle can.
    """
    portfolio, state, _ = _compile()
    o = portfolio.vanilla_european_option
    assert int(o.vol_uix.max()) < state.effective_discount_factors.shape[0], (
        "fixture must make the swap in-range, or it proves nothing"
    )
    broken = dataclasses.replace(
        portfolio,
        vanilla_european_option=dataclasses.replace(
            o, effective_discount_factor_uix=np.ascontiguousarray(o.vol_uix)
        ),
    )
    out = _run(broken, state)
    assert not np.allclose(out, _expected_matrix(), rtol=1e-4, atol=1e-10)


def test_wrong_trade_uid_breaks_grouping_against_the_oracle():
    r = _result()
    mis_tagged = ValuationResult(
        trade_uids=np.ascontiguousarray([9007, 41, 120003, 777, 5], dtype=np.int64),
        values=r.values,
    )
    rows = _oracle_rows()
    want = sum(
        (pv for uid, pv in rows if uid == ws.MULTI_ROW_UID),
        start=np.zeros(ws.N_SCENARIOS),
    )
    assert not np.allclose(
        mis_tagged.get_trade_pv(ws.MULTI_ROW_UID), want, rtol=1e-6, atol=1e-10
    )


def test_wrong_expiry_date_reprices_the_options_only():
    """A date mutation must move the options and leave spot untouched."""
    batch = ws.ingest_batch()
    shifted = dataclasses.replace(
        batch,
        expiry_ord=np.ascontiguousarray(
            [ws.ORD_SPOT, ws.ORD_DELIVERY, ws.ORD_EXP_B, ws.ORD_EXP_B, ws.ORD_EXP_B],
            dtype=np.int32,
        ),
    )
    p2, s2, _ = _compile(shifted)
    out = _run(p2, s2)
    expected = _expected_matrix()
    np.testing.assert_allclose(out[0], expected[0], rtol=RTOL_LINEAR)
    assert not np.allclose(out[2], expected[2], rtol=1e-4)


# =========================================================================
# OUTPUT CONTRACT
# =========================================================================


@pytest.mark.parametrize(
    "shape", [(4, ws.N_SCENARIOS), (6, ws.N_SCENARIOS), (5, ws.N_SCENARIOS - 1)]
)
def test_mis_sized_output_is_rejected(shape):
    portfolio, state, _ = _compile()
    with pytest.raises(ValueError, match="out has shape"):
        run_valuation(
            portfolio=portfolio,
            valuation_state=state,
            out=np.empty(shape, dtype=np.float64),
        )


def test_non_contiguous_output_is_rejected():
    portfolio, state, _ = _compile()
    wide = np.empty((5, ws.N_SCENARIOS + 2))
    with pytest.raises(ValueError):
        run_valuation(
            portfolio=portfolio, valuation_state=state, out=wide[:, : ws.N_SCENARIOS]
        )
