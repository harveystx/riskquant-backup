# tests/valuation/oracle/test_option_production_path.py
"""
Phase 5 acceptance: a REAL cross-currency option, compiled by the REAL option
compiler, priced by the REAL Black-76 kernel, against the independent oracle.

Nothing here hand-builds a `VanillaEuropeanOptionBlock`. The strike-currency
forward, its log cache, the volatility row and the settlement discount row are
all produced by `build_uix_mapper` / `compile_portfolio_from_ingest_batch` /
`assemble_valuation_state`, and the oracle derives the same economics from
covered interest parity without importing a single production symbol.
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
from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionBlock,
)
from riskquant.valuation.engine import run_valuation

# D046: the production price path uses the Abramowitz-Stegun CDF; the oracle uses
# scipy, so every option deviation here is attributable to that approximation.
# D046 measures the envelope against |d2|, not against moneyness: <= 6e-6 for
# |d2| <~ 2, ~5e-5 at |d2| ~ 3. The fixture spans |d2| in [0.41, 2.32], so 5e-5
# is the correct D046 band for its deepest row. Measured worst: 2.22e-05 on the
# short gold call at |d2| = 2.32; every |d2| < 2 row is at or below 2.3e-06.
#
# This tolerance is READ OFF D046 for the fixture's measured moneyness. Do not
# raise it to accommodate a construction change -- if a row moves past |d2| ~ 3.5
# the right response is to record the new band, not to widen this one.
RTOL_CDF = 5e-5
RTOL_LINEAR = 1e-13


def _compile(batch=None, as_of=None):
    batch = batch if batch is not None else of.ingest_batch()
    mapper = build_uix_mapper(
        [batch], as_of_ord32=of.AS_OF_ORD if as_of is None else as_of
    )
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    state = assemble_valuation_state(
        matrix=of.build_market_matrix(mapper), uix_mapper=mapper
    )
    return portfolio, state, mapper


def _production_pv(**kw):
    portfolio, state, _ = _compile(**kw)
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return portfolio.trade_uids, out


def _oracle_pv() -> dict[int, np.ndarray]:
    m = of.economic_market()
    pv: dict[int, np.ndarray] = {}
    for t in of.economic_book():
        if isinstance(t, oracle.ForwardTrade):
            pv[t.trade_uid] = oracle.pv_forward(t, m, of.SETTLE_CCY)
        elif isinstance(t, oracle.CrossCurrencyOptionTrade):
            pv[t.trade_uid] = oracle.pv_cross_currency_option(t, m, of.SETTLE_CCY)
        else:  # pragma: no cover
            raise TypeError(f"unknown economic trade type: {type(t)!r}")
    return pv


def _tol(uid: int) -> float:
    return RTOL_LINEAR if uid == 9007 else RTOL_CDF


# =========================================================================
# REACHABILITY
# =========================================================================


def test_option_is_reachable_through_the_production_compiler():
    portfolio, _, _ = _compile()
    assert isinstance(portfolio.vanilla_european_option, VanillaEuropeanOptionBlock)
    assert portfolio.vanilla_european_option.size == 3


def test_compiled_uids_preserve_hostile_ingest_ordering():
    portfolio, _, _ = _compile()
    assert portfolio.trade_uids.tolist() == [9007, 41, 120003, 5]
    assert portfolio.trade_uids.tolist() != sorted(portfolio.trade_uids.tolist())


def test_the_fold_populates_every_option_factor_universe():
    _, state, _ = _compile()
    assert state.forward_prices_strike_ccy.shape[0] > 0
    assert state.option_log_forward_prices_strike_ccy.shape[0] > 0
    assert state.vols.shape[0] > 0
    assert state.effective_discount_factors.shape[0] > 0


# =========================================================================
# VAL-11 -- the pairing invariant, proved structurally
# =========================================================================


def test_forward_and_log_forward_uix_are_the_same_coordinate():
    """One universe, two materialisations: the compiler stamps one integer."""
    portfolio, _, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    np.testing.assert_array_equal(
        o.forward_price_strike_ccy_uix,
        o.option_log_forward_price_strike_ccy_uix,
    )


def test_log_forward_table_is_the_elementwise_log_of_the_forward_table():
    """`L[k, s] == log(F[k, s])` for EVERY row and scenario, not just the paired ones."""
    _, state, _ = _compile()
    np.testing.assert_allclose(
        state.option_log_forward_prices_strike_ccy,
        np.log(state.forward_prices_strike_ccy),
        rtol=0.0,
        atol=0.0,
    )


def test_strike_ccy_forward_matches_independent_covered_interest_parity():
    """The fold's ratio must equal the oracle's independently derived F_K."""
    _, state, _mapper = _compile()
    m = of.economic_market()
    for t in of.economic_book():
        if not isinstance(t, oracle.CrossCurrencyOptionTrade):
            continue
        want = oracle.forward_in_strike_ccy(t, m, of.SETTLE_CCY)
        row = _forward_row_for(t.trade_uid)
        np.testing.assert_allclose(
            state.forward_prices_strike_ccy[row], want, rtol=RTOL_LINEAR
        )


def _forward_row_for(uid: int) -> int:
    portfolio, _, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    idx = o.trade_uids.tolist().index(uid)
    return int(o.forward_price_strike_ccy_uix[idx])


# =========================================================================
# ACCEPTANCE
# =========================================================================


def test_production_option_matches_the_independent_oracle():
    uids, prod = _production_pv()
    exp = _oracle_pv()
    for row, uid in enumerate(uids.tolist()):
        np.testing.assert_allclose(
            prod[row],
            exp[uid],
            rtol=_tol(uid),
            atol=0.0,
            err_msg=f"row {row} / uid {uid}",
        )


def test_call_and_put_and_long_and_short_are_all_exercised():
    book = {t.trade_uid: t for t in of.economic_book()}
    assert book[41].is_call and book[41].qty < 0  # short call
    assert not book[120003].is_call and book[120003].qty > 0  # long put
    assert not book[5].is_call and book[5].qty > 0  # long put, other strike
    _, prod = _production_pv()
    assert np.all(prod[1] < 0)  # short call is a liability
    assert np.all(prod[2] > 0)  # long put is an asset


def test_deferred_settlement_binds_the_settlement_discount_row():
    """
    uid 120003 settles AFTER expiry, so its discount row must be the settlement
    row -- distinct from the expiry row that denominates its forward.
    """
    portfolio, _, mapper = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    i = o.trade_uids.tolist().index(120003)
    edf_row = int(o.effective_discount_factor_uix[i])
    fk_row = int(o.forward_price_strike_ccy_uix[i])
    expiry_edf_row = int(mapper.fk_effective_pairs[fk_row, 1])
    assert edf_row != expiry_edf_row, (
        "deferred settlement must not reuse the expiry discount row"
    )


def test_forward_block_remains_certified():
    uids, prod = _production_pv()
    exp = _oracle_pv()
    np.testing.assert_allclose(
        prod[uids.tolist().index(9007)], exp[9007], rtol=RTOL_LINEAR
    )


# =========================================================================
# FAIL-FAST (VAL-04, option path)
# =========================================================================


def test_missing_as_of_is_rejected():
    batch = of.ingest_batch()
    mapper = build_uix_mapper([batch])  # no as_of_ord32
    with pytest.raises(ValueError, match="as_of_ord32"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=mapper)


def test_unpopulated_forward_universe_is_rejected():
    batch = of.ingest_batch()
    mapper = build_uix_mapper([batch], as_of_ord32=of.AS_OF_ORD)
    starved = dataclasses.replace(
        mapper, fk_effective_pairs=np.empty((0, 2), dtype=np.int32)
    )
    with pytest.raises(ValueError, match="forward_price_strike_ccy"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=starved)


def test_unpopulated_vol_universe_is_rejected():
    batch = of.ingest_batch()
    mapper = build_uix_mapper([batch], as_of_ord32=of.AS_OF_ORD)
    starved = dataclasses.replace(mapper, vol_effective_uix=np.empty(0, dtype=np.int32))
    with pytest.raises(ValueError, match="volatility"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=starved)


def test_non_positive_strike_is_rejected():
    batch = of.ingest_batch()
    bad = dataclasses.replace(
        batch,
        strike_price=np.ascontiguousarray([61.25, 0.0, 30.0, 2300.0], dtype=np.float64),
    )
    mapper = build_uix_mapper([bad], as_of_ord32=of.AS_OF_ORD)
    with pytest.raises(ValueError, match="strike must be strictly positive"):
        compile_portfolio_from_ingest_batch(ingest_batch=bad, uix_mapper=mapper)


def test_settlement_before_expiry_is_rejected():
    batch = of.ingest_batch()
    bad = dataclasses.replace(
        batch,
        settle_ord=np.ascontiguousarray(
            [of.ORD_DELIVERY, of.ORD_EXP_A - 5, of.ORD_SETTLE_LATE, of.ORD_EXP_A],
            dtype=np.int32,
        ),
    )
    mapper = build_uix_mapper([bad], as_of_ord32=of.AS_OF_ORD)
    with pytest.raises(ValueError, match="settlement precedes expiry"):
        compile_portfolio_from_ingest_batch(ingest_batch=bad, uix_mapper=mapper)


# =========================================================================
# BLOCK PRECOMPUTED INVARIANTS
# =========================================================================


def test_log_strike_is_the_log_of_strike():
    portfolio, _, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    np.testing.assert_allclose(o.log_strike, np.log(o.strike), rtol=0.0, atol=0.0)


def test_half_tte_is_half_the_square_of_sqrt_tte():
    portfolio, _, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    np.testing.assert_allclose(o.half_tte, 0.5 * o.sqrt_tte**2, rtol=1e-15)


def test_tte_matches_the_independent_act_365_25_convention():
    portfolio, _, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    book = {t.trade_uid: t for t in of.economic_book()}
    for i, uid in enumerate(o.trade_uids.tolist()):
        np.testing.assert_allclose(o.sqrt_tte[i] ** 2, book[uid].tte_years, rtol=1e-14)


def test_cp_mult_reaches_the_block_as_int8():
    portfolio, _, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    assert o.cp_mult.dtype == np.int8


# =========================================================================
# MUTATION EVIDENCE
# =========================================================================


def _swap(a: np.ndarray, i: int, j: int) -> np.ndarray:
    b = a.copy()
    b[[i, j]] = b[[j, i]]
    return np.ascontiguousarray(b)


STATE_MUTATIONS = {
    "strike-ccy forward row permutation": lambda s: dataclasses.replace(
        s, forward_prices_strike_ccy=_swap(s.forward_prices_strike_ccy, 0, 1)
    ),
    "log-forward row permutation (VAL-11)": lambda s: dataclasses.replace(
        s,
        option_log_forward_prices_strike_ccy=_swap(
            s.option_log_forward_prices_strike_ccy, 0, 1
        ),
    ),
    "vol row permutation": lambda s: dataclasses.replace(s, vols=_swap(s.vols, 0, 1)),
    "discount row permutation": lambda s: dataclasses.replace(
        s, effective_discount_factors=_swap(s.effective_discount_factors, 1, 2)
    ),
    "strike-leg FX inversion dropped": lambda s: dataclasses.replace(
        s,
        effective_discount_factors=np.ascontiguousarray(
            s.effective_discount_factors * (0.84655**2)
        ),
    ),
    "asset-leg FX substituted": lambda s: dataclasses.replace(
        s,
        forward_prices_strike_ccy=np.ascontiguousarray(
            s.forward_prices_strike_ccy * 1.17
        ),
    ),
}


@pytest.mark.parametrize("name", list(STATE_MUTATIONS))
def test_state_mutation_disagrees_with_the_independent_oracle(name):
    portfolio, state, _ = _compile()
    mutated = STATE_MUTATIONS[name](state)
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=mutated, out=out)

    exp = _oracle_pv()
    expected = np.stack([exp[u] for u in portfolio.trade_uids.tolist()])
    assert not np.allclose(out, expected, rtol=1e-4, atol=1e-10), (
        f"mutation '{name}' was NOT detected against the independent oracle"
    )


def test_log_forward_uix_pointed_at_another_forward_is_detected():
    """
    The decisive VAL-11 test: break the pairing at the BLOCK, pointing the log
    coordinate at a different forward while the level coordinate stays correct.
    The oracle must reject the result.
    """
    portfolio, state, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    wrong = np.ascontiguousarray(
        (o.option_log_forward_price_strike_ccy_uix + 1)
        % state.forward_prices_strike_ccy.shape[0],
        dtype=np.int32,
    )
    assert not np.array_equal(wrong, o.forward_price_strike_ccy_uix)

    broken = dataclasses.replace(
        portfolio,
        vanilla_european_option=dataclasses.replace(
            o, option_log_forward_price_strike_ccy_uix=wrong
        ),
    )
    out = np.empty((broken.total_trades, state.batch_size), dtype=np.float64, order="C")
    run_valuation(portfolio=broken, valuation_state=state, out=out)

    exp = _oracle_pv()
    expected = np.stack([exp[u] for u in portfolio.trade_uids.tolist()])
    assert not np.allclose(out, expected, rtol=1e-4, atol=1e-10)


def test_vol_uix_pointed_at_another_volatility_is_detected():
    """`vol_uix` must be tied to the option's own forward, not "some vol row"."""
    portfolio, state, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    wrong = np.ascontiguousarray((o.vol_uix + 1) % state.vols.shape[0], dtype=np.int32)
    assert not np.array_equal(wrong, o.vol_uix)

    broken = dataclasses.replace(
        portfolio,
        vanilla_european_option=dataclasses.replace(o, vol_uix=wrong),
    )
    out = np.empty((broken.total_trades, state.batch_size), dtype=np.float64, order="C")
    run_valuation(portfolio=broken, valuation_state=state, out=out)

    exp = _oracle_pv()
    expected = np.stack([exp[u] for u in portfolio.trade_uids.tolist()])
    assert not np.allclose(out, expected, rtol=1e-4, atol=1e-10)


def test_expiry_discount_row_substituted_for_settlement_is_detected():
    """
    Bind uid 120003's payoff to its EXPIRY discount row instead of its
    SETTLEMENT row. Economically this discounts a deferred payoff from the wrong
    date; the oracle must see it.
    """
    portfolio, state, mapper = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    i = o.trade_uids.tolist().index(120003)
    fk_row = int(o.forward_price_strike_ccy_uix[i])
    expiry_edf_row = int(mapper.fk_effective_pairs[fk_row, 1])

    wrong = o.effective_discount_factor_uix.copy()
    wrong[i] = expiry_edf_row
    broken = dataclasses.replace(
        portfolio,
        vanilla_european_option=dataclasses.replace(
            o, effective_discount_factor_uix=np.ascontiguousarray(wrong)
        ),
    )
    out = np.empty((broken.total_trades, state.batch_size), dtype=np.float64, order="C")
    run_valuation(portfolio=broken, valuation_state=state, out=out)

    exp = _oracle_pv()
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        out[portfolio.trade_uids.tolist().index(120003)],
        exp[120003],
        rtol=RTOL_CDF,
    )


def test_option_block_row_permutation_is_detected():
    portfolio, state, _ = _compile()
    o = portfolio.vanilla_european_option
    assert o is not None, "Vanilla European option must not be None"
    idx = np.array([2, 0, 1])
    permuted = dataclasses.replace(
        portfolio,
        vanilla_european_option=dataclasses.replace(
            o,
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

    exp = _oracle_pv()
    expected = np.stack([exp[u] for u in portfolio.trade_uids.tolist()])
    assert not np.allclose(out, expected, rtol=1e-4, atol=1e-10)


def test_wrong_as_of_reprices_every_option():
    """
    The as-of must be the one `build_market_data_matrix` used. A mismatch is a
    wiring bug the compiler cannot detect, so prove the fixture is identifying
    for it.
    """
    _uids, correct = _production_pv()
    _, drifted = _production_pv(as_of=of.AS_OF_ORD - 30)
    assert not np.allclose(correct[1:], drifted[1:], rtol=1e-6)
    np.testing.assert_allclose(correct[0], drifted[0], rtol=RTOL_LINEAR)
