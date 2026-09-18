# tests/valuation/oracle/test_forward_production_path.py
"""
Phase 4 acceptance: a REAL forward, compiled by the REAL compiler, priced by the
REAL engine, against the independent economic oracle.

The Phase-3 suite hand-built its blocks. Nothing here does: the
`VanillaForwardBlock` and every UIX it carries are produced by
`build_uix_mapper` and `compile_portfolio_from_ingest_batch` from an
`IngestBatch` of global ids, and the factor rows are produced by the real
`assemble_valuation_state` fold.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from oracle import economic_oracle as oracle
from oracle import forward_fixture as ff

from riskquant.market_data.compile import assemble_valuation_state
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.valuation.blocks.vanilla_forward_block import VanillaForwardBlock
from riskquant.valuation.engine import run_valuation

RTOL = 1e-13


def _compile():
    batch = ff.ingest_batch()
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    matrix = ff.build_market_matrix(mapper)
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    return portfolio, state


def _production_pv():
    portfolio, state = _compile()
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return portfolio.trade_uids, out


def _oracle_pv() -> dict[int, np.ndarray]:
    m = ff.economic_market()
    pv: dict[int, np.ndarray] = {}
    for t in ff.economic_book():
        if isinstance(t, oracle.SpotTrade):
            pv[t.trade_uid] = oracle.pv_spot(t, m, ff.SETTLE_CCY)
        elif isinstance(t, oracle.ForwardTrade):
            pv[t.trade_uid] = oracle.pv_forward(t, m, ff.SETTLE_CCY)
        else:  # pragma: no cover
            raise TypeError(f"unknown economic trade type: {type(t)!r}")
    return pv


# =========================================================================
# REACHABILITY -- the Phase-4 difference
# =========================================================================


def test_forward_is_reachable_through_the_production_compiler():
    portfolio, _ = _compile()
    assert isinstance(portfolio.vanilla_forward, VanillaForwardBlock)
    assert portfolio.vanilla_forward.size == 3


def test_compiled_uids_preserve_hostile_ingest_ordering():
    portfolio, _ = _compile()
    assert portfolio.trade_uids.tolist() == [9007, 41, 120003, 5]
    assert portfolio.trade_uids.tolist() != sorted(portfolio.trade_uids.tolist())


def test_the_fold_populates_both_forward_factor_universes():
    _, state = _compile()
    assert state.effective_forward_prices.shape[0] > 0
    assert state.effective_discount_factors.shape[0] > 0
    assert state.effective_forward_prices.shape[1] == ff.N_SCENARIOS


# =========================================================================
# ACCEPTANCE
# =========================================================================


def test_production_forward_matches_the_independent_oracle():
    uids, prod = _production_pv()
    exp = _oracle_pv()
    for row, uid in enumerate(uids.tolist()):
        np.testing.assert_allclose(
            prod[row], exp[uid], rtol=RTOL, atol=0.0, err_msg=f"row {row} / uid {uid}"
        )


def test_cross_currency_forward_legs_are_independently_correct():
    """uid 41: USD asset leg, GBP strike leg, EUR settlement. Legs proved apart."""
    m = ff.economic_market()
    trade = next(t for t in ff.economic_book() if getattr(t, "trade_uid", None) == 41)

    asset_leg = (
        trade.qty
        * m.forward_native[trade.forward_factor_id]
        * m.discount[trade.asset_discount_factor_id]
        * m.fx_rate("USD", "EUR")
    )
    strike_leg = (
        trade.qty
        * trade.strike
        * m.discount[trade.strike_discount_factor_id]
        * m.fx_rate("GBP", "EUR")
    )

    uids, prod = _production_pv()
    row = uids.tolist().index(41)
    np.testing.assert_allclose(prod[row], asset_leg - strike_leg, rtol=RTOL)

    # The legs must actually differ, or the fixture proves nothing about binding.
    assert not np.allclose(
        m.discount[trade.asset_discount_factor_id],
        m.discount[trade.strike_discount_factor_id],
    )
    assert not np.allclose(m.fx_rate("USD", "EUR"), m.fx_rate("GBP", "EUR"))


def test_single_currency_forward_degenerates_to_q_times_f_minus_k_times_df():
    """
    uid 120003: one curve, one date, one currency on both legs, so the two-leg
    form must collapse to Q (F - K) DF * FX. Proved as an oracle result, not as
    an asserted algebraic identity.
    """
    m = ff.economic_market()
    trade = next(
        t for t in ff.economic_book() if getattr(t, "trade_uid", None) == 120003
    )
    assert trade.asset_discount_factor_id == trade.strike_discount_factor_id
    assert trade.asset_ccy == trade.strike_ccy

    df = m.discount[trade.asset_discount_factor_id]
    fx = m.fx_rate(trade.asset_ccy, ff.SETTLE_CCY)
    degenerate = (
        trade.qty * (m.forward_native[trade.forward_factor_id] - trade.strike) * df * fx
    )

    uids, prod = _production_pv()
    np.testing.assert_allclose(prod[uids.tolist().index(120003)], degenerate, rtol=RTOL)


def test_identity_fx_strike_leg_is_not_converted():
    """uid 5: EUR strike into EUR settlement. `NO_FX_RF_REQUIRED` must mean 1.0."""
    uids, prod = _production_pv()
    exp = _oracle_pv()
    np.testing.assert_allclose(prod[uids.tolist().index(5)], exp[5], rtol=RTOL)


def test_spot_block_is_unaffected_by_the_forward_compiler():
    uids, prod = _production_pv()
    exp = _oracle_pv()
    np.testing.assert_allclose(prod[uids.tolist().index(9007)], exp[9007], rtol=RTOL)


# =========================================================================
# FAIL-FAST (VAL-04, forward path)
# =========================================================================


def test_unresolved_market_leg_is_rejected_by_the_compiler():
    """
    A forward whose underlying is absent from the run's effective-forward
    universe must be refused AT THE COMPILER, not gathered out of range inside
    a boundscheck-free kernel.
    """
    batch = ff.ingest_batch()
    mapper = build_uix_mapper([batch])
    starved = dataclasses.replace(
        mapper, efp_effective_quads=np.empty((0, 4), dtype=np.int32)
    )
    with pytest.raises(ValueError, match="effective_forward_price"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=starved)


def test_unresolved_discount_leg_is_rejected_by_the_compiler():
    batch = ff.ingest_batch()
    mapper = build_uix_mapper([batch])
    starved = dataclasses.replace(
        mapper, edf_effective_triples=np.empty((0, 3), dtype=np.int32)
    )
    with pytest.raises(ValueError, match="effective_discount_factor"):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=starved)


def test_mapper_built_from_a_different_batch_is_rejected():
    """The effective tables are per-run; a foreign mapper must not silently bind."""
    batch = ff.ingest_batch()
    other = dataclasses.replace(
        batch,
        underlying_ids=np.ascontiguousarray(
            np.full(batch.size, ff.RF_BRENT, dtype=np.int32)
        ),
    )
    foreign = build_uix_mapper([other])
    with pytest.raises((RuntimeError, ValueError)):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=foreign)


# =========================================================================
# MUTATION EVIDENCE
# =========================================================================


def _swap(a: np.ndarray, i: int, j: int) -> np.ndarray:
    b = a.copy()
    b[[i, j]] = b[[j, i]]
    return np.ascontiguousarray(b)


MUTATIONS = {
    "asset forward row permutation": lambda s: dataclasses.replace(
        s, effective_forward_prices=_swap(s.effective_forward_prices, 0, 1)
    ),
    "strike discount row substitution": lambda s: dataclasses.replace(
        s, effective_discount_factors=_swap(s.effective_discount_factors, 0, 1)
    ),
    "asset FX substitution": lambda s: dataclasses.replace(
        s,
        effective_forward_prices=np.ascontiguousarray(
            s.effective_forward_prices * 1.17
        ),
    ),
    "strike FX substitution": lambda s: dataclasses.replace(
        s,
        effective_discount_factors=np.ascontiguousarray(
            s.effective_discount_factors * 1.17
        ),
    ),
    "wrong FX inversion on the asset leg": lambda s: dataclasses.replace(
        s,
        effective_forward_prices=np.ascontiguousarray(
            s.effective_forward_prices / (0.91370**2)
        ),
    ),
    "spot row permutation": lambda s: dataclasses.replace(
        s, effective_spot_prices=_swap(s.effective_spot_prices, 0, 1)
    ),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_disagrees_with_the_independent_oracle(name):
    portfolio, state = _compile()
    if name == "spot row permutation" and state.effective_spot_prices.shape[0] < 2:
        pytest.skip("single spot row: permutation is not expressible")
    mutated = MUTATIONS[name](state)
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=mutated, out=out)

    exp = _oracle_pv()
    expected = np.stack([exp[u] for u in portfolio.trade_uids.tolist()])
    assert not np.allclose(out, expected, rtol=1e-9, atol=1e-12), (
        f"mutation '{name}' was NOT detected against the independent oracle"
    )


def test_block_row_permutation_is_detected():
    """Permuting the compiled block's rows must break the uid <-> PV binding."""
    portfolio, state = _compile()
    f = portfolio.vanilla_forward
    assert f is not None, "Vanilla forward must not be None"
    idx = np.array([2, 0, 1])
    permuted = dataclasses.replace(
        portfolio,
        vanilla_forward=dataclasses.replace(
            f,
            qty=np.ascontiguousarray(f.qty[idx]),
            strike=np.ascontiguousarray(f.strike[idx]),
            effective_forward_price_uix=np.ascontiguousarray(
                f.effective_forward_price_uix[idx]
            ),
            effective_discount_factor_uix=np.ascontiguousarray(
                f.effective_discount_factor_uix[idx]
            ),
        ),
    )
    out = np.empty(
        (permuted.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=permuted, valuation_state=state, out=out)

    exp = _oracle_pv()
    expected = np.stack([exp[u] for u in portfolio.trade_uids.tolist()])
    assert not np.allclose(out, expected, rtol=1e-9, atol=1e-12)
