# tests/valuation/oracle/test_end_to_end_oracle.py
"""
Phase 3 acceptance: independent economic oracle vs the production valuation path.

    economic identity -> market state -> run_valuation -> PV
                      \
                       -> independent scalar mathematics -> expected PV

The oracle imports no riskquant symbol (asserted below). The fixture is
hostile by construction: non-ascending sparse trade uids, sparse non-ascending
factor ids where id != row position for every factor, three currencies, an
inverse-quoted FX pair, four distinct non-unit discount factors, mixed signs
and two block types that each bind four separate factor universes.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.valuation.engine import run_valuation

from . import economic_oracle as oracle
from . import fixture as fx

# D046: Abramowitz-Stegun is the accepted price-path CDF. Near-the-money
# relative error is <= 6e-6; these options sit at |d2| < 2. Spot and forward
# carry no CDF and are held to floating-point noise only.
TOL_CDF_RTOL = 1e-5
TOL_LINEAR_RTOL = 1e-13


def _production_pv() -> tuple[np.ndarray, np.ndarray]:
    portfolio = fx.compile_book()
    state = fx.compile_state(fx.economic_market())
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return portfolio.trade_uids, out


def _oracle_pv() -> dict[int, np.ndarray]:
    m = fx.economic_market()
    pv: dict[int, np.ndarray] = {}
    for t in fx.economic_book():
        if isinstance(t, oracle.SpotTrade):
            pv[t.trade_uid] = oracle.pv_spot(t, m, fx.SETTLE_CCY)
        elif isinstance(t, oracle.ForwardTrade):
            pv[t.trade_uid] = oracle.pv_forward(t, m, fx.SETTLE_CCY)
        elif isinstance(t, oracle.OptionTrade):
            pv[t.trade_uid] = oracle.pv_option(t, m, fx.SETTLE_CCY)
        else:  # pragma: no cover
            raise TypeError(f"unknown economic trade type: {type(t)!r}")
    return pv


def test_oracle_is_structurally_independent():
    import ast
    import pathlib

    src = pathlib.Path(oracle.__file__).read_text()
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert not any(m.startswith("riskquant") for m in mods), mods


def test_end_to_end_pv_matches_independent_oracle():
    uids, prod = _production_pv()
    exp = _oracle_pv()
    tol = {
        9007: TOL_LINEAR_RTOL,
        41: TOL_LINEAR_RTOL,
        120003: TOL_CDF_RTOL,
        5: TOL_CDF_RTOL,
    }
    for row, uid in enumerate(uids.tolist()):
        np.testing.assert_allclose(
            prod[row],
            exp[uid],
            rtol=tol[uid],
            atol=0.0,
            err_msg=f"row {row} / trade {uid}",
        )


def test_row_identity_binds_to_originating_trade():
    """PV[row] must belong to trade_uids[row], not to some other row."""
    uids, prod = _production_pv()
    exp = _oracle_pv()
    assert uids.tolist() == [9007, 41, 120003, 5]
    for row, uid in enumerate(uids.tolist()):
        others = [u for u in exp if u != uid]
        for other in others:
            assert not np.allclose(prod[row], exp[other], rtol=1e-3), (
                f"row {row} (uid {uid}) is indistinguishable from uid {other}; "
                "the fixture is not identifying"
            )


# =========================================================================
# MUTATION EVIDENCE -- prove the fixture can detect the defect classes
# =========================================================================


def _pv_with(state_mutator=None, book_mutator=None) -> np.ndarray:
    portfolio = fx.compile_book()
    state = fx.compile_state(fx.economic_market())
    if state_mutator is not None:
        state = state_mutator(state)
    if book_mutator is not None:
        portfolio = book_mutator(portfolio)
    out = np.empty(
        (portfolio.total_trades, state.batch_size), dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return out


def _swap_rows(a: np.ndarray, i: int, j: int) -> np.ndarray:
    b = a.copy()
    b[[i, j]] = b[[j, i]]
    return np.ascontiguousarray(b)


def _replace(state, **kw):
    import dataclasses

    return dataclasses.replace(state, **kw)


MUTATIONS = {
    "spot factor rows permuted": (
        lambda s: _replace(
            s, effective_spot_prices=_swap_rows(s.effective_spot_prices, 0, 2)
        ),
        None,
    ),
    "discount factor row substituted": (
        lambda s: _replace(
            s, effective_discount_factors=_swap_rows(s.effective_discount_factors, 2, 3)
        ),
        None,
    ),
    "effective forward rows permuted": (
        lambda s: _replace(
            s, effective_forward_prices=_swap_rows(s.effective_forward_prices, 0, 1)
        ),
        None,
    ),
    "vol rows permuted": (lambda s: _replace(s, vols=_swap_rows(s.vols, 0, 1)), None),
    "strike-ccy forward rows permuted": (
        lambda s: _replace(
            s, forward_prices_strike_ccy=_swap_rows(s.forward_prices_strike_ccy, 0, 1)
        ),
        None,
    ),
    "FX direction inverted on the settlement leg": (
        lambda s: _replace(
            s,
            effective_spot_prices=np.ascontiguousarray(
                s.effective_spot_prices * (1.0 / 0.91370) ** 2
            ),
        ),
        None,
    ),
    "option block rows permuted": (None, lambda p: _permute_option_rows(p)),
}


def _permute_option_rows(p):
    import dataclasses

    o = p.vanilla_european_option
    idx = np.array([1, 0])
    swapped = dataclasses.replace(
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
    )
    return dataclasses.replace(p, vanilla_european_option=swapped)


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_is_detected_against_the_oracle(name):
    """
    Each mutation must make production disagree with the INDEPENDENTLY computed
    economic expectation -- not merely differ from the unmutated run.
    """
    sm, bm = MUTATIONS[name]
    mutated = _pv_with(state_mutator=sm, book_mutator=bm)
    uids, _ = _production_pv()
    exp = _oracle_pv()
    expected = np.stack([exp[u] for u in uids.tolist()])
    assert not np.allclose(mutated, expected, rtol=1e-6, atol=1e-12), (
        f"mutation '{name}' was NOT detected -- the fixture is not identifying "
        "for this defect class"
    )
