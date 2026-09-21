# tests/valuation/oracle/test_option_delivery_independence.py
"""
A NODE European option's value does not depend on `delivery_ord`.

The option contract is expiry (forward measure, volatility, TTE) plus settlement
(payoff discount). Delivery never enters the Black-76 value and never selects
the underlying market point, which is why ingest stopped requiring it (D065,
option half).

This drives the REAL compiler, fold and kernel on the production-path option
fixture, and reconciles both variants against the independent economic oracle.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from riskquant.constants import NO_DATE_ORD32
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.valuation.engine import run_valuation

from . import economic_oracle as oracle
from . import option_fixture as of

# D046: the production path uses the Abramowitz-Stegun CDF, the oracle SciPy.
# Same band as `test_option_production_path.py` for this fixture's |d2| span.
RTOL_CDF = 5e-5


def _price(batch):
    mapper = build_uix_mapper([batch], as_of_ord32=of.AS_OF_ORD)
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    state = assemble_valuation_state(
        matrix=of.build_market_matrix(mapper), uix_mapper=mapper
    )
    out = np.empty((portfolio.total_trades, of.N_SCENARIOS), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return dict(zip(portfolio.trade_uids.tolist(), out, strict=True)), portfolio


def _oracle_pvs():
    market = of.economic_market()
    pv = {}
    for trade in of.economic_book():
        if isinstance(trade, oracle.ForwardTrade):
            pv[trade.trade_uid] = oracle.pv_forward(trade, market, of.SETTLE_CCY)
        elif isinstance(trade, oracle.CrossCurrencyOptionTrade):
            pv[trade.trade_uid] = oracle.pv_cross_currency_option(
                trade, market, of.SETTLE_CCY
            )
    return pv


def _without_option_delivery(batch):
    """Erase delivery on the OPTION rows only; the forward row keeps its own."""
    from riskquant.portfolio.instruments import VANILLA_EUROPEAN_OPTION_NODE_ID

    delivery = batch.delivery_ord.copy()
    option_rows = batch.instrument_ids == np.int32(VANILLA_EUROPEAN_OPTION_NODE_ID)
    assert bool(option_rows.any())
    delivery[option_rows] = np.int32(NO_DATE_ORD32)
    return dataclasses.replace(
        batch, delivery_ord=np.ascontiguousarray(delivery)
    ), option_rows


def test_option_value_is_identical_without_delivery():
    batch = of.ingest_batch()
    base_pv, base_portfolio = _price(batch)
    stripped, option_rows = _without_option_delivery(batch)
    stripped_pv, stripped_portfolio = _price(stripped)

    option_uids = batch.uids[option_rows].tolist()
    assert set(option_uids) <= set(base_pv)
    for uid in base_pv:
        np.testing.assert_array_equal(stripped_pv[uid], base_pv[uid])

    a = base_portfolio.vanilla_european_option
    b = stripped_portfolio.vanilla_european_option
    for field in ("qty", "strike", "log_strike", "cp_mult", "sqrt_tte", "half_tte"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_both_variants_reconcile_to_the_independent_oracle():
    expected = _oracle_pvs()
    for batch in (of.ingest_batch(), _without_option_delivery(of.ingest_batch())[0]):
        produced, _ = _price(batch)
        for uid, want in expected.items():
            np.testing.assert_allclose(produced[uid], want, rtol=RTOL_CDF)
