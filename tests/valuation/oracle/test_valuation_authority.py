# tests/valuation/oracle/test_valuation_authority.py
"""
Valuation audit: the canonical route as the AUTHORITATIVE producer of V_0 / V_s.

Hostile fixture: `whole_surface_fixture` (all three production blocks - SPOT,
VANILLA FORWARD (NODE), VANILLA EUROPEAN OPTION (NODE) - on non-contiguous
factor ids, three currencies with a reverse-quoted pair, a multi-row UID
spanning two blocks, three scenarios). Oracle: `economic_oracle` (pure scalar
Python/scipy; never imports production pricing). These tests add the output,
buffer, scenario-order, sign, dispatch and as-of properties the existing
oracle suite does not pin.
"""

import dataclasses

import numpy as np
import pytest

import riskquant.valuation.engine as engine
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.valuation.engine import run_valuation

from . import whole_surface_fixture as ws
from .test_whole_surface import _compile, _expected_matrix, _tol

POISON = np.nan


def _poisoned(portfolio, state):
    out = np.full(
        (portfolio.total_trades, state.batch_size), POISON, dtype=np.float64, order="C"
    )
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return out


def _assert_matches_oracle(out) -> None:
    want = _expected_matrix()
    for r in range(out.shape[0]):
        np.testing.assert_allclose(
            out[r], want[r], rtol=_tol(r), atol=0.0, err_msg=f"row {r}"
        )


def test_every_output_cell_is_overwritten() -> None:
    """No reliance on pre-zeroed memory: a NaN-poisoned buffer is fully written."""
    portfolio, state, _ = _compile()
    out = _poisoned(portfolio, state)
    assert np.isfinite(out).all()
    _assert_matches_oracle(out)


def test_the_overwrite_check_detects_an_unwritten_block(monkeypatch) -> None:
    """Negative control: a pricer that writes nothing leaves the poison visible."""
    portfolio, state, _ = _compile()
    fwd = portfolio.vanilla_forward
    pricers = list(engine.BLOCK_PRICERS)
    pricers[fwd.BLOCK_TYPE_ID] = lambda **kw: None
    monkeypatch.setattr(engine, "BLOCK_PRICERS", tuple(pricers))
    out = _poisoned(portfolio, state)
    rows = np.flatnonzero(np.isin(portfolio.trade_uids, fwd.trade_uids))
    assert (
        np.isnan(out[rows]).all() and not np.isnan(np.delete(out, rows, axis=0)).any()
    )


def test_a_missing_pricer_fails_closed(monkeypatch) -> None:
    portfolio, state, _ = _compile()
    pricers = list(engine.BLOCK_PRICERS)
    pricers[portfolio.vanilla_european_option.BLOCK_TYPE_ID] = None
    monkeypatch.setattr(engine, "BLOCK_PRICERS", tuple(pricers))
    with pytest.raises(TypeError):
        _poisoned(portfolio, state)


def test_buffer_reuse_and_repetition_are_bitwise_stable() -> None:
    portfolio, state, _ = _compile()
    first = _poisoned(portfolio, state)
    buf = np.full_like(first, 7.77e300)
    for _ in range(3):
        run_valuation(portfolio=portfolio, valuation_state=state, out=buf)
        assert np.array_equal(buf, first)


def test_scenario_permutation_permutes_output_columns_only(monkeypatch) -> None:
    """Each scenario column is valued independently: permuting every per-scenario
    market input permutes the output columns, bitwise, and nothing else."""
    portfolio, state, _ = _compile()
    base = _poisoned(portfolio, state)
    perm = np.array([2, 0, 1])
    names = [
        n
        for n, v in vars(ws).items()
        if isinstance(v, np.ndarray) and v.shape == (ws.N_SCENARIOS,)
    ]
    assert {
        "FX_USD_EUR",
        "FX_EUR_GBP",
        "LVL_GOLD",
        "VOL_SILVER",
        "DF_GBP_SETTLE_LATE",
    } <= set(names)
    for n in names:
        monkeypatch.setattr(ws, n, getattr(ws, n)[perm])
    portfolio_p, state_p, _ = _compile()
    assert np.array_equal(_poisoned(portfolio_p, state_p), base[:, perm])


def _with_qty(portfolio, factor):
    fields = {
        f.name: getattr(portfolio, f.name)
        for f in dataclasses.fields(portfolio)
        if f.init
    }
    for name, block in list(fields.items()):
        if block is not None and hasattr(block, "qty"):
            fields[name] = dataclasses.replace(
                block, qty=np.ascontiguousarray(block.qty * factor)
            )
    return CompiledPortfolio(**fields)


def test_quantity_sign_and_zero_are_exact() -> None:
    """V is linear in qty for every block: reversal negates exactly, zero is zero."""
    portfolio, state, _ = _compile()
    base = _poisoned(portfolio, state)
    assert np.array_equal(_poisoned(_with_qty(portfolio, -1.0), state), -base)
    assert (_poisoned(_with_qty(portfolio, 0.0), state) == 0.0).all()


def test_as_of_is_load_bearing_and_bound_consistently(monkeypatch) -> None:
    """Moving the mapper as-of moves exactly the time-dependent rows (option
    Black-76 time to expiry), and the new values still equal the oracle at the
    new as-of: valuation and oracle read the same date. The shift is -30 days so
    every |d| stays inside this fixture's certified D046 band (|d2| <= 2.32);
    at +30 days one option reaches |d| ~ 2.4+ and its price error (5.0e-5
    relative) follows the documented D046 CDF envelope, not an as-of defect."""
    portfolio, state, _ = _compile()
    before = _poisoned(portfolio, state)
    monkeypatch.setattr(ws, "AS_OF_ORD", ws.AS_OF_ORD - 30)
    portfolio2, state2, mapper2 = _compile()
    assert mapper2.as_of_ord32 == ws.AS_OF_ORD
    after = _poisoned(portfolio2, state2)
    _assert_matches_oracle(after)
    offset = portfolio.block_offsets[
        list(portfolio.blocks).index(portfolio.vanilla_european_option)
    ]
    option_rows = np.arange(offset, offset + portfolio.vanilla_european_option.size)
    moved = np.abs(after - before).max(axis=1)
    assert (moved[option_rows] > 1e-6).all()
    assert (moved[np.setdiff1d(np.arange(before.shape[0]), option_rows)] == 0.0).all()
