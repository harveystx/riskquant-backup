# tests/valuation/test_result_container.py
"""
`ValuationResult` row/trade cardinality contract (VAL-05).

The block model permits one economic trade to produce many block rows
(`VANILLA_IRS -> FIXED_CASHFLOW + FLOATING_CASHFLOW`), and `CompiledPortfolio`
concatenates uids in block-slot order. So `trade_uids` is neither unique nor
sorted, and the container must not pretend otherwise.

Primary fixture uid set is deliberately hostile: [9007, 41, 120003, 5].
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.valuation.result import ValuationResult

UIDS = np.array([9007, 41, 120003, 5], dtype=np.int64)
VALUES = np.array([[9.007, 90.07], [4.1, 41.0], [12.0003, 120.003], [0.5, 5.0]])


def _r() -> ValuationResult:
    return ValuationResult(trade_uids=UIDS.copy(), values=VALUES.copy())


def test_fixture_is_hostile():
    assert UIDS.tolist() != sorted(UIDS.tolist()), (
        "identity fixture must be non-ascending"
    )
    assert len(set(UIDS.tolist())) == len(UIDS), "distinct uids in the primary fixture"


@pytest.mark.parametrize("row,uid", list(enumerate(UIDS.tolist())))
def test_every_present_uid_resolves_to_its_own_row(row, uid):
    np.testing.assert_array_equal(_r().get_trade_values(uid), VALUES[row : row + 1])


def test_absent_uid_raises_key_error():
    with pytest.raises(KeyError):
        _r().get_trade_values(999_999)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        ValuationResult(trade_uids=UIDS.copy(), values=VALUES[:2].copy())


def test_num_rows_and_num_scenarios():
    r = _r()
    assert r.num_rows == 4
    assert r.num_scenarios == 2


def test_select_preserves_order_and_rows():
    r = _r().select(np.array([120003, 9007], dtype=np.int64))
    assert r.trade_uids.tolist() == [9007, 120003]
    np.testing.assert_array_equal(r.values, VALUES[[0, 2]])


# =========================================================================
# DECOMPOSED INSTRUMENT: one trade, many rows
# =========================================================================

DUP_UIDS = np.array([77, 77, 41, 77], dtype=np.int64)
DUP_VALUES = np.array([[1.0, 10.0], [2.0, 20.0], [100.0, 1000.0], [4.0, 40.0]])


def _dup() -> ValuationResult:
    return ValuationResult(trade_uids=DUP_UIDS.copy(), values=DUP_VALUES.copy())


def test_decomposed_trade_returns_every_row_it_produced():
    got = _dup().get_trade_values(77)
    assert got.shape == (3, 2)
    np.testing.assert_array_equal(got, DUP_VALUES[[0, 1, 3]])


def test_trade_pv_aggregates_the_rows():
    np.testing.assert_allclose(
        _dup().get_trade_pv(77), DUP_VALUES[[0, 1, 3]].sum(axis=0)
    )
    np.testing.assert_allclose(_dup().get_trade_pv(41), DUP_VALUES[2])


def test_num_trades_counts_distinct_trades_not_rows():
    r = _dup()
    assert r.num_rows == 4
    assert r.num_trades == 2


def test_container_equality_and_hashing_do_not_raise():
    """`eq=False` (VAL-09): array-bearing dataclasses need identity semantics."""
    a, b = _r(), _r()
    assert (a == b) is False
    assert a == a
    assert isinstance(hash(a), int)
    assert a in [a]
