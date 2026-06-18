# tests/market_data/calibrate/option_quotes/test_option_quotes.py
"""
Tests for the option-quotes edge: no-arbitrage validation and canonical
compilation (put-call parity / undiscounting, trust-the-master dedup).

Plain-assert, pytest-discoverable, or standalone via ``python test_option_quotes.py``.
"""

from __future__ import annotations

import numpy as np

from riskquant.market_data.calibrate.option_quotes.compile import (
    _compile_canonical_option_quotes_arrays,
    compile_canonical_option_quotes,
)
from riskquant.market_data.calibrate.option_quotes.master import (
    CanonicalOptionQuotes,
    ValidatedOptionQuotes,
)
from riskquant.market_data.calibrate.option_quotes.validate import (
    validate_option_quotes,
)

_CALL = 1
_PUT = 0


def _assert_raises(fn, substr):
    try:
        fn()
    except ValueError as e:
        assert substr in str(e), f"wrong message: {e!r}"
        return
    raise AssertionError(f"expected ValueError containing {substr!r}")


# --------------------------------------------------------------- validation
def test_validate_accepts_valid_calls_and_puts():
    F = np.array([100.0, 100.0, 100.0])
    K = np.array([100.0, 80.0, 120.0])
    T = np.array([1.0, 1.0, 1.0])
    flag = np.array([_CALL, _CALL, _PUT], dtype=np.int8)
    df = np.array([0.98, 0.98, 0.98])
    # all strictly inside the no-arbitrage band (row 2 is an ITM put,
    # intrinsic = K - F = 20, so its price must exceed ~df*20 = 19.6)
    price = np.array([10.0, 25.0, 25.0])

    out = validate_option_quotes(F, K, T, flag, df, price)
    assert isinstance(out, ValidatedOptionQuotes)
    assert out.n_quotes == 3
    assert out.fwd_price.dtype == np.float64
    assert out.call_put_flag.dtype == np.int8


def test_validate_rejects_call_above_forward():
    _assert_raises(
        lambda: validate_option_quotes(
            [100.0], [100.0], [1.0], np.array([_CALL], np.int8), [0.98], [99.0]
        ),
        "upper no-arbitrage",
    )


def test_validate_rejects_call_below_intrinsic():
    # ITM call: discounted intrinsic = 0.98 * (100 - 80) = 19.6; price 10 < that
    _assert_raises(
        lambda: validate_option_quotes(
            [100.0], [80.0], [1.0], np.array([_CALL], np.int8), [0.98], [10.0]
        ),
        "lower no-arbitrage",
    )


def test_validate_rejects_put_above_strike():
    # put upper bound = df * K = 98; price 99 > that
    _assert_raises(
        lambda: validate_option_quotes(
            [100.0], [100.0], [1.0], np.array([_PUT], np.int8), [0.98], [99.0]
        ),
        "upper no-arbitrage",
    )


def test_validate_rejects_put_below_intrinsic():
    # ITM put F<K: discounted intrinsic = 0.98 * (100 - 80) = 19.6; price 10 < that
    _assert_raises(
        lambda: validate_option_quotes(
            [80.0], [100.0], [1.0], np.array([_PUT], np.int8), [0.98], [10.0]
        ),
        "lower no-arbitrage",
    )


def test_validate_accepts_boundaries():
    # exactly at discounted intrinsic (lower) and at df*F / df*K (upper)
    F = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    flag = np.array([_CALL, _PUT], dtype=np.int8)
    df = np.array([0.98, 0.98])
    price = np.array([0.0, 98.0])  # ATM call at intrinsic 0; put at df*K
    out = validate_option_quotes(F, K, T, flag, df, price)
    assert out.n_quotes == 2


def test_validate_error_lists_offending_rows():
    # rows 1 and 3 violate the upper bound
    F = np.array([100.0, 100.0, 100.0, 100.0])
    K = np.array([100.0, 100.0, 100.0, 100.0])
    T = np.array([1.0, 1.0, 1.0, 1.0])
    flag = np.array([_CALL, _CALL, _CALL, _CALL], dtype=np.int8)
    df = np.array([0.98, 0.98, 0.98, 0.98])
    price = np.array([10.0, 99.0, 10.0, 99.0])
    try:
        validate_option_quotes(F, K, T, flag, df, price)
    except ValueError as e:
        msg = str(e)
        assert "2 quote(s)" in msg, msg
        assert "1" in msg and "3" in msg, msg
        return
    raise AssertionError("expected ValueError")


# --------------------------------------------------------------- compilation
def _expected_undisc_call(F, K, flag, df, price):
    undisc = price / df
    return np.where(flag == _CALL, undisc, undisc + F - K)


def test_compile_putcall_parity_and_undiscount():
    F = np.array([100.0, 100.0, 50.0])
    K = np.array([100.0, 120.0, 40.0])
    T = np.array([1.0, 0.5, 2.0])
    flag = np.array([_CALL, _PUT, _PUT], dtype=np.int8)
    df = np.array([0.98, 0.97, 0.95])
    price = np.array([10.0, 18.0, 12.0])

    quotes = ValidatedOptionQuotes(
        fwd_price=F,
        strike=K,
        tte=T,
        call_put_flag=flag,
        discount_factor=df,
        option_price=price,
    )
    out = compile_canonical_option_quotes(quotes)
    assert isinstance(out, CanonicalOptionQuotes)
    expected = _expected_undisc_call(F, K, flag, df, price)
    assert np.allclose(out.undiscounted_call_price, expected, rtol=0, atol=1e-12)
    # canonical coordinates are passed through unchanged
    assert np.array_equal(out.fwd_price, F)
    assert np.array_equal(out.strike, K)
    assert np.array_equal(out.tte, T)


def test_compile_trust_master_matches_raw_seam():
    F = np.array([100.0, 100.0])
    K = np.array([100.0, 120.0])
    T = np.array([1.0, 0.5])
    flag = np.array([_CALL, _PUT], dtype=np.int8)
    df = np.array([0.98, 0.97])
    price = np.array([10.0, 18.0])

    quotes = ValidatedOptionQuotes(
        fwd_price=F,
        strike=K,
        tte=T,
        call_put_flag=flag,
        discount_factor=df,
        option_price=price,
    )
    via_master = compile_canonical_option_quotes(quotes)
    via_raw = _compile_canonical_option_quotes_arrays(F, K, T, flag, df, price)
    assert np.array_equal(
        via_master.undiscounted_call_price, via_raw.undiscounted_call_price
    )


def test_compile_arrays_seam_normalizes_dtype_and_layout():
    # non-contiguous, wrong dtype raw inputs must be normalized by the seam
    F = np.asfortranarray(np.array([[100.0, 100.0]]))[0]  # non-C-contiguous view
    K = np.array([100, 120], dtype=np.int64)  # wrong dtype
    T = np.array([1.0, 0.5])
    flag = np.array([1, 0], dtype=np.int64)  # wrong dtype
    df = np.array([0.98, 0.97])
    price = np.array([10.0, 18.0])

    out = _compile_canonical_option_quotes_arrays(F, K, T, flag, df, price)
    assert out.undiscounted_call_price.dtype == np.float64
    assert out.fwd_price.flags["C_CONTIGUOUS"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
