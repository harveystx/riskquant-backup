# tests/valuation/test_black76.py
"""
Current Black-76 API. Replaces the deleted `tests/valuation/models/test_black_scholes.py`,
which imported `riskquant.valuation.models.black_scholes` -- a package that no
longer exists.

Expected values come from `scipy.stats.norm`, never from the production
`numba_norm_cdf_abrom_steg`. Tolerances are the D046 measured envelope.
"""

from __future__ import annotations

import itertools
import math

import pytest
from scipy.stats import norm

from riskquant.valuation.pricing_models.black76.model import (
    _black76_fv,
    _black76_fv_call,
    _black76_fv_put,
)

# D046: relative price error <= 6e-6 for |d2| <~ 2; grows as the price collapses.
RTOL_NEAR_MONEY = 1e-5
RTOL_MODERATE_TAIL = 1e-3  # |d2| ~ 3-4
ATOL_PER_100_FWD = 2.0e-5  # D046 absolute bound per 100 of forward


def ref_fv(fwd: float, strike: float, vol: float, tte: float, cp: int) -> float:
    vol_sqrt_tte = vol * math.sqrt(tte)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * tte) / vol_sqrt_tte
    d2 = d1 - vol_sqrt_tte
    return cp * (fwd * float(norm.cdf(cp * d1)) - strike * float(norm.cdf(cp * d2)))


def call_impl(f, k, v, t):
    return _black76_fv_call(f, k, math.sqrt(t), 0.5 * t, v)


def put_impl(f, k, v, t):
    return _black76_fv_put(f, k, math.sqrt(t), 0.5 * t, v)


NEAR_MONEY = [
    (100.0, 100.0, 0.20, 1.0),
    (100.0, 90.0, 0.25, 0.5),  # ITM call
    (100.0, 115.0, 0.30, 2.0),  # OTM call
    (2410.55, 2500.0, 0.215, 1.0),
    (28.115, 30.0, 0.341, 0.5),
    (0.85, 0.80, 0.11, 0.25),  # small forward
    (4.2e6, 4.0e6, 0.18, 3.0),  # large forward
]


@pytest.mark.parametrize("f,k,v,t", NEAR_MONEY)
def test_call_matches_independent_reference(f, k, v, t):
    got, want = call_impl(f, k, v, t), ref_fv(f, k, v, t, 1)
    assert got == pytest.approx(
        want, rel=RTOL_NEAR_MONEY, abs=ATOL_PER_100_FWD * f / 100.0
    )


@pytest.mark.parametrize("f,k,v,t", NEAR_MONEY)
def test_put_matches_independent_reference(f, k, v, t):
    got, want = put_impl(f, k, v, t), ref_fv(f, k, v, t, -1)
    assert got == pytest.approx(
        want, rel=RTOL_NEAR_MONEY, abs=ATOL_PER_100_FWD * f / 100.0
    )


@pytest.mark.parametrize("f,k,v,t", NEAR_MONEY)
def test_put_call_parity_is_exact(f, k, v, t):
    """
    Phi_AS(x) + Phi_AS(-x) == 1 identically, so the approximation error cancels
    in C - P. This must hold to floating-point noise, not to the CDF envelope.
    """
    assert call_impl(f, k, v, t) - put_impl(f, k, v, t) == pytest.approx(
        f - k, rel=1e-13, abs=1e-9
    )


def test_cp_mult_dispatch_agrees_with_the_named_helpers():
    """
    `_black76_fv_call/_put` are thin `_black76_fv` wrappers, so they must agree
    to floating-point noise.

    NOT asserted bit-identical. Under `fastmath={"contract","reassoc","nsz"}`
    Numba specialises the inlined body differently depending on whether
    `cp_mult` arrives as a runtime int64 (direct call) or as a literal folded
    inside the wrapper, and the reassociated FMA sequence differs by 1 ULP on
    the put branch. Measured: 13.663205017578976 vs 13.66320501757898, a
    relative difference of 2.6e-16. Recorded as a numerical observation, not a
    defect -- but the wrappers are not interchangeable at the bit level.
    """
    f, k, v, t = 100.0, 105.0, 0.22, 1.5
    assert _black76_fv(f, k, math.sqrt(t), 0.5 * t, v, 1) == pytest.approx(
        call_impl(f, k, v, t), rel=4e-16, abs=0.0
    )
    assert _black76_fv(f, k, math.sqrt(t), 0.5 * t, v, -1) == pytest.approx(
        put_impl(f, k, v, t), rel=4e-16, abs=0.0
    )


def test_fv_is_undiscounted_future_value():
    """
    The primitive returns FV, not PV. Discounting is the caller's job, so the
    value must be independent of any discount factor -- there is nowhere to
    pass one.
    """
    fv = call_impl(100.0, 100.0, 0.20, 1.0)
    assert fv == pytest.approx(ref_fv(100.0, 100.0, 0.20, 1.0, 1), rel=RTOL_NEAR_MONEY)
    for df in (0.5, 0.9, 1.0):
        assert df * fv == pytest.approx(
            df * ref_fv(100.0, 100.0, 0.20, 1.0, 1), rel=RTOL_NEAR_MONEY
        )


def test_deep_itm_call_approaches_intrinsic():
    assert call_impl(100.0, 1.0, 0.20, 1.0) == pytest.approx(99.0, rel=1e-9)


def test_deep_otm_call_is_effectively_zero():
    assert abs(call_impl(100.0, 1000.0, 0.20, 1.0)) < 1e-9


def test_higher_vol_raises_option_value():
    vals = [call_impl(100.0, 100.0, v, 1.0) for v in (0.05, 0.10, 0.20, 0.40)]
    assert all(b > a for a, b in itertools.pairwise(vals))


def test_moderate_tail_stays_inside_the_documented_envelope():
    """|d2| ~ 3-4. D046 records 5e-5 to 5e-4 relative here; assert we are not worse."""
    f, k, v, t = 100.0, 200.0, 0.25, 1.0
    got, want = call_impl(f, k, v, t), ref_fv(f, k, v, t, 1)
    assert got == pytest.approx(want, rel=RTOL_MODERATE_TAIL)


def test_scale_invariance_of_relative_error():
    """A&S has no absolute-scale sensitivity: 1e-4 and 1e7 forwards behave alike."""
    errs = []
    for f in (1e-4, 1e0, 1e4, 1e7):
        got, want = call_impl(f, f, 0.20, 1.0), ref_fv(f, f, 0.20, 1.0, 1)
        errs.append(abs(got - want) / want)
    assert max(errs) - min(errs) < 1e-9
