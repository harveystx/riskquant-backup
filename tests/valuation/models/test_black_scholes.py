# tests/valuation/models/test_black_scholes.py
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from riskquant.valuation.models.black_scholes import (
    fv_black_scholes_vanilla_3d,
    pv_black_scholes_vanilla_3d,
)

# =========================================================================
# CONSTANTS
# =========================================================================
N_TRADES = 4
N_SCENARIOS = 8


# =========================================================================
# HELPERS
# =========================================================================


def _scalar_black76_fv(
    fwd: float, strike: float, vol: float, tte: float, is_call: bool
) -> float:
    """Reference scalar Black-76 implementation for cross-checking."""
    safe_tte = max(tte, 1e-9)
    std = vol * np.sqrt(safe_tte)
    d1 = (np.log(fwd / strike) + 0.5 * vol**2 * safe_tte) / std
    d2 = d1 - std
    cp = 1.0 if is_call else -1.0
    return cp * (fwd * norm.cdf(cp * d1) - strike * norm.cdf(cp * d2))


# =========================================================================
# TESTS — fv_black_scholes_vanilla_3d
# =========================================================================


def test_fv_call_atm_positive() -> None:
    """ATM calls should always have positive future value."""
    fwd = np.full((N_TRADES, N_SCENARIOS), 100.0)
    strike = np.full((N_TRADES, 1), 100.0)
    vol = np.full((N_TRADES, N_SCENARIOS), 0.20)
    tte = np.full((N_TRADES, 1), 1.0)
    is_call = np.ones((N_TRADES, 1), dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    assert result.shape == (N_TRADES, N_SCENARIOS)
    assert np.all(result > 0)


def test_fv_put_atm_positive() -> None:
    """ATM puts should also have positive future value."""
    fwd = np.full((N_TRADES, N_SCENARIOS), 100.0)
    strike = np.full((N_TRADES, 1), 100.0)
    vol = np.full((N_TRADES, N_SCENARIOS), 0.20)
    tte = np.full((N_TRADES, 1), 1.0)
    is_call = np.zeros((N_TRADES, 1), dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    assert result.shape == (N_TRADES, N_SCENARIOS)
    assert np.all(result > 0)


def test_fv_call_put_parity() -> None:
    """Call - Put = Fwd - Strike (Black-76 put-call parity on FV)."""
    fwd = np.full((N_TRADES, N_SCENARIOS), 105.0)
    strike = np.full((N_TRADES, 1), 100.0)
    vol = np.full((N_TRADES, N_SCENARIOS), 0.25)
    tte = np.full((N_TRADES, 1), 0.5)

    fv_call = fv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=strike,
        vol=vol,
        tte=tte,
        is_call=np.ones((N_TRADES, 1), dtype=np.int8),
    )
    fv_put = fv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=strike,
        vol=vol,
        tte=tte,
        is_call=np.zeros((N_TRADES, 1), dtype=np.int8),
    )

    np.testing.assert_allclose(fv_call - fv_put, fwd - strike, rtol=1e-10)


def test_fv_matches_scalar_reference() -> None:
    """Vectorized output must match scalar Black-76 reference for several cases."""
    cases = [
        # (fwd, strike, vol, tte, is_call)
        (100.0, 100.0, 0.20, 1.0, True),
        (110.0, 100.0, 0.30, 0.5, True),
        (90.0, 100.0, 0.15, 2.0, False),
        (100.0, 120.0, 0.40, 0.25, False),
    ]

    n = len(cases)
    s = 1  # single scenario suffices for a scalar check

    fwd = np.array([[c[0]] for c in cases])
    strike = np.array([[c[1]] for c in cases])
    vol = np.array([[c[2]] for c in cases])
    tte = np.array([[c[3]] for c in cases])
    is_call = np.array([[1 if c[4] else 0] for c in cases], dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    for i, c in enumerate(cases):
        expected = _scalar_black76_fv(*c)
        np.testing.assert_allclose(result[i, 0], expected, rtol=1e-12)


def test_fv_deep_itm_call_approaches_intrinsic() -> None:
    """Deep ITM call with low vol should converge to intrinsic value (Fwd - K)."""
    fwd = np.full((1, N_SCENARIOS), 200.0)
    strike = np.full((1, 1), 100.0)
    vol = np.full((1, N_SCENARIOS), 0.001)
    tte = np.full((1, 1), 0.01)
    is_call = np.ones((1, 1), dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    np.testing.assert_allclose(result, 100.0, rtol=1e-4)


def test_fv_deep_otm_call_approaches_zero() -> None:
    """Deep OTM call with low vol should approach zero."""
    fwd = np.full((1, N_SCENARIOS), 50.0)
    strike = np.full((1, 1), 200.0)
    vol = np.full((1, N_SCENARIOS), 0.001)
    tte = np.full((1, 1), 0.01)
    is_call = np.ones((1, 1), dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    np.testing.assert_allclose(result, 0.0, atol=1e-10)


def test_fv_zero_tte_call_equals_intrinsic() -> None:
    """At expiry (tte=0), FV should equal max(Fwd - K, 0) for a call."""
    fwd = np.array([[110.0, 90.0]])
    strike = np.full((1, 1), 100.0)
    vol = np.full((1, 2), 0.20)
    tte = np.full((1, 1), 0.0)
    is_call = np.ones((1, 1), dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    expected = np.maximum(fwd - strike, 0.0)
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_fv_zero_tte_put_equals_intrinsic() -> None:
    """At expiry (tte=0), FV should equal max(K - Fwd, 0) for a put."""
    fwd = np.array([[110.0, 90.0]])
    strike = np.full((1, 1), 100.0)
    vol = np.full((1, 2), 0.20)
    tte = np.full((1, 1), 0.0)
    is_call = np.zeros((1, 1), dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    expected = np.maximum(strike - fwd, 0.0)
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_fv_higher_vol_higher_value() -> None:
    """Higher vol should produce higher ATM option value."""
    fwd = np.full((1, 1), 100.0)
    strike = np.full((1, 1), 100.0)
    tte = np.full((1, 1), 1.0)
    is_call = np.ones((1, 1), dtype=np.int8)

    fv_low = fv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=strike,
        vol=np.full((1, 1), 0.10),
        tte=tte,
        is_call=is_call,
    )
    fv_high = fv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=strike,
        vol=np.full((1, 1), 0.40),
        tte=tte,
        is_call=is_call,
    )

    assert fv_high[0, 0] > fv_low[0, 0]


def test_fv_broadcasting_per_scenario_vol() -> None:
    """Verify per-scenario vol produces different values per scenario."""
    fwd = np.full((1, 3), 100.0)
    strike = np.full((1, 1), 100.0)
    vol = np.array([[0.10, 0.20, 0.40]])
    tte = np.full((1, 1), 1.0)
    is_call = np.ones((1, 1), dtype=np.int8)

    result = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )

    assert result.shape == (1, 3)
    # higher vol => higher value
    assert result[0, 0] < result[0, 1] < result[0, 2]


# =========================================================================
# TESTS — pv_black_scholes_vanilla_3d
# =========================================================================


def test_pv_equals_fv_times_df() -> None:
    """PV must equal FV * df_settle."""
    fwd = np.full((N_TRADES, N_SCENARIOS), 105.0)
    strike = np.full((N_TRADES, 1), 100.0)
    vol = np.full((N_TRADES, N_SCENARIOS), 0.25)
    tte = np.full((N_TRADES, 1), 1.0)
    is_call = np.ones((N_TRADES, 1), dtype=np.int8)
    df = np.full((N_TRADES, N_SCENARIOS), 0.95)

    fv = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )
    pv = pv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call, df_settle=df
    )

    np.testing.assert_allclose(pv, fv * df, rtol=1e-14)


def test_pv_df_one_equals_fv() -> None:
    """With df=1.0, PV must equal FV exactly."""
    fwd = np.full((2, 4), 100.0)
    strike = np.full((2, 1), 95.0)
    vol = np.full((2, 4), 0.30)
    tte = np.full((2, 1), 0.5)
    is_call = np.ones((2, 1), dtype=np.int8)
    df = np.ones((2, 4))

    fv = fv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call
    )
    pv = pv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call, df_settle=df
    )

    np.testing.assert_allclose(pv, fv, rtol=1e-14)


def test_pv_per_scenario_df() -> None:
    """Each scenario gets its own discount factor applied."""
    fwd = np.full((1, 3), 100.0)
    strike = np.full((1, 1), 100.0)
    vol = np.full((1, 3), 0.20)
    tte = np.full((1, 1), 1.0)
    is_call = np.ones((1, 1), dtype=np.int8)
    df = np.array([[0.90, 0.95, 1.00]])

    pv = pv_black_scholes_vanilla_3d(
        fwd=fwd, strike=strike, vol=vol, tte=tte, is_call=is_call, df_settle=df
    )

    # PV should be increasing with higher df
    assert pv[0, 0] < pv[0, 1] < pv[0, 2]
