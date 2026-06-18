# tests/market_data/calibrate/implied_vol/black76/test_calibrate.py
"""
Correctness tests for the Black-76 implied-volatility calibration path.

These exercise the *real* compiled kernels (numba required), cross-checked
against an independent SciPy reference. Runs under pytest, or standalone via
``python test_calibrate.py``.

Reference model
---------------
Undiscounted Black-76 call: C = F*Phi(d1) - K*Phi(d2),
    d1 = (ln(F/K) + 0.5*sig^2*T) / (sig*sqrt(T)),  d2 = d1 - sig*sqrt(T).
Undiscounted vega: dC/dvol = F * phi(d1) * sqrt(T).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from riskquant.constants import IV_SOLVER_FAILED_SENTINEL, MIN_RELIABLE_VEGA_OVER_FWD
from riskquant.market_data.calibrate.implied_vol.black76.calibrate import (
    _calibrate_black76_iv_arrays,
    calibrate_black76_iv_from_undiscounted_call_prices,
)
from riskquant.market_data.calibrate.option_quotes.master import (
    CalibratedOptionQuotes,
    CanonicalOptionQuotes,
)

_LOWER_VOL = 1e-8
_UPPER_VOL = 10.0


# --------------------------------------------------------------------------- refs
def _b76_undisc(fwd, strike, tte, vol):
    fwd = np.asarray(fwd, float)
    strike = np.asarray(strike, float)
    tte = np.asarray(tte, float)
    vol = np.asarray(vol, float)
    vst = vol * np.sqrt(tte)
    d1 = (np.log(fwd / strike) + 0.5 * vol * vol * tte) / vst
    d2 = d1 - vst
    return fwd * norm.cdf(d1) - strike * norm.cdf(d2)


def _vega_ref(fwd, strike, tte, vol):
    """Reference undiscounted vega; 0 where vol <= 0 (failed / zero-vol)."""
    fwd = np.asarray(fwd, float)
    strike = np.asarray(strike, float)
    tte = np.asarray(tte, float)
    vol = np.asarray(vol, float)
    out = np.zeros(np.broadcast(fwd, strike, tte, vol).shape, float)
    m = vol > 0.0
    if np.any(m):
        st = np.sqrt(tte * np.ones_like(out))[m]
        d1 = (
            np.log((fwd / strike) * np.ones_like(out))[m]
            + 0.5 * (vol * vol * tte * np.ones_like(out))[m]
        ) / ((vol * np.ones_like(out))[m] * st)
        out[m] = (fwd * np.ones_like(out))[m] * norm.pdf(d1) * st
    return out


def _scipy_iv(fwd, strike, tte, target):
    f = lambda v: float(_b76_undisc(fwd, strike, tte, v)) - target
    return brentq(f, _LOWER_VOL, _UPPER_VOL, xtol=1e-12, rtol=1e-12, maxiter=200)


def _vega_over_fwd(fwd, strike, tte, vol):
    st = math.sqrt(tte)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * tte) / (vol * st)
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * st


_FWDS = (0.5, 1.0, 50.0, 100.0, 4000.0)
_MONEYNESS = (0.6, 0.8, 0.95, 1.0, 1.05, 1.25, 1.6)
_TTES = (1.0 / 365.0, 0.05, 0.25, 1.0, 5.0)
_VOLS = (0.01, 0.05, 0.20, 0.80, 2.5)


def _build_grid():
    """Solvable (F, K, T, target, true_vol) tuples (above intrinsic, below fwd)."""
    rows = []
    for F in _FWDS:
        for m in _MONEYNESS:
            K = F * m
            for T in _TTES:
                for sig in _VOLS:
                    target = float(_b76_undisc(F, K, T, sig))
                    intrinsic = max(F - K, 0.0)
                    if target <= intrinsic + 1e-12 * max(F, K, 1.0):
                        continue
                    if target > F:
                        continue
                    rows.append((F, K, T, target, sig))
    return rows


# --------------------------------------------------------------------------- tests
def test_grid_reprices():
    """Every solved IV must reprice to its target (price round-trip)."""
    rows = _build_grid()
    F = np.array([r[0] for r in rows])
    K = np.array([r[1] for r in rows])
    T = np.array([r[2] for r in rows])
    tgt = np.array([r[3] for r in rows])

    iv, vega = _calibrate_black76_iv_arrays(F, K, T, tgt)
    assert np.all(iv >= 0.0), "no quote in this grid should fail to invert"

    reprice = _b76_undisc(F, K, T, iv)
    rel = np.abs(reprice - tgt) / np.maximum(tgt, 1.0)
    # solver vol tolerance is 1e-8; the implied price residual stays << 1e-5
    assert rel.max() < 1e-5, f"max rel reprice error {rel.max():.3e}"

    assert not np.isnan(vega).any()
    assert np.all(vega >= 0.0)


def test_scipy_vol_parity_well_conditioned():
    """Where vega is healthy, the kernel IV matches SciPy brentq to ~1e-6."""
    worst = 0.0
    n = 0
    for F, K, T, target, sig in _build_grid():
        if _vega_over_fwd(F, K, T, sig) <= 1e-2:
            continue
        iv, _ = _calibrate_black76_iv_arrays(
            np.array([F]), np.array([K]), np.array([T]), np.array([target])
        )
        ref = _scipy_iv(F, K, T, target)
        worst = max(worst, abs(float(iv[0]) - ref))
        n += 1
    assert n > 0
    assert worst < 1e-6, f"max |kernel - scipy| vol = {worst:.3e}"


def test_vega_matches_reference():
    """Kernel vega matches the closed-form reference at the solved iv."""
    rows = _build_grid()
    F = np.array([r[0] for r in rows])
    K = np.array([r[1] for r in rows])
    T = np.array([r[2] for r in rows])
    tgt = np.array([r[3] for r in rows])

    iv, vega = _calibrate_black76_iv_arrays(F, K, T, tgt)
    ref = _vega_ref(F, K, T, iv)
    rel = np.abs(vega - ref) / np.maximum(ref, 1.0)
    assert rel.max() < 1e-9, f"max rel vega error {rel.max():.3e}"


def test_rotation_regression():
    """
    Guard for the NR-zbrent rotation bug: OTM / short-dated quotes that the
    pre-fix solver resolved to the lower bracket bound (non-repricing). These
    must now invert and reprice.
    """
    cases = [
        (4000.0, 6400.0, 1.0, 0.80),  # deep OTM, 1y
        (4000.0, 6400.0, 5.0, 0.20),  # deep OTM, 5y
        (4000.0, 5000.0, 5.0, 0.05),  # OTM, low vol, long T
        (1.0, 1.6, 0.25, 0.20),  # OTM small notional
        (100.0, 110.0, 1.0 / 365.0, 0.80),  # OTM, 1 day, resolvable time value
    ]
    for F, K, T, sig in cases:
        target = float(_b76_undisc(F, K, T, sig))
        iv, _ = _calibrate_black76_iv_arrays(
            np.array([F]), np.array([K]), np.array([T]), np.array([target])
        )
        v = float(iv[0])
        assert v > _LOWER_VOL, f"regressed to bracket bound: {F, K, T, sig} -> {v}"
        reprice = float(_b76_undisc(F, K, T, v))
        assert abs(reprice - target) / max(target, 1.0) < 1e-5


def test_failure_regimes():
    F = np.array([100.0, 100.0, 100.0])
    K = np.array([90.0, 90.0, 90.0])
    T = np.array([1.0, 1.0, 1.0])
    # [0] target above forward -> no solution
    # [1] target below intrinsic -> arbitrage -> no solution
    # [2] target at intrinsic -> valid zero-vol solve
    tgt = np.array([F[0] + 1.0, 5.0, 10.0])
    iv, vega = _calibrate_black76_iv_arrays(F, K, T, tgt)

    assert iv[0] == IV_SOLVER_FAILED_SENTINEL
    assert iv[1] == IV_SOLVER_FAILED_SENTINEL
    assert iv[2] == 0.0  # at-intrinsic is a SUCCESS, not a failure
    assert not np.isnan(iv).any(), "failures must use the sentinel, never NaN"

    # vega is 0 for failed AND for at-intrinsic zero-vol solves
    assert np.all(vega == 0.0)


def test_master_contract():
    rows = _build_grid()
    # inject two guaranteed failures (target above forward) and one at-intrinsic
    # success (target == intrinsic -> iv 0.0, mask True, vega 0.0)
    F = np.array([r[0] for r in rows] + [100.0, 50.0, 100.0])
    K = np.array([r[1] for r in rows] + [90.0, 40.0, 90.0])
    T = np.array([r[2] for r in rows] + [1.0, 1.0, 1.0])
    undisc = np.array(
        [r[3] for r in rows] + [200.0, 100.0, 10.0]
    )  # fail, fail, intrinsic

    quotes = CanonicalOptionQuotes(
        fwd_price=F, strike=K, tte=T, undiscounted_call_price=undisc
    )
    out = calibrate_black76_iv_from_undiscounted_call_prices(quotes)

    assert isinstance(out, CalibratedOptionQuotes)
    assert out.n_quotes == F.shape[0]
    assert out.n_solved + out.n_failed == out.n_quotes
    assert out.n_failed >= 2
    assert not np.isnan(out.iv).any()
    # iv-sign contract
    assert np.array_equal(out.success_mask, out.iv >= 0.0)
    assert np.all(out.iv[~out.success_mask] == IV_SOLVER_FAILED_SENTINEL)
    assert np.all(out.iv[out.success_mask] >= 0.0)

    # vega contract: non-negative, finite, zero on failure (but NOT necessarily
    # nonzero on success -- at-intrinsic zero-vol is a success with zero vega)
    assert not np.isnan(out.vega).any()
    assert np.all(out.vega >= 0.0)
    assert np.all(out.vega[~out.success_mask] == 0.0)
    intrinsic_success = out.success_mask & (out.iv == 0.0)
    assert np.any(intrinsic_success)
    assert np.all(out.vega[intrinsic_success] == 0.0)


def test_reliable_mask_gates_flat_vega():
    """reliable_mask keeps well-conditioned quotes and drops flat-vega ones."""
    # ATM, healthy vega
    good = (100.0, 100.0, 1.0, 0.20)
    # deep OTM ultra-short, near-zero vega (under-determined iv)
    flat = (100.0, 150.0, 1.0 / 365.0, 0.20)

    F = np.array([good[0], flat[0]])
    K = np.array([good[1], flat[1]])
    T = np.array([good[2], flat[2]])
    undisc = np.array([float(_b76_undisc(*good)), float(_b76_undisc(*flat))])
    quotes = CanonicalOptionQuotes(
        fwd_price=F, strike=K, tte=T, undiscounted_call_price=undisc
    )
    out = calibrate_black76_iv_from_undiscounted_call_prices(quotes)
    rel = out.reliable_mask()

    assert rel.dtype == np.bool_
    assert np.all(rel <= out.success_mask)  # reliable implies solved
    assert rel[0], "ATM healthy-vega quote should be reliable"
    # flat-vega quote: either failed or vega below the floor -> not reliable
    assert not rel[1], "flat-vega quote should not be reliable"
    # default floor is the centralized constant
    assert MIN_RELIABLE_VEGA_OVER_FWD > 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
