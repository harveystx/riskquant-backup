# scripts/demos/market_data/calibrate/implied_vol/black76/demo_kernel_black76_model.py
"""
Demo: the Black-76 model kernels used by implied-vol calibration.

Exercises the undiscounted call-price kernel and its analytic vega, and shows
the properties the calibrator relies on:

    * call price is bounded by [intrinsic, forward] and decreasing in strike,
    * put-call parity holds at the undiscounted (forward) level,
    * vega peaks at-the-money and decays into the wings -- the basis for using
      it as the calibration quality weight,
    * analytic vega matches a central finite-difference of the price.

Run:
    python -m scripts.demos.market_data.calibrate.implied_vol.black76.demo_kernel_black76_model
"""

from __future__ import annotations

import numpy as np

from riskquant.market_data.calibrate.implied_vol.black76.kernels.black76_model import (
    _black76_call_price_undiscounted,
    _black76_undiscounted_vega,
    _numba_norm_cdf,
    _numba_norm_pdf,
)


def _banner(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> None:
    fwd = 100.0
    tte = 0.50
    vol = 0.20
    strikes = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0])

    _banner(f"Undiscounted Black-76 call (F={fwd:g}, T={tte:g}, vol={vol:.0%})")
    print(f"{'K':>8} | {'Call':>12} | {'Vega':>12} | {'ImpliedPut':>12}")
    prev_call = np.inf
    for k in strikes:
        call = _black76_call_price_undiscounted(fwd=fwd, strike=k, tte=tte, vol=vol)
        vega = _black76_undiscounted_vega(fwd=fwd, strike=k, tte=tte, vol=vol)
        put = call - (fwd - k)  # put-call parity, undiscounted (forward) measure
        print(f"{k:8.1f} | {call:12.6f} | {vega:12.6f} | {put:12.6f}")

        # no-arbitrage bounds the calibrator trusts
        assert call >= max(fwd - k, 0.0) - 1e-9, "call below intrinsic"
        assert call <= fwd + 1e-9, "call above forward"
        assert put >= max(k - fwd, 0.0) - 1e-9, "put below intrinsic"
        assert call <= prev_call + 1e-12, "call not monotone decreasing in strike"
        prev_call = call

    _banner("Vega is the quality weight: peaks ATM, decays in the wings")
    vegas = {
        k: _black76_undiscounted_vega(fwd=fwd, strike=k, tte=tte, vol=vol)
        for k in strikes
    }
    atm = max(vegas, key=vegas.get)
    print(
        f"max vega at K={atm:g} (vega={vegas[atm]:.4f}); "
        f"deep-ITM K=60 vega={vegas[60.0]:.4f}, "
        f"deep-OTM K=140 vega={vegas[140.0]:.4f}"
    )
    assert atm == 100.0, "vega should peak at-the-money for this grid"

    _banner("Analytic vega vs central finite-difference of price")
    h = 1e-6
    print(f"{'K':>8} | {'analytic':>14} | {'finite-diff':>14} | {'abs err':>10}")
    for k in (80.0, 100.0, 120.0):
        up = _black76_call_price_undiscounted(fwd=fwd, strike=k, tte=tte, vol=vol + h)
        dn = _black76_call_price_undiscounted(fwd=fwd, strike=k, tte=tte, vol=vol - h)
        fd = (up - dn) / (2.0 * h)
        an = _black76_undiscounted_vega(fwd=fwd, strike=k, tte=tte, vol=vol)
        print(f"{k:8.1f} | {an:14.8f} | {fd:14.8f} | {abs(an - fd):10.2e}")
        assert abs(an - fd) < 1e-4, "analytic vega disagrees with finite difference"

    _banner("Limit behaviour")
    near_zero = _black76_call_price_undiscounted(
        fwd=fwd, strike=90.0, tte=tte, vol=1e-9
    )
    huge_vol = _black76_call_price_undiscounted(fwd=fwd, strike=90.0, tte=tte, vol=8.0)
    print(f"vol->0  call(K=90) = {near_zero:.6f}  (intrinsic = {fwd - 90.0:.6f})")
    print(f"vol=800% call(K=90) = {huge_vol:.6f}  (approaches forward = {fwd:.6f})")
    assert abs(near_zero - max(fwd - 90.0, 0.0)) < 1e-6
    assert huge_vol < fwd

    _banner("Normal helpers")
    print(f"Phi(0) = {_numba_norm_cdf(0.0):.6f}  (expect 0.5)")
    print(f"phi(0) = {_numba_norm_pdf(0.0):.6f}  (expect 1/sqrt(2*pi) = 0.398942)")

    print("\nOK")


if __name__ == "__main__":
    main()
