# scripts/demos/market_data/calibrate/implied_vol/black76/demo_calibrate.py
"""
Demo: the end-to-end Black-76 implied-vol calibration edge.

Pipeline:
    raw market quotes (discounted prices, mixed calls/puts)
      -> validate_option_quotes        (dtype/shape/domain + no-arbitrage)
      -> compile_canonical_option_quotes(put-call parity + undiscounting)
      -> calibrate_black76_iv_...       (implied vol + vega quality weight)

A synthetic volatility smile is generated with the package's own model kernel,
quoted as out-of-the-money options (puts below the forward, calls above),
discounted to market prices, then run back through the edge to recover the
smile. Also shows the vega quality weight, the reliability gate, and the
no-arbitrage guard rejecting a bad quote.

Run:
    python -m scripts.demos.market_data.calibrate.implied_vol.black76.demo_calibrate
"""

from __future__ import annotations

import numpy as np

from riskquant.market_data.calibrate.implied_vol.black76.calibrate import (
    calibrate_black76_iv_from_undiscounted_call_prices,
)
from riskquant.market_data.calibrate.implied_vol.black76.kernels.black76_model import (
    _black76_call_price_undiscounted,
)
from riskquant.market_data.calibrate.option_quotes.compile import (
    compile_canonical_option_quotes,
)
from riskquant.market_data.calibrate.option_quotes.validate import (
    validate_option_quotes,
)


def _banner(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _synthetic_quotes():
    """A skewed smile quoted OTM (puts K<F, calls K>=F) at a single tenor."""
    fwd = 100.0
    tte = 0.50
    df = 0.985

    strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    true_iv = np.array([0.34, 0.29, 0.245, 0.20, 0.215, 0.25, 0.30])  # skew

    # undiscounted call for every strike, from the known smile
    undisc_call = np.array(
        [
            _black76_call_price_undiscounted(fwd=fwd, strike=k, tte=tte, vol=v)
            for k, v in zip(strikes, true_iv)
        ]
    )

    # quote OTM: calls at/above the forward, puts below it (parity to put leg)
    flag = (strikes >= fwd).astype(np.int8)
    undisc_quote = np.where(flag == 1, undisc_call, undisc_call - (fwd - strikes))

    option_price = df * undisc_quote  # discounted market price

    return {
        "fwd_price": np.full_like(strikes, fwd),
        "strike": strikes,
        "tte": np.full_like(strikes, tte),
        "call_put_flag": flag,
        "discount_factor": np.full_like(strikes, df),
        "option_price": option_price,
        "_true_iv": true_iv,
    }


def main() -> None:
    q = _synthetic_quotes()
    true_iv = q.pop("_true_iv")

    validated = validate_option_quotes(
        fwd_price=q["fwd_price"],
        strike=q["strike"],
        tte=q["tte"],
        call_put_flag=q["call_put_flag"],
        discount_factor=q["discount_factor"],
        option_price=q["option_price"],
    )
    canonical = compile_canonical_option_quotes(validated)
    calibrated = calibrate_black76_iv_from_undiscounted_call_prices(canonical)

    _banner("Calibrated option quotes")
    print(calibrated.to_string())

    _banner("Smile recovery (calibrated iv vs true iv)")
    print(
        f"{'K':>8} | {'flag':>4} | {'true iv':>9} | {'calib iv':>9} | {'abs err':>10}"
    )
    for i in range(calibrated.n_quotes):
        flag = "C" if q["call_put_flag"][i] == 1 else "P"
        print(
            f"{q['strike'][i]:8.1f} | {flag:>4} | "
            f"{true_iv[i]:9.5f} | {calibrated.iv[i]:9.5f} | "
            f"{abs(calibrated.iv[i] - true_iv[i]):10.2e}"
        )
    max_err = float(np.max(np.abs(calibrated.iv - true_iv)))
    print(f"\nmax |calibrated - true| iv = {max_err:.2e}")

    _banner("Quality gate")
    reliable = calibrated.reliable_mask()
    print(f"solved   : {calibrated.n_solved} / {calibrated.n_quotes}")
    print(f"failed   : {calibrated.n_failed}")
    print(f"reliable : {int(reliable.sum())} (vega above the default floor)")

    _banner("No-arbitrage guard rejects a bad quote")
    try:
        validate_option_quotes(
            fwd_price=[100.0],
            strike=[100.0],
            tte=[0.5],
            call_put_flag=np.array([1], dtype=np.int8),
            discount_factor=[0.985],
            option_price=[99.0],  # exceeds df * F = 98.5
        )
    except ValueError as exc:
        print(f"rejected: {exc}")

    print("\nOK")


if __name__ == "__main__":
    main()
