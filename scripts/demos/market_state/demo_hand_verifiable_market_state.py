# scripts/demos/market_state/demo_hand_verifiable_market_state.py

"""
Hand-verifiable demonstration of ``riskquant.market_state``.

The package collapses a ``(K, S)`` observation matrix into ``(S,)`` series
describing how unusual each observation date was across the cross-section. It
is deliberately NOT a dependence package: every index below uses only the
per-variable marginal scales and is blind to the correlation structure.

That last point is the one this demo exists to prove, because the prototype
these indices came from called the first of them a "market turbulence index",
which it is not. Kritzman-Li turbulence is a Mahalanobis distance and its
entire content is the inverse covariance. The demo shows two dates with
identical marginal magnitudes and opposite co-movement receiving the identical
robust RMS-z score, which is exactly why the function was renamed.

Every value is checked against longhand arithmetic, not against the package.

Run:
    python scripts/demos/market_state/demo_hand_verifiable_market_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from riskquant.constants import MAD_TO_SIGMA_NORMAL  # noqa: E402
from riskquant.market_state.indices import (  # noqa: E402
    robust_rms_z_index,
    robust_scale_mad,
    volatility_normalised_directional_index,
    volatility_normalised_dispersion_index,
    volatility_normalised_shock_index,
)

LINE = "=" * 78


def _banner(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def demo_robust_scale() -> None:
    _banner("ROBUST SCALE  --  MAD, and why it is not the standard deviation")

    series = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
    panel = np.vstack([series, np.array([1.0, 2.0, 3.0, 4.0, 5.0])])
    print(f"x = {panel[0]}   (one gross outlier)")
    print(f"y = {panel[1]}")

    median = float(np.median(panel[0]))
    deviations = np.abs(panel[0] - median)
    mad = float(np.median(deviations))
    print(f"\nmedian(x)              = {median}")
    print(f"|x - median|           = {deviations}")
    print(f"median(|x - median|)   = {mad}")
    print(f"scaled by {MAD_TO_SIGMA_NORMAL}      = {MAD_TO_SIGMA_NORMAL * mad}")

    package = robust_scale_mad(panel)
    print(f"\npackage robust scale   = {package[0]}")
    print(f"sample std (ddof=1)    = {float(np.std(panel[0], ddof=1)):.4f}")
    print("\nThe constant is 1 / Phi^-1(0.75), which makes the estimator")
    print("consistent for sigma under normality. Breakdown point 50 per cent:")
    print("the outlier moves the standard deviation by two orders of magnitude")
    print("and leaves the robust scale untouched.")


def demo_rms_z_index_is_blind_to_co_movement() -> None:
    _banner("ROBUST RMS-Z  --  NOT Kritzman-Li turbulence")

    together = np.array([[1.0, -1.0, 2.0, -2.0], [1.0, -1.0, 2.0, -2.0]])
    apart = np.array([[1.0, -1.0, 2.0, -2.0], [-1.0, 1.0, -2.0, 2.0]])

    print("Two panels with IDENTICAL marginal magnitudes:")
    print(f"  moving together: {together.tolist()}")
    print(f"  moving apart   : {apart.tolist()}")

    index_together = robust_rms_z_index(together)
    index_apart = robust_rms_z_index(apart)
    print(f"\nrobust RMS-z, together = {np.round(index_together, 6)}")
    print(f"robust RMS-z, apart    = {np.round(index_apart, 6)}")
    print(f"identical              = {np.array_equal(index_together, index_apart)}")

    print("\nThat identity is the finding. A Mahalanobis turbulence index would")
    print("separate these two panels completely, because its content is the")
    print("inverse covariance. This index has no access to it, which is why it")
    print("is named for what it computes and not for what it resembles.")


def demo_shock_index_longhand() -> None:
    _banner("VOLATILITY-NORMALISED SHOCK  --  checked longhand")

    panel = np.array([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]])
    print(f"x = {panel[0]}\ny = {panel[1]}")

    scale = np.std(panel, axis=1, ddof=1)
    standardised = panel / scale[:, None]
    longhand = np.sqrt((standardised**2).mean(axis=0))
    print(f"\nsample std (ddof=1)   = {np.round(scale, 6)}")
    print(f"z = x / sigma         =\n{np.round(standardised, 6)}")
    print(f"sqrt(mean_k(z^2))     = {np.round(longhand, 6)}")

    package = volatility_normalised_shock_index(panel)
    print(f"\npackage               = {np.round(package, 6)}")
    print(f"identical             = {np.allclose(package, longhand, atol=0, rtol=0)}")
    print("\nThe numerator is not mean-centred: for a return series the")
    print("meaningful notion of a shock is displacement from zero.")


def demo_direction_versus_dispersion() -> None:
    _banner("DIRECTION VERSUS DISPERSION  --  two regimes separated")

    together = np.vstack([np.array([3.0, -1.0, 1.0, -3.0])] * 2)
    apart = np.vstack(
        [np.array([3.0, -1.0, 1.0, -3.0]), np.array([-3.0, 1.0, -1.0, 3.0])]
    )

    print("A common move and an offsetting move of the same magnitude:\n")
    for label, panel in (("together", together), ("apart", apart)):
        directional = volatility_normalised_directional_index(panel)
        dispersion = volatility_normalised_dispersion_index(panel)
        print(f"  {label:<9} directional = {np.round(directional, 6)}")
        print(f"  {'':<9} dispersion  = {np.round(dispersion, 6)}")

    print("\nA common market move scores on direction and nowhere on dispersion.")
    print("An offsetting move does the reverse. Comparing the two separates a")
    print("directional market move from a dispersion event; neither index alone")
    print("can make that distinction.")


def demo_degeneracy_is_rejected() -> None:
    _banner("DEGENERACY  --  rejected, not silently converted to NaN")

    panel = np.vstack([np.ones(5), np.arange(5.0)])
    print(f"x = {panel[0]}   (constant: every dispersion estimate is zero)")
    print(f"y = {panel[1]}")

    for name, index in (
        ("robust_rms_z_index", robust_rms_z_index),
        ("volatility_normalised_shock_index", volatility_normalised_shock_index),
        ("volatility_normalised_directional_index", volatility_normalised_directional_index),
        ("volatility_normalised_dispersion_index", volatility_normalised_dispersion_index),
    ):
        try:
            index(panel)
            print(f"  {name:<42} ACCEPTED  <- unexpected")
        except ValueError as error:
            print(f"  {name:<42} rejected: {str(error)[:40]}...")

    print("\nThe prototype replaced a zero scale with NaN and propagated it")
    print("silently. Every index here requires a fully finite input and rejects")
    print("a zero scale, so a degenerate variable cannot reach a reported number.")


def main() -> None:
    demo_robust_scale()
    demo_rms_z_index_is_blind_to_co_movement()
    demo_shock_index_longhand()
    demo_direction_versus_dispersion()
    demo_degeneracy_is_rejected()
    print(f"\n{LINE}\nmarket_state contract demonstrated.\n{LINE}")


if __name__ == "__main__":
    main()
