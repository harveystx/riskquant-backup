# src/riskquant/credit/exposure/add_on/resolve.py
"""
Trade-level exposure resolve (the edge).

Produces the per-trade scalars the CEM and SA-CCR views consume.

CEM (unchanged): effective notional = adjusted notional, with supervisory delta
and maturity factor both fixed at 1 (CEM does not use them):

  effective_notional  = adjusted_notional = VOLUME * current price
                        (marked at the CURRENT price, never the inception price)
  supervisory_factor  = add-on factor for the trade's asset subclass.

SA-CCR (P2-11A/B): the trade-level effective notional is the full CRE52 product
D_i = d_i * MF_i * delta_i, where d_i is the adjusted notional above, delta_i is
the supervisory delta (+1 long / -1 short for the supported linear equity/
commodity population) and MF_i is the unmargined maturity factor. delta and MF
are STRUCTURAL SA-CCR terms, not house parameters. The signed D_i flows into the
signed subclass aggregation (P2-03) unchanged.

The kernel stays dumb: it multiplies these and compresses. All market-state
dependence (the current price) is resolved here, at the edge.
"""

from __future__ import annotations

__all__ = [
    "SACCR_MF_ONE_YEAR_CAP",
    "SACCR_MF_TEN_BUSINESS_DAY_FLOOR",
    "resolve_quantity_price_effective_notional",
    "resolve_saccr_effective_notional",
    "resolve_supervisory_factors",
]

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_raise
from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors


def resolve_quantity_price_effective_notional(
    *,  # force keyword-only
    volume: npt.NDArray[np.float64],
    current_price: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Effective notional via the current-price rule: ``volume * current price``.

    For CEM the effective notional is the adjusted notional (delta = maturity
    factor = 1), marked at the *current* market price (never inception) so
    exposure tracks the live market, not the trade date. This is the book's sole
    effective-notional derivation; were a second asset class ever added, its rule
    (e.g. rates ``notional * supervisory_duration``) would join here without
    moving the ``effective_notional`` contract the engine consumes.
    """
    if volume.shape != current_price.shape:
        raise ValueError(
            "resolve_effective_notional: volume / current_price misaligned."
        )
    return np.ascontiguousarray(
        volume.astype(np.float64, copy=False)
        * current_price.astype(np.float64, copy=False)
    )


#: SA-CCR unmargined maturity-factor bounds (CRE52.48-49). MF = sqrt(clamp(M, .)):
#: the 10-business-day floor (10/250 years, 250 business days per year) and the
#: one-year cap. Structural CRE52 terms, not house-calibrated parameters.
SACCR_MF_TEN_BUSINESS_DAY_FLOOR: float = 10.0 / 250.0
SACCR_MF_ONE_YEAR_CAP: float = 1.0


def resolve_saccr_effective_notional(
    *,  # force keyword-only
    volume: npt.NDArray[np.float64],
    current_price: npt.NDArray[np.float64],
    direction_sign: npt.NDArray[np.float64],
    remaining_maturity_years: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """SA-CCR trade-level effective notional ``D_i = d_i * MF_i * delta_i`` (CRE52.30).

    Structural CRE52 mathematics, for the supported unmargined linear equity/
    commodity population:

      d_i    = volume_i * current_price_i          (adjusted notional, CRE52.36)
      delta_i = direction_sign_i in {+1, -1}       (supervisory delta, CRE52.39)
      MF_i   = sqrt(clamp(M_i, 10/250, 1))         (unmargined MF, CRE52.48-49)

    ``direction_sign`` and ``remaining_maturity_years`` are already-canonical
    primitive arrays (the +-1 sign and M in years); string direction and any date
    representation die at the Pydantic boundary before this point.

    Fail-fast contract (no silent repair):
      * a non-{+1,-1} direction sign is rejected;
      * a non-positive remaining maturity is rejected (an expired/negative trade
        is a caller error, NOT something to clip into the floor).
    A valid but very-short maturity below 10/250y IS clamped up to the floor:
    that clamp is the CRE52 floor, applied only to legal positive maturities.
    """
    if not (
        volume.shape
        == current_price.shape
        == direction_sign.shape
        == remaining_maturity_years.shape
    ):
        raise ValueError(
            "resolve_saccr_effective_notional: volume / current_price / "
            "direction_sign / remaining_maturity_years misaligned."
        )

    sign = direction_sign.astype(np.float64, copy=False)
    if not np.all((sign == 1.0) | (sign == -1.0)):
        raise ValueError(
            "resolve_saccr_effective_notional: direction_sign must be +1.0 (long) "
            "or -1.0 (short) for every trade; supervisory delta is structural."
        )

    m = remaining_maturity_years.astype(np.float64, copy=False)
    if not np.all(m > 0.0):
        raise ValueError(
            "resolve_saccr_effective_notional: remaining_maturity_years must be "
            "strictly positive for every trade; an expired or negative maturity "
            "is a boundary error, not a value to clamp into the 10-business-day "
            "floor."
        )

    adjusted_notional = volume.astype(np.float64, copy=False) * current_price.astype(
        np.float64, copy=False
    )
    maturity_factor = np.sqrt(
        np.clip(m, SACCR_MF_TEN_BUSINESS_DAY_FLOOR, SACCR_MF_ONE_YEAR_CAP)
    )
    return np.ascontiguousarray(adjusted_notional * maturity_factor * sign)


def resolve_supervisory_factors(
    asset_subclass_uix: npt.NDArray[np.int32],
    add_on_factors: CompiledAddOnFactors,
) -> npt.NDArray[np.float64]:
    """Per-trade supervisory add-on factor by asset-subclass UIX (strict join).

    Every trade subclass MUST exist in the table — an unpriced subclass is a gap
    in the supervisory config, not something to default silently.
    """
    pos = searchsorted_or_raise(
        universe=add_on_factors.subclass_uix,
        raw=asset_subclass_uix,
        ctx="resolve_supervisory_factors: asset subclass uix(s)",
    )
    return np.ascontiguousarray(add_on_factors.factor[pos])
