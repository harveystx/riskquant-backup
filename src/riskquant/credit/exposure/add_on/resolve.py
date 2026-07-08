# src/riskquant/credit/exposure/add_on/resolve.py
"""
CEM trade-level exposure resolve (the edge).

Produces the two per-trade scalars the CEM view consumes, kept separable for the
SA-CCR upgrade path (`effective_notional` gains supervisory delta and maturity
factor; `supervisory_factor` moves to a per-hedging-set application):

  effective_notional  = adjusted_notional * delta * maturity_factor   (CEM: d=MF=1)
                        adjusted notional is marked at the CURRENT price
                        (VOLUME * current price), never the inception price.
  supervisory_factor  = add-on factor for the trade's asset subclass.

The kernel stays dumb: it multiplies these and compresses. All market-state
dependence (the current price) is resolved here, at the edge.
"""

from __future__ import annotations

__all__ = [
    "resolve_quantity_price_effective_notional",
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
