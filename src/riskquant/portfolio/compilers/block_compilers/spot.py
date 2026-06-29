# src/riskquant/portfolio/compilers/block_compilers/spot.py
"""
Spot block compiler.

Translates the spot-instrument sub-range of an `IngestBatch` (global IDs) into
a `SpotBlock` (execution UIXs) via the run's `UixMapper`. The only field that
needs translation is the single effective-spot coordinate:

    underlying_ids                 -> price_uix      (market_id_universe)
    (fx_u2s_ids, fx_u2s_inv)       -> fx_eff_uix     (fx_spots_effective row)
    (price_uix, fx_eff_uix)        -> effective_spot_price_uix
                                      (ValuationState.effective_spot_prices row)

The kernel then gathers one pre-folded `effective_spot_prices` row and
multiplies by `qty` -- no FX inversion, no branching. `qty` and `trade_uids`
are pure economics, no translation needed; slicing returns NumPy views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.valuation.blocks.spot_block import SpotBlock

if TYPE_CHECKING:
    from riskquant.universe import UixMapper


def compile_spot_block(
    batch: IngestBatch,
    start: int,
    end: int,
    uix_mapper: UixMapper,
) -> SpotBlock | None:
    """Slice + UIX-translate a spot run from `batch[start:end]`."""
    if start == end:
        return None

    # Underlying price IDs -> market-universe UIXs.
    price_uix = uix_mapper.map_market_ids_to_uix(batch.underlying_ids[start:end])

    # Under->settle FX leg -> fx_spots_effective row (INVALID_UIX for identity).
    fx_eff_uix = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_u2s_ids[start:end],
        raw_fx_inv=batch.fx_u2s_inv[start:end],
    )

    # Fold the (price, fx) pair into the single effective-spot coordinate.
    effective_spot_price_uix = uix_mapper.map_spot_to_effective_uix(
        price_uix=price_uix,
        fx_eff_uix=fx_eff_uix,
    )

    return SpotBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        effective_spot_price_uix=effective_spot_price_uix,
    )
