# src/riskquant/portfolio/compilers/instruments/spot.py
"""
Spot block compiler.

Translates the spot-instrument sub-range of an `IngestBatch` (global IDs) into
a `SpotBlock` (execution UIXs) via the run's `UixMapper`. The only fields
that need translation are:
  - `price_uix`            : underlying_ids        -> market_id_universe
  - `fx_spots_effective_uix`: (fx_u2s_ids, fx_u2s_inv) -> fx_effective_pairs

`qty` and `trade_uids` are pure economics — no translation needed; slicing
returns NumPy views (zero copy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.valuation.blocks.spot_block import SpotBlock

if TYPE_CHECKING:
    from riskquant.universe import UixMapper


def compile_spot(
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

    # FX leg (only the under->settle leg matters for spot).
    fx_spots_effective_uix = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_u2s_ids[start:end],
        raw_fx_inv=batch.fx_u2s_inv[start:end],
    )

    return SpotBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        spot_price_uix=price_uix,
        fx_spot_effective_uix=fx_spots_effective_uix,
    )
