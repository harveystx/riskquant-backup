# src/riskquant/portfolio/compilers/block_compilers/vanilla_forward.py
"""
Vanilla forward block compilers (node and curve variants).

Translate a sub-range of an `IngestBatch` into a `VanillaForward*Block` via
the run's `UixMapper`:
  - Underlying market ID  -> market_id_universe        (price_uix / price_curve_uix)
  - Discount curve IDs    -> discount_curve_id_universe
  - FX leg pairs          -> fx_effective_pairs
  - Settlement / delivery ordinals -> date_ord32_universe

`qty`, `strike`, `trade_uids` are pure economics — zero-copy slicing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from riskquant.valuation.blocks.vanilla_forward_block import (
    VanillaForwardCurveBlock,
    VanillaForwardNodeBlock,
)
from riskquant.portfolio.ingest.batch import IngestBatch

if TYPE_CHECKING:
    from riskquant.universe import UixMapper


def compile_vanilla_forward_node(
    batch: IngestBatch,
    start: int,
    end: int,
    uix_mapper: UixMapper,
) -> VanillaForwardNodeBlock | None:
    """Slice + UIX-translate a vanilla-forward-node run."""
    if start == end:
        return None

    price_uix = uix_mapper.map_market_ids_to_uix(batch.underlying_ids[start:end])
    settlement_date_uix = uix_mapper.map_dates_to_uix(batch.settle_ord[start:end])

    fx_u2s_eff = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_u2s_ids[start:end],
        raw_fx_inv=batch.fx_u2s_inv[start:end],
    )
    fx_k2s_eff = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_k2s_ids[start:end],
        raw_fx_inv=batch.fx_k2s_inv[start:end],
    )

    under_df_uix = uix_mapper.map_discount_ids_to_uix(
        batch.under_discount_curve_ids[start:end]
    )
    strike_df_uix = uix_mapper.map_discount_ids_to_uix(
        batch.strike_discount_curve_ids[start:end]
    )

    return VanillaForwardNodeBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        strike=batch.strike_price[start:end],
        settlement_date_uix=settlement_date_uix,
        fx_under_to_settle_effective_uix=fx_u2s_eff,
        fx_strike_to_settle_effective_uix=fx_k2s_eff,
        under_discount_curve_uix=under_df_uix,
        strike_discount_curve_uix=strike_df_uix,
        price_uix=price_uix,
    )


def compile_vanilla_forward_curve(
    batch: IngestBatch,
    start: int,
    end: int,
    uix_mapper: UixMapper,
) -> VanillaForwardCurveBlock | None:
    """Slice + UIX-translate a vanilla-forward-curve run."""
    if start == end:
        return None

    price_curve_uix = uix_mapper.map_market_ids_to_uix(batch.underlying_ids[start:end])
    settlement_date_uix = uix_mapper.map_dates_to_uix(batch.settle_ord[start:end])
    delivery_date_uix = uix_mapper.map_dates_to_uix(batch.delivery_ord[start:end])

    fx_u2s_eff = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_u2s_ids[start:end],
        raw_fx_inv=batch.fx_u2s_inv[start:end],
    )
    fx_k2s_eff = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_k2s_ids[start:end],
        raw_fx_inv=batch.fx_k2s_inv[start:end],
    )

    under_df_uix = uix_mapper.map_discount_ids_to_uix(
        batch.under_discount_curve_ids[start:end]
    )
    strike_df_uix = uix_mapper.map_discount_ids_to_uix(
        batch.strike_discount_curve_ids[start:end]
    )

    return VanillaForwardCurveBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        strike=batch.strike_price[start:end],
        settlement_date_uix=settlement_date_uix,
        delivery_date_uix=delivery_date_uix,
        fx_under_to_settle_effective_uix=fx_u2s_eff,
        fx_strike_to_settle_effective_uix=fx_k2s_eff,
        under_discount_curve_uix=under_df_uix,
        strike_discount_curve_uix=strike_df_uix,
        price_curve_uix=price_curve_uix,
    )
