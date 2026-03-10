# src/riskquant/portfolio/compilers/vanilla_european_option.py
from __future__ import annotations
from typing import Optional
import numpy as np

from riskquant.portfolio.instruments import (
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionNodeBlock,
    VanillaEuropeanOptionCurveBlock,
)


def compile_vanilla_euro_option_node(
    batch: IngestBatch,
) -> Optional[VanillaEuropeanOptionNodeBlock]:
    mask = batch.instrument_ids == VANILLA_EUROPEAN_OPTION_NODE_ID
    if not np.any(mask):
        return None

    return VanillaEuropeanOptionNodeBlock(
        trade_uids=batch.uids[mask],
        qty=batch.qty[mask],
        strike=batch.strike_price[mask],
        is_call=batch.is_call[mask],
        expiry_date_ordinal32=batch.expiry_ord[mask],
        payoff_settlement_date_ordinal32=batch.settle_ord[mask],
        fx_under_to_settle_uix=batch.fx_under_to_settle_ids[mask],
        fx_strike_to_settle_uix=batch.fx_strike_to_settle_ids[mask],
        fx_under_to_settle_invert=batch.fx_under_to_settle_invert[mask].astype(
            bool
        ),
        fx_strike_to_settle_invert=batch.fx_strike_to_settle_invert[
            mask
        ].astype(bool),
        under_discount_curve_uix=batch.under_discount_curve_ids[mask],
        strike_discount_curve_uix=batch.strike_discount_curve_ids[mask],
        underlying_price_uix=batch.price_rf_ids[mask],
        vol_uix=batch.vol_rf_ids[mask],
    )


def compile_vanilla_euro_option_curve(
    batch: IngestBatch,
) -> Optional[VanillaEuropeanOptionCurveBlock]:
    mask = batch.instrument_ids == VANILLA_EUROPEAN_OPTION_CURVE_ID
    if not np.any(mask):
        return None

    return VanillaEuropeanOptionCurveBlock(
        trade_uids=batch.uids[mask],
        qty=batch.qty[mask],
        strike=batch.strike_price[mask],
        is_call=batch.is_call[mask],
        expiry_date_ordinal32=batch.expiry_ord[mask],
        payoff_settlement_date_ordinal32=batch.settle_ord[mask],
        fx_under_to_settle_uix=batch.fx_under_to_settle_ids[mask],
        fx_strike_to_settle_uix=batch.fx_strike_to_settle_ids[mask],
        fx_under_to_settle_invert=batch.fx_under_to_settle_invert[mask].astype(
            bool
        ),
        fx_strike_to_settle_invert=batch.fx_strike_to_settle_invert[
            mask
        ].astype(bool),
        under_discount_curve_uix=batch.under_discount_curve_ids[mask],
        strike_discount_curve_uix=batch.strike_discount_curve_ids[mask],
        underlying_price_curve_uix=batch.price_curve_ids[mask],
        delivery_date_ordinal32=batch.delivery_ord[mask],
        vol_surface_uix=batch.vol_surface_rf_ids[mask],
    )
