# src/riskquant/portfolio/compilers/forward.py
from __future__ import annotations

import numpy as np
from typing import Optional

from riskquant.portfolio.instruments import FORWARD_NODE_ID, FORWARD_CURVE_ID
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.blocks.forward_block import (
    ForwardNodeBlock,
    ForwardCurveBlock,
)


def compile_forward_node(batch: IngestBatch) -> Optional[ForwardNodeBlock]:
    # 1. Create the boolean mask for this instrument
    mask = batch.instrument_ids == FORWARD_NODE_ID

    # 2. Early exit if no trades found
    if not np.any(mask):
        return None

    # 3. Vectorized slice directly into the SoA block
    return ForwardNodeBlock(
        trade_uids=batch.uids[mask],
        qty=batch.qty[mask],
        strike=batch.strike_price[mask],
        settlement_date_ordinal32=batch.settle_ord[mask],
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
        price_uix=batch.price_rf_ids[mask],
    )


def compile_forward_curve(batch: IngestBatch) -> Optional[ForwardCurveBlock]:
    mask = batch.instrument_ids == FORWARD_CURVE_ID
    if not np.any(mask):
        return None

    return ForwardCurveBlock(
        trade_uids=batch.uids[mask],
        qty=batch.qty[mask],
        strike=batch.strike_price[mask],
        settlement_date_ordinal32=batch.settle_ord[mask],
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
        price_curve_uix=batch.price_curve_ids[mask],
        delivery_date_ordinal32=batch.delivery_ord[mask],
    )
