# src/riskquant/portfolio/compilers/instruments/forward.py
from __future__ import annotations
from typing import Optional

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.blocks.forward_block import (
    ForwardNodeBlock,
    ForwardCurveBlock,
)


def compile_forward_node(
    batch: IngestBatch,
    start: int,
    end: int,
) -> Optional[ForwardNodeBlock]:
    if start == end:
        return None

    return ForwardNodeBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        strike=batch.strike_price[start:end],
        settlement_date_uix=batch.settle_date_uix[start:end],
        fx_under_to_settle_uix=batch.fx_under_to_settle_ids[start:end],
        fx_strike_to_settle_uix=batch.fx_strike_to_settle_ids[start:end],
        fx_under_to_settle_invert=batch.fx_under_to_settle_invert[
            start:end
        ].astype(bool, copy=False),
        fx_strike_to_settle_invert=batch.fx_strike_to_settle_invert[
            start:end
        ].astype(bool, copy=False),
        under_discount_curve_uix=batch.under_discount_curve_ids[start:end],
        strike_discount_curve_uix=batch.strike_discount_curve_ids[start:end],
        price_uix=batch.price_rf_ids[start:end],
    )


def compile_forward_curve(
    batch: IngestBatch,
    start: int,
    end: int,
) -> Optional[ForwardCurveBlock]:
    if start == end:
        return None

    return ForwardCurveBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        strike=batch.strike_price[start:end],
        settlement_date_uix=batch.settle_date_uix[start:end],
        fx_under_to_settle_uix=batch.fx_under_to_settle_ids[start:end],
        fx_strike_to_settle_uix=batch.fx_strike_to_settle_ids[start:end],
        fx_under_to_settle_invert=batch.fx_under_to_settle_invert[
            start:end
        ].astype(bool, copy=False),
        fx_strike_to_settle_invert=batch.fx_strike_to_settle_invert[
            start:end
        ].astype(bool, copy=False),
        under_discount_curve_uix=batch.under_discount_curve_ids[start:end],
        strike_discount_curve_uix=batch.strike_discount_curve_ids[start:end],
        price_curve_uix=batch.price_curve_ids[start:end],
        delivery_date_uix=batch.delivery_date_uix[start:end],
    )
