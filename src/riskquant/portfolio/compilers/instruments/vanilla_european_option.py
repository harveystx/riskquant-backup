# src/riskquant/portfolio/compilers/instruments/vanilla_european_option.py
from __future__ import annotations
from typing import Optional

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionNodeBlock,
    VanillaEuropeanOptionCurveBlock,
)


def compile_vanilla_euro_option_node(
    batch: IngestBatch,
    start: int,
    end: int,
) -> Optional[VanillaEuropeanOptionNodeBlock]:
    if start == end:
        return None

    return VanillaEuropeanOptionNodeBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        strike=batch.strike_price[start:end],
        is_call=batch.is_call[start:end],
        expiry_date_uix=batch.expiry_date_uix[start:end],
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
        underlying_price_uix=batch.price_rf_ids[start:end],
        vol_uix=batch.vol_rf_ids[start:end],
    )


def compile_vanilla_euro_option_curve(
    batch: IngestBatch,
    start: int,
    end: int,
) -> Optional[VanillaEuropeanOptionCurveBlock]:
    if start == end:
        return None

    return VanillaEuropeanOptionCurveBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        strike=batch.strike_price[start:end],
        is_call=batch.is_call[start:end],
        expiry_date_uix=batch.expiry_date_uix[start:end],
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
        underlying_price_curve_uix=batch.price_curve_ids[start:end],
        delivery_date_uix=batch.delivery_date_uix[start:end],
        vol_surface_uix=batch.vol_surface_rf_ids[start:end],
        vol_surface_y_axis_type_id=batch.vol_surface_y_axis_type_ids[start:end],
    )
