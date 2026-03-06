# src/riskquant/portfolio/ingest/builders/vanilla_european_option.py
"""
Module for building vanilla European option trade records from ingested batch data.
"""

from __future__ import annotations

import numpy as np

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.instruments import (
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
)
from riskquant.portfolio.payloads.vanilla_european_option import (
    VanillaEuropeanOptionNodePayload,
    VanillaEuropeanOptionCurvePayload,
)
from riskquant.portfolio.trade import TradeRecord


def build_vanilla_european_option_node(
    *,
    ix: int,
    batch: IngestBatch,
) -> TradeRecord:
    """
    Build a vanilla European option trade record priced off a single node (requires expiry date but not delivery date).
    """
    return TradeRecord(
        trade_uid=int(batch.uids[ix]),
        instrument_id=VANILLA_EUROPEAN_OPTION_NODE_ID,
        payload=VanillaEuropeanOptionNodePayload(
            quantity=np.float64(batch.qty[ix]),
            strike=np.float64(batch.strike_price[ix]),
            is_call=np.int8(int(batch.is_call[ix])),
            expiry_date_ordinal32=np.int32(batch.expiry_ord[ix]),
            payoff_settlement_date_ordinal32=np.int32(batch.settle_ord[ix]),
            fx_under_to_settle_rf_id=np.int32(batch.fx_under_to_settle[ix]),
            fx_strike_to_settle_rf_id=np.int32(batch.fx_strike_to_settle[ix]),
            underlying_price_rf_id=np.int32(batch.price_rf_ids[ix]),
            vol_rf_id=np.int32(batch.vol_rf_ids[ix]),
        ),
    )


def build_vanilla_european_option_curve(
    *,
    ix: int,
    batch: IngestBatch,
) -> TradeRecord:
    """
    Build a vanilla European option trade record priced off curves/surfaces.
    """
    return TradeRecord(
        trade_uid=int(batch.uids[ix]),
        instrument_id=VANILLA_EUROPEAN_OPTION_CURVE_ID,
        payload=VanillaEuropeanOptionCurvePayload(
            quantity=np.float64(batch.qty[ix]),
            strike=np.float64(batch.strike_price[ix]),
            is_call=np.int8(int(batch.is_call[ix])),
            expiry_date_ordinal32=np.int32(batch.expiry_ord[ix]),
            payoff_settlement_date_ordinal32=np.int32(batch.settle_ord[ix]),
            fx_under_to_settle_rf_id=np.int32(batch.fx_under_to_settle[ix]),
            fx_strike_to_settle_rf_id=np.int32(batch.fx_strike_to_settle[ix]),
            underlying_price_curve_rf_id=np.int32(batch.price_curve_ids[ix]),
            delivery_date_ordinal32=np.int32(batch.delivery_ord[ix]),
            vol_surface_rf_id=np.int32(batch.vol_surface_rf_ids[ix]),
        ),
    )
