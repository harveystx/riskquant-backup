# src/riskquant/portfolio/ingest/builders/forward.py
"""
Module for building forward trade records from ingested batch data.
"""

from __future__ import annotations

import numpy as np

from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio import instruments
from riskquant.portfolio.ingest.utils import is_na_helper
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.payloads.forward import (
    ForwardNodePayload,
    ForwardCurvePayload,
)
from riskquant.portfolio.trades import TradeRecord
from riskquant.portfolio.constants import NO_DELIVERY_DATE_ORDINAL32


def build_forward_node(
    *,
    ix: int,
    batch: IngestBatch,
) -> TradeRecord:
    """
    Build a forward node trade record priced off a single node (requires settlement date but not delivery date).
    """
    return TradeRecord(
        trade_uid=int(batch.uids[ix]),
        instrument_id=np.int32(batch.forward_node_id),
        payload=ForwardNodePayload(
            quantity=np.float64(batch.qty[ix]),
            strike=np.float64(batch.strike_price[ix]),
            settlement_date_ordinal32=np.int32(batch.settle_ord[ix]),
            fx_under_to_settle_rf_id=np.int32(batch.fx_under_to_settle[ix]),
            fx_strike_to_settle_rf_id=np.int32(batch.fx_strike_to_settle[ix]),
            price_rf_id=np.int32(batch.price_rf_ids[ix]),
        ),
    )


def build_forward_curve(
    *,
    ix: int,
    batch: IngestBatch,
) -> TradeRecord:
    """
    Build a forward trade record priced off a curve (requires delivery date).
    """
    return TradeRecord(
        trade_uid=int(batch.uids[ix]),
        instrument_id=np.int32(batch.forward_curve_id),
        payload=ForwardCurvePayload(
            quantity=np.float64(batch.qty[ix]),
            strike=np.float64(batch.strike_price[ix]),
            settlement_date_ordinal32=np.int32(batch.settle_ord[ix]),
            fx_under_to_settle_rf_id=np.int32(batch.fx_under_to_settle[ix]),
            fx_strike_to_settle_rf_id=np.int32(batch.fx_strike_to_settle[ix]),
            price_curve_id=np.int32(batch.price_curve_ids[ix]),
            delivery_date_ordinal32=np.int32(batch.delivery_ord[ix]),
        ),
    )
