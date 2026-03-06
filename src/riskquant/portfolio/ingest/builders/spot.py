# src/riskquant/portfolio/ingest/builders/spot.py
"""
Module for building spot trade records from ingested batch data.
"""


import numpy as np
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.instruments import SPOT_ID
from riskquant.portfolio.payloads.spot import SpotPayload
from riskquant.portfolio.trade import TradeRecord


def build_spot(*, ix: int, batch: IngestBatch) -> TradeRecord:
    return TradeRecord(
        trade_uid=int(batch.uids[ix]),
        instrument_id=SPOT_ID,
        payload=SpotPayload(
            quantity=np.float64(batch.qty[ix]),
            price_rf_id=np.int32(batch.price_rf_ids[ix]),
            fx_rf_id=np.int32(batch.fx_under_to_settle[ix]),
        ),
    )
