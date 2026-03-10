# src/riskquant/portfolio/compilers/spot.py
from __future__ import annotations
from typing import Optional
import numpy as np

from riskquant.portfolio.instruments import SPOT_ID
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.blocks.spot_block import SpotBlock


def compile_spot(batch: IngestBatch) -> Optional[SpotBlock]:
    mask = batch.instrument_ids == SPOT_ID

    if not np.any(mask):
        return None

    return SpotBlock(
        trade_uids=batch.uids[mask],
        qty=batch.qty[mask],
        price_uix=batch.price_rf_ids[mask],
        fx_uix=batch.fx_under_to_settle_ids[mask],
        fx_invert=batch.fx_under_to_settle_invert[mask].astype(bool),
    )
