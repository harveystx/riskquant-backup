# src/riskquant/portfolio/compilers/instruments/spot.py
from __future__ import annotations
from typing import Optional

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.blocks.spot_block import SpotBlock


def compile_spot(
    batch: IngestBatch,
    start: int,
    end: int,
) -> Optional[SpotBlock]:
    if start == end:
        return None

    # Slicing a contiguous numpy array [start:end] returns a view.
    # Zero memory copy. O(1) CPU time.
    return SpotBlock(
        trade_uids=batch.uids[start:end],
        qty=batch.qty[start:end],
        price_uix=batch.price_rf_ids[start:end],
        fx_uix=batch.fx_under_to_settle_ids[start:end],
        fx_invert=batch.fx_under_to_settle_invert[start:end].astype(
            bool, copy=False
        ),
    )
