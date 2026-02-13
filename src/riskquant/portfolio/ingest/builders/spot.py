"""
Module for building spot trade records from ingested batch data.

This module provides functionality to construct TradeRecord objects for spot
transactions, extracting relevant position data from an IngestBatch and
packaging it into standardized trade record format.
"""

from __future__ import annotations

import numpy as np

# Import the SpotPayload data structure for spot trade metadata
from riskquant.portfolio.payloads.spot import SpotPayload

# Import the TradeRecord class for creating standardized trade objects
from riskquant.portfolio.trades import TradeRecord

# Import IngestBatch for type hinting and batch data structure
from riskquant.portfolio.ingest.precomputer import IngestBatch


def build_spot(
    *,
    ix: int,
    batch: IngestBatch,
) -> TradeRecord:
    """
    Build a spot trade record from batch ingestion data.

    Constructs a TradeRecord object for a spot transaction by extracting
    position data from a precompiled IngestBatch at a specific index and
    wrapping it in a SpotPayload.

    Args:
        ix: Index into the batch arrays to identify the specific position
                within the batch to process.
        batch: IngestBatch object containing precomputed arrays of trade data,
               including UIDs, prices, quantities, and FX conversion rates.

    Returns:
        TradeRecord: A complete trade record for the spot position containing
                     the unique identifier, instrument ID, and payload with
                     pricing, quantity, and FX rate information.
    """
    # Extract the unique identifier (uid) for this position by indexing into the batch UIDs array
    uid = int(batch.uids[ix])

    # Construct and return a TradeRecord with all required spot trade information
    return TradeRecord(
        # Set the unique trade identifier from the extracted uid
        trade_uid=uid,
        # Set the instrument identifier from the batch's spot_id (common for all spots in batch)
        instrument_id=batch.spot_id,
        # Create a SpotPayload containing the price, quantity, and FX risk factor data
        payload=SpotPayload(
            # Store the price risk factor ID as a numpy int32 from the batch array
            price_rf_id=np.int32(batch.price_rf_ids[ix]),
            # Store the trade quantity as a numpy float64 from the batch array
            quantity=np.float64(batch.qty[ix]),
            # Store the FX underlying-to-settlement risk factor ID as a numpy int32 from the batch array
            fx_rf_id=np.int32(batch.fx_under_to_settle[ix]),
        ),
    )
