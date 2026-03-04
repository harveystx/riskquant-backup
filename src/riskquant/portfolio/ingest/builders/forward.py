"""
Module for building forward trade records from ingested batch data.

This module provides functionality to construct TradeRecord objects for
forward transactions, extracting position data from an IngestBatch and
validating required fields (such as strike price) before packaging the
data into standardized trade record format.
"""

from __future__ import annotations

import numpy as np

# Import custom exception for pretty error formatting
from riskquant.exceptions import PrettyKeyError

# Import helper function to check for missing/NA values in batch data
from riskquant.portfolio.ingest.parser import _is_na_helper as is_na

# Import IngestBatch for type hinting and batch data structure
from riskquant.portfolio.ingest.precomputer import IngestBatch

# Import the ForwardPayload data structure for forward trade metadata
from riskquant.portfolio.payloads.forward import ForwardPayload

# Import the TradeRecord class for creating standardized trade objects
from riskquant.portfolio.trades import TradeRecord


def build_forward(*, ix: int, batch: IngestBatch) -> TradeRecord:
    """
    Build a forward trade record from batch ingestion data with validation.

    Constructs a TradeRecord object for a forward transaction by extracting
    position data from a precompiled IngestBatch at a specific index. Validates
    that required fields (such as strike price) are present before wrapping
    the data in a ForwardPayload.

    Args:
        ix: Index into the batch arrays to identify the specific position
            within the batch to process.
        batch: IngestBatch object containing precomputed arrays of trade data,
               including UIDs, prices, quantities, strikes, settlement dates,
               and FX conversion rates.

    Returns:
        TradeRecord: A complete trade record for the forward position containing
                     the unique identifier, instrument ID, and payload with
                     pricing, quantity, strike, settlement date, and FX
                     conversion rate information.

    Raises:
        PrettyKeyError: If the strike price is missing (NA) for the forward
                        position, includes trade UID and underlying for context.
    """
    # Extract the unique identifier (uid) for this position by indexing into the batch UIDs array
    uid = int(batch.uids[ix])

    # Extract the strike price from the batch by row index using .iat[] for positional access
    K = batch.strike_price.iat[ix]

    # Validate that the strike price is present; raise error if missing or NA
    if is_na(K):
        raise PrettyKeyError(
            "FORWARD row missing strike_price:\n"
            f"- uid={uid}\n"
            f"- underlying={batch.rf_names.iat[ix]!r}\n"
        )

    # Construct and return a TradeRecord with all required forward trade information
    return TradeRecord(
        # Set the unique trade identifier from the extracted uid
        trade_uid=uid,
        # Set the instrument identifier from the batch's fwd_id (common for all forwards in batch)
        instrument_id=batch.fwd_id,
        # Create a ForwardPayload containing the price, quantity, strike, settlement, and FX data
        payload=ForwardPayload(
            # Store the price risk factor ID as a numpy int32 from the batch array
            price_rf_id=np.int32(batch.price_rf_ids[ix]),
            # Store the forward quantity as a numpy float64 from the batch array
            quantity=np.float64(batch.qty[ix]),
            # Store the validated strike price as a numpy float64
            strike=np.float64(K),
            # Store the settlement date as a numpy int32 ordinal (days since epoch)
            settlement_date_ordinal32=np.int32(batch.settle_ord[ix]),
            # Store the FX underlying-to-settlement risk factor ID as a numpy int32 from the batch array
            fx_under_to_settle_rf_id=np.int32(batch.fx_under_to_settle[ix]),
            # Store the FX strike-to-settlement risk factor ID as a numpy int32 from the batch array
            fx_strike_to_settle_rf_id=np.int32(batch.fx_strike_to_settle[ix]),
        ),
    )
