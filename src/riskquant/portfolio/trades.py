"""
Module for defining standardized trade record data structures.

This module provides the core TradeRecord class which represents a single
trade with its unique identifier, associated instrument, and type-specific
payload containing all relevant pricing and valuation data.
"""

# Import dataclass decorator for creating immutable data structures
from dataclasses import dataclass

# Import numpy for numeric type definitions
import numpy as np
from openpyxl.descriptors import Integer

from riskquant.portfolio import instruments
from riskquant.portfolio.payloads.base_payload import Payload


@dataclass(frozen=True, slots=True, kw_only=True)
class TradeRecord:
    """
    Immutable data class representing a single trade record.

    TradeRecord encapsulates the fundamental information for any trade:
    a unique identifier, an instrument reference, and a payload containing
    instrument-specific data (pricing, quantity, strike prices, etc.).

    Attributes:
        trade_uid: Unique integer identifier for this specific trade.
        instrument_id: Numpy int32 identifier linking to the instrument
                       definition (e.g., spot, forward, option).
        payload: Polymorphic Payload object containing instrument-specific
                 data.
    """

    trade_uid: int
    instrument_id: np.int32
    payload: Payload

    def __post_init__(self):
        # trade_uid is expected to be a Python int (not numpy int) for easier handling in dicts and logs
        if self.trade_uid is None:
            raise ValueError("trade_uid cannot be None")
        if not isinstance(self.trade_uid, int):
            raise TypeError(
                f"trade_uid must be an int, got {type(self.trade_uid)}"
            )

        # instrument_id should be a numpy int32 for consistency with batch data
        if self.instrument_id is None:
            raise ValueError("instrument_id cannot be None")
        if not isinstance(self.instrument_id, (int, np.integer)):
            raise TypeError(
                f"instrument_id must be a numpy int32, got {type(self.instrument_id)}"
            )
        object.__setattr__(self, "instrument_id", np.int32(self.instrument_id))
        if self.instrument_id not in instruments.ALL_INSTRUMENT_IDS:
            raise ValueError(
                f"instrument_id = {self.instrument_id} is not a valid instrument ID. Valid IDs: {instruments.ALL_INSTRUMENT_IDS}"
            )

        # Payload must be an instance of a subclass of Payload
        if self.payload is None:
            raise ValueError("payload cannot be None")
        if not isinstance(self.payload, Payload):
            raise TypeError(
                f"payload must be an instance of Payload, "
                f"got {type(self.payload)}"
            )

    def __str__(self) -> str:
        """
        Return a formatted string representation of the trade record.

        Produces a multi-line human-readable string showing the trade UID,
        instrument ID, and the string representation of the payload object.
        Useful for debugging, logging, and interactive inspection.

        Returns:
            str: A formatted string representation of all trade record fields.
        """
        # Format the trade record as a multi-line string with proper indentation
        return (
            f"TradeRecord(\n"
            f"  trade_uid={self.trade_uid},\n"
            f"  instrument_id={self.instrument_id},\n"
            f"  payload={self.payload}\n"
            f")"
        )
