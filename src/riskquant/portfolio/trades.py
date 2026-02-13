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

# Import the abstract Payload base class for type polymorphism
from riskquant.portfolio.payloads.base import Payload


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """
    Immutable data class representing a single trade record.

    TradeRecord encapsulates the fundamental information for any trade:
    a unique identifier, an instrument reference, and a payload containing
    instrument-specific data (pricing, quantity, strike prices, etc.).

    The class is frozen to ensure immutability and uses slots for memory
    efficiency. String representations are provided for debugging and
    logging purposes.

    Attributes:
        trade_uid: Unique integer identifier for this specific trade.
                   Used to track and reference the trade throughout
                   risk calculations and portfolio analysis.
        instrument_id: Numpy int32 identifier linking to the instrument
                       definition (e.g., spot, forward, option). This ID
                       is common across all trades of the same instrument
                       within a batch.
        payload: Polymorphic Payload object containing instrument-specific
                 data. The concrete type (SpotPayload, ForwardPayload, etc.)
                 is determined by the instrument type and contains all
                 necessary valuation parameters.
    """

    # Unique identifier for this trade
    trade_uid: int
    # Instrument definition identifier (references instrument metadata)
    instrument_id: np.int32
    # Instrument-specific data payload (polymorphic, subclass of Payload)
    payload: Payload

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

    def __repr__(self) -> str:
        """
        Return the official string representation of the trade record.

        Delegates to __str__() to provide a consistent, human-readable
        representation. This is called by Python's repr() function and
        used when displaying the object in the REPL or logs.

        Returns:
            str: The same formatted representation as __str__().
        """
        # Use the string representation for repr (common pattern for display-friendly objects)
        return self.__str__()
