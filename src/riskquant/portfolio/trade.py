# src/riskquant/portfolio/trade.py
"""
Module for defining standardized trade record data structures.
"""

from dataclasses import dataclass
import numpy as np

from riskquant.portfolio.payloads.base_payload import Payload


@dataclass(frozen=True, slots=True, kw_only=True)
class TradeRecord:
    """
    Immutable, high-performance container representing a single trade record.

    Assumes all data has been strictly validated and typed by the upstream
    IngestBatch vectorizer before instantiation.

    Attributes:
        trade_uid: Unique integer identifier for this specific trade.
        instrument_id: Numpy int32 identifier linking to the instrument Enum.
        payload: Polymorphic Payload object containing instrument-specific data.
    """

    trade_uid: int
    instrument_id: np.int32
    payload: Payload

    def __str__(self) -> str:
        """
        Return a formatted string representation of the trade record.
        """
        return (
            f"TradeRecord(\n"
            f"  trade_uid={self.trade_uid},\n"
            f"  instrument_id={self.instrument_id},\n"
            f"  payload={self.payload}\n"
            f")"
        )
