# src/riskquant/portfolio/instruments.py
from __future__ import annotations

from enum import IntEnum

import numpy as np


class Instrument(IntEnum):
    """Enum for financial instruments in the risk quant portfolio."""

    SPOT = 0
    FORWARD_NODE = 1
    FORWARD_CURVE = 2
    VANILLA_EUROPEAN_OPTION_NODE = 3
    VANILLA_EUROPEAN_OPTION_CURVE = 4

    @classmethod
    def all_instrument_ids(cls) -> set[int]:
        """Get a set of all valid instrument IDs."""
        return {member.value for member in cls}

    @classmethod
    def from_id(cls, id_: int) -> "Instrument":
        """Get the Instrument enum member corresponding to the given ID."""
        return cls(id_)

    @classmethod
    def num_instruments(cls) -> int:
        """Get the total number of instruments defined."""
        return len(cls)


# Get integer values for instruments for easy reference when building trade records in builders/ modules
SPOT_ID: int = Instrument.SPOT.value
FORWARD_NODE_ID: int = Instrument.FORWARD_NODE.value
FORWARD_CURVE_ID: int = Instrument.FORWARD_CURVE.value
VANILLA_EUROPEAN_OPTION_NODE_ID: int = (
    Instrument.VANILLA_EUROPEAN_OPTION_NODE.value
)
VANILLA_EUROPEAN_OPTION_CURVE_ID: int = (
    Instrument.VANILLA_EUROPEAN_OPTION_CURVE.value
)


# All valid instrument IDs for quick validation in trade record creation
ALL_INSTRUMENT_IDS: set[int] = Instrument.all_instrument_ids()


# Total number of instruments defined (for dispatch table sizing)
NUM_INSTRUMENTS: int = Instrument.num_instruments()
