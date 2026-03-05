# src/riskquant/portfolio/instruments.py
from __future__ import annotations

from enum import Enum

import numpy as np


class Instrument(Enum):
    """Enum for financial instruments in the risk quant portfolio."""

    SPOT = np.int32(1)
    FORWARD_NODE = np.int32(2)
    FORWARD_CURVE = np.int32(3)
    VANILLA_EUROPEAN_OPTION_NODE = np.int32(4)
    VANILLA_EUROPEAN_OPTION_CURVE = np.int32(5)

    @classmethod
    def all_instrument_ids(cls) -> set[np.int32]:
        """Get a set of all valid instrument IDs."""
        return {member.value for member in cls}

    @classmethod
    def from_id(cls, id_: np.int32) -> "Instrument":
        """Get the Instrument enum member corresponding to the given ID."""
        return cls(id_)


# Get integer values for instruments for easy reference when building trade records in builders/ modules
SPOT_ID: np.int32 = Instrument.SPOT.value
FORWARD_NODE_ID: np.int32 = Instrument.FORWARD_NODE.value
FORWARD_CURVE_ID: np.int32 = Instrument.FORWARD_CURVE.value
VANILLA_EUROPEAN_OPTION_NODE_ID: np.int32 = (
    Instrument.VANILLA_EUROPEAN_OPTION_NODE.value
)
VANILLA_EUROPEAN_OPTION_CURVE_ID: np.int32 = (
    Instrument.VANILLA_EUROPEAN_OPTION_CURVE.value
)


# All valid instrument IDs for quick validation in trade record creation
ALL_INSTRUMENT_IDS: set[np.int32] = Instrument.all_instrument_ids()
