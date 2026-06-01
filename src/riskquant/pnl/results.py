# src/riskquant/pnl/results.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class PnlMatrix:
    """
    Result container for position-level PnL simulations.

    Attributes:
        trade_uids: (M,) int64 array; row i of pnl corresponds to trade_uids[i]
        pnl:        (M, S) float32 PnL matrix.
    """

    trade_uids: np.ndarray
    pnl: np.ndarray

    def __post_init__(self) -> None:
        # Strict DOD enforcement: Asserts only, no hidden allocations.
        if self.pnl.dtype != np.float32:
            raise TypeError(
                "PnlResult requires pnl to be explicitly cast to float32."
            )
        if not self.pnl.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "PnlResult pnl matrix must be C-contiguous for optimal downstream aggregation."
            )
