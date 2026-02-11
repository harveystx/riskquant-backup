# riskquant/portfolio/pnl/results.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PnlResult:
    """
    Result container for position-level PnL simulations.

    Attributes:
        trade_uids: (M,) int64 array; row i of pnl corresponds to trade_uids[i]
        pnl:        (M, S) PnL matrix, expressed in each trade's settlement_ccy
    """

    trade_uids: np.ndarray  # int64, shape (M,)
    pnl: np.ndarray  # float32/64, shape (M, S), contiguous
