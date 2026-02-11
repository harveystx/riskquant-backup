# riskquant/valuation/pnl/pnl_writer.py
from __future__ import annotations

from typing import Protocol, Optional

import numpy as np


class PnlBatchWriter(Protocol):
    def write_batch(
        self,
        *,
        run_id: str,
        trade_uids: np.ndarray,  # (M,) int64
        scenario_offset: int,  # start scenario index
        pnl: np.ndarray,  # (M,B) float32/64 contiguous
        scenario_count_total: Optional[int],
    ) -> None:
        ...
