# src/riskquant/valuation/result.py
"""
Valuation results container.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class ValuationResult:
    trade_uids: npt.NDArray[np.int64]  # (n_trades,)
    values: npt.NDArray[np.float64]  # (n_trades, n_scenarios)

    def __post_init__(self) -> None:
        if self.values.shape[0] != self.trade_uids.shape[0]:
            raise ValueError("UIDs and values row mismatch")

    @property
    def num_trades(self) -> int:
        return self.values.shape[0]

    @property
    def num_scenarios(self) -> int:
        return self.values.shape[1]

    def get_trade_values(self, uid: int) -> npt.NDArray[np.float64]:
        idx = np.searchsorted(self.trade_uids, uid)
        if self.trade_uids[idx] != uid:
            raise KeyError(uid)
        return self.values[idx]

    def select(self, uids: npt.NDArray[np.int64]) -> "ValuationResult":
        idx = np.isin(self.trade_uids, uids)
        return ValuationResult(
            trade_uids=self.trade_uids[idx],
            values=self.values[idx],
        )
