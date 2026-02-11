# riskquant/portfolio/trades.py
from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.payloads.base import Payload


@dataclass(frozen=True, slots=True)
class TradeRecord:
    trade_uid: int
    instrument_id: np.int32
    payload: Payload

    def __str__(self) -> str:
        return (
            f"TradeRecord(\n"
            f"  trade_uid={self.trade_uid},\n"
            f"  instrument_id={self.instrument_id},\n"
            f"  payload={self.payload}\n"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()
