# riskquant/portfolio/payloads/spot.py
from dataclasses import dataclass

import numpy as np

from riskquant.portfolio.constants import FX_NONE
from riskquant.portfolio.payloads.base import Payload


@dataclass(frozen=True, slots=True)
class SpotPayload(Payload):
    quantity: np.float64
    price_rf_id: np.int32
    fx_rf_id: np.int32 = FX_NONE

    def __str__(self) -> str:
        return (
            f"SpotPayload("
            f"quantity={self.quantity}, "
            f"price_rf_id={self.price_rf_id}, "
            f"fx_rf_id={self.fx_rf_id}"
            f")"
        )
