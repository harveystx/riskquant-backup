# riskquant/market_data/risk_factors/snapshot.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True, slots=True)
class RiskFactorLevelSnapshot:
    as_of_date: date
    factor_ids: np.ndarray  # (N0,) int32
    factor_levels: np.ndarray  # (N0,) float32/float64

    def __post_init__(self) -> None:
        # Enforce minimal invariants for downstream kernels.
        # This is a one-time cost at snapshot creation, and prevents hidden slow paths later.
        ids = self.factor_ids
        lv = self.factor_levels

        if ids.ndim != 1:
            raise ValueError(f"factor_ids must be 1D; got shape={ids.shape}")
        if lv.ndim != 1:
            raise ValueError(f"factor_levels must be 1D; got shape={lv.shape}")
        if ids.shape[0] != lv.shape[0]:
            raise ValueError(
                f"factor_ids and factor_levels length mismatch: "
                f"{ids.shape[0]} vs {lv.shape[0]}"
            )
        if ids.dtype != np.int32:
            raise TypeError(f"factor_ids must be int32; got {ids.dtype}")
        if lv.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"factor_levels must be float32 or float64; got {lv.dtype}"
            )

        # Require contiguous to keep later gathers fast and predictable.
        if not ids.flags["C_CONTIGUOUS"]:
            raise ValueError("factor_ids must be C-contiguous")
        if not lv.flags["C_CONTIGUOUS"]:
            raise ValueError("factor_levels must be C-contiguous")
