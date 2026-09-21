# src/riskquant/credit/masters/obligors/compiled.py
from __future__ import annotations

__all__ = [
    "CompiledObligorCredit",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CompiledObligorCredit:
    """
    Universal obligor credit data.
    Consumed by ALL credit kernels (Deterministic JTD, Expected Loss, Copula).
    """

    uid: npt.NDArray[np.int64]  # obligor ID; sorted ascending, unique
    pd: npt.NDArray[np.float64]
    lgd: npt.NDArray[np.float64]
    rating_id: npt.NDArray[np.int8]  # int enum from config/enums/credit_ratings
    sector_id: npt.NDArray[np.int32]  # int enum, future
    country_id: npt.NDArray[np.int32]  # int enum, future
    parent_uid: npt.NDArray[np.int64]  # NO_PARENT_OBLIGOR sentinel for ultimates
