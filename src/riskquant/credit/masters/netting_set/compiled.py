# src/riskquant/credit/masters/netting_set/compiled.py
from __future__ import annotations

__all__ = [
    "CompiledNettingSetMaster",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledNettingSetMaster:
    """
    Compiled netting-set master data.

    This only contains information relevant to netting-set-level collateral, which is
    all that's needed for exposure calculation. The master is keyed by `uid`, which
    is the netting-set id ingested from the batch and joined onto the portfolio's
    `trade_uids` to align collateral to priced trades.
    """

    # --- NETTING SET UID ---
    # ingest column netting_set_id; sorted ascending, unique
    uid: npt.NDArray[np.int64]

    # --- OBLIGOR UID ---
    # Obligor uid of the netting set; aligned to `uid`
    obligor_uid: npt.NDArray[np.int64]

    # --- COLLATERAL PER NETTING SET ---
    # net collateral per netting set; aligned to `uid`
    net_collateral: npt.NDArray[np.float64]
