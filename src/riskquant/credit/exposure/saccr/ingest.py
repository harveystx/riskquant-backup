# src/riskquant/credit/exposure/saccr/ingest.py
"""SA-CCR config ingest (the edge): config tables -> CompiledSaccrConfig."""

from __future__ import annotations

__all__ = ["compile_saccr_config"]

import numpy as np
import pandas as pd

from riskquant.constants import SACCR_ALPHA_DEFAULT, SACCR_MULTIPLIER_FLOOR_DEFAULT
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig


def compile_saccr_config(
    subclass_hedging_set: pd.DataFrame,
    hedging_set_rho: pd.DataFrame,
    *,
    alpha: float = SACCR_ALPHA_DEFAULT,
    multiplier_floor: float = SACCR_MULTIPLIER_FLOOR_DEFAULT,
    subclass_col: str = "subclass_uix",
    hedging_set_col: str = "hedging_set_uix",
    rho_col: str = "rho",
) -> CompiledSaccrConfig:
    """Compile the subclass->hedging-set map and hedging-set->rho table."""
    for col in (subclass_col, hedging_set_col):
        if col not in subclass_hedging_set.columns:
            raise ValueError(
                f"compile_saccr_config: subclass_hedging_set missing '{col}'."
            )
    for col in (hedging_set_col, rho_col):
        if col not in hedging_set_rho.columns:
            raise ValueError(f"compile_saccr_config: hedging_set_rho missing '{col}'.")

    sc = subclass_hedging_set[subclass_col].to_numpy(dtype=np.int32)
    sc_hs = subclass_hedging_set[hedging_set_col].to_numpy(dtype=np.int32)
    if sc.shape[0] != np.unique(sc).shape[0]:
        raise ValueError("compile_saccr_config: duplicate subclass_uix.")
    order = np.argsort(sc, kind="stable")

    hs = hedging_set_rho[hedging_set_col].to_numpy(dtype=np.int32)
    rho = hedging_set_rho[rho_col].to_numpy(dtype=np.float64)
    if hs.shape[0] != np.unique(hs).shape[0]:
        raise ValueError("compile_saccr_config: duplicate hedging_set_uix.")
    h_order = np.argsort(hs, kind="stable")

    return CompiledSaccrConfig(
        subclass_uix=np.ascontiguousarray(sc[order]),
        subclass_hedging_set=np.ascontiguousarray(sc_hs[order]),
        hedging_set_uix=np.ascontiguousarray(hs[h_order]),
        hedging_set_rho=np.ascontiguousarray(rho[h_order]),
        alpha=float(alpha),
        multiplier_floor=float(multiplier_floor),
    )
