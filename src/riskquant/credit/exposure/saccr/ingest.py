# src/riskquant/credit/exposure/saccr/ingest.py
"""SA-CCR config ingest (the edge): config tables -> CompiledSaccrConfig.

Both tables are ``pl.DataFrame``. The column rules are those of the add-on
ingest (WS_CREDIT_POLARS C.7, C.8): the uix columns are integer (or integral
float) columns that fit int32, ``rho`` is a required float column.
"""

from __future__ import annotations

__all__ = ["compile_saccr_config"]

import numpy as np
import polars as pl

from riskquant.constants import SACCR_ALPHA_DEFAULT, SACCR_MULTIPLIER_FLOOR_DEFAULT
from riskquant.credit.exposure.add_on.ingest import (
    _float_column,
    _int_column,
    _require_polars_frame,
)
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig

_ENTRY = "compile_saccr_config"


def compile_saccr_config(
    subclass_hedging_set: pl.DataFrame,
    hedging_set_rho: pl.DataFrame,
    *,
    alpha: float = SACCR_ALPHA_DEFAULT,
    multiplier_floor: float = SACCR_MULTIPLIER_FLOOR_DEFAULT,
    subclass_col: str = "subclass_uix",
    hedging_set_col: str = "hedging_set_uix",
    rho_col: str = "rho",
) -> CompiledSaccrConfig:
    """Compile the subclass->hedging-set map and hedging-set->rho table.

    Raises
    ------
    TypeError
        A table is not a ``pl.DataFrame`` (both are checked before any column
        is read); a column has a dtype that its rule does not admit; a uix
        value is fractional or outside int32.
    ValueError
        A column is missing; a null (or NaN) cell; a duplicate subclass or
        hedging set.
    """
    shs = _require_polars_frame(
        subclass_hedging_set, entry=_ENTRY, arg="subclass_hedging_set"
    )
    hsr = _require_polars_frame(hedging_set_rho, entry=_ENTRY, arg="hedging_set_rho")
    for col in (subclass_col, hedging_set_col):
        if col not in shs.columns:
            raise ValueError(f"{_ENTRY}: subclass_hedging_set missing '{col}'.")
    for col in (hedging_set_col, rho_col):
        if col not in hsr.columns:
            raise ValueError(f"{_ENTRY}: hedging_set_rho missing '{col}'.")

    sc = _int_column(shs, subclass_col, np.int32, entry=_ENTRY)
    sc_hs = _int_column(shs, hedging_set_col, np.int32, entry=_ENTRY)
    if sc.shape[0] != np.unique(sc).shape[0]:
        raise ValueError(f"{_ENTRY}: duplicate subclass_uix.")
    order = np.argsort(sc, kind="stable")

    hs = _int_column(hsr, hedging_set_col, np.int32, entry=_ENTRY)
    rho = _float_column(hsr, rho_col, entry=_ENTRY)
    if hs.shape[0] != np.unique(hs).shape[0]:
        raise ValueError(f"{_ENTRY}: duplicate hedging_set_uix.")
    h_order = np.argsort(hs, kind="stable")

    return CompiledSaccrConfig(
        subclass_uix=np.ascontiguousarray(sc[order]),
        subclass_hedging_set=np.ascontiguousarray(sc_hs[order]),
        hedging_set_uix=np.ascontiguousarray(hs[h_order]),
        hedging_set_rho=np.ascontiguousarray(rho[h_order]),
        alpha=float(alpha),
        multiplier_floor=float(multiplier_floor),
    )
