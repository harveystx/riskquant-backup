# src/riskquant/credit/masters/netting_set/ingest.py
"""
Netting-set master DataFrame -> `CompiledNettingSetMaster`.

The netting-set edge: a human-facing master frame is cast to strict dtypes,
sorted by netting-set uid ascending, validated, and collapsed into a single
immutable SoA container. The ascending uid invariant is what
`compile_exposure._resolve_collateral` relies on for its `searchsorted`
join from `netting_set_uids` -> `(obligor_uid, net_collateral)`.
"""

from __future__ import annotations

import warnings

__all__ = [
    "compile_netting_set_master",
]

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.credit.masters.netting_set.validate import (
    validate_netting_set_master_arrays,
)

_REQUIRED_COLS: tuple[str, ...] = ("uid", "obligor_uid", "net_collateral")


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"compile_netting_set_master: missing required column(s) {missing}."
        )


def _i64(df: pd.DataFrame, col: str) -> npt.NDArray[np.int64]:
    arr = df[col].to_numpy(dtype=np.int64)
    return np.ascontiguousarray(arr)


def _f64(df: pd.DataFrame, col: str) -> npt.NDArray[np.float64]:
    arr = df[col].to_numpy(dtype=np.float64)
    if np.isnan(arr).any():
        raise ValueError(
            f"compile_netting_set_master: NaN in non-nullable column '{col}'."
        )
    return np.ascontiguousarray(arr)


def compile_netting_set_master(
    netting_set_df: pd.DataFrame,
) -> CompiledNettingSetMaster:
    """
    Compile a validated netting-set master frame into `CompiledNettingSetMaster`.

    Parameters
    ----------
    netting_set_df
        One row per effective netting set. Required columns:
        - `uid` (int64): netting-set identifier; matches the
          `netting_set_id` column on the portfolio ingest frame.
        - `obligor_uid` (int64): obligor that owns the netting set; the
          join key from the exposure result into the obligor space.
        - `net_collateral` (float64): haircut-adjusted collateral, >= 0.

    Returns
    -------
    CompiledNettingSetMaster
        `uid` sorted ascending; other columns reordered by the same
        permutation to stay row-aligned.
    """
    if netting_set_df.empty:
        raise ValueError("compile_netting_set_master: cannot compile an empty frame.")

    _check_required_columns(netting_set_df)

    df = netting_set_df.sort_values("uid", kind="stable").reset_index(drop=True)

    uid = _i64(df, "uid")
    obligor_uid = _i64(df, "obligor_uid")
    net_collateral = _f64(df, "net_collateral")

    validate_netting_set_master_arrays(
        uid=uid,
        obligor_uid=obligor_uid,
        net_collateral=net_collateral,
    )

    return CompiledNettingSetMaster(
        uid=uid,
        obligor_uid=obligor_uid,
        net_collateral=net_collateral,
    )

