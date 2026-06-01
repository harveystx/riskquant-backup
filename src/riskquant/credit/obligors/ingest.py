# src/riskquant/credit/obligors/ingest.py
"""
Obligor master DataFrame -> `CompiledObligorCredit`.

The obligor edge: a human-facing master frame is cast to strict dtypes,
sorted by uid ascending, validated as one self-consistent universe, and
collapsed into a single immutable SoA container.

`uid` ascending sorted is a structural invariant the downstream layers
rely on for vectorized joins (`searchsorted` in `compile_jtd_portfolio_from_df`,
`aggregate_ead_to_obligor`, and `_resolve_collateral`). The same ordering
permutation is applied to every column so the SoA stays aligned row-wise.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.constants import NO_PARENT_OBLIGOR
from riskquant.credit.obligors.compiled import CompiledObligorCredit
from riskquant.credit.obligors.validate import validate_obligor_master_arrays

_REQUIRED_COLS: tuple[str, ...] = (
    "uid",
    "pd",
    "lgd",
    "rating_id",
    "sector_id",
    "country_id",
    "parent_uid",
)


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"compile_obligor_master: missing required column(s) {missing}."
        )


def _i64(df: pd.DataFrame, col: str) -> npt.NDArray[np.int64]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.int64))


def _f64(df: pd.DataFrame, col: str) -> npt.NDArray[np.float64]:
    arr = df[col].to_numpy(dtype=np.float64)
    if np.isnan(arr).any():
        raise ValueError(f"compile_obligor_master: NaN in non-nullable column '{col}'.")
    return np.ascontiguousarray(arr)


def _i8(df: pd.DataFrame, col: str) -> npt.NDArray[np.int8]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.int8))


def _i32(df: pd.DataFrame, col: str) -> npt.NDArray[np.int32]:
    return np.ascontiguousarray(df[col].to_numpy(dtype=np.int32))


def _parent_uid(df: pd.DataFrame) -> npt.NDArray[np.int64]:
    """NA / None -> NO_PARENT_OBLIGOR; numeric values pass through as int64."""
    filled = (
        df["parent_uid"]
        .astype("Int64")  # pandas nullable integer
        .fillna(NO_PARENT_OBLIGOR)
    )
    return cast(
        npt.NDArray[np.int64],
        np.ascontiguousarray(filled.to_numpy(dtype=np.int64)),
    )


def compile_obligor_master_from_df(
    obligor_df: pd.DataFrame,
) -> CompiledObligorCredit:
    """
    Compile a validated obligor-master frame into `CompiledObligorCredit`.

    Parameters
    ----------
    obligor_df
        One row per obligor. Required columns: uid, pd, lgd, rating_id,
        sector_id, country_id, parent_uid. `parent_uid` is nullable —
        NA/None marks an ultimate (no parent) and is mapped to
        `NO_PARENT_OBLIGOR`.

    Returns
    -------
    CompiledObligorCredit
        `uid` sorted ascending; all other columns reordered by the same
        permutation to stay row-aligned.
    """
    if obligor_df.empty:
        raise ValueError("compile_obligor_master: cannot compile an empty frame.")

    _check_required_columns(obligor_df)

    # Sort once by uid; one stable permutation applied to every column. Sorting
    # the frame (not the arrays) keeps the parent_uid fillna semantics intact.
    df = obligor_df.sort_values("uid", kind="stable").reset_index(drop=True)

    uid = _i64(df, "uid")
    pd_ = _f64(df, "pd")
    lgd = _f64(df, "lgd")
    rating_id = _i8(df, "rating_id")
    sector_id = _i32(df, "sector_id")
    country_id = _i32(df, "country_id")
    parent_uid = _parent_uid(df)

    validate_obligor_master_arrays(
        uid=uid,
        pd=pd_,
        lgd=lgd,
        rating_id=rating_id,
        sector_id=sector_id,
        country_id=country_id,
        parent_uid=parent_uid,
    )

    return CompiledObligorCredit(
        uid=uid,
        pd=pd_,
        lgd=lgd,
        rating_id=rating_id,
        sector_id=sector_id,
        country_id=country_id,
        parent_uid=parent_uid,
    )
