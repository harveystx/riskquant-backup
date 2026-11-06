# src/riskquant/credit/masters/netting_set/ingest.py
"""
Netting-set master frame -> `CompiledNettingSetMaster`.

The netting-set edge: a human-facing ``pl.DataFrame`` is read column by column
into strict dtypes, sorted by netting-set uid ascending, validated, and
collapsed into a single immutable SoA container. The ascending uid invariant is
what `compile_exposure._resolve_collateral` relies on for its `searchsorted`
join from `netting_set_uids` -> `(obligor_uid, net_collateral)`.

Column rules (WS_CREDIT_POLARS C.8): ``uid`` and ``obligor_uid`` are integer
columns (an integer dtype, or a float dtype with integral values);
``net_collateral`` is a float column.
"""

from __future__ import annotations

__all__ = [
    "compile_netting_set_master",
]

import numpy as np
import polars as pl

from riskquant.credit.exposure.add_on.ingest import (
    _float_column,
    _int_column,
    _require_polars_frame,
)
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.credit.masters.netting_set.validate import (
    validate_netting_set_master_arrays,
)

_ENTRY = "compile_netting_set_master"

_REQUIRED_COLS: tuple[str, ...] = ("uid", "obligor_uid", "net_collateral")


def _check_required_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"{_ENTRY}: missing required column(s) {missing}.")


def compile_netting_set_master(
    netting_set_df: pl.DataFrame,
) -> CompiledNettingSetMaster:
    """
    Compile a validated netting-set master frame into `CompiledNettingSetMaster`.

    Parameters
    ----------
    netting_set_df
        ``pl.DataFrame``, one row per effective netting set. Required columns:
        - `uid` (integer): netting-set identifier; matches the
          `netting_set_id` column on the portfolio ingest frame.
        - `obligor_uid` (integer): obligor that owns the netting set; the
          join key from the exposure result into the obligor space.
        - `net_collateral` (float): haircut-adjusted collateral.
        The caller's frame is not changed.

    Returns
    -------
    CompiledNettingSetMaster
        `uid` sorted ascending; other columns reordered by the same
        permutation to stay row-aligned. Every array is C-contiguous and
        writeable.

    Raises
    ------
    TypeError
        ``netting_set_df`` is not a ``pl.DataFrame``; a column dtype that its
        rule does not admit; a fractional or out-of-range integer value.
    ValueError
        An empty frame; a missing required column; a null (or NaN) cell; a
        failed master validation.
    """
    frame = _require_polars_frame(netting_set_df, entry=_ENTRY, arg="netting_set_df")
    if frame.height == 0:
        raise ValueError(f"{_ENTRY}: cannot compile an empty frame.")

    _check_required_columns(frame)

    uid_raw = _int_column(frame, "uid", np.int64, entry=_ENTRY)
    obligor_raw = _int_column(frame, "obligor_uid", np.int64, entry=_ENTRY)
    collateral_raw = _float_column(frame, "net_collateral", entry=_ENTRY)

    order = np.argsort(uid_raw, kind="stable")
    uid = np.ascontiguousarray(uid_raw[order])
    obligor_uid = np.ascontiguousarray(obligor_raw[order])
    net_collateral = np.ascontiguousarray(collateral_raw[order])

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
