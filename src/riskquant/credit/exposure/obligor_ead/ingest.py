# src/riskquant/credit/exposure/obligor_ead/ingest.py
"""
Obligor-EAD ingest (the edge).

`compile_obligor_ead` is the perimeter where a human- or vendor-authored EAD
table becomes a validated, uid-sorted SoA container. It takes only the uid + EAD
columns, so the same single sheet a desk authors (uid, ead, pd, lgd, ...) can be
fed here for exposure and to `compile_obligor_master` for credit, each picking
the columns it owns. Fail-fast on anything that would corrupt the downstream
uid-join or the loss identity: a frame that is not ``pl.DataFrame``, missing
columns, a column of the wrong dtype, null cells, non-finite EAD, negative EAD,
duplicate uids (WS_CREDIT_POLARS C.7, C.8).
"""

from __future__ import annotations

__all__ = [
    "compile_obligor_ead",
]

import numpy as np
import polars as pl

from riskquant.credit.exposure.add_on.ingest import (
    _float_column,
    _int_column,
    _require_polars_frame,
)
from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad

_ENTRY = "compile_obligor_ead"


def compile_obligor_ead(
    df: pl.DataFrame,
    *,
    uid_col: str = "uid",
    ead_col: str = "ead",
) -> CompiledObligorEad:
    """
    Compile a uid + EAD table into a validated `CompiledObligorEad`.

    Parameters
    ----------
    df
        ``pl.DataFrame`` carrying at least `uid_col` (integer column) and
        `ead_col` (float column); other columns are ignored, so the same sheet
        can also feed `compile_obligor_master`. The caller's frame is not
        changed.
    uid_col, ead_col
        Column names for the obligor uid and its per-obligor exposure.

    Raises
    ------
    TypeError
        ``df`` is not a ``pl.DataFrame``; a column dtype that its rule does not
        admit; a fractional or out-of-range uid.
    ValueError
        A missing column; a null or NaN cell; an infinite or negative EAD; a
        duplicate uid.

    Notes
    -----
    The rows are sorted by uid here, so callers need not pre-sort. Scenario-
    varying EAD is not authored through this scalar-table edge; build a
    `CompiledObligorEad` with an ``(N, S)`` `ead` directly for that.
    """
    frame = _require_polars_frame(df, entry=_ENTRY, arg="df")
    for col in (uid_col, ead_col):
        if col not in frame.columns:
            raise ValueError(f"{_ENTRY}: missing required column '{col}'.")

    uid = _int_column(frame, uid_col, np.int64, entry=_ENTRY)
    ead_raw = _float_column(frame, ead_col, entry=_ENTRY)
    if not np.isfinite(ead_raw).all():
        raise ValueError(f"{_ENTRY}: non-finite value in EAD column '{ead_col}'.")
    if (ead_raw < 0.0).any():
        raise ValueError(
            f"{_ENTRY}: negative EAD in column '{ead_col}' "
            "(exposure-at-default must be >= 0)."
        )

    if uid.shape[0] != np.unique(uid).shape[0]:
        raise ValueError(f"{_ENTRY}: duplicate obligor uid(s).")

    # Sort by uid so the container's join-key invariant holds regardless of the
    # authoring order.
    order = np.argsort(uid, kind="stable")
    uid_sorted = np.ascontiguousarray(uid[order])
    ead_sorted = np.ascontiguousarray(ead_raw[order])

    return CompiledObligorEad(uid=uid_sorted, ead=ead_sorted)
