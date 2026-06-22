# src/riskquant/credit/exposure/obligor_ead/ingest.py
"""
Obligor-EAD ingest (the edge).

`compile_obligor_ead` is the perimeter where a human- or vendor-authored EAD
table becomes a validated, uid-sorted SoA container. It takes only the uid + EAD
columns, so the same single sheet a desk authors (uid, ead, pd, lgd, ...) can be
fed here for exposure and to `compile_obligor_master` for credit, each picking
the columns it owns. Fail-fast on anything that would corrupt the downstream
uid-join or the loss identity: missing columns, non-finite EAD, negative EAD,
duplicate uids.
"""

from __future__ import annotations

__all__ = [
    "compile_obligor_ead",
]

import numpy as np
import pandas as pd

from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad


def compile_obligor_ead(
    df: pd.DataFrame,
    *,
    uid_col: str = "uid",
    ead_col: str = "ead",
) -> CompiledObligorEad:
    """
    Compile a uid + EAD table into a validated `CompiledObligorEad`.

    Parameters
    ----------
    df
        Table carrying at least `uid_col` and `ead_col`; other columns are
        ignored, so the same sheet can also feed `compile_obligor_master`.
    uid_col, ead_col
        Column names for the obligor uid and its per-obligor exposure.

    Notes
    -----
    The rows are sorted by uid here, so callers need not pre-sort. Scenario-
    varying EAD is not authored through this scalar-table edge; build a
    `CompiledObligorEad` with an ``(N, S)`` `ead` directly for that.
    """
    for col in (uid_col, ead_col):
        if col not in df.columns:
            raise ValueError(f"compile_obligor_ead: missing required column '{col}'.")

    uid = np.ascontiguousarray(df[uid_col].to_numpy(dtype=np.int64))

    ead_raw = df[ead_col].to_numpy(dtype=np.float64)
    if not np.isfinite(ead_raw).all():
        raise ValueError(
            f"compile_obligor_ead: non-finite value in EAD column '{ead_col}'."
        )
    if (ead_raw < 0.0).any():
        raise ValueError(
            f"compile_obligor_ead: negative EAD in column '{ead_col}' "
            "(exposure-at-default must be >= 0)."
        )

    if uid.shape[0] != np.unique(uid).shape[0]:
        raise ValueError("compile_obligor_ead: duplicate obligor uid(s).")

    # Sort by uid so the container's join-key invariant holds regardless of the
    # authoring order.
    order = np.argsort(uid, kind="stable")
    uid_sorted = np.ascontiguousarray(uid[order])
    ead_sorted = np.ascontiguousarray(ead_raw[order])

    return CompiledObligorEad(uid=uid_sorted, ead=ead_sorted)
