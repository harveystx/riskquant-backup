# src/riskquant/credit/jtd/ingest/builders/mf_students_t.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`): the loadings frame is present and
routable, every obligor covered, no factor duplicated, sum-beta^2 <= 1, and the
chi/nu reconciliation holds. This builder packs CSR (band order), precomputes the
idio weight, and sets Student-t thresholds (mirrors mf_gaussian + sf_students_t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.credit.jtd.blocks.mf_copula_students_t import MfStudentsTCopulaJtdBlock
from riskquant.credit.jtd.ingest._shared import (
    _f64,
    _i64,
    _join_pd_lgd_from_master,
    _resolve_factor_uix,
    _students_t_thresholds,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_mf_students_t_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
    mf_loadings_df: pd.DataFrame,
) -> MfStudentsTCopulaJtdBlock:
    obligor_uids = _i64(df, "uid")
    n = obligor_uids.shape[0]
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    nu = _f64(df, "nu")

    idiosyncratic_uix = _resolve_factor_uix(
        df["idiosyncratic_factor"].to_numpy(copy=False),
        factor_name_to_uix,
        ctx="idiosyncratic_factor",
    )
    chi_squared_uix = _resolve_factor_uix(
        df["chi_squared_factor"].to_numpy(copy=False),
        factor_name_to_uix,
        ctx="chi_squared_factor",
    )

    # Band-order position of each obligor; filter the shared loadings frame to
    # this block's obligors (others belong to the MF-gaussian block).
    uid_to_pos = {int(uid): pos for pos, uid in enumerate(obligor_uids)}
    load_uids = mf_loadings_df["obligor_uid"].to_numpy()
    keep = np.array([int(u) in uid_to_pos for u in load_uids], dtype=bool)

    pos = np.array([uid_to_pos[int(u)] for u in load_uids[keep]], dtype=np.int64)
    loadings = _f64(mf_loadings_df.loc[keep], "factor_loading")
    sys_names = mf_loadings_df.loc[keep, "factor_name"].to_numpy(copy=False)

    # Stable sort by obligor position -> CSR-contiguous per obligor, band order.
    order = np.argsort(pos, kind="stable")
    pos_sorted = pos[order]
    loading_values = np.ascontiguousarray(loadings[order])
    systemic_uix = _resolve_factor_uix(
        sys_names[order], factor_name_to_uix, ctx="mf systemic factor_name"
    )

    counts = np.bincount(pos_sorted, minlength=n)
    loading_indptr = np.empty(n + 1, dtype=np.int64)
    loading_indptr[0] = 0
    np.cumsum(counts, out=loading_indptr[1:])

    # Independent systemic factors -> systemic variance = sum of squared loadings.
    sum_sq = np.add.reduceat(loading_values * loading_values, loading_indptr[:-1])
    idio_weight = np.sqrt(np.maximum(0.0, 1.0 - sum_sq))

    return MfStudentsTCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        thresholds=_students_t_thresholds(pd_arr, nu),
        loading_indptr=loading_indptr,
        loading_values=loading_values,
        systemic_uix=systemic_uix,
        idiosyncratic_uix=idiosyncratic_uix,
        idio_weight=np.ascontiguousarray(idio_weight),
        chi_squared_uix=chi_squared_uix,
        sqrt_nu=np.sqrt(nu),
    )
