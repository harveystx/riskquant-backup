# src/riskquant/credit/jtd/ingest/builders/mf_gaussian.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`): the loadings frame is present and
routable, every obligor is covered, no factor is duplicated, and sum-beta^2 <= 1.
This builder only packs CSR (in band order) and precomputes the idio weight."""

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.credit.jtd.blocks.mf_copula_gaussian import MfGaussianCopulaJtdBlock
from riskquant.credit.jtd.ingest._shared import (
    _f64,
    _gaussian_thresholds,
    _i64,
    _join_pd_lgd_from_master,
    _names,
    _resolve_factor_uix,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_mf_gaussian_block(
    df: pl.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
    mf_loadings_df: pl.DataFrame,
) -> MfGaussianCopulaJtdBlock:
    obligor_uids = _i64(df, "uid")
    n = obligor_uids.shape[0]
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)

    idiosyncratic_uix = _resolve_factor_uix(
        _names(df, "idiosyncratic_factor"),
        factor_name_to_uix,
        ctx="idiosyncratic_factor",
    )

    # Band-order position of each obligor; filter the shared loadings frame to
    # this block's obligors (other obligors belong to other MF blocks).
    uid_to_pos = {int(uid): pos for pos, uid in enumerate(obligor_uids)}
    load_uids = _i64(mf_loadings_df, "obligor_uid")
    keep = np.array([int(u) in uid_to_pos for u in load_uids], dtype=bool)

    pos = np.array([uid_to_pos[int(u)] for u in load_uids[keep]], dtype=np.int64)
    loadings = _f64(mf_loadings_df, "factor_loading")[keep]
    sys_names = _names(mf_loadings_df, "factor_name")[keep]

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

    # Independent systemic factors -> systemic variance = sum of squared
    # loadings; idio weight is the residual (sum beta^2 <= 1 guaranteed upstream).
    sum_sq = np.add.reduceat(loading_values * loading_values, loading_indptr[:-1])
    idio_weight = np.sqrt(np.maximum(0.0, 1.0 - sum_sq))

    return MfGaussianCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        thresholds=_gaussian_thresholds(pd_arr),
        loading_indptr=loading_indptr,
        loading_values=loading_values,
        systemic_uix=systemic_uix,
        idiosyncratic_uix=idiosyncratic_uix,
        idio_weight=np.ascontiguousarray(idio_weight),
    )
