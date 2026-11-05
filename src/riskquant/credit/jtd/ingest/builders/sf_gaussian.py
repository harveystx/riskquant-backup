# src/riskquant/credit/jtd/ingest/builders/sf_gaussian.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`); builders only resolve and construct."""

from __future__ import annotations

import polars as pl

from riskquant.credit.jtd.blocks.sf_copula_gaussian import SfGaussianCopulaJtdBlock
from riskquant.credit.jtd.ingest._shared import (
    _f64,
    _gaussian_thresholds,
    _i64,
    _join_pd_lgd_from_master,
    _names,
    _resolve_factor_uix,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_sf_gaussian_block(
    df: pl.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> SfGaussianCopulaJtdBlock:
    obligor_uids = _i64(df, "uid")
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    return SfGaussianCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        thresholds=_gaussian_thresholds(pd_arr),
        factor_loading=_f64(df, "factor_loading"),
        systemic_uix=_resolve_factor_uix(
            _names(df, "systemic_factor"),
            factor_name_to_uix,
            ctx="systemic_factor",
        ),
        idiosyncratic_uix=_resolve_factor_uix(
            _names(df, "idiosyncratic_factor"),
            factor_name_to_uix,
            ctx="idiosyncratic_factor",
        ),
    )
