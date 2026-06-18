# src/riskquant/credit/jtd/ingest/builders/sf_students_t.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`); builders only resolve and construct."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.credit.jtd.blocks.sf_copula_students_t import SfStudentsTCopulaJtdBlock
from riskquant.credit.jtd.ingest._shared import (
    _f64,
    _i64,
    _join_pd_lgd_from_master,
    _resolve_factor_uix,
    _students_t_thresholds,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_sf_students_t_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> SfStudentsTCopulaJtdBlock:
    obligor_uids = _i64(df, "uid")
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    nu = _f64(df, "nu")
    return SfStudentsTCopulaJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        thresholds=_students_t_thresholds(pd_arr, nu),
        factor_loading=_f64(df, "factor_loading"),
        systemic_uix=_resolve_factor_uix(
            df["systemic_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="systemic_factor",
        ),
        idiosyncratic_uix=_resolve_factor_uix(
            df["idiosyncratic_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="idiosyncratic_factor",
        ),
        chi_squared_uix=_resolve_factor_uix(
            df["chi_squared_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="chi_squared_factor",
        ),
        sqrt_nu=np.sqrt(nu),
    )
