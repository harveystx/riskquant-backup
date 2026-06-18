# src/riskquant/credit/jtd/ingest/builders/independent_bernoulli.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`); builders only resolve and construct."""

from __future__ import annotations

import pandas as pd

from riskquant.credit.jtd.blocks.independent_bernoulli import (
    IndependentBernoulliJtdBlock,
)
from riskquant.credit.jtd.ingest._shared import (
    _i64,
    _join_pd_lgd_from_master,
    _resolve_factor_uix,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_independent_bernoulli_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> IndependentBernoulliJtdBlock:
    obligor_uids = _i64(df, "uid")
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    return IndependentBernoulliJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        pd=pd_arr,
        uniform_uix=_resolve_factor_uix(
            df["uniform_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="uniform_factor",
        ),
    )
