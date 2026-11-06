# src/riskquant/credit/jtd/ingest/builders/beta_binomial.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`); builders only resolve and construct."""

from __future__ import annotations

import polars as pl

from riskquant.credit.jtd.blocks.beta_binomial import BetaBinomialJtdBlock
from riskquant.credit.jtd.ingest._shared import (
    _i64,
    _join_pd_lgd_from_master,
    _names,
    _resolve_factor_uix,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_beta_binomial_block(
    df: pl.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> BetaBinomialJtdBlock:
    obligor_uids = _i64(df, "uid")
    # PD is stochastic (P_s ~ Beta drawn into the beta row); only LGD is joined.
    _, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    return BetaBinomialJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        uniform_uix=_resolve_factor_uix(
            _names(df, "uniform_factor"),
            factor_name_to_uix,
            ctx="uniform_factor",
        ),
        beta_uix=_resolve_factor_uix(
            _names(df, "beta_factor"),
            factor_name_to_uix,
            ctx="beta_factor",
        ),
    )
