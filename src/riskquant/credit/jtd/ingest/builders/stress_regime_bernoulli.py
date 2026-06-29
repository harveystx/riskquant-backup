# src/riskquant/credit/jtd/ingest/builders/stress_regime_bernoulli.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`); builders only resolve and construct.
The stress-regime indicator is resolved per obligor: obligors may name distinct
stress factors (block-diagonal regimes) or share one (the global-stress case)."""

from __future__ import annotations

import pandas as pd

from riskquant.credit.jtd.blocks.stress_regime_bernoulli import (
    StressRegimeBernoulliJtdBlock,
)
from riskquant.credit.jtd.ingest._shared import (
    _f64,
    _i64,
    _join_pd_lgd_from_master,
    _resolve_factor_uix,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_stress_regime_bernoulli_block(
    df: pd.DataFrame,
    obligor_master: CompiledObligorCredit,
    factor_name_to_uix: dict[str, int],
) -> StressRegimeBernoulliJtdBlock:
    obligor_uids = _i64(df, "uid")
    pd_arr, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)

    return StressRegimeBernoulliJtdBlock(
        obligor_uids=obligor_uids,
        lgd=lgd_arr,
        pd=pd_arr,
        uniform_uix=_resolve_factor_uix(
            df["uniform_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="uniform_factor",
        ),
        stress_regime_indicator_uix=_resolve_factor_uix(
            df["stress_regime_factor"].to_numpy(copy=False),
            factor_name_to_uix,
            ctx="stress_regime_factor",
        ),
        stress_pd_multiplier=_f64(df, "stress_pd_multiplier"),
    )
