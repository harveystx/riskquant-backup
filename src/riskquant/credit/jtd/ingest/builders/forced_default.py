# src/riskquant/credit/jtd/ingest/builders/forced_default.py
"""Per-model JTD block builder. Input is assumed structurally and semantically
valid (validated upstream in `validate.py`); builders only resolve and construct."""

from __future__ import annotations

import polars as pl

from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock
from riskquant.credit.jtd.ingest._shared import _i64, _join_pd_lgd_from_master
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit


def build_forced_default_block(
    df: pl.DataFrame,
    obligor_master: CompiledObligorCredit,
) -> ForcedDefaultJtdBlock:
    """Forced-default JTD: loss = ead * lgd, default forced to 1. The kernel
    never reads PD, so the master join is for LGD only."""
    obligor_uids = _i64(df, "uid")
    _, lgd_arr = _join_pd_lgd_from_master(obligor_uids, obligor_master)
    return ForcedDefaultJtdBlock(obligor_uids=obligor_uids, lgd=lgd_arr)
