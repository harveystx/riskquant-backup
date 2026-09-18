# src/riskquant/portfolio/ingest/pipeline.py

"""
Top-level ingest orchestrator.This wires the full
edge pipeline so a raw, human-facing frame is normalized and validated before
any SoA payload is produced.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.canonicalize import canonicalize_raw_ingest_df
from riskquant.portfolio.ingest.compile import (
    compile_ingest_soa_payload_from_df,
)
from riskquant.portfolio.ingest.dispatcher import dispatch_ingest_validation
from riskquant.portfolio.ingest.materialize import (
    materialize_canonicalized_ingest_df,
)


def compile_raw_df_into_ingest_batch(
    *,
    raw_df: pd.DataFrame,
    static_config: RiskStaticConfig,
    copy: bool = False,
    as_of_date: date | None = None,
) -> IngestBatch:
    """
    Run the full ingest pipeline: raw frame -> validated SoA `IngestBatch`.

    Stages
    ------
    1. canonicalize : column names, ordering, string normalization (in place)
    2. materialize  : strict dtype coercion (in place)
    3. dispatch     : table-driven per-instrument validation; returns int32 ids
    4. compile      : DataFrame -> flat SoA `CompiledIngestSoA`
    5. pack         : wrap into the immutable `IngestBatch` contract

    Parameters
    ----------
    raw_df
        Human-facing positions frame (market-risk and/or JTD credit rows).
    static_config
        The full `RiskStaticConfig` is required to validate the raw frame and
        compile the SoA arrays.
    copy
        When True (default) the caller's frame is left untouched: stages 1-2
        mutate a private copy. Set False only when the caller already owns a
        throwaway frame and wants to avoid the edge allocation.
    """
    # Canonicalize / materialize mutate in place; copy once at the edge so the
    # caller's frame is never clobbered.
    df = raw_df.copy() if copy else raw_df

    canonicalize_raw_ingest_df(df)
    materialize_canonicalized_ingest_df(df)

    # Validation routing also yields the int32 instrument ids, so the
    # `instrument` strings are hash-mapped exactly once for the whole pipeline.
    instrument_ids = dispatch_ingest_validation(df)

    # Compile the canonicalized, validated frame into the flat SoA payloads expected
    # by the portfolio engine. This is the last step before packing into the immutable
    # `IngestBatch` contract.
    compiled_ingest_soa = compile_ingest_soa_payload_from_df(
        df=df,
        instrument_ids=instrument_ids,
        static_config=static_config,
        as_of_date=as_of_date,
    )
    # compiled_ingest_soa.print_soa()

    return IngestBatch.from_compiled_ingest_soa(compiled_ingest_soa)
