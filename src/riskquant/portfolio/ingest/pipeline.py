# src/riskquant/portfolio/ingest/pipeline.py
"""
Top-level ingest orchestrator.

Wires the full edge pipeline so a raw, human-facing Polars frame is normalized
and validated before any SoA payload is produced.

Pandas migration (D-INGEST-01)
------------------------------
``compile_raw_df_into_ingest_batch`` previously accepted ``pd.DataFrame``.
It now requires ``pl.DataFrame``.

Rationale:
- Polars is the approved edge technology. Pandas frames never enter the hot
  path.
- Polars DataFrames are immutable; the ``copy`` parameter is meaningless and
  has been removed.

Caller update pattern
---------------------
Simple fixture callers:
    # was: raw_df=pd.DataFrame(data)
    # now: raw_df=pl.from_dict(data)

Filter-then-call callers (pandas workflow upstream):
    # add a one-time conversion at the call site
    batch = compile_raw_df_into_ingest_batch(
        raw_df=pl.from_pandas(sub), static_config=rsc
    )
"""

from __future__ import annotations

from datetime import date

import polars as pl

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
    raw_df: pl.DataFrame,
    static_config: RiskStaticConfig,
    as_of_date: date | None = None,
) -> IngestBatch:
    """
    Run the full ingest pipeline: raw Polars frame -> validated SoA ``IngestBatch``.

    Stages
    ------
    1. canonicalize : column names, ordering, string normalization (returns new frame)
    2. materialize  : strict dtype coercion (returns new frame)
    3. dispatch     : table-driven per-instrument validation; returns int32 ids
    4. compile      : DataFrame -> flat SoA ``CompiledIngestSoA``
    5. pack         : wrap into the immutable ``IngestBatch`` contract

    Parameters
    ----------
    raw_df
        Human-facing positions frame (``pl.DataFrame``). Not mutated.
    static_config
        The full ``RiskStaticConfig``. Required to validate the raw frame and
        compile the SoA arrays.
    as_of_date
        Optional run as-of date (D067). None -> ``NO_DATE_ORD32`` sentinel.

    Notes
    -----
    ``copy`` is removed. Polars DataFrames are immutable; all stages return new
    frames. The caller's frame is never mutated.
    """
    df = canonicalize_raw_ingest_df(raw_df)
    df = materialize_canonicalized_ingest_df(df)

    # Validation routing also yields the int32 instrument ids, so the
    # ``instrument`` strings are hash-mapped exactly once for the whole pipeline.
    instrument_ids = dispatch_ingest_validation(df)

    compiled_ingest_soa = compile_ingest_soa_payload_from_df(
        df=df,
        instrument_ids=instrument_ids,
        static_config=static_config,
        as_of_date=as_of_date,
    )

    return IngestBatch.from_compiled_ingest_soa(compiled_ingest_soa)
