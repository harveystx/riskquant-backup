# src/riskquant/portfolio/engine.py

from __future__ import annotations

import numpy as np

from riskquant.portfolio.compilers.registry import COMPILERS, FIELD_NAMES
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.compiled import CompiledPortfolio


def compile_portfolio_from_batch(batch: IngestBatch) -> CompiledPortfolio:
    instrument_ids = batch.instrument_ids
    n = instrument_ids.size

    if n == 0:
        return CompiledPortfolio()

    # ------------------------------------------------------------
    # 1. Sort once
    # ------------------------------------------------------------
    order = np.argsort(instrument_ids, kind="stable")
    batch = batch.reorder(order)

    sorted_ids = batch.instrument_ids

    # ------------------------------------------------------------
    # 2. Find boundaries
    # ------------------------------------------------------------
    change_idx = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    boundaries = np.concatenate(([0], change_idx, [n]))

    # ------------------------------------------------------------
    # 3. Compile blocks
    # ------------------------------------------------------------
    compiled_fields = {}

    # Local binding for faster access inside loop
    compilers = COMPILERS
    field_names = FIELD_NAMES

    n_groups = len(boundaries) - 1

    for i in range(n_groups):
        start = boundaries[i]
        end = boundaries[i + 1]

        instrument_id = sorted_ids[start]

        compiler = compilers[instrument_id]
        if compiler is None:
            continue

        field_name = field_names[instrument_id]

        block = compiler(batch, start, end)

        if block is not None:
            compiled_fields[field_name] = block

    # Pass the unique dates reference directly into the container mapping
    compiled_fields["unique_dates_ordinal32"] = batch.unique_dates_ordinal32

    # ------------------------------------------------------------
    # 4. Construct result
    # ------------------------------------------------------------
    return CompiledPortfolio(**compiled_fields)
