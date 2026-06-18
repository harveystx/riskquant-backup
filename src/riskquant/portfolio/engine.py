# src/riskquant/portfolio/engine.py
"""
Top-level portfolio orchestrator and compiler.

The engine sits exactly on the boundary between Global State (sparse
`IngestBatch`es indexed by `static_config` Risk Factor IDs) and Local
Execution State (`CompiledPortfolio` blocks indexed by zero-based
contiguous UIX coordinates aligned with the `MarketDataMatrix`).

Pipeline
--------
1. `build_uix_mapper(ingest_batches)`
      Discovers the global universe touched by the run (market IDs,
      discount-curve IDs, dates) AND deduplicates every distinct
      `(fx_rf_uix, invert)` pair across all FX legs of every trade. The
      result, a `UixMapper`, is the single source of truth for ID->UIX
      translation downstream.

2. (Caller fetches market data using the discovered universes.)

3. `compile_portfolio_from_ingest_batch(ingest_batch, uix_mapper)`
      Translates every global ID and ordinal date in the batch into UIX
      execution coordinates, then partitions into per-instrument compiled
      blocks via the registered compiler table. Returns a frozen
      `CompiledPortfolio`.

Design invariants
-----------------
- `UixMapper` is immutable. Its `_universe` arrays are sorted, unique, and
  validated once at construction; runtime `map_*` calls trust the contract
  and use `np.searchsorted` for O(log N) vectorized lookup.
- `fx_effective_pairs` is built in the same pass as the market universe so
  the FX-effective UIX space is part of the global contract. Pricers will
  index a `(K, S)` `fx_spots_effective` matrix built by the market-data
  builder; `INVALID_UIX` (< 0) signals "no FX conversion required" and is
  the only branch in the pricer hot path.
- Compilers take `(batch, start, end, mapper)`. They are the ONLY place
  global IDs are translated to UIXs; blocks never see global IDs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from riskquant.constants import NO_FX_RF_REQUIRED
from riskquant.portfolio.compiled import (
    BlockT,
    CompiledPortfolio,
    CompiledPortfolioFields,
)
from riskquant.portfolio.compilers.registry import COMPILERS, FIELD_NAMES
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.universe import UixMapper

# `UixMapper` (the run's UIX universe contract type) was relocated to the leaf
# module `riskquant.universe` (item C2) so market_data can import it at runtime
# without a layering inversion. It is re-exported here so existing
# `from riskquant.portfolio.engine import UixMapper` sites keep working; the
# PRODUCER `build_uix_mapper` (below) stays here because it consumes IngestBatch.
__all__ = [
    "UixMapper",
    "build_uix_mapper",
    "compile_portfolio_from_ingest_batch",
]


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _dedup_fx_effective_pairs(
    *,
    market_id_universe: npt.NDArray[np.int32],
    raw_fx_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_fx_inv_arrays: tuple[npt.NDArray[np.int8], ...],
) -> npt.NDArray[np.int32]:
    """
    Build the deduplicated `(fx_uix, invert)` pair table covering every FX
    leg in the run.

    Inputs are the raw `(fx_id, inv)` arrays from every IngestBatch leg
    (under->settle, strike->settle, ...). The function:
      1. Concatenates them into one stream.
      2. Drops sentinel rows (`fx_id == NO_FX_RF_REQUIRED`).
      3. Translates `fx_id` -> `market_id_universe` UIX.
      4. `np.unique(axis=0)` to get the sorted-unique pair table.

    Returned shape is `(K, 2) int32`. The two columns are
    `(fx_uix_in_market_universe, invert_flag)`. The table is sorted
    lexicographically by `(fx_uix, invert)`, which is critical for the
    `np.searchsorted` lookup that `map_fx_legs_to_effective_uix` performs
    via the packed `key = (uix << 1) | inv` trick.
    """
    if len(raw_fx_id_arrays) != len(raw_fx_inv_arrays):
        raise ValueError(
            f"FX leg array count mismatch: ids={len(raw_fx_id_arrays)}, "
            f"inv={len(raw_fx_inv_arrays)}"
        )

    if not raw_fx_id_arrays:
        return np.empty((0, 2), dtype=np.int32)

    all_ids = np.concatenate(raw_fx_id_arrays)
    all_inv = np.concatenate(raw_fx_inv_arrays)

    mask = all_ids != int(NO_FX_RF_REQUIRED)
    if not mask.any():
        return np.empty((0, 2), dtype=np.int32)

    real_ids = all_ids[mask]
    real_inv = all_inv[mask]

    # Translate raw IDs -> market-universe UIXs.
    real_uix = np.searchsorted(market_id_universe, real_ids).astype(
        np.int32, copy=False
    )

    pairs = np.stack(
        (real_uix, real_inv.astype(np.int32, copy=False)),
        axis=1,
    )
    unique_pairs = np.unique(pairs, axis=0)
    # np.unique already returns sorted lex order; ensure C-contiguous int32.
    return np.ascontiguousarray(unique_pairs.astype(np.int32, copy=False))


# ==============================================================================
# ORCHESTRATOR PHASE (UNIVERSE DISCOVERY)
# ==============================================================================


def build_uix_mapper(ingest_batches: Sequence[IngestBatch]) -> UixMapper:
    """
    Global universe discovery — the "orchestrator pause".

    Walks every `IngestBatch` once to collect:
      - the union of every market ID (underlyings, vols, FX spots)
      - the union of every discount curve ID
      - the union of every trade-relevant date ordinal
      - the deduplicated `(fx_uix, invert)` pair table

    Returns a frozen `UixMapper` that downstream consumers (block compilers,
    market-data builder, validation) use as the single source of truth for
    UIX <-> global-ID translation.
    """
    if not ingest_batches:
        raise ValueError(
            "No ingest batches provided to build_uix_mapper. At least one "
            "IngestBatch is required."
        )

    # ---- MARKET IDs ----
    market_id_arrays = [
        arr
        for b in ingest_batches
        for arr in (b.underlying_ids, b.volatility_ids, b.fx_u2s_ids, b.fx_k2s_ids)
    ]
    raw_market = np.concatenate(market_id_arrays)
    market_id_universe = np.unique(raw_market[raw_market >= 0]).astype(
        np.int32, copy=False
    )

    # ---- DISCOUNT CURVE IDs ----
    discount_id_arrays = [
        arr
        for b in ingest_batches
        for arr in (b.under_discount_curve_ids, b.strike_discount_curve_ids)
    ]
    raw_discount = np.concatenate(discount_id_arrays)
    discount_curve_id_universe = np.unique(raw_discount[raw_discount >= 0]).astype(
        np.int32, copy=False
    )

    # ---- DATES ----
    date_arrays = [
        arr
        for b in ingest_batches
        for arr in (b.settle_ord, b.delivery_ord, b.expiry_ord)
    ]
    raw_dates = np.concatenate(date_arrays)
    date_ord32_universe = np.unique(raw_dates[raw_dates >= 0]).astype(
        np.int32, copy=False
    )

    # ---- FX EFFECTIVE PAIRS ----
    raw_fx_id_arrays = tuple(
        arr for b in ingest_batches for arr in (b.fx_u2s_ids, b.fx_k2s_ids)
    )
    raw_fx_inv_arrays = tuple(
        arr for b in ingest_batches for arr in (b.fx_u2s_inv, b.fx_k2s_inv)
    )
    fx_effective_pairs = _dedup_fx_effective_pairs(
        market_id_universe=market_id_universe,
        raw_fx_id_arrays=raw_fx_id_arrays,
        raw_fx_inv_arrays=raw_fx_inv_arrays,
    )

    return UixMapper(
        market_id_universe=market_id_universe,
        discount_curve_id_universe=discount_curve_id_universe,
        date_ord32_universe=date_ord32_universe,
        fx_effective_pairs=fx_effective_pairs,
    )


# ==============================================================================
# COMPILER PHASE (UIX INJECTION)
# ==============================================================================


def compile_portfolio_from_ingest_batch(
    *,
    ingest_batch: IngestBatch,
    uix_mapper: UixMapper,
) -> CompiledPortfolio:
    """
    Sort an `IngestBatch` by `instrument_id` and compile each contiguous run
    into an execution-ready block using the provided `UixMapper`.

    The function performs the only global-ID -> UIX translation in the
    portfolio layer: it hands each instrument-specific compiler the
    sub-batch and the mapper, and the compiler does the actual gather.
    """
    instrument_ids = ingest_batch.instrument_ids
    n = instrument_ids.size

    if n == 0:
        raise ValueError("Cannot compile an empty batch into a portfolio.")

    # ---- 1. Sort once by instrument_id (stable so original ordering wins
    #         within each instrument group). ----
    order = np.argsort(instrument_ids, kind="stable")
    ingest_batch = ingest_batch.reorder(order)
    sorted_ids = ingest_batch.instrument_ids

    # ---- 2. Find group boundaries via one diff + one concat. ----
    change_idx = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    boundaries = np.concatenate(([0], change_idx, [n]))

    # ---- 3. Compile per-instrument blocks. ----
    compiled_fields: CompiledPortfolioFields = {}
    compilers = COMPILERS
    field_names = FIELD_NAMES

    n_groups = len(boundaries) - 1
    for i in range(n_groups):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        instrument_id = int(sorted_ids[start])

        compiler = compilers[instrument_id]
        if compiler is None:
            continue

        field_name = field_names[instrument_id]
        if field_name is None:
            raise RuntimeError(
                f"Compiler registered for instrument_id={instrument_id} but "
                "no FIELD_NAME mapping. Wire it in compilers.registry."
            )

        block: BlockT | None = compiler(ingest_batch, start, end, uix_mapper)
        if block is not None:
            compiled_fields[field_name] = block

    # ---- 4. Forward portfolio-level metadata to the compiled artifact. ----
    compiled_fields["unique_dates_ord32"] = uix_mapper.date_ord32_universe

    return CompiledPortfolio(**compiled_fields)
