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
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.constants import INVALID_UIX, NO_FX_RF_REQUIRED
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.portfolio.compilers.registry import COMPILERS, FIELD_NAMES
from riskquant.portfolio.ingest.batch import IngestBatch

# ==============================================================================
# UIX TRANSLATION CONTEXT
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class UixMapper:
    """
    Immutable execution-space translator.

    Holds the four global universes touched by the run plus the deduplicated
    FX-effective pair table. All universe arrays are sorted ascending and
    unique; the `__post_init__` validation runs once so the runtime `map_*`
    methods can stay allocation-light and branch-light.

    Field layout
    ------------
    market_id_universe          : sorted-unique int32 IDs used by
                                  prices / forwards / vols / fx-spots.
                                  Indexed by `price_uix`, `vol_uix`,
                                  `underlying_price_curve_uix`, etc.
    discount_curve_id_universe  : sorted-unique int32 IDs used by discount
                                  curves. Kept distinct from `market_id_universe`
                                  because the discount-curve ID range is
                                  disjoint and `MarketDataMatrix.discount_factor_grid`
                                  has its own axis-0 universe.
    date_ord32_universe         : sorted-unique int32 day-ordinals
                                  (days-since-epoch) touched by the run.
                                  Indexed by `*_date_uix` on blocks.
    fx_effective_pairs          : (K, 2) int32 deduplicated `(fx_rf_uix, invert)`
                                  table. `fx_rf_uix` is itself an index into
                                  `market_id_universe`; `invert` is 0 or 1.
                                  The market-data builder turns this into a
                                  `(K, S)` `fx_spots_effective` matrix.
    """

    market_id_universe: npt.NDArray[np.int32]
    discount_curve_id_universe: npt.NDArray[np.int32]
    date_ord32_universe: npt.NDArray[np.int32]
    fx_effective_pairs: npt.NDArray[np.int32]

    # --- INVARIANTS (CHECKED ONCE) -----------------------------------------

    def __post_init__(self) -> None:
        self._validate_universe(self.market_id_universe, "market_id_universe")
        self._validate_universe(
            self.discount_curve_id_universe, "discount_curve_id_universe"
        )
        self._validate_universe(self.date_ord32_universe, "date_ord32_universe")
        self._validate_fx_effective_pairs(self.fx_effective_pairs)

    @staticmethod
    def _validate_universe(arr: npt.NDArray[np.int32], name: str) -> None:
        if arr.dtype != np.int32:
            raise TypeError(f"{name} must be int32; got {arr.dtype}.")
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D; got ndim={arr.ndim}.")
        if not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous.")
        if arr.size > 1 and np.any(arr[1:] <= arr[:-1]):
            raise ValueError(f"{name} must be strictly increasing and unique.")

    @staticmethod
    def _validate_fx_effective_pairs(pairs: npt.NDArray[np.int32]) -> None:
        if pairs.dtype != np.int32:
            raise TypeError(f"fx_effective_pairs must be int32; got {pairs.dtype}.")
        if pairs.ndim != 2 or (pairs.size and pairs.shape[1] != 2):
            raise ValueError(
                f"fx_effective_pairs must have shape (K, 2); got {pairs.shape}"
            )
        if not pairs.flags.c_contiguous:
            raise ValueError("fx_effective_pairs must be C-contiguous.")

    # --- TRANSLATION API ---------------------------------------------------

    def map_market_ids_to_uix(
        self,
        raw_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate global market IDs into `market_id_universe` UIXs.

        Sentinel rows (`raw_ids < 0`) map to `INVALID_UIX`.
        """
        return _searchsorted_with_sentinel(
            universe=self.market_id_universe,
            raw=raw_ids,
        )

    def map_discount_ids_to_uix(
        self,
        raw_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate global discount-curve IDs into
        `discount_curve_id_universe` UIXs.

        Sentinel rows (`raw_ids < 0`) map to `INVALID_UIX`.
        """
        return _searchsorted_with_sentinel(
            universe=self.discount_curve_id_universe,
            raw=raw_ids,
        )

    def map_dates_to_uix(
        self,
        ordinals: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate global day-ordinals into `date_ord32_universe` UIXs.

        Sentinel rows (`ordinals < 0`) map to `INVALID_UIX`.
        """
        return _searchsorted_with_sentinel(
            universe=self.date_ord32_universe,
            raw=ordinals,
        )

    def map_fx_legs_to_effective_uix(
        self,
        *,
        raw_fx_ids: npt.NDArray[np.int32],
        raw_fx_inv: npt.NDArray[np.int8],
    ) -> npt.NDArray[np.int32]:
        """
        Translate per-trade `(raw_fx_id, raw_fx_inv)` into per-trade
        `fx_spots_effective_uix` indices.

        Steps (all vectorized, no Python loop):
          1. Translate `raw_fx_ids` into `market_id_universe` UIXs.
          2. Look the resulting `(fx_uix, invert)` pair up in
             `fx_effective_pairs` to obtain the effective UIX.

        Sentinel rows (`raw_fx_ids == NO_FX_RF_REQUIRED`) map to `INVALID_UIX`.
        """
        if raw_fx_ids.shape != raw_fx_inv.shape:
            raise ValueError(
                f"fx legs shape mismatch: ids={raw_fx_ids.shape}, "
                f"inv={raw_fx_inv.shape}"
            )

        # Stage 1: market-id universe lookup. Sentinels collapse to INVALID_UIX.
        fx_market_uix = _searchsorted_with_sentinel(
            universe=self.market_id_universe,
            raw=raw_fx_ids,
        )

        # Stage 2: dedup-pair lookup. We pack each (uix, inv) pair into a
        # single int64 key for a vectorized one-shot searchsorted into the
        # equivalently-packed effective-pairs table.
        n = raw_fx_ids.shape[0]
        out = np.full(n, INVALID_UIX, dtype=np.int32)

        if self.fx_effective_pairs.size == 0:
            return out

        needs_fx = fx_market_uix != INVALID_UIX
        if not needs_fx.any():
            return out

        needed_uix = fx_market_uix[needs_fx].astype(np.int64, copy=False)
        needed_inv = raw_fx_inv[needs_fx].astype(np.int64, copy=False)
        needed_key = (needed_uix << 1) | needed_inv

        pair_uix = self.fx_effective_pairs[:, 0].astype(np.int64, copy=False)
        pair_inv = self.fx_effective_pairs[:, 1].astype(np.int64, copy=False)
        pair_key = (pair_uix << 1) | pair_inv

        idx = np.searchsorted(pair_key, needed_key)
        # Every needed_key MUST exist in pair_key (it was deduplicated from
        # exactly this input). Validate to fail loud on contract drift.
        if idx.size and (
            idx.max() >= pair_key.size or np.any(pair_key[idx] != needed_key)
        ):
            raise RuntimeError(
                "FX-effective dedup contract violated: per-trade pair not "
                "found in fx_effective_pairs. Mapper was built from a "
                "different batch than this one."
            )

        out[needs_fx] = idx.astype(np.int32, copy=False)
        return out


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _searchsorted_with_sentinel(
    *,
    universe: npt.NDArray[np.int32],
    raw: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """
    Vectorized O(log N) ID translation with sentinel preservation.

    `raw < 0` is treated as a sentinel and mapped to `INVALID_UIX` in the
    output. All other IDs MUST exist in `universe` (the universe is built
    from the same IDs upstream); a violation indicates a wiring bug and is
    caught by `validate_ingest_batches_against_uix_universe`.

    The universe itself MUST be sorted ascending and unique — guaranteed by
    `UixMapper.__post_init__`.
    """
    uix = np.searchsorted(universe, raw).astype(np.int32, copy=False)
    uix[raw < 0] = INVALID_UIX
    return uix


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
    compiled_fields: dict[str, object] = {}
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

        block = compiler(ingest_batch, start, end, uix_mapper)
        if block is not None:
            compiled_fields[field_name] = block

    # ---- 4. Forward portfolio-level metadata to the compiled artifact. ----
    compiled_fields["unique_dates_ord32"] = uix_mapper.date_ord32_universe

    return CompiledPortfolio(**compiled_fields)
