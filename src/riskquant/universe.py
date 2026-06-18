# src/riskquant/universe.py
"""
Run-level UIX universe contract.

`UixMapper` is the immutable single source of truth for global-ID -> UIX
translation across a run. It is deliberately a LEAF module: it depends only
on `_arrays` and `constants`, so any tier (portfolio, market_data, simlab)
can import the contract type at runtime without a layering inversion. The
PRODUCER, `build_uix_mapper(ingest_batches)`, stays in `portfolio.engine`
because it consumes portfolio internals (`IngestBatch`); only the contract
type lives here.

This module was extracted from `portfolio.engine` (item C2) specifically to
let `market_data.builder` / `market_data.assemble` drop their `TYPE_CHECKING`
deferral of `UixMapper` — the deferral existed only to dodge the
`market_data -> portfolio` import cycle, which no longer exists now that the
contract type is a leaf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_propagate_sentinel_int32
from riskquant.constants import INVALID_UIX


@dataclass(frozen=True, slots=True, kw_only=True)
class UixMapper:
    """
    Immutable execution-space translator.

    This is the single source of truth for global-ID → UIX translation.

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
        return searchsorted_or_propagate_sentinel_int32(
            universe=self.market_id_universe,
            raw=raw_ids,
            sentinel=INVALID_UIX,
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
        return searchsorted_or_propagate_sentinel_int32(
            universe=self.discount_curve_id_universe,
            raw=raw_ids,
            sentinel=INVALID_UIX,
        )

    def map_dates_to_uix(
        self,
        ordinals: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate global day-ordinals into `date_ord32_universe` UIXs.

        Sentinel rows (`ordinals < 0`) map to `INVALID_UIX`.
        """
        return searchsorted_or_propagate_sentinel_int32(
            universe=self.date_ord32_universe,
            raw=ordinals,
            sentinel=INVALID_UIX,
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
        fx_market_uix = searchsorted_or_propagate_sentinel_int32(
            universe=self.market_id_universe,
            raw=raw_fx_ids,
            sentinel=INVALID_UIX,
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
