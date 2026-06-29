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

Effective-factor pair tables
----------------------------
Beyond the raw universes, the mapper carries deduplicated *effective-factor*
pair tables. Each is a compile-time dedup of the distinct raw-factor tuples a
class of effective factor folds, so the market-data compiler can materialise a
dense `(K, S)` effective matrix and each block carries a single `_uix` into it.
`fx_effective_pairs` is the FX-spot table; `spot_effective_pairs` is the
effective-spot-price table built one level up on it. Both share the same
`np.unique(axis=0)` build / packed-key `searchsorted` lookup mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_propagate_sentinel_int32
from riskquant.constants import INVALID_UIX

# ==============================================================================
# FREE LOOKUP HELPERS (shared by the mapper methods and the engine producer)
# ==============================================================================


def lookup_fx_effective_uix(
    *,
    market_id_universe: npt.NDArray[np.int32],
    fx_effective_pairs: npt.NDArray[np.int32],
    raw_fx_ids: npt.NDArray[np.int32],
    raw_fx_inv: npt.NDArray[np.int8],
) -> npt.NDArray[np.int32]:
    """
    Translate per-trade `(raw_fx_id, raw_fx_inv)` into `fx_spots_effective`
    row indices, against an explicit `market_id_universe` / `fx_effective_pairs`.

    Pulled out of `UixMapper.map_fx_legs_to_effective_uix` so the engine's
    effective-spot dedup can compute the same `fx_eff_uix` the compiler will
    stamp, without a `portfolio -> ... -> portfolio` import cycle (this is a
    leaf module). Sentinel rows (`raw_fx_ids == NO_FX_RF_REQUIRED`, i.e. any
    `raw < 0`) collapse to `INVALID_UIX` ("no FX conversion required").
    """
    if raw_fx_ids.shape != raw_fx_inv.shape:
        raise ValueError(
            f"fx legs shape mismatch: ids={raw_fx_ids.shape}, inv={raw_fx_inv.shape}"
        )

    # Stage 1: market-id universe lookup. Sentinels collapse to INVALID_UIX.
    fx_market_uix = searchsorted_or_propagate_sentinel_int32(
        universe=market_id_universe,
        raw=raw_fx_ids,
        sentinel=INVALID_UIX,
    )

    # Stage 2: dedup-pair lookup. Pack each (uix, inv) pair into a single int64
    # key for a vectorized one-shot searchsorted into the equivalently-packed
    # effective-pairs table.
    n = raw_fx_ids.shape[0]
    out = np.full(n, INVALID_UIX, dtype=np.int32)

    if fx_effective_pairs.size == 0:
        return out

    needs_fx = fx_market_uix != INVALID_UIX
    if not needs_fx.any():
        return out

    needed_uix = fx_market_uix[needs_fx].astype(np.int64, copy=False)
    needed_inv = raw_fx_inv[needs_fx].astype(np.int64, copy=False)
    needed_key = (needed_uix << 1) | needed_inv

    pair_uix = fx_effective_pairs[:, 0].astype(np.int64, copy=False)
    pair_inv = fx_effective_pairs[:, 1].astype(np.int64, copy=False)
    pair_key = (pair_uix << 1) | pair_inv

    idx = np.searchsorted(pair_key, needed_key)
    # Every needed_key MUST exist in pair_key (it was deduplicated from exactly
    # this input). Validate to fail loud on contract drift.
    if idx.size and (idx.max() >= pair_key.size or np.any(pair_key[idx] != needed_key)):
        raise RuntimeError(
            "FX-effective dedup contract violated: per-trade pair not found in "
            "fx_effective_pairs. Mapper was built from a different batch than "
            "this one."
        )

    out[needs_fx] = idx.astype(np.int32, copy=False)
    return out


def _pack_spot_effective_key(
    price_uix: npt.NDArray[np.int32],
    fx_eff_uix: npt.NDArray[np.int32],
) -> npt.NDArray[np.int64]:
    """
    Pack `(price_uix, fx_eff_uix)` into one int64 key that is strictly monotone
    in the SIGNED `(price_uix, fx_eff_uix)` lexicographic order produced by
    `np.unique(axis=0)` on the pair table -- so the build (np.unique) and the
    lookup (searchsorted) agree.

    `price_uix >= 0` always (a spot trade's underlying is always present).
    `fx_eff_uix` is in `{INVALID_UIX} u [0, K_fx)`; the only negative value is
    the identity-FX sentinel, which must sort first. Remap `<0 -> 0`, valid
    `v -> v + 1`: monotone, and the remapped lower word stays in `[0, K_fx]`,
    well clear of the high word at bit 32.
    """
    eff_key = np.where(fx_eff_uix < 0, 0, fx_eff_uix.astype(np.int64) + 1)
    return (price_uix.astype(np.int64) << 32) | eff_key


@dataclass(frozen=True, slots=True, kw_only=True)
class UixMapper:
    """
    Immutable execution-space translator.

    This is the single source of truth for global-ID → UIX translation.

    Holds the four global universes touched by the run plus the deduplicated
    effective-factor pair tables. All universe arrays are sorted ascending and
    unique; pair tables are sorted by `np.unique(axis=0)`. The `__post_init__`
    validation runs once so the runtime `map_*` methods can stay
    allocation-light and branch-light.

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
    spot_effective_pairs        : (K, 2) int32 deduplicated
                                  `(price_uix, fx_eff_uix)` table. `price_uix`
                                  indexes `market_id_universe`; `fx_eff_uix`
                                  indexes `fx_effective_pairs` rows (i.e. the
                                  `fx_spots_effective` matrix), or `INVALID_UIX`
                                  for identity FX. The market-data fold turns
                                  this into `ValuationState.effective_spot_prices`
                                  via `prices[price_uix] * fx_spots_effective[
                                  fx_eff_uix]` (sentinel rows multiply by 1.0).
    """

    market_id_universe: npt.NDArray[np.int32]
    discount_curve_id_universe: npt.NDArray[np.int32]
    date_ord32_universe: npt.NDArray[np.int32]
    fx_effective_pairs: npt.NDArray[np.int32]
    spot_effective_pairs: npt.NDArray[np.int32]

    # --- INVARIANTS (CHECKED ONCE) -----------------------------------------

    def __post_init__(self) -> None:
        self._validate_universe(self.market_id_universe, "market_id_universe")
        self._validate_universe(
            self.discount_curve_id_universe, "discount_curve_id_universe"
        )
        self._validate_universe(self.date_ord32_universe, "date_ord32_universe")
        self._validate_pair_table(self.fx_effective_pairs, "fx_effective_pairs")
        self._validate_pair_table(self.spot_effective_pairs, "spot_effective_pairs")

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
    def _validate_pair_table(pairs: npt.NDArray[np.int32], name: str) -> None:
        if pairs.dtype != np.int32:
            raise TypeError(f"{name} must be int32; got {pairs.dtype}.")
        if pairs.ndim != 2 or (pairs.size and pairs.shape[1] != 2):
            raise ValueError(f"{name} must have shape (K, 2); got {pairs.shape}")
        if not pairs.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous.")

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

        Sentinel rows (`raw_fx_ids == NO_FX_RF_REQUIRED`) map to `INVALID_UIX`.
        """
        return lookup_fx_effective_uix(
            market_id_universe=self.market_id_universe,
            fx_effective_pairs=self.fx_effective_pairs,
            raw_fx_ids=raw_fx_ids,
            raw_fx_inv=raw_fx_inv,
        )

    def map_spot_to_effective_uix(
        self,
        *,
        price_uix: npt.NDArray[np.int32],
        fx_eff_uix: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate per-trade `(price_uix, fx_eff_uix)` into per-trade
        `effective_spot_price_uix` indices.

        `price_uix` is a `market_id_universe` index (from
        `map_market_ids_to_uix`); `fx_eff_uix` is an `fx_spots_effective` row
        (from `map_fx_legs_to_effective_uix`), `INVALID_UIX` for identity FX.
        The packed `(price_uix, fx_eff_uix)` key is searchsorted into
        `spot_effective_pairs`; every per-trade pair MUST exist there (the
        table was deduplicated from exactly these legs).
        """
        if price_uix.shape != fx_eff_uix.shape:
            raise ValueError(
                f"spot legs shape mismatch: price_uix={price_uix.shape}, "
                f"fx_eff_uix={fx_eff_uix.shape}"
            )

        n = price_uix.shape[0]
        out = np.full(n, INVALID_UIX, dtype=np.int32)
        if self.spot_effective_pairs.size == 0 or n == 0:
            return out

        needed_key = _pack_spot_effective_key(price_uix, fx_eff_uix)
        pair_key = _pack_spot_effective_key(
            self.spot_effective_pairs[:, 0],
            self.spot_effective_pairs[:, 1],
        )

        idx = np.searchsorted(pair_key, needed_key)
        if idx.size and (
            idx.max() >= pair_key.size or np.any(pair_key[idx] != needed_key)
        ):
            raise RuntimeError(
                "spot-effective dedup contract violated: per-trade "
                "(price_uix, fx_eff_uix) pair not found in spot_effective_pairs. "
                "Mapper was built from a different batch than this one."
            )

        return idx.astype(np.int32, copy=False)
