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

import hashlib
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from riskquant.constants import (
    DAYS_PER_YR_ACT_365_25,
    INVALID_UIX,
    NO_FX_RF_REQUIRED,
    TTE_FLOOR,
)

# ==============================================================================
# FREE LOOKUP HELPERS (shared by the mapper methods and the engine producer)
# ==============================================================================


def translate_ids_to_uix_or_raise(
    *,
    universe: npt.NDArray[np.int32],
    raw: npt.NDArray[np.int32],
    ctx: str,
) -> npt.NDArray[np.int32]:
    """
    Translate global ids into dense `universe` rows with a membership proof.

    Negative `raw` values are the absent-linkage sentinels (NO_ID, NO_DATE_ORD32,
    NO_FX_RF_REQUIRED) and map to `INVALID_UIX`. Every non-negative id must be
    an element of `universe`; otherwise this raises before any coordinate is
    returned.

    Why not a plain `np.searchsorted`: for an absent id it returns the insertion
    position, which is the dense row of a DIFFERENT, valid id (or one past the
    end). That row passes every range check (D058) and silently re-points the
    trade to another factor (AUD-02). Membership is therefore the contract of
    every id -> uix translation, not an optional extra check.
    """
    out = np.full(raw.shape[0], INVALID_UIX, dtype=np.int32)
    present = raw >= 0
    if not bool(present.any()):
        return out
    ids = raw[present]
    n = universe.shape[0]
    if n == 0:
        raise ValueError(
            f"{ctx}: {int(ids.shape[0])} id(s) absent from the UixMapper universe "
            f"(AUD-02; universe is empty): {np.unique(ids)[:10].tolist()}."
        )
    pos = np.searchsorted(universe, ids)
    probe = np.minimum(pos, n - 1)
    member = universe[probe] == ids
    if not bool(member.all()):
        missing = np.unique(ids[~member])[:10].tolist()
        raise ValueError(
            f"{ctx}: {int((~member).sum())} id(s) absent from the UixMapper "
            f"universe (AUD-02): {missing}. The portfolio must be compiled "
            "against a mapper built from the batches that contain it."
        )
    out[present] = pos.astype(np.int32, copy=False)
    return out


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
    leaf module). Only the canonical sentinel (`raw_fx_ids == NO_FX_RF_REQUIRED`)
    collapses to `INVALID_UIX` ("no FX conversion required"); any other negative
    id, or an invert flag outside {0, 1}, raises (D060, D062).
    """
    if raw_fx_ids.shape != raw_fx_inv.shape:
        raise ValueError(
            f"fx legs shape mismatch: ids={raw_fx_ids.shape}, inv={raw_fx_inv.shape}"
        )
    # D060: the ONLY admissible negative FX id is the canonical
    # NO_FX_RF_REQUIRED. Any other negative (e.g. NO_ID from an unresolved pair)
    # is a missing quote, and a missing quote must never become identity FX.
    bad_sentinel = (raw_fx_ids < 0) & (raw_fx_ids != NO_FX_RF_REQUIRED)
    if bool(bad_sentinel.any()):
        raise ValueError(
            "FX leg carries a negative rf id that is not NO_FX_RF_REQUIRED: "
            f"{np.unique(raw_fx_ids[bad_sentinel])[:10].tolist()}. An unresolved "
            "FX quote cannot be treated as 'no FX required'."
        )
    # D062: the invert flag packs into bit 0 of the pair key; a value outside
    # {0, 1} would alias the neighbouring FX rf.
    if bool(((raw_fx_inv != 0) & (raw_fx_inv != 1)).any()):
        raise ValueError("FX invert flags must be 0 or 1.")

    # Stage 1: market-id universe lookup with membership proof. The
    # NO_FX_RF_REQUIRED sentinel collapses to INVALID_UIX.
    fx_market_uix = translate_ids_to_uix_or_raise(
        universe=market_id_universe,
        raw=raw_fx_ids,
        ctx="FX rf ids",
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
    the identity-FX sentinel, which must sort first. `_remap_optional_uix`
    maps `INVALID_UIX -> 0` and valid `v -> v + 1` (any other negative raises):
    monotone, and the remapped lower word stays in `[0, K_fx]`, well clear of
    the high word at bit 32. A negative `price_uix` raises (D062).
    """
    if price_uix.size and int(price_uix.min()) < 0:
        raise ValueError(
            "spot-effective key: price_uix holds an unresolved (negative) coordinate."
        )
    eff_key = _remap_optional_uix(fx_eff_uix)
    return (price_uix.astype(np.int64) << 32) | eff_key


def _mixed_radix_key(
    columns: tuple[npt.NDArray[np.int64], ...],
    radices: tuple[int, ...],
    *,
    ctx: str,
) -> npt.NDArray[np.int64]:
    """
    Pack non-negative integer columns into one int64 key that is strictly
    monotone in their LEXICOGRAPHIC order.

    `np.unique(axis=0)` sorts a pair/tuple table lexicographically by column, so
    a mixed-radix key built with the same column order reproduces that order
    exactly -- which is what lets the build (`np.unique`) and the lookup
    (`np.searchsorted`) agree without a second sort.

    Radices are the RUNTIME sizes of each column's universe, not hard-coded bit
    widths: a hard-coded shift silently aliases two distinct factors the day a
    universe outgrows its field. The product is asserted to fit in int64.
    """
    if len(columns) != len(radices):
        raise ValueError(f"{ctx}: {len(columns)} columns vs {len(radices)} radices")

    product = 1
    for r in radices:
        if r <= 0:
            raise ValueError(f"{ctx}: non-positive radix {r}")
        product *= r
        if product > np.iinfo(np.int64).max:
            raise ValueError(
                f"{ctx}: packed key overflows int64 (radices={radices}). The "
                "effective-factor universes are larger than a single int64 key "
                "can address lexicographically."
            )

    # D062: every packed coordinate must lie in [0, radix). A sentinel in a
    # non-leading column otherwise packs to a VALID key of a different factor
    # (MD-03: a missing settlement date aliased another trade's EDF row).
    for i, (col, radix) in enumerate(zip(columns, radices, strict=True)):
        if col.size and (int(col.min()) < 0 or int(col.max()) >= radix):
            raise ValueError(
                f"{ctx}: packed column {i} holds a coordinate outside [0, {radix}) "
                f"(min={int(col.min())}, max={int(col.max())}). An unresolved or "
                "sentinel coordinate cannot be packed into an effective-factor key."
            )

    key = np.zeros(columns[0].shape[0], dtype=np.int64)
    for col, radix in zip(columns, radices, strict=True):
        key = key * np.int64(radix) + col
    return key


def _remap_optional_uix(uix: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
    """
    Map an optional `_uix` (INVALID_UIX for "not required") onto `[0, K]`.

    Only `INVALID_UIX` is the identity/absent sentinel (D062); it must sort
    FIRST, so `INVALID_UIX -> 0` and valid `v -> v + 1`. Any other negative value
    is a corrupted coordinate and raises.
    """
    if bool(((uix < 0) & (uix != INVALID_UIX)).any()):
        raise ValueError(
            "optional fx_eff_uix holds a negative value other than INVALID_UIX."
        )
    return np.where(uix < 0, 0, uix.astype(np.int64) + 1)


def _pack_edf_effective_key(
    *,
    curve_uix: npt.NDArray[np.int32],
    date_uix: npt.NDArray[np.int32],
    fx_eff_uix: npt.NDArray[np.int32],
    n_curves: int,
    n_dates: int,
    n_fx: int,
) -> npt.NDArray[np.int64]:
    """Lexicographic key for the `(curve_uix, date_uix, fx_eff_uix)` EDF table."""
    return _mixed_radix_key(
        (
            curve_uix.astype(np.int64),
            date_uix.astype(np.int64),
            _remap_optional_uix(fx_eff_uix),
        ),
        (max(n_curves, 1), max(n_dates, 1), n_fx + 1),
        ctx="effective_discount_factors",
    )


def _pack_efp_effective_key(
    *,
    price_uix: npt.NDArray[np.int32],
    curve_uix: npt.NDArray[np.int32],
    date_uix: npt.NDArray[np.int32],
    fx_eff_uix: npt.NDArray[np.int32],
    n_market: int,
    n_curves: int,
    n_dates: int,
    n_fx: int,
) -> npt.NDArray[np.int64]:
    """Lexicographic key for the `(price, curve, date, fx)` EFP table."""
    return _mixed_radix_key(
        (
            price_uix.astype(np.int64),
            curve_uix.astype(np.int64),
            date_uix.astype(np.int64),
            _remap_optional_uix(fx_eff_uix),
        ),
        (max(n_market, 1), max(n_curves, 1), max(n_dates, 1), n_fx + 1),
        ctx="effective_forward_prices",
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
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
    edf_effective_triples: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32),
    )
    efp_effective_quads: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.int32),
    )
    vol_effective_uix: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
    )
    fk_effective_pairs: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32),
    )
    as_of_ord32: int | None = None
    # D064: deterministic content fingerprint of every universe/table plus the
    # as-of. `MarketDataMatrix` carries the fingerprint of the mapper it was
    # built against and the fold refuses any other mapper. Derived, never
    # supplied.
    fingerprint: str = field(init=False, default="")

    # --- INVARIANTS (CHECKED ONCE) -----------------------------------------

    def __post_init__(self) -> None:
        self._validate_universe(
            self.market_id_universe,
            "market_id_universe",
        )
        self._validate_universe(
            self.discount_curve_id_universe,
            "discount_curve_id_universe",
        )
        self._validate_universe(
            self.date_ord32_universe,
            "date_ord32_universe",
        )
        self._validate_pair_table(
            self.fx_effective_pairs,
            "fx_effective_pairs",
        )
        self._validate_pair_table(
            self.spot_effective_pairs,
            "spot_effective_pairs",
        )
        self._validate_tuple_table(
            self.edf_effective_triples,
            3,
            "edf_effective_triples",
        )
        self._validate_tuple_table(
            self.efp_effective_quads,
            4,
            "efp_effective_quads",
        )
        self._validate_tuple_table(
            self.fk_effective_pairs,
            2,
            "fk_effective_pairs",
        )
        if self.vol_effective_uix.dtype != np.int32:
            raise TypeError(
                f"vol_effective_uix must be int32; got {self.vol_effective_uix.dtype}."
            )
        if self.vol_effective_uix.ndim != 1:
            raise ValueError("vol_effective_uix must be 1-D.")
        self._validate_universe(self.vol_effective_uix, "vol_effective_uix")
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

    def _compute_fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for name in (
            "market_id_universe",
            "discount_curve_id_universe",
            "date_ord32_universe",
            "fx_effective_pairs",
            "spot_effective_pairs",
            "edf_effective_triples",
            "efp_effective_quads",
            "vol_effective_uix",
            "fk_effective_pairs",
        ):
            arr = getattr(self, name)
            h.update(name.encode())
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr, dtype=np.int32).tobytes())
        h.update(repr(self.as_of_ord32).encode())
        return h.hexdigest()

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
    def _validate_tuple_table(
        table: npt.NDArray[np.int32], width: int, name: str
    ) -> None:
        if table.dtype != np.int32:
            raise TypeError(f"{name} must be int32; got {table.dtype}.")
        if table.ndim != 2 or table.shape[1] != width:
            raise ValueError(f"{name} must have shape (K, {width}); got {table.shape}")
        if not table.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous.")

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

        Sentinel rows (`raw_ids < 0`) map to `INVALID_UIX`; a non-negative id
        absent from the universe raises (AUD-02).
        """
        return translate_ids_to_uix_or_raise(
            universe=self.market_id_universe,
            raw=raw_ids,
            ctx="market ids",
        )

    def map_discount_ids_to_uix(
        self,
        raw_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate global discount-curve IDs into
        `discount_curve_id_universe` UIXs.

        Sentinel rows (`raw_ids < 0`) map to `INVALID_UIX`; a non-negative id
        absent from the universe raises (AUD-02).
        """
        return translate_ids_to_uix_or_raise(
            universe=self.discount_curve_id_universe,
            raw=raw_ids,
            ctx="discount curve ids",
        )

    def map_dates_to_uix(
        self,
        ordinals: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate global day-ordinals into `date_ord32_universe` UIXs.

        Sentinel rows (`ordinals < 0`) map to `INVALID_UIX`; a non-negative
        ordinal absent from the universe raises (AUD-02).
        """
        return translate_ids_to_uix_or_raise(
            universe=self.date_ord32_universe,
            raw=ordinals,
            ctx="date ordinals",
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

    # ----------------------------------------------------------------- #
    # FORWARD EFFECTIVE-FACTOR COORDINATES                              #
    # ----------------------------------------------------------------- #

    def map_edf_to_effective_uix(
        self,
        *,
        curve_uix: npt.NDArray[np.int32],
        date_uix: npt.NDArray[np.int32],
        fx_eff_uix: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate a per-trade `(curve_uix, date_uix, fx_eff_uix)` discounting leg
        into a `ValuationState.effective_discount_factors` row.

        `curve_uix` indexes `discount_curve_id_universe`, `date_uix` indexes
        `date_ord32_universe`, `fx_eff_uix` is an `fx_spots_effective` row or
        `INVALID_UIX` for identity FX. Every per-trade triple MUST exist in
        `edf_effective_triples` -- the table was deduplicated from exactly these
        legs, so a miss means the mapper and the batch disagree.
        """
        return self._lookup_effective(
            needed=_pack_edf_effective_key(
                curve_uix=curve_uix,
                date_uix=date_uix,
                fx_eff_uix=fx_eff_uix,
                n_curves=self.discount_curve_id_universe.shape[0],
                n_dates=self.date_ord32_universe.shape[0],
                n_fx=self.fx_effective_pairs.shape[0],
            ),
            table=self.edf_effective_triples,
            table_key=_pack_edf_effective_key(
                curve_uix=self.edf_effective_triples[:, 0],
                date_uix=self.edf_effective_triples[:, 1],
                fx_eff_uix=self.edf_effective_triples[:, 2],
                n_curves=self.discount_curve_id_universe.shape[0],
                n_dates=self.date_ord32_universe.shape[0],
                n_fx=self.fx_effective_pairs.shape[0],
            ),
            n=curve_uix.shape[0],
            ctx="effective-discount-factor",
        )

    def map_efp_to_effective_uix(
        self,
        *,
        price_uix: npt.NDArray[np.int32],
        curve_uix: npt.NDArray[np.int32],
        date_uix: npt.NDArray[np.int32],
        fx_eff_uix: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate a per-trade forward underlying leg into a
        `ValuationState.effective_forward_prices` row.

        The quad is `(market_id_uix, discount_curve_uix, delivery_date_uix,
        fx_eff_uix)`: the forward NODE level, the curve that discounts it, the
        delivery date, and its asset->settle FX. The fold pre-multiplies all
        four into one PV row.
        """
        n_market = self.market_id_universe.shape[0]
        n_curves = self.discount_curve_id_universe.shape[0]
        n_dates = self.date_ord32_universe.shape[0]
        n_fx = self.fx_effective_pairs.shape[0]
        return self._lookup_effective(
            needed=_pack_efp_effective_key(
                price_uix=price_uix,
                curve_uix=curve_uix,
                date_uix=date_uix,
                fx_eff_uix=fx_eff_uix,
                n_market=n_market,
                n_curves=n_curves,
                n_dates=n_dates,
                n_fx=n_fx,
            ),
            table=self.efp_effective_quads,
            table_key=_pack_efp_effective_key(
                price_uix=self.efp_effective_quads[:, 0],
                curve_uix=self.efp_effective_quads[:, 1],
                date_uix=self.efp_effective_quads[:, 2],
                fx_eff_uix=self.efp_effective_quads[:, 3],
                n_market=n_market,
                n_curves=n_curves,
                n_dates=n_dates,
                n_fx=n_fx,
            ),
            n=price_uix.shape[0],
            ctx="effective-forward-price",
        )

    @staticmethod
    def _lookup_effective(
        *,
        needed: npt.NDArray[np.int64],
        table: npt.NDArray[np.int32],
        table_key: npt.NDArray[np.int64],
        n: int,
        ctx: str,
    ) -> npt.NDArray[np.int32]:
        """Shared searchsorted-with-membership-proof for the effective tables."""
        out = np.full(n, INVALID_UIX, dtype=np.int32)
        if table.shape[0] == 0 or n == 0:
            return out

        idx = np.searchsorted(table_key, needed)
        if idx.size and (
            idx.max() >= table_key.size or np.any(table_key[idx] != needed)
        ):
            raise RuntimeError(
                f"{ctx} dedup contract violated: a per-trade leg is not present "
                "in the effective table. The mapper was built from a different "
                "batch than the one being compiled."
            )
        return idx.astype(np.int32, copy=False)

        # ----------------------------------------------------------------- #

    # OPTION EFFECTIVE-FACTOR COORDINATES                                #
    # ----------------------------------------------------------------- #

    def map_vol_to_effective_uix(
        self,
        market_uix: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate a per-trade volatility market-uix into a `ValuationState.vols`
        row.

        `vol_effective_uix` is the deduplicated set of `market_id_universe`
        coordinates whose `MarketDataMatrix.vol_nodes` row a `vols` row mirrors.
        It is sorted ascending, so a plain searchsorted plus a membership proof
        resolves it.
        """
        if market_uix.size and int(market_uix.min()) < 0:
            raise ValueError(
                "volatility: a trade's volatility factor did not resolve to a "
                "market_id_universe coordinate."
            )
        return self._lookup_effective(
            needed=market_uix.astype(np.int64),
            table=self.vol_effective_uix.reshape(-1, 1),
            table_key=self.vol_effective_uix.astype(np.int64),
            n=market_uix.shape[0],
            ctx="volatility",
        )

    def map_fk_to_effective_uix(
        self,
        *,
        efp_uix: npt.NDArray[np.int32],
        edf_expiry_uix: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        """
        Translate a per-trade `(efp_uix, edf_expiry_uix)` pair into a
        `ValuationState.forward_prices_strike_ccy` row.

        The strike-currency forward is not a new market quantity: it is the
        ratio of two effective rows this mapper already carries,

            F_K(T) = effective_forward_prices[efp] / effective_discount_factors[edf]

        with BOTH legs taken at the option's EXPIRY date. Deduplicating on that
        pair is therefore the whole of its identity.

        The same row index addresses `option_log_forward_prices_strike_ccy`: the
        log cache is generated as an elementwise log of this table, row for row,
        so the VAL-11 pairing invariant `L[k] == log(F[k])` holds by
        construction rather than by assertion.
        """
        n_efp = max(self.efp_effective_quads.shape[0], 1)
        n_edf = max(self.edf_effective_triples.shape[0], 1)
        return self._lookup_effective(
            needed=_mixed_radix_key(
                (efp_uix.astype(np.int64), edf_expiry_uix.astype(np.int64)),
                (n_efp, n_edf),
                ctx="forward_prices_strike_ccy",
            ),
            table=self.fk_effective_pairs,
            table_key=_mixed_radix_key(
                (
                    self.fk_effective_pairs[:, 0].astype(np.int64),
                    self.fk_effective_pairs[:, 1].astype(np.int64),
                ),
                (n_efp, n_edf),
                ctx="forward_prices_strike_ccy",
            ),
            n=efp_uix.shape[0],
            ctx="strike-currency-forward",
        )

    def tte_years(self, ordinals: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
        """
        Canonical act/365.25 time-to-expiry from the run's as-of date.

        This MUST use the same `as_of_ord32` that `build_market_data_matrix`
        used to build `tte_grid`; a mismatch silently reprices every option
        against a different maturity. The compiler cannot detect that, so the
        as-of is carried on the mapper -- one per-run value, one source.
        """
        if self.as_of_ord32 is None:
            raise ValueError(
                "UixMapper was built without `as_of_ord32`, so time-to-expiry "
                "cannot be resolved. Options require it; pass it to "
                "`build_uix_mapper`."
            )
        raw = (
            ordinals.astype(np.float64) - float(self.as_of_ord32)
        ) / DAYS_PER_YR_ACT_365_25
        return np.maximum(raw, TTE_FLOOR)

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
