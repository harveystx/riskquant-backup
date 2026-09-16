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
      discount-curve IDs, dates) AND deduplicates every distinct effective-
      factor tuple across all trades: the `(fx_rf_uix, invert)` FX pairs and
      the `(price_uix, fx_eff_uix)` effective-spot pairs. The result, a
      `UixMapper`, is the single source of truth for ID->UIX translation
      downstream.

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
- The effective-factor pair tables are built in the same pass as the market
  universe so the effective UIX spaces are part of the global contract.
  Pricers index dense `(K, S)` effective matrices built/folded by the
  market-data layer; `INVALID_UIX` (< 0) signals "no FX conversion required".
- Compilers take `(batch, start, end, mapper)`. They are the ONLY place
  global IDs are translated to UIXs; blocks never see global IDs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_propagate_sentinel_int32
from riskquant.constants import INVALID_UIX, NO_FX_RF_REQUIRED
from riskquant.portfolio.compiled import (
    CompiledPortfolio,
    CompiledPortfolioFields,
)
from riskquant.portfolio.compilers.registry import COMPILERS, FIELD_NAMES
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.instruments import (
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_NODE_ID,
)
from riskquant.universe import UixMapper, lookup_fx_effective_uix

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
    `np.searchsorted` lookup that `lookup_fx_effective_uix` performs via the
    packed `key = (uix << 1) | inv` trick.
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


def _dedup_spot_effective_pairs(
    *,
    market_id_universe: npt.NDArray[np.int32],
    fx_effective_pairs: npt.NDArray[np.int32],
    raw_underlying_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_fx_u2s_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_fx_u2s_inv_arrays: tuple[npt.NDArray[np.int8], ...],
) -> npt.NDArray[np.int32]:
    """
    Build the deduplicated `(price_uix, fx_eff_uix)` effective-spot table.

    For every trade's spot leg (underlying price + its under->settle FX),
    resolve `price_uix` (into `market_id_universe`) and `fx_eff_uix` (an
    `fx_effective_pairs` row, or `INVALID_UIX` for identity FX), then
    `np.unique(axis=0)`. Built as a global superset over ALL trades' underlying
    legs -- exactly mirroring `_dedup_fx_effective_pairs`; rows referenced by no
    spot block are simply never gathered (and are bounded by distinct
    underlying x FX combinations).

    Resolution uses `lookup_fx_effective_uix` against the already-built
    `fx_effective_pairs`, guaranteeing the `fx_eff_uix` here is byte-identical
    to the one the spot compiler stamps via `map_fx_legs_to_effective_uix`.

    Returned shape is `(K, 2) int32`, sorted lexicographically by
    `(price_uix, fx_eff_uix)` -- the order the packed-key searchsorted in
    `map_spot_to_effective_uix` relies on.
    """
    if not raw_underlying_id_arrays:
        return np.empty((0, 2), dtype=np.int32)

    underlying_ids = np.concatenate(raw_underlying_id_arrays)
    if underlying_ids.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    fx_ids = np.concatenate(raw_fx_u2s_id_arrays)
    fx_inv = np.concatenate(raw_fx_u2s_inv_arrays)

    # Underlyings are always present (never sentinel), so a plain searchsorted
    # matches `map_market_ids_to_uix` for these rows.
    price_uix = np.searchsorted(market_id_universe, underlying_ids).astype(
        np.int32, copy=False
    )
    fx_eff_uix = lookup_fx_effective_uix(
        market_id_universe=market_id_universe,
        fx_effective_pairs=fx_effective_pairs,
        raw_fx_ids=fx_ids,
        raw_fx_inv=fx_inv,
    )

    pairs = np.stack((price_uix, fx_eff_uix), axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    return np.ascontiguousarray(unique_pairs.astype(np.int32, copy=False))


def _forward_node_mask(
    raw_instrument_id_arrays: tuple[npt.NDArray[np.int32], ...],
) -> npt.NDArray[np.bool_]:
    """Rows whose instrument is a NODE-sourced vanilla forward."""
    if not raw_instrument_id_arrays:
        return np.empty(0, dtype=np.bool_)
    ids = np.concatenate(raw_instrument_id_arrays)
    return ids == np.int32(VANILLA_FORWARD_NODE_ID)


def _dedup_edf_effective_triples(
    *,
    market_id_universe: npt.NDArray[np.int32],
    discount_curve_id_universe: npt.NDArray[np.int32],
    date_ord32_universe: npt.NDArray[np.int32],
    fx_effective_pairs: npt.NDArray[np.int32],
    raw_curve_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_date_ord_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_fx_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_fx_inv_arrays: tuple[npt.NDArray[np.int8], ...],
    row_mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.int32]:
    """
    Build the deduplicated `(curve_uix, date_uix, fx_eff_uix)` table backing
    `ValuationState.effective_discount_factors`.

    Rows whose curve or date is a sentinel carry no discounting leg and are
    dropped: a spot trade has neither, and including it would mint a row the
    fold cannot value. `row_mask` narrows the pool further where a caller knows
    only some instruments contribute.

    Returns `(K, 3) int32` sorted lexicographically -- the order the mixed-radix
    searchsorted in `map_edf_to_effective_uix` relies on.
    """
    if not raw_curve_id_arrays:
        return np.empty((0, 3), dtype=np.int32)

    curve_ids = np.concatenate(raw_curve_id_arrays)
    date_ords = np.concatenate(raw_date_ord_arrays)
    fx_ids = np.concatenate(raw_fx_id_arrays)
    fx_inv = np.concatenate(raw_fx_inv_arrays)
    if curve_ids.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    if row_mask is not None:
        curve_ids, date_ords = curve_ids[row_mask], date_ords[row_mask]
        fx_ids, fx_inv = fx_ids[row_mask], fx_inv[row_mask]

    curve_uix = searchsorted_or_propagate_sentinel_int32(
        universe=discount_curve_id_universe, raw=curve_ids, sentinel=INVALID_UIX
    )
    date_uix = searchsorted_or_propagate_sentinel_int32(
        universe=date_ord32_universe, raw=date_ords, sentinel=INVALID_UIX
    )
    fx_eff_uix = lookup_fx_effective_uix(
        market_id_universe=market_id_universe,
        fx_effective_pairs=fx_effective_pairs,
        raw_fx_ids=fx_ids,
        raw_fx_inv=fx_inv,
    )

    keep = (curve_uix >= 0) & (date_uix >= 0)
    if not bool(keep.any()):
        return np.empty((0, 3), dtype=np.int32)

    triples = np.stack((curve_uix[keep], date_uix[keep], fx_eff_uix[keep]), axis=1)
    return np.ascontiguousarray(np.unique(triples, axis=0).astype(np.int32, copy=False))


def _dedup_efp_effective_quads(
    *,
    market_id_universe: npt.NDArray[np.int32],
    discount_curve_id_universe: npt.NDArray[np.int32],
    date_ord32_universe: npt.NDArray[np.int32],
    fx_effective_pairs: npt.NDArray[np.int32],
    raw_underlying_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_curve_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_date_ord_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_fx_id_arrays: tuple[npt.NDArray[np.int32], ...],
    raw_fx_inv_arrays: tuple[npt.NDArray[np.int8], ...],
    row_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int32]:
    """
    Build the deduplicated `(price_uix, curve_uix, date_uix, fx_eff_uix)` table
    backing `ValuationState.effective_forward_prices`.

    `row_mask` restricts the pool to NODE-sourced forwards. Curve-sourced
    forwards read `MarketDataMatrix.price_curve_grid` instead of `prices` and
    need a source-kind column here; that is deliberately NOT built in this phase
    (see the module docstring of the forward block compiler).
    """
    if not raw_underlying_id_arrays or row_mask.size == 0:
        return np.empty((0, 4), dtype=np.int32)

    under_ids = np.concatenate(raw_underlying_id_arrays)[row_mask]
    curve_ids = np.concatenate(raw_curve_id_arrays)[row_mask]
    date_ords = np.concatenate(raw_date_ord_arrays)[row_mask]
    fx_ids = np.concatenate(raw_fx_id_arrays)[row_mask]
    fx_inv = np.concatenate(raw_fx_inv_arrays)[row_mask]
    if under_ids.size == 0:
        return np.empty((0, 4), dtype=np.int32)

    price_uix = searchsorted_or_propagate_sentinel_int32(
        universe=market_id_universe, raw=under_ids, sentinel=INVALID_UIX
    )
    curve_uix = searchsorted_or_propagate_sentinel_int32(
        universe=discount_curve_id_universe, raw=curve_ids, sentinel=INVALID_UIX
    )
    date_uix = searchsorted_or_propagate_sentinel_int32(
        universe=date_ord32_universe, raw=date_ords, sentinel=INVALID_UIX
    )
    fx_eff_uix = lookup_fx_effective_uix(
        market_id_universe=market_id_universe,
        fx_effective_pairs=fx_effective_pairs,
        raw_fx_ids=fx_ids,
        raw_fx_inv=fx_inv,
    )

    keep = (price_uix >= 0) & (curve_uix >= 0) & (date_uix >= 0)
    if not bool(keep.any()):
        return np.empty((0, 4), dtype=np.int32)

    quads = np.stack(
        (price_uix[keep], curve_uix[keep], date_uix[keep], fx_eff_uix[keep]), axis=1
    )
    return np.ascontiguousarray(np.unique(quads, axis=0).astype(np.int32, copy=False))


def _option_node_mask(
    raw_instrument_id_arrays: tuple[npt.NDArray[np.int32], ...],
) -> npt.NDArray[np.bool_]:
    """Rows whose instrument is a NODE-sourced vanilla European option."""
    if not raw_instrument_id_arrays:
        return np.empty(0, dtype=np.bool_)
    ids = np.concatenate(raw_instrument_id_arrays)
    return ids == np.int32(VANILLA_EUROPEAN_OPTION_NODE_ID)


def _dedup_vol_effective_uix(
    *,
    market_id_universe: npt.NDArray[np.int32],
    raw_volatility_id_arrays: tuple[npt.NDArray[np.int32], ...],
    row_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int32]:
    """
    Build the deduplicated market-uix set backing `ValuationState.vols`.

    A `vols` row mirrors a `MarketDataMatrix.vol_nodes` row, so its whole
    identity is the market-universe coordinate of the volatility risk factor.
    Sorted ascending -- the order `map_vol_to_effective_uix` searchsorts
    against.
    """
    if not raw_volatility_id_arrays or row_mask.size == 0:
        return np.empty(0, dtype=np.int32)

    vol_ids = np.concatenate(raw_volatility_id_arrays)[row_mask]
    if vol_ids.size == 0:
        return np.empty(0, dtype=np.int32)

    uix = searchsorted_or_propagate_sentinel_int32(
        universe=market_id_universe, raw=vol_ids, sentinel=INVALID_UIX
    )
    keep = uix >= 0
    if not bool(keep.any()):
        return np.empty(0, dtype=np.int32)
    return np.ascontiguousarray(np.unique(uix[keep]).astype(np.int32, copy=False))


def _dedup_fk_effective_pairs(
    *,
    efp_uix: npt.NDArray[np.int32],
    edf_expiry_uix: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """
    Build the deduplicated `(efp_uix, edf_expiry_uix)` table backing
    `ValuationState.forward_prices_strike_ccy` AND its log cache.

    The strike-currency forward is the ratio of two effective rows that already
    exist:

        F_K(T) = effective_forward_prices[efp] / effective_discount_factors[edf]

    both taken at the option's EXPIRY. Deduplicating on that pair is the whole
    of its identity, and because the log cache is generated elementwise from
    this same table, one row index addresses both -- which is what makes the
    VAL-11 pairing structural rather than asserted.
    """
    keep = (efp_uix >= 0) & (edf_expiry_uix >= 0)
    if not bool(keep.any()):
        return np.empty((0, 2), dtype=np.int32)
    pairs = np.stack((efp_uix[keep], edf_expiry_uix[keep]), axis=1)
    return np.ascontiguousarray(np.unique(pairs, axis=0).astype(np.int32, copy=False))


def _build_fk_pairs(
    *,
    mapper: UixMapper,
    ingest_batches: Sequence[IngestBatch],
    opt_mask: npt.NDArray[np.bool_],
    expiry_ord: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """
    Resolve every option's `(efp_uix, edf_expiry_uix)` pair and deduplicate.

    Uses the provisional mapper so the coordinates resolved here are the exact
    ones `compile_vanilla_european_option_block` will resolve per trade.
    """
    if not bool(opt_mask.any()):
        return np.empty((0, 2), dtype=np.int32)

    under_ids = np.concatenate(tuple(b.underlying_ids for b in ingest_batches))[
        opt_mask
    ]
    under_curve = np.concatenate(
        tuple(b.under_discount_curve_ids for b in ingest_batches)
    )[opt_mask]
    fx_u2s_id = np.concatenate(tuple(b.fx_u2s_ids for b in ingest_batches))[opt_mask]
    fx_u2s_inv = np.concatenate(tuple(b.fx_u2s_inv for b in ingest_batches))[opt_mask]
    strike_curve = np.concatenate(
        tuple(b.strike_discount_curve_ids for b in ingest_batches)
    )[opt_mask]
    fx_k2s_id = np.concatenate(tuple(b.fx_k2s_ids for b in ingest_batches))[opt_mask]
    fx_k2s_inv = np.concatenate(tuple(b.fx_k2s_inv for b in ingest_batches))[opt_mask]
    exp_ord = expiry_ord[opt_mask]

    expiry_uix = mapper.map_dates_to_uix(np.ascontiguousarray(exp_ord))
    efp_uix = mapper.map_efp_to_effective_uix(
        price_uix=mapper.map_market_ids_to_uix(np.ascontiguousarray(under_ids)),
        curve_uix=mapper.map_discount_ids_to_uix(np.ascontiguousarray(under_curve)),
        date_uix=expiry_uix,
        fx_eff_uix=mapper.map_fx_legs_to_effective_uix(
            raw_fx_ids=np.ascontiguousarray(fx_u2s_id),
            raw_fx_inv=np.ascontiguousarray(fx_u2s_inv),
        ),
    )
    edf_uix = mapper.map_edf_to_effective_uix(
        curve_uix=mapper.map_discount_ids_to_uix(np.ascontiguousarray(strike_curve)),
        date_uix=expiry_uix,
        fx_eff_uix=mapper.map_fx_legs_to_effective_uix(
            raw_fx_ids=np.ascontiguousarray(fx_k2s_id),
            raw_fx_inv=np.ascontiguousarray(fx_k2s_inv),
        ),
    )
    return _dedup_fk_effective_pairs(efp_uix=efp_uix, edf_expiry_uix=edf_uix)


# ==============================================================================
# ORCHESTRATOR PHASE (UNIVERSE DISCOVERY)
# ==============================================================================


def build_uix_mapper(
    ingest_batches: Sequence[IngestBatch],
    *,
    as_of_ord32: int | None = None,
) -> UixMapper:
    """
    Global universe discovery — the "orchestrator pause".

    Walks every `IngestBatch` once to collect:
      - the union of every market ID (underlyings, vols, FX spots)
      - the union of every discount curve ID
      - the union of every trade-relevant date ordinal
      - the deduplicated `(fx_uix, invert)` FX-effective table
      - the deduplicated `(price_uix, fx_eff_uix)` effective-spot table

    Returns a frozen `UixMapper` that downstream consumers (block compilers,
    market-data builder/fold, validation) use as the single source of truth for
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

    # ---- SPOT EFFECTIVE PAIRS (built on top of fx_effective_pairs) ----
    spot_effective_pairs = _dedup_spot_effective_pairs(
        market_id_universe=market_id_universe,
        fx_effective_pairs=fx_effective_pairs,
        raw_underlying_id_arrays=tuple(b.underlying_ids for b in ingest_batches),
        raw_fx_u2s_id_arrays=tuple(b.fx_u2s_ids for b in ingest_batches),
        raw_fx_u2s_inv_arrays=tuple(b.fx_u2s_inv for b in ingest_batches),
    )

    # ---- FORWARD + OPTION EFFECTIVE UNIVERSES ----
    # All are built on fx_effective_pairs, so the fx_eff_uix stamped here is
    # byte-identical to the one each compiler resolves per trade.
    instrument_id_arrays = tuple(b.instrument_ids for b in ingest_batches)
    fwd_mask = _forward_node_mask(instrument_id_arrays)
    opt_mask = _option_node_mask(instrument_id_arrays)

    # A forward's underlying leg matures at DELIVERY; an option's strike-ccy
    # forward is taken at EXPIRY. One pooled date column, chosen per row.
    delivery_ord = np.concatenate(tuple(b.delivery_ord for b in ingest_batches))
    expiry_ord = np.concatenate(tuple(b.expiry_ord for b in ingest_batches))
    settle_ord = np.concatenate(tuple(b.settle_ord for b in ingest_batches))
    underlying_date_ord = np.where(opt_mask, expiry_ord, delivery_ord)

    # The option strike leg needs an effective discount row at BOTH dates: the
    # EXPIRY row denominates the strike-ccy forward, the SETTLEMENT row
    # discounts the payoff.
    strike_curve_ids = np.concatenate(
        tuple(b.strike_discount_curve_ids for b in ingest_batches)
    )
    fx_k2s_ids = np.concatenate(tuple(b.fx_k2s_ids for b in ingest_batches))
    fx_k2s_inv = np.concatenate(tuple(b.fx_k2s_inv for b in ingest_batches))

    edf_effective_triples = _dedup_edf_effective_triples(
        market_id_universe=market_id_universe,
        discount_curve_id_universe=discount_curve_id_universe,
        date_ord32_universe=date_ord32_universe,
        fx_effective_pairs=fx_effective_pairs,
        raw_curve_id_arrays=(np.concatenate((strike_curve_ids, strike_curve_ids)),),
        raw_date_ord_arrays=(np.concatenate((settle_ord, expiry_ord)),),
        raw_fx_id_arrays=(np.concatenate((fx_k2s_ids, fx_k2s_ids)),),
        raw_fx_inv_arrays=(np.concatenate((fx_k2s_inv, fx_k2s_inv)),),
    )

    efp_effective_quads = _dedup_efp_effective_quads(
        market_id_universe=market_id_universe,
        discount_curve_id_universe=discount_curve_id_universe,
        date_ord32_universe=date_ord32_universe,
        fx_effective_pairs=fx_effective_pairs,
        raw_underlying_id_arrays=tuple(b.underlying_ids for b in ingest_batches),
        raw_curve_id_arrays=tuple(b.under_discount_curve_ids for b in ingest_batches),
        raw_date_ord_arrays=(underlying_date_ord,),
        raw_fx_id_arrays=tuple(b.fx_u2s_ids for b in ingest_batches),
        raw_fx_inv_arrays=tuple(b.fx_u2s_inv for b in ingest_batches),
        row_mask=fwd_mask | opt_mask,
    )

    vol_effective_uix = _dedup_vol_effective_uix(
        market_id_universe=market_id_universe,
        raw_volatility_id_arrays=tuple(b.volatility_ids for b in ingest_batches),
        row_mask=opt_mask,
    )

    # The strike-ccy forward universe is a pair of the two tables above, so it
    # is resolved through a provisional mapper carrying them.
    provisional = UixMapper(
        market_id_universe=market_id_universe,
        discount_curve_id_universe=discount_curve_id_universe,
        date_ord32_universe=date_ord32_universe,
        fx_effective_pairs=fx_effective_pairs,
        spot_effective_pairs=spot_effective_pairs,
        edf_effective_triples=edf_effective_triples,
        efp_effective_quads=efp_effective_quads,
        vol_effective_uix=vol_effective_uix,
        as_of_ord32=as_of_ord32,
    )
    fk_effective_pairs = _build_fk_pairs(
        mapper=provisional,
        ingest_batches=ingest_batches,
        opt_mask=opt_mask,
        expiry_ord=expiry_ord,
    )

    return UixMapper(
        market_id_universe=market_id_universe,
        discount_curve_id_universe=discount_curve_id_universe,
        date_ord32_universe=date_ord32_universe,
        fx_effective_pairs=fx_effective_pairs,
        spot_effective_pairs=spot_effective_pairs,
        edf_effective_triples=edf_effective_triples,
        efp_effective_quads=efp_effective_quads,
        vol_effective_uix=vol_effective_uix,
        fk_effective_pairs=fk_effective_pairs,
        as_of_ord32=as_of_ord32,
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
            # Fail loud rather than silently drop trades. During buildout some
            # instruments have no compiler yet; a batch that actually contains
            # one must not vanish from the compiled portfolio.
            raise RuntimeError(
                f"No compiler registered for instrument_id={instrument_id} "
                f"(run [{start}:{end}], {end - start} trade(s)). Register it in "
                "compilers.registry before compiling portfolios that use it."
            )

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
