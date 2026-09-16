# src/riskquant/portfolio/compilers/block_compilers/vanilla_forward.py
"""
Vanilla forward block compiler.

Translates the vanilla-forward sub-range of an `IngestBatch` (global IDs) into a
`VanillaForwardBlock` (execution UIXs) via the run's `UixMapper`.

Supported economic contract
---------------------------
A PHYSICALLY SETTLED forward: at delivery the holder receives one unit of the
asset and pays `strike` in the strike currency. Each leg is discounted on its
own curve and converted at its own spot FX, so for a long quantity `Q`:

    PV(0) = Q * [ F_a(0,T) DF_a(0,T) FX_spot(a -> settle)
                  - K      DF_k(0,T) FX_spot(k -> settle) ]

This degenerates EXACTLY to the single-currency form `Q (F - K) DF` when both
legs share a curve and a currency, which is also the cash-settled value in that
case -- a cash-settled forward has one net cashflow discounted once.

The block therefore does NOT carry a settlement enum (D045/VAL-15). One block,
one economic meaning. What the compiler must guarantee is that a cash-settled
forward is never compiled with two DIFFERENT curves, because the two-leg form
would then silently produce the physical value instead.

Source kind
-----------
NODE-sourced forwards only. `VANILLA_FORWARD_CURVE_ID` reads
`MarketDataMatrix.price_curve_grid` rather than `forward_nodes` and needs a
source-kind column in the effective-forward universe; that is deliberately out
of scope for this phase and stays unregistered, so the engine fails loud on
encounter rather than pricing it from the wrong array.

UIX translation
---------------
    underlying_ids               -> price_uix   (market_id_universe)
    under_discount_curve_ids     -> curve_uix   (discount_curve_id_universe)
    delivery_ord                 -> date_uix    (date_ord32_universe)
    (fx_u2s_ids, fx_u2s_inv)     -> fx_eff_uix  (fx_spots_effective row)
    the four above               -> effective_forward_price_uix

    strike_discount_curve_ids    -> curve_uix
    settle_ord                   -> date_uix
    (fx_k2s_ids, fx_k2s_inv)     -> fx_eff_uix
    the three above              -> effective_discount_factor_uix

`qty`, `strike` and `trade_uids` are pure economics and need no translation;
slicing returns NumPy views.

Fail-fast ownership (D045/VAL-04, forward path)
-----------------------------------------------
This compiler is the layer that proves every UIX it stamps is a valid dense row
of a POPULATED factor universe. It refuses to emit a block otherwise, because
the kernel runs under `boundscheck=False` and cannot tell an out-of-range row
from a real one. The two effective universes it indexes are built by
`build_uix_mapper` from exactly these legs, so a miss here means the mapper and
the batch disagree -- which is a wiring bug, not a market-data gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.valuation.blocks.vanilla_forward_block import VanillaForwardBlock

if TYPE_CHECKING:
    from riskquant.universe import UixMapper


def _require_resolved(
    uix: np.ndarray,
    *,
    n_rows: int,
    leg: str,
    uids: np.ndarray,
) -> None:
    """
    Prove `0 <= uix < n_rows` for every row before the block is emitted.

    `n_rows` is the row count of the effective universe the kernel will gather
    from. A negative value means the leg did not resolve (sentinel curve, date
    or factor); a value at or past `n_rows` means the universe is smaller than
    the coordinate -- the production form of the empty/unpopulated-state defect.
    """
    bad = (uix < 0) | (uix >= n_rows)
    if not bool(bad.any()):
        return
    offenders = uids[bad][:10].tolist()
    raise ValueError(
        f"vanilla forward: {int(bad.sum())} row(s) have an unresolved or "
        f"out-of-range {leg} coordinate against a universe of {n_rows} row(s). "
        f"First offending trade uid(s): {offenders}. A forward cannot be "
        "compiled until its market-data legs are present in the run's "
        "effective-factor universes."
    )


def compile_vanilla_forward_block(
    batch: IngestBatch,
    start: int,
    end: int,
    uix_mapper: UixMapper,
) -> VanillaForwardBlock | None:
    """Slice + UIX-translate a vanilla-forward run from `batch[start:end]`."""
    if start == end:
        return None

    uids = batch.uids[start:end]

    # ---- UNDERLYING LEG -> effective_forward_prices row -------------------
    price_uix = uix_mapper.map_market_ids_to_uix(batch.underlying_ids[start:end])
    under_curve_uix = uix_mapper.map_discount_ids_to_uix(
        batch.under_discount_curve_ids[start:end]
    )
    delivery_uix = uix_mapper.map_dates_to_uix(batch.delivery_ord[start:end])
    fx_u2s_eff_uix = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_u2s_ids[start:end],
        raw_fx_inv=batch.fx_u2s_inv[start:end],
    )

    effective_forward_price_uix = uix_mapper.map_efp_to_effective_uix(
        price_uix=price_uix,
        curve_uix=under_curve_uix,
        date_uix=delivery_uix,
        fx_eff_uix=fx_u2s_eff_uix,
    )
    _require_resolved(
        effective_forward_price_uix,
        n_rows=uix_mapper.efp_effective_quads.shape[0],
        leg="effective_forward_price",
        uids=uids,
    )

    # ---- STRIKE LEG -> effective_discount_factors row ---------------------
    strike_curve_uix = uix_mapper.map_discount_ids_to_uix(
        batch.strike_discount_curve_ids[start:end]
    )
    settle_uix = uix_mapper.map_dates_to_uix(batch.settle_ord[start:end])
    fx_k2s_eff_uix = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_k2s_ids[start:end],
        raw_fx_inv=batch.fx_k2s_inv[start:end],
    )

    effective_discount_factor_uix = uix_mapper.map_edf_to_effective_uix(
        curve_uix=strike_curve_uix,
        date_uix=settle_uix,
        fx_eff_uix=fx_k2s_eff_uix,
    )
    _require_resolved(
        effective_discount_factor_uix,
        n_rows=uix_mapper.edf_effective_triples.shape[0],
        leg="effective_discount_factor",
        uids=uids,
    )

    return VanillaForwardBlock(
        trade_uids=uids,
        qty=batch.qty[start:end],
        strike=batch.strike_price[start:end],
        effective_forward_price_uix=effective_forward_price_uix,
        effective_discount_factor_uix=effective_discount_factor_uix,
    )
