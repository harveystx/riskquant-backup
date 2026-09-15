# src/riskquant/sensitivities/sensitizers/spot.py
"""
Spot block sensitizer.

Sensitivity-side counterpart of `valuation.pricers.spot`: unpacks the block
and the compiled identity graph into flat arrays, calls the dumb kernel, and
resolves execution coordinates back to global `rf_id`s.

Why this takes a `MarketDataMatrix`, not a `ValuationState`
-----------------------------------------------------------
`ValuationState` deliberately carries only the FOLDED
`effective_spot_prices = spot_price * fx_effective` (its `spot_prices` and
`fx_spots_effective` fields are commented out precisely because pricing never
needs them separately). Sensitivities must DECOMPOSE that product -- dPV/dSpot
and dPV/dFX are different numbers against different risk factors -- so the
folded array is insufficient by construction.

`MarketDataMatrix` is the correct tier: it is the raw, universe-aligned input
to that fold, and it carries `prices` (N_mkt, S) and `fx_spots_effective`
(K_fx, S) unfolded, keyed by exactly the universes `UixMapper` indexes. This
is why the sensitivity path is a sibling of valuation rather than a consumer
of it -- both read the same compiled portfolio and the same market state, at
different points in the fold.

`RiskStaticConfig` is NOT required and is deliberately not accepted: every
identity this layer needs is already compiled into `UixMapper`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from riskquant.constants import INVALID_UIX
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivities.sensitizer_kernels.spot import _kernel_sensitize_spot

if TYPE_CHECKING:
    from riskquant.market_data.state import MarketDataMatrix
    from riskquant.universe import UixMapper
    from riskquant.valuation.blocks.spot_block import SpotBlock


def sensitize_spot_block(
    *,
    block: SpotBlock,
    uix_mapper: UixMapper,
    market_data: MarketDataMatrix,
    scenario: int = 0,
) -> BlockSensitivities:
    """
    Compute exact closed-form sensitivities for a compiled `SpotBlock`.

    Parameters
    ----------
    block
        Compiled spot block (SoA).
    uix_mapper
        The portfolio's own compiled identity graph. `spot_effective_pairs`
        maps each `effective_spot_price_uix` to `(price_uix, fx_eff_uix)`;
        `fx_effective_pairs` maps `fx_eff_uix` to `(fx_rf_uix, invert)`;
        `market_id_universe` maps a market uix to a global `rf_id`. No FX
        relationship is inferred from a name or a settlement-currency string.
    market_data
        Raw, universe-aligned market grids (see module docstring).
    scenario
        Scenario column to evaluate against. Sensitivities are a point-in-
        state quantity; the base snapshot is column 0.

    Returns
    -------
    BlockSensitivities
        Flat SoA arrays keyed by global `rf_id`.
    """
    n_trades = int(block.qty.shape[0])

    # ---- 1. Unpack the compiled identity graph into flat per-trade arrays --
    # One fancy-index gather per table; no Python loop over trades.
    eff_uix = block.effective_spot_price_uix
    spot_pairs = uix_mapper.spot_effective_pairs[eff_uix]
    price_uix = np.ascontiguousarray(spot_pairs[:, 0], dtype=np.int32)
    fx_eff_uix = np.ascontiguousarray(spot_pairs[:, 1], dtype=np.int32)

    has_fx = fx_eff_uix != INVALID_UIX

    # `fx_eff_uix` indexes `fx_effective_pairs` rows; that row's column 0 is
    # the market uix of the FX factor itself. Sentinel rows would index
    # garbage, so they are pointed at row 0 and masked out afterwards -- the
    # kernel never reads them for sentinel trades.
    safe_fx_eff_uix = np.where(has_fx, fx_eff_uix, 0).astype(np.int32, copy=False)

    if uix_mapper.fx_effective_pairs.size:
        fx_rows = uix_mapper.fx_effective_pairs[safe_fx_eff_uix]
        fx_market_uix = np.ascontiguousarray(fx_rows[:, 0], dtype=np.int32)
        fx_invert = fx_rows[:, 1]
    else:
        fx_market_uix = np.zeros(n_trades, dtype=np.int32)
        fx_invert = np.zeros(n_trades, dtype=np.int32)

    # FAIL FAST: an inverted FX-effective leg means PV is qty * S / FX, whose
    # partials are NOT the bilinear ones this kernel implements. Rather than
    # silently emit wrong risk, reject it until the inverted case is authored.
    if bool(np.any(has_fx & (fx_invert != 0))):
        raise NotImplementedError(
            "Inverted FX-effective spot legs (invert=1) are not supported by "
            "the closed-form spot sensitizer: PV = qty * spot / fx has "
            "different partials than the direct (invert=0) case. Author the "
            "inverted branch before pricing risk on such a trade."
        )

    # ---- 2. Dumb math ------------------------------------------------------
    spot_first_order = np.empty(n_trades, dtype=np.float64)
    fx_first_order = np.empty(n_trades, dtype=np.float64)
    cross_second_order = np.empty(n_trades, dtype=np.float64)

    _kernel_sensitize_spot(
        spot_first_order,
        fx_first_order,
        cross_second_order,
        block.qty,
        price_uix,
        np.ascontiguousarray(
            np.where(has_fx, safe_fx_eff_uix, INVALID_UIX), dtype=np.int32
        ),
        market_data.prices,
        market_data.fx_spots_effective,
        np.int32(INVALID_UIX),
        np.int64(scenario),
    )

    # ---- 3. Resolve execution coordinates -> global rf_id ------------------
    # This is the single point where uix becomes rf_id. Downstream consumers
    # never need the mapper to interpret a sensitivity row.
    market_universe = uix_mapper.market_id_universe
    price_rf_ids = np.ascontiguousarray(market_universe[price_uix], dtype=np.int32)
    fx_rf_ids = np.ascontiguousarray(market_universe[fx_market_uix], dtype=np.int32)

    trade_uids = np.ascontiguousarray(block.trade_uids, dtype=np.int64)

    # ---- 4. Assemble entries ----------------------------------------------
    # Every trade contributes its own price-factor row; only FX-legged trades
    # contribute an FX row and a cross term.
    position_uids = np.concatenate((trade_uids, trade_uids[has_fx]))
    factor_rf_ids = np.concatenate((price_rf_ids, fx_rf_ids[has_fx]))
    first_order = np.concatenate((spot_first_order, fx_first_order[has_fx]))
    # Own-factor second-order terms are identically zero for a bilinear payoff.
    second_order = np.zeros(position_uids.shape[0], dtype=np.float64)

    return BlockSensitivities(
        position_uids=np.ascontiguousarray(position_uids, dtype=np.int64),
        factor_rf_ids=np.ascontiguousarray(factor_rf_ids, dtype=np.int32),
        first_order=np.ascontiguousarray(first_order, dtype=np.float64),
        second_order=second_order,
        cross_position_uids=np.ascontiguousarray(trade_uids[has_fx], dtype=np.int64),
        cross_factor_rf_ids_i=np.ascontiguousarray(
            price_rf_ids[has_fx], dtype=np.int32
        ),
        cross_factor_rf_ids_j=np.ascontiguousarray(fx_rf_ids[has_fx], dtype=np.int32),
        cross_second_order=np.ascontiguousarray(
            cross_second_order[has_fx], dtype=np.float64
        ),
    )
