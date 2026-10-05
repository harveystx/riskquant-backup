# src/riskquant/sensitivities/sensitizers/vanilla_forward.py
"""
Vanilla forward sensitizer (layer 2; D103, WS-C).

One sensitizer serves every `VanillaForwardBlock`: block type drives dispatch,
and the instrument source is not recovered from it (D102.1). Run admission is
not decided here (D102, D103.4).

Coordinate (D069, D081 §4): partials are taken with respect to the D069
normalised LEVEL of each factor. For a Proxy V1 target A <- beta * Y, the
emitted coefficient is dPV/dx_A; beta enters level generation only, and this
protocol has no access to it.

Sequence:

    1. unpack `efp_effective_quads` and `edf_effective_triples` through the
       block's uix columns;
    2. read `fx_effective_pairs[:, 1]` of every used FX leg, and refuse any
       inverted leg (D103.3) BEFORE any market level is read;
    3. call the layer-1 kernel for the primitive partials;
    4. map each primitive to its global rf id through `market_id_universe`;
    5. pre-aggregate by (uid, rf) and by (uid, lo_rf, hi_rf): primitives that
       share an rf merge here (the same-rf product contributes 2 * w to the
       diagonal), and every pair of each position's factor set F(p) is
       emitted, with an explicit zero where the pair is mathematically zero
       (SECOND_ORDER_FULL completeness).

`DuplicatePolicy.RAISE` downstream is a guard only: this block's output has
unique keys. Outputs of different blocks are not merged here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from riskquant.constants import INVALID_UIX
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivities.sensitizer_kernels.vanilla_forward import (
    _kernel_sensitize_vanilla_forward,
)

if TYPE_CHECKING:
    from riskquant.market_data.state import MarketDataMatrix
    from riskquant.universe import UixMapper
    from riskquant.valuation.blocks.vanilla_forward_block import VanillaForwardBlock


def _group_starts(*keys: npt.NDArray[np.int64]) -> npt.NDArray[np.intp]:
    """Start index of each run of equal keys in lexicographically sorted arrays."""
    change = np.zeros(keys[0].shape[0], dtype=np.bool_)
    change[0] = True
    for key in keys:
        change[1:] |= key[1:] != key[:-1]
    return np.flatnonzero(change)


def sensitize_vanilla_forward_block(
    *,
    block: VanillaForwardBlock,
    uix_mapper: UixMapper,
    market_data: MarketDataMatrix,
    scenario: int = 0,
) -> BlockSensitivities:
    """
    Compute exact closed-form level-coordinate sensitivities of a compiled
    `VanillaForwardBlock`.

    Parameters
    ----------
    block
        Compiled forward block (SoA).
    uix_mapper
        The portfolio's compiled identity graph. `efp_effective_quads` maps an
        `effective_forward_price_uix` to `(price_uix, curve_uix, date_uix,
        fx_eff_uix)`; `edf_effective_triples` maps an
        `effective_discount_factor_uix` to `(curve_uix, date_uix, fx_eff_uix)`;
        `fx_effective_pairs` maps `fx_eff_uix` to `(fx_rf_uix, invert)`;
        `market_id_universe` maps a market uix to a global `rf_id`.
    market_data
        Raw, universe-aligned market grids.
    scenario
        Scenario column to evaluate against; the base snapshot is column 0.

    Returns
    -------
    BlockSensitivities
        Flat SoA arrays keyed by global `rf_id`, unique per (uid, rf) and per
        (uid, lo_rf, hi_rf), with `lo_rf < hi_rf`.

    Raises
    ------
    NotImplementedError
        D103.3: an FX leg with `invert == 1`. The partials of the inverted
        branch are mathematically defined but outside the admission contract.
    """
    quads = uix_mapper.efp_effective_quads[block.effective_forward_price_uix]
    triples = uix_mapper.edf_effective_triples[block.effective_discount_factor_uix]
    fx_a = np.ascontiguousarray(quads[:, 3], dtype=np.int32)
    fx_k = np.ascontiguousarray(triples[:, 2], dtype=np.int32)
    has_a = fx_a != INVALID_UIX
    has_k = fx_k != INVALID_UIX
    pairs = uix_mapper.fx_effective_pairs
    if bool(np.any(pairs[fx_a[has_a], 1] != 0) or np.any(pairs[fx_k[has_k], 1] != 0)):
        raise NotImplementedError(
            "D103.3: a VANILLA_FORWARD row has an inverted FX leg (invert=1). "
            "Its partials are mathematically defined but outside the admission "
            "contract; inverted-FX TAYLOR admission needs a separate decision."
        )

    trade_uids = np.ascontiguousarray(block.trade_uids, dtype=np.int64)
    n_rows = int(trade_uids.shape[0])
    if n_rows == 0:
        none_i64, none_i32 = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
        none_f64 = np.empty(0, dtype=np.float64)
        return BlockSensitivities(
            position_uids=none_i64,
            factor_rf_ids=none_i32,
            first_order=none_f64,
            second_order=none_f64.copy(),
            cross_position_uids=none_i64.copy(),
            cross_factor_rf_ids_i=none_i32.copy(),
            cross_factor_rf_ids_j=none_i32.copy(),
            cross_second_order=none_f64.copy(),
        )

    d_f = np.empty(n_rows, dtype=np.float64)
    d_x_a = np.empty(n_rows, dtype=np.float64)
    d_x_k = np.empty(n_rows, dtype=np.float64)
    d_f_x_a = np.empty(n_rows, dtype=np.float64)
    _kernel_sensitize_vanilla_forward(
        d_f,
        d_x_a,
        d_x_k,
        d_f_x_a,
        np.ascontiguousarray(block.qty, dtype=np.float64),
        np.ascontiguousarray(block.strike, dtype=np.float64),
        np.ascontiguousarray(quads[:, 0], dtype=np.int32),
        np.ascontiguousarray(quads[:, 1], dtype=np.int32),
        np.ascontiguousarray(quads[:, 2], dtype=np.int32),
        fx_a,
        np.ascontiguousarray(triples[:, 0], dtype=np.int32),
        np.ascontiguousarray(triples[:, 1], dtype=np.int32),
        market_data.forward_nodes,
        market_data.discount_factor_grid,
        market_data.fx_spots_effective,
        np.int32(INVALID_UIX),
        np.int64(scenario),
    )

    universe = uix_mapper.market_id_universe
    rf_f = universe[quads[:, 0]].astype(np.int64)
    rf_a = universe[pairs[fx_a[has_a], 0]].astype(np.int64)
    rf_k = universe[pairs[fx_k[has_k], 0]].astype(np.int64)
    uid_a = trade_uids[has_a]
    w_a = d_f_x_a[has_a]
    same = rf_f[has_a] == rf_a

    # ---- diagonal: (uid, rf) -> first order, second order -------------------------
    # A same-rf (F, X_a) product is w * x^2: d2PV/dx2 = 2 w.
    diag_uid = np.concatenate((trade_uids, uid_a, trade_uids[has_k], uid_a[same]))
    diag_rf = np.concatenate((rf_f, rf_a, rf_k, rf_a[same]))
    diag_d1 = np.concatenate((
        d_f,
        d_x_a[has_a],
        d_x_k[has_k],
        np.zeros(int(same.sum())),
    ))
    diag_d2 = np.concatenate((
        np.zeros(n_rows + uid_a.size + int(has_k.sum())),
        2.0 * w_a[same],
    ))
    order = np.lexsort((diag_rf, diag_uid))
    diag_uid, diag_rf = diag_uid[order], diag_rf[order]
    starts = _group_starts(diag_uid, diag_rf)
    first_order = np.add.reduceat(diag_d1[order], starts)
    second_order = np.add.reduceat(diag_d2[order], starts)
    key_uid, key_rf = diag_uid[starts], diag_rf[starts]

    # ---- complete pair set of every F(p), lo < hi (keys are sorted per uid) ---------
    group = np.cumsum(np.r_[True, key_uid[1:] != key_uid[:-1]]) - 1
    size = np.bincount(group)[group]
    local = (
        np.arange(key_uid.size) - np.r_[0, np.cumsum(np.bincount(group))[:-1]][group]
    )
    partners = size - 1 - local
    lo_idx = np.repeat(np.arange(key_uid.size), partners)
    hi_idx = (
        lo_idx
        + 1
        + (np.arange(lo_idx.size) - np.repeat(np.cumsum(partners) - partners, partners))
    )

    # ---- pairs: (uid, lo, hi) -> cross second order ------------------------------------
    diff = ~same
    rf_f_a = rf_f[has_a][diff]
    rf_x_a = rf_a[diff]
    pair_uid = np.concatenate((key_uid[lo_idx], uid_a[diff]))
    pair_lo = np.concatenate((key_rf[lo_idx], np.minimum(rf_f_a, rf_x_a)))
    pair_hi = np.concatenate((key_rf[hi_idx], np.maximum(rf_f_a, rf_x_a)))
    pair_v = np.concatenate((np.zeros(lo_idx.size), w_a[diff]))
    if pair_uid.size:
        order = np.lexsort((pair_hi, pair_lo, pair_uid))
        pair_uid, pair_lo, pair_hi = pair_uid[order], pair_lo[order], pair_hi[order]
        starts = _group_starts(pair_uid, pair_lo, pair_hi)
        cross = np.add.reduceat(pair_v[order], starts)
        pair_uid, pair_lo, pair_hi = pair_uid[starts], pair_lo[starts], pair_hi[starts]
    else:
        cross = pair_v

    return BlockSensitivities(
        position_uids=np.ascontiguousarray(key_uid, dtype=np.int64),
        factor_rf_ids=np.ascontiguousarray(key_rf, dtype=np.int32),
        first_order=np.ascontiguousarray(first_order, dtype=np.float64),
        second_order=np.ascontiguousarray(second_order, dtype=np.float64),
        cross_position_uids=np.ascontiguousarray(pair_uid, dtype=np.int64),
        cross_factor_rf_ids_i=np.ascontiguousarray(pair_lo, dtype=np.int32),
        cross_factor_rf_ids_j=np.ascontiguousarray(pair_hi, dtype=np.int32),
        cross_second_order=np.ascontiguousarray(cross, dtype=np.float64),
    )
