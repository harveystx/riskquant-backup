# src/riskquant/credit/exposure/saccr/aggregate.py
"""
SA-CCR add-on aggregation (scenario-independent, vectorized at the edge).

The add-on depends only on effective notionals, never on marks, so it is computed
once here, not in the per-scenario kernel. Chain of segment reductions:

  trade -> (netting set, hedging set, subclass):  D_jk = sum(effective_notional)
  add-on per (ns,hs,subclass):                    A_jk = SF_subclass * D_jk (signed)
  hedging-set correlation:  A_hs = sqrt( (rho*sum_k A_jk)^2 + (1-rho^2)*sum_k A_jk^2 )
  netting set:              AddOn = sum_hs A_hs

Pure NumPy (bincount over composite keys); the nonlinear sqrt is why this cannot
be a per-trade kernel sum. Returns the add-on aligned to a supplied
``netting_set_uids`` order (the view's CSR order).
"""

from __future__ import annotations

__all__ = ["compute_saccr_add_on"]

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_raise
from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig


def compute_saccr_add_on(
    *,
    netting_set_uids: npt.NDArray[np.int64],
    trade_netting_set_id: npt.NDArray[np.int64],
    trade_effective_notional: npt.NDArray[np.float64],
    trade_subclass_uix: npt.NDArray[np.int32],
    add_on_factors: CompiledAddOnFactors,
    saccr_config: CompiledSaccrConfig,
) -> npt.NDArray[np.float64]:
    """Per-netting-set SA-CCR add-on, aligned to ``netting_set_uids``."""
    sf = add_on_factors.factor[
        searchsorted_or_raise(
            universe=add_on_factors.subclass_uix,
            raw=trade_subclass_uix,
            ctx="saccr add-on: trade subclass(es)",
        )
    ]
    hedging_set = saccr_config.subclass_hedging_set[
        searchsorted_or_raise(
            universe=saccr_config.subclass_uix,
            raw=trade_subclass_uix,
            ctx="saccr add-on: subclass->hedging-set",
        )
    ]

    # D_jk over (netting set, hedging set, subclass)
    key3 = np.stack(
        [
            trade_netting_set_id,
            hedging_set.astype(np.int64),
            trade_subclass_uix.astype(np.int64),
        ],
        axis=1,
    )
    uniq3, inv3 = np.unique(key3, axis=0, return_inverse=True)
    inv3 = inv3.ravel()
    d_jk = np.bincount(inv3, weights=trade_effective_notional, minlength=uniq3.shape[0])
    sf_jk = add_on_factors.factor[
        searchsorted_or_raise(
            universe=add_on_factors.subclass_uix,
            raw=uniq3[:, 2].astype(np.int32),
            ctx="saccr add-on: group subclass->SF",
        )
    ]
    add_on_jk = sf_jk * d_jk

    # hedging-set correlation over (netting set, hedging set)
    key2 = uniq3[:, :2]
    uniq2, inv2 = np.unique(key2, axis=0, return_inverse=True)
    inv2 = inv2.ravel()
    sum_a = np.bincount(inv2, weights=add_on_jk, minlength=uniq2.shape[0])
    sum_a_sq = np.bincount(
        inv2, weights=add_on_jk * add_on_jk, minlength=uniq2.shape[0]
    )
    rho = saccr_config.hedging_set_rho[
        searchsorted_or_raise(
            universe=saccr_config.hedging_set_uix,
            raw=uniq2[:, 1].astype(np.int32),
            ctx="saccr add-on: hedging-set->rho",
        )
    ]
    add_on_hs = np.sqrt((rho * sum_a) ** 2 + (1.0 - rho * rho) * sum_a_sq)

    # sum hedging sets up to the netting set
    ns_of_hs = uniq2[:, 0]
    add_on_by_ns = np.zeros(netting_set_uids.shape[0], dtype=np.float64)
    pos = searchsorted_or_raise(
        universe=netting_set_uids,
        raw=ns_of_hs,
        ctx="saccr add-on: hedging-set netting set",
    )
    np.add.at(add_on_by_ns, pos, add_on_hs)
    return np.ascontiguousarray(add_on_by_ns)
