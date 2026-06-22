# src/riskquant/credit/exposure/obligor_ead/align.py
"""
Obligor-EAD -> JTD portfolio alignment.

The obligor-grain analogue of `jtd.bridge.aggregate_ead_to_obligor`: that bridge
reindexes netting-set EAD onto a JTD portfolio's band order; this reindexes EAD
that is already at obligor grain. Same join tool (`searchsorted_or_raise`), same
target order (`jtd_portfolio.obligor_uids`, which is band order — an internal
compaction artifact, not the caller's order).

The semantics here are STRICT in the opposite direction to the bridge: every
obligor the JTD portfolio models must have an EAD row (a modelled obligor with no
exposure is a gap in the inputs, not something to silently zero). Obligors
present in the EAD but absent from the portfolio are simply unused.
"""

from __future__ import annotations

__all__ = [
    "align_obligor_ead_to_portfolio",
]

import numpy as np
import numpy.typing as npt

from riskquant._arrays import searchsorted_or_raise
from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad


def align_obligor_ead_to_portfolio(
    *,
    obligor_ead: CompiledObligorEad,
    obligor_uids_target: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """
    Reindex obligor EAD onto a target obligor-uid order (band order).

    Parameters
    ----------
    obligor_ead
        Validated, uid-sorted obligor EAD; `ead` is ``(N,)`` or ``(N, S)``.
    obligor_uids_target
        Target obligor-uid order — typically `jtd_portfolio.obligor_uids`.

    Returns
    -------
    The EAD reindexed onto `obligor_uids_target`: ``(n_target,)`` for scalar EAD,
    ``(n_target, S)`` for scenario-varying. C-contiguous, ready to feed (after any
    broadcast) the bare-array engine core.

    Raises
    ------
    ValueError
        If any target obligor has no EAD row (`searchsorted_or_raise`).
    """
    pos = searchsorted_or_raise(
        universe=obligor_ead.uid,
        raw=obligor_uids_target,
        ctx="align_obligor_ead_to_portfolio: obligor uid(s)",
    )
    return np.ascontiguousarray(obligor_ead.ead[pos])
