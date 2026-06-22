# src/riskquant/credit/jtd/engine.py
"""
JTD engine: model-id -> evaluator dispatch and the loss-resolution loop.

The dispatch table is now owned by the registry: `JTD_EVALUATORS` is just
`registry.evaluator_table()`. Adding a model no longer touches this file — the
model's evaluator is wired the moment its `JtdModelSpec` is registered, and the
registry validates completeness/contiguity at import.

`run_jtd` walks the compiled obligor blocks in band order, dispatches each on its
`JTD_MODEL_ID`, and writes into contiguous bands of one pre-allocated
(n_obligors x n_scenarios) output.
"""

from __future__ import annotations

__all__ = [
    "run_jtd",
    "run_jtd_for_obligor_ead",
]

from typing import Any, Final

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.obligor_ead.align import (
    align_obligor_ead_to_portfolio,
)
from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad
from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.evaluators.base import JtdEvaluator
from riskquant.credit.jtd.registry import evaluator_table
from riskquant.credit.jtd.results import JtdLossResult

JTD_EVALUATORS: Final[tuple[JtdEvaluator[Any], ...]] = evaluator_table()


def run_jtd(
    *,
    jtd_portfolio: CompiledJtdPortfolio,
    ead_by_obligor: npt.NDArray[np.float64],
    credit_factors: CreditFactorMatrix,
) -> JtdLossResult:
    """
    Resolve jump-to-default loss across all JTD models.

    Parameters
    ----------
    jtd_portfolio
        Compiled obligor blocks; `obligor_uids` defines the output row order.
    ead_by_obligor
        (n_obligors, S) EAD, rows aligned to `jtd_portfolio.obligor_uids` —
        the output of `aggregate_ead_to_obligor`.
    credit_factors
        Latent-factor matrix; `num_scenarios` must match `ead_by_obligor`.
        Ignored by the forced-default model, which is scenario-independent.
    """
    n_obligors = jtd_portfolio.total_obligors
    n_scenarios = credit_factors.num_scenarios

    if ead_by_obligor.shape != (n_obligors, n_scenarios):
        raise ValueError(
            f"ead_by_obligor shape {ead_by_obligor.shape} != expected "
            f"{(n_obligors, n_scenarios)}."
        )

    loss: npt.NDArray[np.float64] = np.empty(
        (n_obligors, n_scenarios), dtype=np.float64, order="C"
    )

    blocks = jtd_portfolio.blocks
    offsets = jtd_portfolio.block_offsets
    evaluators_table = JTD_EVALUATORS

    for i in range(jtd_portfolio.total_blocks):
        block = blocks[i]
        start = int(offsets[i])
        end = start + block.num_obligors

        evaluator = evaluators_table[block.JTD_MODEL_ID]
        evaluator(
            block=block,
            ead=ead_by_obligor[start:end, :],
            credit_factors=credit_factors,
            out=loss[start:end, :],
        )

    return JtdLossResult(uid=jtd_portfolio.obligor_uids, loss=loss)


def run_jtd_for_obligor_ead(
    *,
    jtd_portfolio: CompiledJtdPortfolio,
    obligor_ead: CompiledObligorEad,
    credit_factors: CreditFactorMatrix,
) -> JtdLossResult:
    """
    uid-safe entry point: resolve JTD loss from EAD keyed by obligor uid.

    The safe outer tier over `run_jtd`. `run_jtd` takes a bare ``(N, S)`` array
    that must already be in the portfolio's band order — correct for the in-
    pipeline caller (the exposure stage feeds it aligned), but a silent-miscalc
    footgun for an external caller who has EAD keyed by uid and no reason to know
    band order. This entry closes that gap: it reindexes `obligor_ead` onto
    `jtd_portfolio.obligor_uids` by uid with a fail-fast join (every modelled
    obligor must have an EAD row), then runs the core. Scalar ``(N,)`` EAD is
    broadcast across the scenario axis; scenario-varying ``(N, S)`` EAD must match
    `credit_factors.num_scenarios`.
    """
    aligned = align_obligor_ead_to_portfolio(
        obligor_ead=obligor_ead,
        obligor_uids_target=jtd_portfolio.obligor_uids,
    )
    n_scenarios = credit_factors.num_scenarios

    if aligned.ndim == 1:
        # Scalar per-obligor EAD -> materialise (N, S) for the bare-array core.
        ead_by_obligor = np.ascontiguousarray(
            np.broadcast_to(aligned[:, None], (aligned.shape[0], n_scenarios))
        )
    else:
        if aligned.shape[1] != n_scenarios:
            raise ValueError(
                f"run_jtd_for_obligor_ead: scenario-varying EAD has "
                f"{aligned.shape[1]} scenarios, but credit_factors has "
                f"{n_scenarios}."
            )
        ead_by_obligor = aligned

    return run_jtd(
        jtd_portfolio=jtd_portfolio,
        ead_by_obligor=ead_by_obligor,
        credit_factors=credit_factors,
    )
