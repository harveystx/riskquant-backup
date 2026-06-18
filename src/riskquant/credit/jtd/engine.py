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
]

from typing import Any, Final

import numpy as np
import numpy.typing as npt

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
