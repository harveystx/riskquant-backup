# src/riskquant/credit/jtd/engine.py
"""
JTD engine: model-id -> evaluator dispatch and the loss-resolution loop.

`JTD_EVALUATORS` is the credit-layer analogue of `EAD_EVALUATORS`: a frozen
tuple indexed by `JtdModel` id. `run_jtd` walks the compiled obligor blocks in
band order, dispatches each on its `JTD_MODEL_ID`, and writes into contiguous
bands of one pre-allocated (n_obligors x n_scenarios) output. The import-time
guard rejects any model that ships without an evaluator.
"""

from __future__ import annotations

from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.evaluators.base import JtdEvaluator
from riskquant.credit.jtd.evaluators.copula import (
    evaluate_gaussian_copula_jtd,
    evaluate_t_copula_jtd,
)
from riskquant.credit.jtd.evaluators.deterministic import (
    evaluate_deterministic_jtd,
)
from riskquant.credit.jtd.models import (
    COPULA_SF_GAUSSIAN_ID,
    COPULA_SF_T_ID,
    DETERMINISTIC_ID,
    NUM_JTD_MODELS,
    JtdModel,
)
from riskquant.credit.jtd.results import JtdLossResult

_JTD_EVALUATORS_MUT: list[JtdEvaluator[Any] | None] = [None] * NUM_JTD_MODELS


def _register(model_id: int, evaluator: JtdEvaluator[Any]) -> None:
    if not isinstance(evaluator, JtdEvaluator):
        raise TypeError(
            f"Evaluator for JTD model_id={model_id} does not satisfy the "
            f"JtdEvaluator protocol: {evaluator!r}"
        )
    _JTD_EVALUATORS_MUT[model_id] = evaluator


_register(COPULA_SF_GAUSSIAN_ID, evaluate_gaussian_copula_jtd)
_register(COPULA_SF_T_ID, evaluate_t_copula_jtd)
_register(DETERMINISTIC_ID, evaluate_deterministic_jtd)

JTD_EVALUATORS: Final[tuple[JtdEvaluator[Any], ...]] = cast(
    tuple[JtdEvaluator[Any], ...], tuple(_JTD_EVALUATORS_MUT)
)

_unregistered = [JtdModel(i).name for i, ev in enumerate(JTD_EVALUATORS) if ev is None]
if _unregistered:
    raise RuntimeError(
        f"No evaluator registered for JTD models: {_unregistered}. "
        f"Register them in credit/jtd/engine.py."
    )


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
        Ignored by the deterministic model, which is scenario-independent.
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
