# src/riskquant/credit/jtd/streaming.py
"""
Streaming (batched) JTD execution.

`run_jtd` materialises the full per-obligor ``(N, S)`` loss; at institutional
obligor x scenario counts that — and the full ``(F, S)`` factor matrix feeding it
— is the memory ceiling. `run_jtd_streaming` removes both: it drives the loop off
the factor generator's scenario batches and, per batch, runs the SAME evaluator
dispatch into one reused ``(N, batch)`` scratch buffer, then collapses that batch
to its per-scenario totals. The only state that persists across batches is the
``(S,)`` portfolio loss vector (plus, optionally, an ``(n_models, S)`` per-model
breakdown) — never ``(N, S)`` or ``(F, S)``.

That ``(S,)`` total is exactly what `riskquant.credit.aggregate` was built to
consume, so the streaming path produces a `StreamingJtdResult` that the
aggregation layer turns into the loss distribution and risk metrics.

The per-scenario *portfolio* vector is the largest object kept (S floats), which
is cheap even at very large S; what streaming gives up is per-obligor attribution
(e.g. top contributors in the VaR scenario), since obligor-level loss is never
retained. Use `run_jtd` when that attribution is needed and the portfolio fits.
"""

from __future__ import annotations

__all__ = [
    "constant_ead_provider",
    "run_jtd_streaming",
    "run_jtd_streaming_from_generator",
]

from collections.abc import Callable, Iterable
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from riskquant.credit._scenario_coverage import (
    assert_scenario_coverage_complete,
    new_scenario_coverage,
    record_scenario_span,
)
from riskquant.credit.factors.simulate.base import (
    CreditFactorBatch,
    CreditFactorGenerator,
)
from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.evaluators.base import JtdEvaluator
from riskquant.credit.jtd.registry import evaluator_table
from riskquant.credit.jtd.results import StreamingJtdResult

JTD_EVALUATORS: Final[tuple[JtdEvaluator[Any], ...]] = evaluator_table()

# (scenario_offset, batch_size) -> (n_obligors, batch_size) EAD, rows aligned to
# jtd_portfolio.obligor_uids. The caller owns EAD generation so it, too, streams
# per batch rather than materialising (N, S).
EadProvider = Callable[[int, int], npt.NDArray[np.float64]]


def run_jtd_streaming(
    *,
    jtd_portfolio: CompiledJtdPortfolio,
    factor_batches: Iterable[CreditFactorBatch],
    ead_provider: EadProvider,
    total_scenarios: int,
    collect_by_model: bool = False,
) -> StreamingJtdResult:
    """
    Resolve JTD loss over scenario batches, accumulating the portfolio total.

    Parameters
    ----------
    jtd_portfolio
        Compiled obligor blocks; band order / `obligor_uids` defines the EAD row
        order the `ead_provider` must match.
    factor_batches
        Scenario batches from a credit-factor generator
        (`generator.iter_batches(batch_size=...)`). Each batch carries
        ``(batch, F)`` scenario-major latent draws and its scenario offset; the
        batches must partition ``[0, total_scenarios)`` — enforced, not assumed
        (the ``(S,)`` accumulator is `np.empty`, so an uncovered scenario would
        otherwise carry uninitialised memory into VaR/ES).
    ead_provider
        ``(scenario_offset, batch_size) -> (n_obligors, batch_size)`` EAD for that
        scenario slice, rows aligned to `jtd_portfolio.obligor_uids`.
    total_scenarios
        Total scenario count S; sizes the persistent accumulator(s).
    collect_by_model
        If True, also accumulate an ``(n_models, S)`` per-model breakdown (rows
        sum to the portfolio total). Costs one ``(S,)`` row per distinct model.

    Raises
    ------
    ValueError
        If the batches do not partition ``[0, total_scenarios)`` exactly, or the
        EAD provider returns the wrong shape for a batch.
    """
    n_obligors = jtd_portfolio.total_obligors
    blocks = jtd_portfolio.blocks
    offsets = jtd_portfolio.block_offsets
    n_blocks = jtd_portfolio.total_blocks
    evaluators_table = JTD_EVALUATORS

    # Persistent accumulators: (S,) portfolio total, written once per scenario as
    # the batches partition the scenario axis.
    portfolio_loss: npt.NDArray[np.float64] = np.empty(total_scenarios, np.float64)

    model_loss: npt.NDArray[np.float64] | None = None
    model_order: tuple[int, ...] = ()
    block_model_row: list[int] = []
    if collect_by_model:
        # One row per block (one block per model in band order).
        model_order = tuple(int(blocks[i].JTD_MODEL_ID) for i in range(n_blocks))
        block_model_row = list(range(n_blocks))
        model_loss = np.empty((n_blocks, total_scenarios), np.float64)

    scratch: npt.NDArray[np.float64] | None = None  # (N, batch) reused across batches

    ctx = "run_jtd_streaming"
    coverage = new_scenario_coverage(total_scenarios, ctx=ctx)

    for batch in factor_batches:
        offset = int(batch.scenario_offset)
        batch_size = int(batch.latent_factors.shape[0])
        record_scenario_span(coverage, offset=offset, count=batch_size, ctx=ctx)
        end_s = offset + batch_size

        # (batch, F) scenario-major -> (F, batch) factor-major for the kernels.
        factors = CreditFactorMatrix(
            latent_factors=np.ascontiguousarray(batch.latent_factors.T)
        )
        ead = ead_provider(offset, batch_size)
        if ead.shape != (n_obligors, batch_size):
            raise ValueError(
                f"ead_provider returned {ead.shape}, expected "
                f"{(n_obligors, batch_size)} at scenario offset {offset}."
            )
        # Kernels are typed for C-contiguous EAD (float64[:, ::1]); a provider
        # returning a strided view (e.g. a column slice) would otherwise fail
        # kernel dispatch. No-op when the provider already returns C-contiguous.
        ead = np.ascontiguousarray(ead)

        if scratch is None or scratch.shape[1] != batch_size:
            scratch = np.empty((n_obligors, batch_size), dtype=np.float64, order="C")

        for i in range(n_blocks):
            block = blocks[i]
            start = int(offsets[i])
            stop = start + block.num_obligors
            evaluators_table[block.JTD_MODEL_ID](
                block=block,
                ead=ead[start:stop, :],
                credit_factors=factors,
                out=scratch[start:stop, :],
            )

        # Collapse this batch to per-scenario totals into the persistent (S,).
        portfolio_loss[offset:end_s] = scratch.sum(axis=0)
        if model_loss is not None:
            for i in range(n_blocks):
                start = int(offsets[i])
                stop = start + blocks[i].num_obligors
                model_loss[block_model_row[i], offset:end_s] = scratch[
                    start:stop, :
                ].sum(axis=0)

    assert_scenario_coverage_complete(coverage, ctx=ctx)

    return StreamingJtdResult(
        portfolio_loss=portfolio_loss,
        total_obligors=n_obligors,
        n_scenarios=total_scenarios,
        model_loss=model_loss,
        model_order=model_order,
    )


def run_jtd_streaming_from_generator(
    *,
    jtd_portfolio: CompiledJtdPortfolio,
    factor_generator: CreditFactorGenerator,
    ead_provider: EadProvider,
    batch_size: int,
    collect_by_model: bool = False,
) -> StreamingJtdResult:
    """Drive `run_jtd_streaming` straight off a credit-factor generator.

    Derives the scenario batches (`generator.iter_batches(batch_size=...)`) and the
    total scenario count from the generator, so the caller supplies only the
    portfolio, an EAD provider, and a batch size — no hand-wiring of the batch
    stream or `total_scenarios`.
    """
    return run_jtd_streaming(
        jtd_portfolio=jtd_portfolio,
        factor_batches=factor_generator.iter_batches(batch_size=batch_size),
        ead_provider=ead_provider,
        total_scenarios=factor_generator.total_scenarios,
        collect_by_model=collect_by_model,
    )


def constant_ead_provider(
    ead_by_obligor: npt.NDArray[np.float64],
) -> EadProvider:
    """Build an `EadProvider` for scenario-INDEPENDENT EAD.

    `ead_by_obligor` is the ``(N,)`` per-obligor exposure (rows aligned to
    `jtd_portfolio.obligor_uids`); the returned provider broadcasts it across each
    batch's scenarios as a read-only ``(N, batch)`` view — zero per-batch
    allocation. Use this when EAD does not vary by scenario; supply your own
    provider when it does (e.g. exposure simulated jointly with the factors).
    """
    base = np.ascontiguousarray(ead_by_obligor, dtype=np.float64).reshape(-1, 1)

    def _provider(scenario_offset: int, batch_size: int) -> npt.NDArray[np.float64]:
        return np.broadcast_to(base, (base.shape[0], batch_size))

    return _provider
