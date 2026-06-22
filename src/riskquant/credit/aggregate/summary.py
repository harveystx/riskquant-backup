# src/riskquant/credit/aggregate/summary.py
"""
Resolve a JTD loss into the credit package's headline outputs.

`build_portfolio_loss_distribution` collapses the per-obligor ``(n, S)`` loss to
the per-scenario portfolio total. `summarize_jtd_loss` produces the headline risk
summary (EL, plus VaR/ES at a set of confidence levels) and — when the compiled
portfolio is supplied — the additive expected-loss contribution of each JTD
model. `model_tail_var_es` exposes per-model STANDALONE tail risk via one
parallel pass of the shared tail kernel over a per-model loss stack.

EL contributions are additive (sum to portfolio EL). Per-model VaR/ES are
standalone (a model's own tail) and do NOT sum to portfolio VaR/ES — additive
tail allocation (Euler) is a separate, deliberately deferred concern.
"""

from __future__ import annotations

__all__ = [
    "build_portfolio_loss_distribution",
    "summarize_jtd_loss",
    "summarize_streaming_jtd_loss",
    "JtdRiskSummary",
    "model_tail_var_es",
    "streaming_model_tail_var_es",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.credit.aggregate.loss_metrics import (
    PortfolioLossDistribution,
    expected_loss,
    loss_var_es,
    tail_var_es_by_group,
)
from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.jtd.results import JtdLossResult, StreamingJtdResult

_DEFAULT_ALPHAS: tuple[float, ...] = (0.9500, 0.9750, 0.9900, 0.9990)


def build_portfolio_loss_distribution(
    result: JtdLossResult,
) -> PortfolioLossDistribution:
    """Collapse the per-obligor ``(n, S)`` loss to the ``(S,)`` portfolio total."""
    portfolio_loss = result.loss.sum(axis=0)
    return PortfolioLossDistribution(portfolio_loss=portfolio_loss)


@dataclass(frozen=True, slots=True, kw_only=True)
class JtdRiskSummary:
    """
    Headline portfolio risk numbers resolved from a JTD loss.

    ``var`` / ``es`` are keyed by confidence level. ``expected_loss_by_model`` is
    populated only when the compiled portfolio is provided; its values sum to
    ``expected_loss``.
    """

    n_scenarios: int
    expected_loss: float
    var: dict[float, float]
    es: dict[float, float]
    expected_loss_by_model: dict[str, float]


def _expected_loss_by_model(
    result: JtdLossResult, jtd_portfolio: CompiledJtdPortfolio
) -> dict[str, float]:
    """Per-model EL contribution via the portfolio's model bands (additive)."""
    by_model: dict[str, float] = {}
    for i in range(jtd_portfolio.total_blocks):
        block = jtd_portfolio.blocks[i]
        start = int(jtd_portfolio.block_offsets[i])
        end = start + block.num_obligors
        name = JtdModel(block.JTD_MODEL_ID).name
        contribution = float(result.loss[start:end, :].sum(axis=0).mean())
        by_model[name] = by_model.get(name, 0.0) + contribution
    return by_model


def _model_loss_stack(
    result: JtdLossResult, jtd_portfolio: CompiledJtdPortfolio
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Per-model portfolio loss rows: model names + a ``(n_models, S)`` stack."""
    names: list[str] = []
    rows: list[npt.NDArray[np.float64]] = []
    for i in range(jtd_portfolio.total_blocks):
        block = jtd_portfolio.blocks[i]
        start = int(jtd_portfolio.block_offsets[i])
        end = start + block.num_obligors
        names.append(JtdModel(block.JTD_MODEL_ID).name)
        rows.append(result.loss[start:end, :].sum(axis=0))
    return names, np.ascontiguousarray(np.stack(rows))


def summarize_jtd_loss(
    *,
    result: JtdLossResult,
    jtd_portfolio: CompiledJtdPortfolio | None = None,
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS,
) -> JtdRiskSummary:
    """
    Headline risk summary for a resolved JTD loss.

    Parameters
    ----------
    result
        Resolved per-obligor JTD loss.
    jtd_portfolio
        Optional compiled portfolio; when supplied, per-model expected-loss
        contributions are computed from its model bands.
    alphas
        Confidence levels for VaR and ES.
    """
    portfolio_loss = result.loss.sum(axis=0)
    var: dict[float, float] = {}
    es: dict[float, float] = {}
    for a in alphas:
        var[a], es[a] = loss_var_es(portfolio_loss, a)
    by_model = (
        _expected_loss_by_model(result, jtd_portfolio)
        if jtd_portfolio is not None
        else {}
    )
    return JtdRiskSummary(
        n_scenarios=int(portfolio_loss.shape[0]),
        expected_loss=expected_loss(portfolio_loss),
        var=var,
        es=es,
        expected_loss_by_model=by_model,
    )


def summarize_streaming_jtd_loss(
    *,
    result: StreamingJtdResult,
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS,
) -> JtdRiskSummary:
    """
    Headline risk summary for a streamed JTD loss.

    Operates on the accumulated ``(S,)`` portfolio total directly (no per-obligor
    matrix). Per-model expected-loss contributions are filled when the streaming
    run collected the ``(n_models, S)`` breakdown (`collect_by_model=True`); they
    remain additive (sum to the portfolio EL).
    """
    portfolio_loss = result.portfolio_loss
    var: dict[float, float] = {}
    es: dict[float, float] = {}
    for a in alphas:
        var[a], es[a] = loss_var_es(portfolio_loss, a)

    by_model: dict[str, float] = {}
    if result.model_loss is not None:
        model_el = result.model_loss.mean(axis=1)
        for row, model_id in enumerate(result.model_order):
            name = JtdModel(model_id).name
            by_model[name] = by_model.get(name, 0.0) + float(model_el[row])

    return JtdRiskSummary(
        n_scenarios=int(portfolio_loss.shape[0]),
        expected_loss=expected_loss(portfolio_loss),
        var=var,
        es=es,
        expected_loss_by_model=by_model,
    )


def model_tail_var_es(
    *,
    result: JtdLossResult,
    jtd_portfolio: CompiledJtdPortfolio,
    alpha: float,
) -> dict[str, tuple[float, float]]:
    """Standalone per-model ``(VaR, ES)`` at ``alpha`` — one parallel kernel pass.

    These are each model's own tail risk; they do not sum to the portfolio's.
    """
    names, stack = _model_loss_stack(result, jtd_portfolio)
    var, es = tail_var_es_by_group(stack, alpha)
    return {name: (float(var[i]), float(es[i])) for i, name in enumerate(names)}


def streaming_model_tail_var_es(
    *,
    result: StreamingJtdResult,
    alpha: float,
) -> dict[str, tuple[float, float]]:
    """Standalone per-model ``(VaR, ES)`` at ``alpha`` from a streamed run.

    Requires the run to have collected the per-model breakdown
    (`collect_by_model=True`); one parallel kernel pass over the ``(n_models, S)``
    stack. As with the non-streaming variant these are each model's own tail and
    do not sum to the portfolio's.
    """
    if result.model_loss is None:
        raise ValueError(
            "streaming_model_tail_var_es requires a run with collect_by_model=True "
            "(result.model_loss is None)."
        )
    var, es = tail_var_es_by_group(result.model_loss, alpha)
    return {
        JtdModel(model_id).name: (float(var[i]), float(es[i]))
        for i, model_id in enumerate(result.model_order)
    }
