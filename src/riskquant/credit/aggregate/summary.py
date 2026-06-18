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
    "JtdRiskSummary",
    "model_tail_var_es",
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
from riskquant.credit.jtd.results import JtdLossResult

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
