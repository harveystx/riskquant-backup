# src/riskquant/credit/aggregate/loss_metrics.py
"""
Portfolio-loss risk metrics.

VaR and ES are computed by the shared per-row tail kernel
(`riskquant.scenario_metrics`), which operates on the PER-SCENARIO loss vector(s) — a
``(S,)`` row, or a ``(G, S)`` stack of group rows — never the full per-obligor
``(n, S)`` matrix. That ``(S,)`` total is the quantity a batched engine can
accumulate online, so the metric layer never assumes a materialised loss matrix.

Tail-convention bridge: the kernel reports the LOWER tail (worst = most negative,
the P&L convention). Credit loss is the opposite (worst = largest). The single
``_loss_tail_var_es`` adapter negates on the way in and out, so everything
public here is loss-oriented: larger loss is worse, VaR/ES are upper-tail.
``scenario_metrics``'s VaR is the order statistic at ``floor((1 - alpha) * S)`` and ES
the mean beyond it.
"""

from __future__ import annotations

__all__ = [
    "expected_loss",
    "loss_var_es",
    "value_at_risk",
    "expected_shortfall",
    "tail_var_es_by_group",
    "loss_var_es_contribution",
    "loss_quantiles",
    "PortfolioLossDistribution",
]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.scenario_metrics.api.contribution import (
    compute_var_es_contribution,
)
from riskquant.scenario_metrics.api.var_es import compute_var_es

_F64_1D = npt.NDArray[np.float64]
_F64_2D = npt.NDArray[np.float64]


def _loss_tail_var_es(
    *,
    loss_by_group: _F64_2D,
    alpha: float,
) -> tuple[_F64_1D, _F64_1D]:
    """Per-row upper-tail loss VaR/ES via the lower-tail kernel.

    Negate so the largest losses become the most-negative values the kernel
    treats as its tail, then negate the results back into loss space.

    Parameters
    ----------
    loss_by_group
        Shape (G, S) stack of group loss rows. G groups, S scenarios.
    alpha
        Confidence level, e.g. 0.99.

    Returns
    -------
    tuple[_F64_1D, _F64_1D]
        VaR and ES for each group. Each shape (G,).
    """
    var_es_result = compute_var_es(-loss_by_group, confidence_level=alpha)
    return -var_es_result.var, -var_es_result.es


def expected_loss(portfolio_loss: _F64_1D) -> float:
    """Mean portfolio loss across scenarios (EL)."""
    return float(portfolio_loss.mean())


def loss_var_es(portfolio_loss: _F64_1D, alpha: float) -> tuple[float, float]:
    """Upper-tail VaR and ES of a single ``(S,)`` portfolio loss vector."""
    var, es = _loss_tail_var_es(
        loss_by_group=portfolio_loss.reshape(1, -1), alpha=alpha
    )
    return float(var[0]), float(es[0])


def value_at_risk(portfolio_loss: _F64_1D, alpha: float) -> float:
    """Value-at-Risk at confidence ``alpha`` (upper-tail loss quantile)."""
    return loss_var_es(portfolio_loss, alpha)[0]


def expected_shortfall(portfolio_loss: _F64_1D, alpha: float) -> float:
    """Expected shortfall at ``alpha`` (mean loss beyond VaR)."""
    return loss_var_es(portfolio_loss, alpha)[1]


def tail_var_es_by_group(
    loss_by_group: _F64_2D, alpha: float
) -> tuple[_F64_1D, _F64_1D]:
    """Per-group upper-tail VaR/ES for a ``(G, S)`` stack of group loss rows.

    One parallel kernel pass over all groups — the basis for standalone
    per-segment (e.g. per-model) tail risk.
    """
    return _loss_tail_var_es(loss_by_group=loss_by_group, alpha=alpha)


def loss_var_es_contribution(
    component_loss: _F64_2D,
    portfolio_loss: _F64_1D,
    alpha: float,
) -> tuple[_F64_1D, _F64_1D]:
    """Per-component upper-tail loss VaR/ES contributions (Euler allocation).

    ``component_loss`` is ``(K, S)`` per-component loss; ``portfolio_loss`` is its
    ``(S,)`` column sum. Returns ``(var_contribution, es_contribution)``, each
    ``(K,)`` in loss space, which foot to the headline loss VaR/ES exactly.

    Delegates to the shared ``scenario_metrics`` tail-allocation kernel — the
    single home for VaR/ES contribution across market and credit risk — via the
    same lower-tail negation bridge ``loss_var_es`` uses. Negate losses into P&L
    on the way in, negate the contributions back into loss space on the way out.
    """
    result = compute_var_es_contribution(
        component_matrix=np.ascontiguousarray(-component_loss),
        total_row=np.ascontiguousarray(-portfolio_loss),
        confidence_level=alpha,
    )
    return (
        np.ascontiguousarray(-result.var_contribution),
        np.ascontiguousarray(-result.es_contribution),
    )


def loss_quantiles(portfolio_loss: _F64_1D, alphas: _F64_1D) -> _F64_1D:
    """General loss quantiles at each level in ``alphas`` (interpolated)."""
    return np.ascontiguousarray(np.quantile(portfolio_loss, alphas), dtype=np.float64)


@dataclass(frozen=True, slots=True, kw_only=True)
class PortfolioLossDistribution:
    """
    The portfolio loss distribution — a per-scenario total-loss vector with the
    standard risk metrics as accessors. The natural carrier for the ``(S,)``
    quantity the metric layer needs, whether collapsed from a resolved JTD loss
    or accumulated online.
    """

    portfolio_loss: _F64_1D  # (S,) total loss per scenario

    @property
    def n_scenarios(self) -> int:
        return int(self.portfolio_loss.shape[0])

    def expected_loss(self) -> float:
        return expected_loss(self.portfolio_loss)

    def var_es(self, alpha: float) -> tuple[float, float]:
        return loss_var_es(self.portfolio_loss, alpha)

    def value_at_risk(self, alpha: float) -> float:
        return value_at_risk(self.portfolio_loss, alpha)

    def expected_shortfall(self, alpha: float) -> float:
        return expected_shortfall(self.portfolio_loss, alpha)

    def quantiles(self, alphas: _F64_1D) -> _F64_1D:
        return loss_quantiles(self.portfolio_loss, alphas)
