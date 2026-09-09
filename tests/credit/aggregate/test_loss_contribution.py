# tests/credit/aggregate/test_loss_contribution.py
"""
Regression tests for loss-space VaR/ES contribution and the credit_var
attribution footing property (RQ-AUD-009).

Confirms the loss-oriented bridge in ``credit.aggregate.loss_metrics`` and the
``solutions.credit_var`` attribution both foot to the headline loss ES for any
(alpha, S), including the non-integral tail-mass case.
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.credit.aggregate.loss_metrics import (
    loss_var_es,
    loss_var_es_contribution,
)
from riskquant.solutions.credit_var._attribution import obligor_tail_attribution


@pytest.mark.parametrize(
    "N,S,alpha,desc",
    [
        (5, 250, 0.99, "non-integral tail_mass=2.5"),
        (5, 100, 0.99, "integral tail_mass=1"),
        (6, 1000, 0.975, "integral 25"),
        (4, 300, 0.999, "full_tail_count==0"),
        (8, 251, 0.99, "non-integral 2.51"),
        (3, 1, 0.99, "S==1 boundary"),
    ],
)
def test_loss_contribution_foots(N, S, alpha, desc):
    rng = np.random.default_rng(N * 97 + S)
    # positive loss matrix with a heavy tail
    obligor_loss = np.ascontiguousarray(rng.exponential(scale=10.0, size=(N, S)))
    portfolio_loss = np.ascontiguousarray(obligor_loss.sum(axis=0))

    var_tot, es_tot = loss_var_es(portfolio_loss, alpha)
    var_c, es_c = loss_var_es_contribution(obligor_loss, portfolio_loss, alpha)

    assert abs(es_c.sum() - es_tot) < 1e-9, f"{desc}: ES foots"
    assert abs(var_c.sum() - var_tot) < 1e-9, f"{desc}: VaR foots"


@pytest.mark.parametrize(
    "N,S,alpha",
    [(5, 250, 0.99), (5, 100, 0.99), (4, 300, 0.999), (8, 251, 0.99)],
)
def test_attribution_es_contribution_foots_to_es_total(N, S, alpha):
    rng = np.random.default_rng(1234 + S)
    obligor_loss = np.ascontiguousarray(rng.exponential(scale=8.0, size=(N, S)))
    portfolio_losses = np.ascontiguousarray(obligor_loss.sum(axis=0))

    attr = obligor_tail_attribution(
        obligor_loss=obligor_loss,
        portfolio_losses=portfolio_losses,
        confidence_level=alpha,
    )
    assert abs(attr.es_contribution.sum() - attr.es_total) < 1e-9


def test_loss_contribution_matches_manual_fractional_tail():
    """Cross-check the loss ES contribution against an explicit fractional-tail
    reference implementation on a small deterministic example."""
    # 4 scenarios, 2 obligors, alpha=0.5 -> tail_mass = 0.5*4 = 2.0 (integral)
    # Use alpha=0.6 -> tail_mass = 0.4*4 = 1.6 (non-integral): 1 full + 0.6 boundary
    obligor_loss = np.array(
        [[10.0, 2.0, 8.0, 4.0],
         [ 1.0, 3.0, 9.0, 5.0]],
        dtype=np.float64,
    )
    portfolio = obligor_loss.sum(axis=0)  # [11, 5, 17, 9]
    alpha = 0.6
    S = 4
    tail_mass = (1.0 - alpha) * S  # 1.6
    full = int(np.floor(tail_mass))  # 1
    frac = tail_mass - full  # 0.6
    # worst loss scenario is index 2 (17); next worst is index 0 (11)
    order = np.argsort(-portfolio)  # descending: [2, 0, 3, 1]
    full_idx = order[:full]  # [2]
    boundary = order[full]   # 0
    ref_es = (obligor_loss[:, full_idx].sum(axis=1) + frac * obligor_loss[:, boundary]) / tail_mass

    _, es_c = loss_var_es_contribution(
        np.ascontiguousarray(obligor_loss), np.ascontiguousarray(portfolio), alpha
    )
    np.testing.assert_allclose(es_c, ref_es, atol=1e-12)
