# tests/scenario_metrics/test_contribution.py
"""
Regression tests for the shared Euler VaR/ES contribution kernel (RQ-AUD-009).

The contribution allocation lives in ``riskquant.scenario_metrics`` so that
market-risk and credit-risk callers share one tail-allocation kernel. The
defining property is FOOTING: the per-component contributions must sum to the
headline VaR/ES computed by ``compute_var_es`` over the same total row, for any
(alpha, S) — including the non-integral tail-mass case that the previous
floor-count implementation got wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.scenario_metrics.api.contribution import compute_var_es_contribution
from riskquant.scenario_metrics.api.var_es import compute_var_es


def _headline(total: np.ndarray, alpha: float) -> tuple[float, float]:
    res = compute_var_es(np.ascontiguousarray(total.reshape(1, -1)), alpha)
    return float(res.var[0]), float(res.es[0])


@pytest.mark.parametrize(
    "K,S,alpha",
    [
        (5, 250, 0.99),    # non-integral tail_mass = 2.5 (the RQ-AUD-009 case)
        (5, 100, 0.99),    # integral tail_mass = 1
        (8, 1000, 0.975),  # integral 25
        (8, 1000, 0.999),  # integral 1
        (4, 300, 0.999),   # non-integral -> full_tail_count == 0
        (3, 1, 0.99),      # S == 1 boundary
        (6, 500, 0.95),    # integral 25
        (10, 251, 0.99),   # non-integral 2.51
    ],
)
def test_contribution_foots_to_headline(K, S, alpha):
    rng = np.random.default_rng(K * 1000 + S)
    comp = np.ascontiguousarray(
        rng.standard_normal((K, S)) * rng.uniform(0.5, 3.0, (K, 1))
    )
    total = np.ascontiguousarray(comp.sum(axis=0))

    res = compute_var_es_contribution(comp, total, alpha)
    var_tot, es_tot = _headline(total, alpha)

    assert abs(res.es_contribution.sum() - es_tot) < 1e-9, (
        f"ES column does not foot: {res.es_contribution.sum()} vs {es_tot}"
    )
    assert abs(res.var_contribution.sum() - var_tot) < 1e-9, (
        f"VaR column does not foot: {res.var_contribution.sum()} vs {var_tot}"
    )


def test_nonintegral_tail_mass_is_the_regression_case():
    """Explicit demonstration that the floor-count method would NOT foot but the
    shared fractional kernel does (alpha=0.99, S=250, tail_mass=2.5)."""
    rng = np.random.default_rng(42)
    comp = np.ascontiguousarray(rng.exponential(1.0, (5, 250)) - 1.0)
    total = np.ascontiguousarray(comp.sum(axis=0))
    _, es_tot = _headline(total, 0.99)

    # Shared kernel: foots.
    res = compute_var_es_contribution(comp, total, 0.99)
    assert abs(res.es_contribution.sum() - es_tot) < 1e-9

    # Old floor-count method: mean over worst floor(2.5)=2 scenarios. Demonstrably
    # does NOT foot (different denominator + dropped fractional boundary term).
    tail_count = max(1, int((1.0 - 0.99) * 250))  # == 2
    tail_idx = np.argpartition(total, 250 - tail_count)[250 - tail_count :]
    old = comp[:, tail_idx].mean(axis=1).sum()
    assert abs(old - es_tot) > 1e-6, "expected the old method to mis-foot here"


def test_contribution_result_is_read_only():
    rng = np.random.default_rng(7)
    comp = np.ascontiguousarray(rng.standard_normal((4, 64)))
    total = np.ascontiguousarray(comp.sum(axis=0))
    res = compute_var_es_contribution(comp, total, 0.95)
    with pytest.raises(ValueError):
        res.es_contribution[0] = 1.0
    with pytest.raises(ValueError):
        res.var_contribution[0] = 1.0


def test_single_component_contribution_equals_headline():
    rng = np.random.default_rng(11)
    comp = np.ascontiguousarray(rng.standard_normal((1, 500)))
    total = np.ascontiguousarray(comp[0].copy())
    res = compute_var_es_contribution(comp, total, 0.99)
    var_tot, es_tot = _headline(total, 0.99)
    assert abs(float(res.es_contribution[0]) - es_tot) < 1e-9
    assert abs(float(res.var_contribution[0]) - var_tot) < 1e-9


# --- validation ---

def test_rejects_non_contiguous_component_matrix():
    comp = np.asfortranarray(np.random.default_rng(1).standard_normal((4, 32)))
    total = np.ascontiguousarray(comp.sum(axis=0))
    with pytest.raises(ValueError):
        compute_var_es_contribution(comp, total, 0.99)


def test_rejects_scenario_axis_mismatch():
    comp = np.ascontiguousarray(np.random.default_rng(2).standard_normal((4, 32)))
    total = np.ascontiguousarray(np.zeros(31))  # wrong S
    with pytest.raises(ValueError):
        compute_var_es_contribution(comp, total, 0.99)


def test_rejects_non_finite_total():
    comp = np.ascontiguousarray(np.random.default_rng(3).standard_normal((4, 32)))
    total = np.ascontiguousarray(comp.sum(axis=0))
    total[0] = np.nan
    with pytest.raises(ValueError):
        compute_var_es_contribution(comp, total, 0.99)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_rejects_confidence_outside_open_interval(bad_alpha):
    comp = np.ascontiguousarray(np.random.default_rng(4).standard_normal((4, 32)))
    total = np.ascontiguousarray(comp.sum(axis=0))
    with pytest.raises(ValueError):
        compute_var_es_contribution(comp, total, bad_alpha)
