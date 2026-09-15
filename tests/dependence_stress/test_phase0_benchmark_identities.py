# tests/dependence_stress/test_phase0_benchmark_identities.py
"""
Phase 0 §5 — benchmark mathematical identities, using the repository VaR/ES
convention. These lock behaviour that must survive the Phase 4 rename.

  5.1  VaR(comonotonic)        == sum standalone VaR
  5.2  ES(comonotonic)         == sum standalone ES
  5.3  VaR(lower-tail comono)  == sum standalone VaR   for every confidence with alpha <= q

An optional cross-check against the real scenario_metrics kernel runs when it is
importable; it is skipped otherwise, so the identities stay executable in a
minimal environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.permutations import (
    reconstruct_stressed,
    transform_empirical_matrix,
)
from riskquant.dependence_stress.tails import Comonotonic, LowerTailAlignment

from phase0_support import (  # type: ignore[import-not-found]
    heterogeneous_matrix,
    standalone_sum_var_es,
    var_es_lower_tail,
)

# heterogeneous + mixed-sign samples at several shapes
_SAMPLES = [
    (k, s) for k in (2, 3, 6) for s in (100, 250, 1000)
]
_CONFIDENCES = (0.90, 0.95, 0.975, 0.99)


def _portfolio(data, transformer, rng=None):
    result = transform_empirical_matrix(data, transformer, rng, audit=False)
    return reconstruct_stressed(data, result.permutations).sum(axis=0), result


@pytest.mark.parametrize("k,s", _SAMPLES)
@pytest.mark.parametrize("cl", _CONFIDENCES)
def test_comonotonic_var_equals_sum_standalone(k, s, cl):
    data = heterogeneous_matrix(k, s)
    ptf, _ = _portfolio(data, Comonotonic())
    como_var, _ = var_es_lower_tail(ptf, cl)
    ss_var, _ = standalone_sum_var_es(data, cl)
    assert np.isclose(como_var, ss_var, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("k,s", _SAMPLES)
@pytest.mark.parametrize("cl", _CONFIDENCES)
def test_comonotonic_es_equals_sum_standalone(k, s, cl):
    data = heterogeneous_matrix(k, s)
    ptf, _ = _portfolio(data, Comonotonic())
    _, como_es = var_es_lower_tail(ptf, cl)
    _, ss_es = standalone_sum_var_es(data, cl)
    assert np.isclose(como_es, ss_es, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("k,s", _SAMPLES)
@pytest.mark.parametrize("q", (0.05, 0.10))
@pytest.mark.parametrize("cl", (0.99, 0.975, 0.95))  # alpha in {0.01,0.025,0.05} <= q
def test_lower_tail_comonotone_reproduces_undiversified_var(k, s, q, cl):
    alpha = 1.0 - cl
    if alpha > q:
        pytest.skip("identity only claimed for alpha <= q")
    data = heterogeneous_matrix(k, s)
    ptf, _ = _portfolio(data, LowerTailAlignment(tail_probability=q, tail_order="comonotonic"))
    stress_var, _ = var_es_lower_tail(ptf, cl)
    ss_var, _ = standalone_sum_var_es(data, cl)
    assert np.isclose(stress_var, ss_var, rtol=0.0, atol=1e-9), (
        "lower-tail comonotone deletes diversification at every alpha <= q"
    )


def test_cross_check_against_real_scenario_metrics_if_available():
    """Bind the convention replica to the real kernel when importable."""
    try:
        from riskquant.scenario_metrics.api.var_es import compute_var_es  # type: ignore
    except Exception:
        pytest.skip("riskquant.scenario_metrics not importable in this environment")
    data = heterogeneous_matrix(4, 500)
    ptf = data.sum(axis=0)
    for cl in _CONFIDENCES:
        ref = compute_var_es(np.ascontiguousarray(ptf.reshape(1, -1)), confidence_level=cl)
        ref_var = float(ref.var[0])
        ref_es = float(ref.es[0])
        var, es = var_es_lower_tail(ptf, cl)
        assert np.isclose(var, ref_var, atol=1e-9)
        assert np.isclose(es, ref_es, atol=1e-9)
