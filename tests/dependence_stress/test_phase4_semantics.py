# tests/dependence_stress/test_phase4_semantics.py

"""
Phase 4 — the benchmark vs dependence-stress distinction is observable.

These lock the renamed public vocabulary and the role separation at the level
of observable contracts, not docstring text or implementation details:

  * the extremal benchmarks are deterministic in comonotonic mode and declare
    they consume no randomness;
  * the copula dependence stresses declare they DO consume randomness;
  * the engine exposes copula stresses and benchmarks as distinct methods;
  * the old ``*Alignment`` / ``apply_*_tail_alignment`` names are gone.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress import benchmarks
from riskquant.dependence_stress.benchmarks import (
    Comonotonic,
    DualTailComonotone,
    LowerTailComonotone,
    UpperTailComonotone,
)
from riskquant.dependence_stress.copulas import GaussianCopula, TCopula
from riskquant.dependence_stress.engine import DependenceStressEngine

_BENCHMARKS_COMONOTONIC = [
    Comonotonic(),
    LowerTailComonotone(tail_probability=0.1),
    UpperTailComonotone(tail_probability=0.1),
    DualTailComonotone(lower_probability=0.1, upper_probability=0.1),
]


def _data(k: int = 5, s: int = 120) -> np.ndarray:
    return np.ascontiguousarray(np.random.default_rng(0).standard_normal((k, s)))


@pytest.mark.parametrize("benchmark", _BENCHMARKS_COMONOTONIC)
def test_benchmarks_are_deterministic_in_comonotonic_mode(benchmark) -> None:
    """A benchmark in comonotonic mode consumes no randomness and repeats."""
    data = _data()
    assert benchmark.requires_rng is False
    first = benchmark.build_permutation(data)
    second = benchmark.build_permutation(data, np.random.default_rng(999))
    assert np.array_equal(first, second)  # RNG cannot change a benchmark


def test_copula_stresses_declare_they_consume_randomness() -> None:
    """The dependence stresses are stochastic; benchmarks are not."""
    k = 5
    corr = np.eye(k)
    assert GaussianCopula(latent_correlation=corr).requires_rng is True
    assert TCopula(latent_correlation=corr, degrees_of_freedom=4.0).requires_rng is True


def test_engine_exposes_stresses_and_benchmarks_as_distinct_methods() -> None:
    surface = {m for m in dir(DependenceStressEngine) if m.startswith("apply_")}
    dependence_stresses = {"apply_gaussian_copula", "apply_t_copula"}
    benchmark_methods = {
        "apply_comonotonic",
        "apply_lower_tail_comonotone",
        "apply_upper_tail_comonotone",
        "apply_dual_tail_comonotone",
    }
    assert dependence_stresses <= surface
    assert benchmark_methods <= surface
    assert dependence_stresses.isdisjoint(benchmark_methods)


def test_old_alignment_vocabulary_is_gone() -> None:
    for stale in (
        "LowerTailAlignment",
        "UpperTailAlignment",
        "DualTailAlignment",
    ):
        assert not hasattr(benchmarks, stale), stale
    for stale in (
        "apply_lower_tail_alignment",
        "apply_upper_tail_alignment",
        "apply_dual_tail_alignment",
    ):
        assert not hasattr(DependenceStressEngine, stale), stale


def test_renamed_engine_benchmark_methods_run() -> None:
    engine = DependenceStressEngine(data=_data())
    assert engine.apply_lower_tail_comonotone(0.05).permutations.dtype == np.int32
    assert engine.apply_upper_tail_comonotone(0.05).permutations.dtype == np.int32
    assert engine.apply_dual_tail_comonotone(0.05, 0.05).permutations.dtype == np.int32
