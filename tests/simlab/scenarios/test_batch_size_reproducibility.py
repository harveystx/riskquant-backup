# tests/simlab/scenarios/test_batch_size_reproducibility.py
"""
R5 — F6 / D021 batch-size reproducibility.

Contract: for a fixed model + data + parameters + seed + total_scenarios, the
concatenated scenario stream is identical regardless of batch_size. batch_size
is an orchestration/memory parameter, never a stochastic-model parameter.

The oracle here is invariance itself: batches are concatenated in yield order
and compared with EXACT equality across batch sizes 1, 7, and full. Covers every
stochastic generator, including all ParametricGenerator distribution kinds
(the auxiliary-variate draw was the F6 defect).
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.bootstrap_fhs import FilteredBootstrapGenerator
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.simlab.scenarios.parametric import ParametricGenerator

_TOTAL = 20
_NREQ = 2


def _matrix(k=2, t=16):
    a = np.asfortranarray(
        np.array(
            [[100.0 + i + 3 * j for i in range(t)] for j in range(k)], np.float64
        ).T
    )
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.arange(k, dtype=np.int32),
        values=a,
    )


def _drive(gen, batch_size):
    chunks = []
    for b in gen.iter_batches(
        chart_ids=np.full(_NREQ, SHIFT_LOG_RETURN_ID, np.int8),
        driver_rf_ids=np.arange(_NREQ, dtype=np.int32),
        batch_size=batch_size,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


_PARAM_KW = {
    "standard_normal": {},
    "student_t": {"student_t_df": 5.0},
    "laplace": {},
    "variance_gamma": {"vg_nu": 0.5},
    "regime_switching_gmm": {"prob_crisis": 0.1, "crisis_vol_multipliers": 3.0},
}


def _make(spec, seed):
    """spec is a generator key; returns a fresh instance seeded with `seed`."""
    m = _matrix()
    if spec == "sbb":
        return StationaryBlockBootstrapGenerator(
            matrix=m,
            total_scenarios=_TOTAL,
            expected_block_size=2.0,
            horizon=2,
            seed=seed,
        )
    if spec == "filtboot":
        return FilteredBootstrapGenerator(
            matrix=m, total_scenarios=_TOTAL, horizon=2, seed=seed
        )
    if spec.startswith("param:"):
        kind = spec.split(":", 1)[1]
        return ParametricGenerator(
            available_factor_ids=np.arange(2, dtype=np.int32),
            cholesky_matrix=np.eye(2, dtype=np.float64),
            volatilities=np.array([0.1, 0.2], np.float64),
            total_scenarios=_TOTAL,
            distribution_kind=kind,
            seed=seed,
            **_PARAM_KW[kind],
        )
    raise AssertionError(spec)


_GENERATORS = [
    "sbb",
    "filtboot",
    *[f"param:{k}" for k in _PARAM_KW],
]


@pytest.mark.parametrize("spec", _GENERATORS)
def test_batch_size_invariance(spec) -> None:
    """Concatenated stream is identical for batch_size in {1, 7, full}."""
    r1 = _drive(_make(spec, 11), 1)
    r7 = _drive(_make(spec, 11), 7)
    rfull = _drive(_make(spec, 11), _TOTAL)
    assert r1.shape == (_TOTAL, _NREQ)
    assert np.array_equal(r1, r7)
    assert np.array_equal(r1, rfull)


@pytest.mark.parametrize("spec", _GENERATORS)
def test_repeated_runs_same_seed_identical(spec) -> None:
    """iter_batches rebuilds the RNG from the seed each call, so a second call
    on the SAME instance reproduces the stream exactly (idempotent)."""
    gen = _make(spec, 7)
    first = _drive(gen, 5)
    second = _drive(gen, 5)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("spec", _GENERATORS)
def test_same_seed_separate_instances_identical(spec) -> None:
    """Two instances with the SAME seed are intentionally IDENTICAL, not
    independent. Independent streams require DIFFERENT seeds (documented: do
    not rely on co-seeded independence)."""
    a = _drive(_make(spec, 99), 4)
    b = _drive(_make(spec, 99), 4)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("spec", _GENERATORS)
def test_different_seeds_differ(spec) -> None:
    """Different seeds give different streams (independence via seed choice)."""
    a = _drive(_make(spec, 1), 4)
    b = _drive(_make(spec, 2), 4)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("spec", _GENERATORS)
def test_deterministic_ordering_prefix_stable(spec) -> None:
    """Scenario ordering is deterministic: the first m scenarios of a full run
    equal the first m of a smaller batched run (order is offset-stable)."""
    full = _drive(_make(spec, 5), _TOTAL)
    batched = _drive(_make(spec, 5), 3)
    assert np.array_equal(full[:7], batched[:7])


@pytest.mark.parametrize("spec", _GENERATORS)
def test_seed_none_runs_without_reproducibility_claim(spec) -> None:
    """seed=None must still produce a valid, correctly shaped stream. No
    reproducibility is claimed: two None-seeded instances draw fresh entropy
    and (with overwhelming probability) differ."""
    a = _drive(_make(spec, None), 6)
    assert a.shape == (_TOTAL, _NREQ)
    assert a.dtype == np.float32
    assert np.all(np.isfinite(a))
    b = _drive(_make(spec, None), 6)
    assert not np.array_equal(a, b)  # fresh entropy each construction
