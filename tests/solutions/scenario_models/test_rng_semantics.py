# tests/solutions/scenario_models/test_rng_semantics.py
"""
Adversarial RNG / seed-semantics contract for the stochastic scenario models.

Pins the guarantees a production risk run depends on, so a future change cannot
silently weaken them:

  * ``seed`` is bounded at the SOLUTION boundary (non-negative), not deep inside
    NumPy's bit generator during iteration;
  * ``seed=int`` gives restart-identical, cross-instance-identical,
    batch-invariant, construction-order-independent streams;
  * ``seed=None`` re-randomizes per ``iter_batches`` call (no reproducibility);
  * results do not depend on NumPy's global random state.

Bitwise equality is asserted because the contract permits it: batching and
seeding change no arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)

_K = 3
_T = 60
_S = 12


def _matrix() -> RiskFactorMatrix:
    rng = np.random.default_rng(0)
    vals = np.asfortranarray(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (_T, _K)), axis=0))
    )
    return RiskFactorMatrix(
        dates=np.arange(_T, dtype=np.int32),
        factor_ids=np.arange(_K, dtype=np.int32),
        values=vals,
    )


def _parametric_data(crisis_vol_multipliers=None) -> ParametricRiskData:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (_K, _K))
    cov = a @ a.T / _K + np.eye(_K) * 0.1
    inv_sd = 1.0 / np.sqrt(np.diag(cov))
    chol = np.ascontiguousarray(np.linalg.cholesky(cov * np.outer(inv_sd, inv_sd)))
    return ParametricRiskData(
        factor_ids=np.arange(_K, dtype=np.int32),
        cholesky_matrix=chol,
        volatilities=np.ascontiguousarray(np.full(_K, 0.02)),
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(_K, SHIFT_LOG_RETURN_ID, dtype=np.int8),
        crisis_vol_multipliers=crisis_vol_multipliers,
    )


def _cfg(model: str, seed):
    if model == "sbb":
        return {
            "model": "sbb",
            "horizon": 1,
            "total_scenarios": _S,
            "expected_block_size": 5.0,
            "seed": seed,
        }
    if model == "filtered_bootstrap":
        return {
            "model": "filtered_bootstrap",
            "horizon": 1,
            "total_scenarios": _S,
            "seed": seed,
        }
    return {
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": _S,
        "distribution": {"kind": "student_t", "df": 5.0},
        "seed": seed,
    }


_STOCHASTIC = ["sbb", "filtered_bootstrap", "parametric"]


def _build(model: str, seed):
    data = _parametric_data() if model == "parametric" else _matrix()
    return build_scenario_generator(
        parse_scenario_model_config(_cfg(model, seed)), data=data
    )


def _drive(gen, batch_size: int = 4) -> np.ndarray:
    chunks = []
    for b in gen.iter_batches(
        chart_ids=np.array([SHIFT_LOG_RETURN_ID] * _K, np.int8),
        driver_rf_ids=np.arange(_K, dtype=np.int32),
        batch_size=batch_size,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


# ---- seed contract at the solution boundary ----------------------------------


@pytest.mark.parametrize("model", _STOCHASTIC)
@pytest.mark.parametrize("bad_seed", [-1, -42])
def test_negative_seed_rejected_at_boundary(model, bad_seed) -> None:
    """Without a boundary bound this fails inside numpy's bit generator during
    iteration, long after the config and generator were accepted."""
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(model, bad_seed))


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_zero_and_none_seed_remain_valid(model) -> None:
    assert parse_scenario_model_config(_cfg(model, 0)).seed == 0
    assert parse_scenario_model_config(_cfg(model, None)).seed is None


@pytest.mark.parametrize("model", _STOCHASTIC)
@pytest.mark.parametrize("bad", [True, 1.0, "7"], ids=["bool", "float", "str"])
def test_seed_is_strictly_integral(model, bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(model, bad))


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_large_seed_accepted_and_usable(model) -> None:
    """Seeds beyond 64 bits are legitimate SeedSequence entropy."""
    gen = _build(model, 2**128 + 7)
    assert _drive(gen).shape[0] > 0


# ---- determinism semantics ---------------------------------------------------


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_restart_is_identical(model) -> None:
    gen = _build(model, 123)
    assert np.array_equal(_drive(gen), _drive(gen))


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_co_seeded_instances_are_identical(model) -> None:
    """Current contract: same seed == same stream (NOT independent streams)."""
    assert np.array_equal(_drive(_build(model, 123)), _drive(_build(model, 123)))


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_different_seed_differs(model) -> None:
    assert not np.array_equal(_drive(_build(model, 123)), _drive(_build(model, 456)))


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_seed_none_rerandomizes_per_iteration(model) -> None:
    """seed=None carries no reproducibility claim, not even within one instance."""
    gen = _build(model, None)
    assert not np.array_equal(_drive(gen), _drive(gen))


@pytest.mark.parametrize("model", _STOCHASTIC)
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 7, 11, _S - 1, _S, _S + 50])
def test_batch_invariance_adversarial_sizes(model, batch_size) -> None:
    """D029 across primes, non-divisors, exact and over-sized batches."""
    ref = _drive(_build(model, 123), 1)
    assert np.array_equal(ref, _drive(_build(model, 123), batch_size))


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_construction_order_independence(model) -> None:
    g1, g2 = _build(model, 123), _build(model, 456)
    a1, a2 = _drive(g1), _drive(g2)
    h2, h1 = _build(model, 456), _build(model, 123)
    b2, b1 = _drive(h2), _drive(h1)
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2)


@pytest.mark.parametrize("model", _STOCHASTIC)
def test_no_dependence_on_numpy_global_state(model) -> None:
    np.random.seed(1)  # noqa: NPY002 - intentionally exercises legacy global state
    first = _drive(_build(model, 7))
    np.random.seed(999)  # noqa: NPY002 - intentionally exercises legacy global state
    for _ in range(50):
        np.random.random()  # noqa: NPY002
    assert np.array_equal(first, _drive(_build(model, 7)))


@pytest.mark.parametrize(
    "distribution",
    [
        {"kind": "standard_normal"},
        {"kind": "laplace"},
        {"kind": "student_t", "df": 5.0},
        {"kind": "variance_gamma", "nu": 0.5},
        {
            "kind": "regime_switching_gmm",
            "prob_crisis": 0.3,
            "crisis_vol_multipliers": 3.0,
        },
        {
            "kind": "regime_switching_gmm",
            "prob_crisis": 0.3,
            "crisis_vol_multipliers": "per_factor",
        },
    ],
    ids=["normal", "laplace", "student_t", "vg", "regime_scalar", "regime_vector"],
)
def test_parametric_every_distribution_is_batch_invariant(distribution) -> None:
    per_factor = distribution.get("crisis_vol_multipliers") == "per_factor"
    data = _parametric_data(np.array([2.0, 3.0, 4.0]) if per_factor else None)
    raw = {
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": _S,
        "distribution": distribution,
        "seed": 99,
    }

    def run(bs):
        return _drive(
            build_scenario_generator(parse_scenario_model_config(raw), data=data), bs
        )

    ref = run(1)
    for bs in (2, 3, 5, 7, _S, _S + 50):
        assert np.array_equal(ref, run(bs))
