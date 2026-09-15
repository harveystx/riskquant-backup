# tests/dependence_stress/test_phase0_validation_and_result_contract.py
"""
Phase 0 §10 (validation boundary) + §11 (result-object contract).

Locks fail-fast behaviour at public entry points and the observable result
contract. Two NEWLY SURFACED defects (not in the accepted audit) are captured as
xfail(strict=True), documenting the target contract without asserting an accepted
production bug; strict flips them to failures once fixed, forcing marker removal:

  * n_realizations silently coerced (3.7 -> 3, True -> 1) instead of rejected;
  * free-function entry points raise incidental AttributeError, not explicit
    TypeError, for a malformed transformer.

Phase 0 does NOT change the permutation dtype: current contract (int64) is
locked here; Phase 2 will flip it to int32 and update these assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
from phase0_support import (  # type: ignore[import-not-found]
    equicorrelation,
    heterogeneous_matrix,
)

from riskquant.dependence_stress.benchmarks import Comonotonic, LowerTailComonotone
from riskquant.dependence_stress.copulas import GaussianCopula
from riskquant.dependence_stress.engine import DependenceStressEngine
from riskquant.dependence_stress.permutations import (
    iter_transform_empirical_matrix,
    transform_empirical_matrix,
)

_DATA = heterogeneous_matrix(4, 100)


# ---- §10 boundary ------------------------------------------------------------
@pytest.mark.parametrize(
    "bad,match",
    [
        (np.full((4, 100), np.nan), "must be finite"),
        (np.full((4, 100), np.inf), "must be finite"),
        (np.zeros((1, 100)), "at least 2 variables"),
        (np.zeros((4, 1)), "at least 2 observations"),
        (np.zeros((4,)), "2-dimensional"),
    ],
)
def test_observation_matrix_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        transform_empirical_matrix(bad, Comonotonic())


@pytest.mark.parametrize("bad_rng", [42, "rng", np.random.RandomState(0)])
def test_stochastic_rejects_malformed_rng(bad_rng):
    with pytest.raises(TypeError):
        transform_empirical_matrix(
            _DATA, GaussianCopula(latent_correlation=equicorrelation(4)), bad_rng
        )


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.5, 2.0, np.nan, np.inf])
def test_invalid_tail_probability(bad_q):
    with pytest.raises(ValueError, match="probability"):
        LowerTailComonotone(tail_probability=bad_q)


def test_invalid_correlation_not_pd():
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    with pytest.raises(ValueError, match="positive definite"):
        GaussianCopula(latent_correlation=bad)


def test_negative_realization_count_raises():
    with pytest.raises(ValueError, match="non-negative"):
        list(iter_transform_empirical_matrix(_DATA, Comonotonic(), n_realizations=-1))


def test_malformed_transformer_rejected_by_engine():
    class NotATransformer:
        pass

    engine = DependenceStressEngine(data=_DATA)
    with pytest.raises(TypeError, match="DependenceTransformer"):
        engine.apply_transformer(NotATransformer())


@pytest.mark.parametrize(
    "bad_count, exc",
    [
        (3.7, TypeError),
        (True, TypeError),
        ("3", TypeError),
        (-1, ValueError),
    ],
)
def test_invalid_realization_count_is_rejected(bad_count, exc):
    """Phase 5: n_realizations fails fast; no silent int() coercion."""
    with pytest.raises(exc):
        list(
            iter_transform_empirical_matrix(
                _DATA, Comonotonic(), n_realizations=bad_count
            )
        )


@pytest.mark.parametrize("good_count", [0, 1, 3, np.int64(3)])
def test_valid_realization_count_is_accepted(good_count):
    results = list(
        iter_transform_empirical_matrix(_DATA, Comonotonic(), n_realizations=good_count)
    )
    assert len(results) == int(good_count)


def test_free_functions_reject_malformed_transformer_explicitly():
    """Phase 5: both free-function entries reject with TypeError, not AttributeError."""

    class NotATransformer:
        pass

    with pytest.raises(TypeError, match="DependenceTransformer"):
        transform_empirical_matrix(_DATA, NotATransformer())

    with pytest.raises(TypeError, match="DependenceTransformer"):
        list(
            iter_transform_empirical_matrix(_DATA, NotATransformer(), n_realizations=1)
        )


# ---- §11 result contract -----------------------------------------------------
def test_result_shapes_dtypes_layout_and_readonly():
    result = transform_empirical_matrix(_DATA, Comonotonic())
    k, s = _DATA.shape
    assert result.permutations.shape == (k, s)
    assert result.portfolio_pnl.shape == (s,)
    # Phase 2: permutations migrated to canonical estate int32
    assert result.permutations.dtype == np.int32
    assert result.portfolio_pnl.dtype == np.float64
    assert result.permutations.flags.c_contiguous
    assert result.portfolio_pnl.flags.c_contiguous
    assert not result.permutations.flags.writeable
    assert not result.portfolio_pnl.flags.writeable


def test_result_write_raises():
    result = transform_empirical_matrix(_DATA, Comonotonic())
    with pytest.raises(ValueError):
        result.permutations[0, 0] = 0
    with pytest.raises(ValueError):
        result.portfolio_pnl[0] = 0.0


def test_engine_isolates_from_caller_mutation():
    data = heterogeneous_matrix(4, 100).copy()
    engine = DependenceStressEngine(data=data)
    before = engine.apply_comonotonic().portfolio_pnl.copy()
    data[:] = 0.0  # mutate the caller's array after construction
    after = engine.apply_comonotonic().portfolio_pnl
    assert np.array_equal(before, after), "engine must own a decoupled copy"


def test_result_equality_is_identity_based():
    a = transform_empirical_matrix(_DATA, Comonotonic())
    b = transform_empirical_matrix(_DATA, Comonotonic())
    assert a == a
    assert a != b
    assert isinstance(a == b, bool)
