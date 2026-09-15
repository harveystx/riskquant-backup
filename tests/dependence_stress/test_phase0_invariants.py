# tests/dependence_stress/test_phase0_invariants.py
"""
Phase 0 §4 — universal structural invariants across a shape/probability grid.

Locks, for every current transformer at overlap-SAFE parameters:
  4.1 permutation validity (each row is a permutation of 0..S-1)
  4.2 exact marginal preservation on the production path (audit=False)
  4.3 exact reconstruction (stressed == take_along_axis(data, perm))
  4.4 exact aggregation (portfolio_pnl == column sum of the reconstruction)

DualTail OVERLAP is deliberately excluded here; it is isolated in
test_phase0_dualtail_overlap_regression.py (accepted P0 defect).
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.permutations import (
    reconstruct_stressed,
    transform_empirical_matrix,
)

from phase0_support import (  # type: ignore[import-not-found]
    K_GRID,
    S_GRID,
    SEED,
    heterogeneous_matrix,
    safe_transformers,
)


def _cases():
    for k in K_GRID:
        for s in S_GRID:
            data = heterogeneous_matrix(k, s)
            for tid, transformer, needs_rng in safe_transformers(k):
                yield pytest.param(data, transformer, needs_rng, id=f"K{k}_S{s}_{tid}")


@pytest.mark.parametrize("data,transformer,needs_rng", list(_cases()))
def test_permutation_is_valid(data, transformer, needs_rng):
    rng = np.random.default_rng(SEED) if needs_rng else None
    result = transform_empirical_matrix(data, transformer, rng, audit=False)
    perm = result.permutations
    k, s = data.shape
    assert perm.shape == (k, s)
    expected = np.broadcast_to(np.arange(s), (k, s))
    assert np.array_equal(np.sort(perm, axis=1), expected), (
        "each row must be a permutation of 0..S-1"
    )


@pytest.mark.parametrize("data,transformer,needs_rng", list(_cases()))
def test_marginals_preserved_on_production_path(data, transformer, needs_rng):
    rng = np.random.default_rng(SEED) if needs_rng else None
    result = transform_empirical_matrix(data, transformer, rng, audit=False)
    stressed = reconstruct_stressed(data, result.permutations)
    assert np.array_equal(np.sort(stressed, axis=1), np.sort(data, axis=1)), (
        "row-wise empirical marginal must be preserved exactly, without audit"
    )


@pytest.mark.parametrize("data,transformer,needs_rng", list(_cases()))
def test_exact_reconstruction(data, transformer, needs_rng):
    rng = np.random.default_rng(SEED) if needs_rng else None
    result = transform_empirical_matrix(data, transformer, rng, audit=False)
    stressed = reconstruct_stressed(data, result.permutations)
    assert np.array_equal(
        stressed, np.take_along_axis(data, result.permutations, axis=1)
    )


@pytest.mark.parametrize("data,transformer,needs_rng", list(_cases()))
def test_exact_aggregation(data, transformer, needs_rng):
    rng = np.random.default_rng(SEED) if needs_rng else None
    result = transform_empirical_matrix(data, transformer, rng, audit=False)
    stressed = reconstruct_stressed(data, result.permutations)
    # direct production aggregate must equal reconstruction-and-sum, bitwise
    assert np.array_equal(result.portfolio_pnl, stressed.sum(axis=0))
    # explicit definition cross-check
    k, s = data.shape
    manual = np.zeros(s, dtype=np.float64)
    for row in range(k):
        manual += data[row][result.permutations[row]]
    assert np.array_equal(result.portfolio_pnl, manual)
