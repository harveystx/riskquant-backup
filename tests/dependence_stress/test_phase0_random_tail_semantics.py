# tests/dependence_stress/test_phase0_random_tail_semantics.py
"""
Phase 0 §5.4 — the property actually justified for tail_order="random":

  * tail MEMBERSHIP is preserved (the tail block holds the same set of
    observations as comonotonic mode);
  * random mode only changes severity matching INSIDE the selected tail;
  * ES over the selected tail is INVARIANT to intra-tail permutation;
  * VaR is NOT invariant and is NOT a calibrated target (only a bound relation
    holds: random tail boundary <= comonotonic tail boundary).

No unjustified monotonic VaR relationship is asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.permutations import (
    reconstruct_stressed,
    transform_empirical_matrix,
)
from riskquant.dependence_stress.tails import LowerTailAlignment

from phase0_support import heterogeneous_matrix  # type: ignore[import-not-found]


def _stressed(data, transformer, rng=None):
    r = transform_empirical_matrix(data, transformer, rng, audit=False)
    return reconstruct_stressed(data, r.permutations)


@pytest.mark.parametrize("k,s,q", [(6, 400, 0.05), (4, 200, 0.10), (3, 101, 0.05)])
def test_random_preserves_tail_membership_per_row(k, s, q):
    data = heterogeneous_matrix(k, s)
    n_tail = max(1, int(np.ceil(q * s)))
    como = _stressed(data, LowerTailAlignment(tail_probability=q, tail_order="comonotonic"))
    rnd = _stressed(
        data,
        LowerTailAlignment(tail_probability=q, tail_order="random"),
        np.random.default_rng(1),
    )
    for row in range(k):
        assert np.array_equal(
            np.sort(como[row, :n_tail]), np.sort(rnd[row, :n_tail])
        ), "random mode must keep the SAME tail observations, only re-paired"


@pytest.mark.parametrize("k,s,q", [(6, 400, 0.05), (4, 200, 0.10)])
def test_random_leaves_es_over_tail_invariant(k, s, q):
    data = heterogeneous_matrix(k, s)
    n_tail = max(1, int(np.ceil(q * s)))
    como = _stressed(data, LowerTailAlignment(tail_probability=q, tail_order="comonotonic")).sum(0)
    seeds = [1, 7, 42, 100]
    como_tail_mean = np.sort(como)[:n_tail].mean()
    for seed in seeds:
        rnd = _stressed(
            data,
            LowerTailAlignment(tail_probability=q, tail_order="random"),
            np.random.default_rng(seed),
        ).sum(0)
        rnd_tail_mean = np.sort(rnd)[:n_tail].mean()
        assert np.isclose(rnd_tail_mean, como_tail_mean, rtol=0.0, atol=1e-9), (
            "mean of the n_tail smallest portfolio values (tail ES) is invariant"
        )


def test_random_var_boundary_not_calibrated_and_bounded_by_comonotonic():
    """
    The comonotonic boundary (Sum_k X_{k,(n_tail-1)}) is the LARGEST tail-block
    portfolio value. Any random intra-tail pairing produces a boundary <= it, and
    can be strictly smaller — so VaR is neither invariant nor a calibrated target.
    """
    data = heterogeneous_matrix(6, 400)
    q = 0.05
    n_tail = max(1, int(np.ceil(q * data.shape[1])))
    como = _stressed(data, LowerTailAlignment(tail_probability=q, tail_order="comonotonic")).sum(0)
    como_boundary = np.sort(como)[n_tail - 1]  # == Sum_k X_{k,(n_tail-1)}
    strictly_smaller = 0
    for seed in range(25):
        rnd = _stressed(
            data,
            LowerTailAlignment(tail_probability=q, tail_order="random"),
            np.random.default_rng(seed),
        ).sum(0)
        rnd_boundary = np.sort(rnd)[n_tail - 1]
        assert rnd_boundary <= como_boundary + 1e-9  # bound always holds
        if rnd_boundary < como_boundary - 1e-9:
            strictly_smaller += 1
    assert strictly_smaller > 0, "random VaR boundary must vary below the comonotonic one"
