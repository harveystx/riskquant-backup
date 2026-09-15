# tests/dependence_stress/test_phase5_result_construction.py

"""
Phase 5 — trusted no-copy result construction and hardened boundaries.

Locks:
  * the trusted ``_from_owned`` constructor keeps the passed buffers without a
    copy and freezes them read-only (internal test of the trusted path only);
  * the public constructor stays safe: it copies, isolates from caller mutation,
    canonicalises, and freezes;
  * both construction paths expose identical values;
  * the streaming result buffers are exclusively owned (no aliasing of the
    engine's shared RankRepresentation.sort_order), including the comonotonic
    benchmark whose permutation equals that order.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.benchmarks import Comonotonic, LowerTailComonotone
from riskquant.dependence_stress.copulas import GaussianCopula
from riskquant.dependence_stress.engine import DependenceStressEngine
from riskquant.dependence_stress.types import DependenceStressResult


def _data(k: int = 5, s: int = 200) -> np.ndarray:
    return np.ascontiguousarray(np.random.default_rng(0).standard_normal((k, s)))


# ---- A. no-copy identity (internal trusted constructor) ----------------------
def test_from_owned_keeps_buffers_without_copy():
    perm = np.empty((3, 4), dtype=np.int32, order="C")
    perm[:] = np.array([[0, 1, 2, 3]], dtype=np.int32)
    pnl = np.zeros(4, dtype=np.float64)
    result = DependenceStressResult._from_owned(perm, pnl)
    assert result.permutations is perm  # no copy
    assert result.portfolio_pnl is pnl  # no copy


def test_from_owned_freezes_read_only():
    perm = np.zeros((2, 3), dtype=np.int32, order="C")
    pnl = np.zeros(3, dtype=np.float64)
    result = DependenceStressResult._from_owned(perm, pnl)
    assert result.permutations.flags["WRITEABLE"] is False
    assert result.portfolio_pnl.flags["WRITEABLE"] is False
    with pytest.raises(ValueError):
        result.permutations[0, 0] = 1
    with pytest.raises(ValueError):
        result.portfolio_pnl[0] = 1.0


# ---- C. public constructor remains safe --------------------------------------
def test_public_constructor_copies_and_isolates():
    perm = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    pnl = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = DependenceStressResult(permutations=perm, portfolio_pnl=pnl)
    # a copy, not the same object
    assert result.permutations is not perm
    assert result.portfolio_pnl is not pnl
    # caller mutation cannot reach the result
    perm[0, 0] = 99
    pnl[0] = -99.0
    assert result.permutations[0, 0] == 0
    assert result.portfolio_pnl[0] == 1.0
    # read-only outputs
    assert result.permutations.flags["WRITEABLE"] is False
    assert result.portfolio_pnl.flags["WRITEABLE"] is False


def test_public_constructor_canonicalises_dtype():
    # int64 in -> int32 out; float32 in -> float64 out
    perm = np.array([[0, 1], [1, 0]], dtype=np.int64)
    pnl = np.array([1.0, 2.0], dtype=np.float32)
    result = DependenceStressResult(permutations=perm, portfolio_pnl=pnl)
    assert result.permutations.dtype == np.int32
    assert result.portfolio_pnl.dtype == np.float64


# ---- D. numerical identity: trusted vs public --------------------------------
def test_trusted_and_public_construction_expose_identical_values():
    perm = np.array([[0, 2, 1, 3], [3, 1, 2, 0]], dtype=np.int32)
    pnl = np.array([0.5, -1.0, 2.0, 3.5], dtype=np.float64)
    trusted = DependenceStressResult._from_owned(perm.copy(), pnl.copy())
    public = DependenceStressResult(permutations=perm, portfolio_pnl=pnl)
    assert np.array_equal(trusted.permutations, public.permutations)
    assert np.array_equal(trusted.portfolio_pnl, public.portfolio_pnl)
    assert trusted.permutations.dtype == public.permutations.dtype == np.int32
    assert trusted.portfolio_pnl.dtype == public.portfolio_pnl.dtype == np.float64


# ---- result buffers are exclusively owned (no sort_order aliasing) -----------
@pytest.mark.parametrize(
    "transformer, needs_rng",
    [
        (Comonotonic(), False),
        (LowerTailComonotone(tail_probability=0.1), False),
        (GaussianCopula(latent_correlation=np.eye(5)), True),
    ],
)
def test_streaming_result_permutations_do_not_alias_engine_sort_order(
    transformer, needs_rng
):
    engine = DependenceStressEngine(data=_data())
    sort_order = engine.rank_representation.sort_order
    rng = np.random.default_rng(1) if needs_rng else None
    result = engine._run(transformer, rng=rng, audit=False)
    # the comonotonic permutation EQUALS sort_order in value, but must be a
    # distinct, exclusively-owned buffer, never the shared read-only sort_order
    assert result.permutations is not sort_order
    assert not np.shares_memory(result.permutations, sort_order)
    assert result.permutations.dtype == np.int32
    assert result.permutations.flags["C_CONTIGUOUS"]
    assert result.permutations.flags["WRITEABLE"] is False


def test_two_comonotonic_results_have_independent_buffers():
    engine = DependenceStressEngine(data=_data())
    a = engine.apply_comonotonic()
    b = engine.apply_comonotonic()
    assert a.permutations is not b.permutations
    assert not np.shares_memory(a.permutations, b.permutations)
    assert np.array_equal(a.permutations, b.permutations)  # same values
