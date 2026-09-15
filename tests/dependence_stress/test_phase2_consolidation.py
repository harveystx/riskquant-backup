# tests/dependence_stress/test_phase2_consolidation.py
"""
Phase 2 — observation/rank consolidation + int32 index migration.

Locks the migration invariants:
  §18A canonical order equivalence: RankRepresentation.sort_order == stable argsort
  §18B transformer output equivalence: injected sort_order == internal argsort fallback
  §18C int32 result contract: permutations int32, C-contiguous, read-only
  §18D index safety: 0 <= perm < S < 2**31
  §18F input ownership: mutating caller input after construction cannot change results

These bind dependence_stress to riskquant.observations.compile as the canonical
observation/rank owner. The old-vs-new bitwise comparison against the pre-Phase-2
implementation is performed out-of-band in the migration report; here we lock the
equivalence between the two live code paths (injected vs fallback), which is the
property that must hold for every future change.
"""

from __future__ import annotations

import numpy as np
import pytest
from phase0_support import (  # type: ignore[import-not-found]
    SEED,
    heterogeneous_matrix,
    safe_transformers,
)

from riskquant.dependence_stress.engine import DependenceStressEngine
from riskquant.dependence_stress.permutations import (
    transform_empirical_matrix,
)
from riskquant.observations.compile import compile_ranks, compile_sample
from riskquant.observations.orientation import ObservationOrientation

_ORIENT = ObservationOrientation.VARIABLES_ON_ROWS


# ---- §18A canonical order equivalence ----------------------------------------
@pytest.mark.parametrize("k,s", [(2, 2), (3, 3), (4, 7), (6, 101), (8, 250)])
def test_sort_order_equals_stable_argsort(k, s):
    data = heterogeneous_matrix(k, s)
    sample = compile_sample(data, orientation=_ORIENT)
    ranks = compile_ranks(sample)
    assert ranks.sort_order.dtype == np.int32
    assert np.array_equal(
        ranks.sort_order,
        np.argsort(sample.observations, axis=1, kind="stable"),
    )


def test_sort_order_equivalence_under_ties_and_mixed_sign():
    data = np.array([
        [1.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        [-3.0, -3.0, 5.0, 5.0, -1.0, 0.0],
        [7.0, 7.0, 7.0, 7.0, 7.0, 8.0],
    ])
    sample = compile_sample(data, orientation=_ORIENT)
    ranks = compile_ranks(sample)
    assert np.array_equal(ranks.sort_order, np.argsort(data, axis=1, kind="stable"))


# ---- §18B injected sort_order == internal argsort fallback -------------------
@pytest.mark.parametrize("k", [2, 3, 8])
def test_injected_order_matches_fallback_for_every_transformer(k):
    data = heterogeneous_matrix(k, 200)
    engine = DependenceStressEngine(data=data)
    for tid, transformer, needs_rng in safe_transformers(k):
        rng_kw = (np.random.default_rng(SEED), np.random.default_rng(SEED))
        # engine path: sort_order injected from the compiled RankRepresentation
        via_engine = engine._run(
            transformer,
            rng=rng_kw[0] if needs_rng else None,
            audit=False,
        ).permutations
        # direct path: no data_order -> transformer computes its own argsort
        direct = transformer.build_permutation(
            engine.data, rng_kw[1] if needs_rng else None
        )
        assert np.array_equal(via_engine, direct), f"{tid}: injected != fallback"


# ---- §18C int32 result contract ----------------------------------------------
def test_result_permutations_are_int32_contiguous_readonly():
    data = heterogeneous_matrix(4, 100)
    result = transform_empirical_matrix(data, safe_transformers(4)[2][1])  # comonotonic
    assert result.permutations.dtype == np.int32
    assert result.permutations.flags["C_CONTIGUOUS"]
    assert result.permutations.flags["WRITEABLE"] is False


# ---- §18D index safety -------------------------------------------------------
@pytest.mark.parametrize("k,s", [(2, 2), (3, 7), (6, 101)])
def test_indices_in_range_and_representable(k, s):
    data = heterogeneous_matrix(k, s)
    result = transform_empirical_matrix(data, safe_transformers(k)[2][1])
    perm = result.permutations
    assert perm.min() >= 0
    assert perm.max() < s
    assert s < 2**31  # CompiledSample guarantees this; int32 is always sufficient


# ---- §18F input ownership ----------------------------------------------------
def test_caller_mutation_after_construction_cannot_change_results():
    data = heterogeneous_matrix(4, 120).copy()
    engine = DependenceStressEngine(data=data)
    before = engine.apply_comonotonic().portfolio_pnl.copy()
    perm_before = engine.apply_comonotonic().permutations.copy()
    data[:] = 0.0  # mutate the caller's array
    after = engine.apply_comonotonic().portfolio_pnl
    perm_after = engine.apply_comonotonic().permutations
    assert np.array_equal(before, after)
    assert np.array_equal(perm_before, perm_after)
    # the compiled sample owns a decoupled read-only snapshot
    assert engine.sample.observations.flags["WRITEABLE"] is False


def test_engine_owns_compiled_sample_and_rank_representation():
    engine = DependenceStressEngine(data=heterogeneous_matrix(3, 50))
    assert engine.sample.observations.shape == (3, 50)
    assert engine.rank_representation.sort_order.dtype == np.int32
    # .data is the owned observations, read-only
    assert not engine.data.flags.writeable
    assert np.shares_memory(engine.data, engine.sample.observations)
