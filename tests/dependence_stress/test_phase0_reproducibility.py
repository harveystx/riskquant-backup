# tests/dependence_stress/test_phase0_reproducibility.py
"""
Phase 0 §7 — stochastic reproducibility vs deterministic transformation.

Stochastic transformers: same input + same parameters + same INITIAL rng state
=> same permutation; and for streaming, same initial state => same SEQUENCE.
This is reproducibility of a stochastic sequence, not determinism.

Deterministic transformers: same input => same permutation regardless of rng
(including None), per the existing contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.permutations import (
    iter_transform_empirical_matrix,
    transform_empirical_matrix,
)

from phase0_support import (  # type: ignore[import-not-found]
    SEED,
    heterogeneous_matrix,
    safe_transformers,
)

_DATA = heterogeneous_matrix(6, 200)
_ALL = safe_transformers(6)
_STOCH = [(tid, t) for tid, t, needs in _ALL if needs]
_DET = [(tid, t) for tid, t, needs in _ALL if not needs]


@pytest.mark.parametrize("tid,transformer", _STOCH, ids=[t for t, _ in _STOCH])
def test_stochastic_same_state_same_permutation(tid, transformer):
    a = transform_empirical_matrix(_DATA, transformer, np.random.default_rng(SEED))
    b = transform_empirical_matrix(_DATA, transformer, np.random.default_rng(SEED))
    assert np.array_equal(a.permutations, b.permutations)


@pytest.mark.parametrize("tid,transformer", _STOCH, ids=[t for t, _ in _STOCH])
def test_stochastic_different_state_generally_differs(tid, transformer):
    a = transform_empirical_matrix(_DATA, transformer, np.random.default_rng(1))
    b = transform_empirical_matrix(_DATA, transformer, np.random.default_rng(2))
    assert not np.array_equal(a.permutations, b.permutations)


@pytest.mark.parametrize("tid,transformer", _STOCH, ids=[t for t, _ in _STOCH])
def test_stochastic_streaming_sequence_reproducible(tid, transformer):
    seq_a = [
        r.permutations.copy()
        for r in iter_transform_empirical_matrix(
            _DATA, transformer, np.random.default_rng(SEED), n_realizations=5
        )
    ]
    seq_b = [
        r.permutations.copy()
        for r in iter_transform_empirical_matrix(
            _DATA, transformer, np.random.default_rng(SEED), n_realizations=5
        )
    ]
    assert all(np.array_equal(x, y) for x, y in zip(seq_a, seq_b))
    # a stochastic SEQUENCE advances: consecutive draws are not all identical
    assert not all(np.array_equal(seq_a[0], s) for s in seq_a[1:])


@pytest.mark.parametrize("tid,transformer", _DET, ids=[t for t, _ in _DET])
def test_deterministic_ignores_rng(tid, transformer):
    none = transform_empirical_matrix(_DATA, transformer, None)
    r1 = transform_empirical_matrix(_DATA, transformer, np.random.default_rng(1))
    r2 = transform_empirical_matrix(_DATA, transformer, np.random.default_rng(999))
    assert np.array_equal(none.permutations, r1.permutations)
    assert np.array_equal(none.permutations, r2.permutations)
