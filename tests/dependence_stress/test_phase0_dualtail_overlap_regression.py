# tests/dependence_stress/test_phase0_dualtail_overlap_regression.py
"""
Phase 0 §6 — DualTail discretized-tail-count OVERLAP (accepted P0 defect).

Valid-partition condition (S known):

    n = max(1, ceil(p * S)),   require   n_lower + n_upper <= S

The probability guard (lower + upper < 1) does NOT imply it. When it is violated
under tail_order="random", the two random blocks overwrite each other; the result
is not a permutation and marginals are silently corrupted on the production path.

Design so the regression self-updates:
  * OVERLAP cases (n_lower + n_upper > S): the construction MUST fail loud. This
    is asserted with pytest.raises and FAILS on the current code (no guard yet) —
    the executable capture of the accepted P0. Phase 1 adds the count guard and
    these turn green.
  * NON-OVERLAP admitted cases: must yield a valid permutation for every seed —
    passes now and after the fix.

The discriminator is the mathematical count condition, not a lucky seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.tails import DualTailAlignment

_S_GRID = (2, 3, 5, 7)
_P_GRID = (0.2, 0.3, 0.34, 0.4)
_SEEDS = (0, 1, 2, 3, 7)


def _counts(p_lower, p_upper, s):
    return max(1, int(np.ceil(p_lower * s))), max(1, int(np.ceil(p_upper * s)))


def _admitted_cases():
    for s in _S_GRID:
        for lo in _P_GRID:
            for up in _P_GRID:
                if lo + up >= 1.0:
                    continue  # rejected at construction; outside the target set
                n_lower, n_upper = _counts(lo, up, s)
                overlaps = (n_lower + n_upper) > s
                yield pytest.param(
                    s,
                    lo,
                    up,
                    overlaps,
                    id=f"S{s}_lo{lo}_up{up}_{'OVERLAP' if overlaps else 'ok'}",
                )


@pytest.mark.parametrize("s,lo,up,overlaps", list(_admitted_cases()))
def test_overlap_must_fail_loud_else_valid_permutation(s, lo, up, overlaps):
    transformer = DualTailAlignment(
        lower_probability=lo, upper_probability=up, tail_order="random"
    )
    if overlaps:
        # ACCEPTED P0: currently produces a malformed permutation instead of
        # raising. Expected to FAIL now; Phase 1 count guard makes it pass.
        data = np.random.default_rng(s).standard_normal((4, s))
        with pytest.raises(ValueError):
            transformer.build_permutation(data, np.random.default_rng(s))
    else:
        expected = np.broadcast_to(np.arange(s), (4, s))
        for seed in _SEEDS:
            data = np.random.default_rng(seed).standard_normal((4, s))
            perm = transformer.build_permutation(data, np.random.default_rng(seed))
            assert np.array_equal(np.sort(perm, axis=1), expected)


def test_overlap_minimal_example_fails_loud():
    """
    Deterministic minimal capture: S=3, lower=upper=0.34 (n_lower=n_upper=2,
    sum=4>3). The construction must fail loud with an explicit message naming
    the counts and S, before any permutation is built. This replaces the
    Phase 0 corruption-demonstration once Phase 1 enforces the contract.
    """
    data = np.random.default_rng(2).standard_normal((4, 3))
    transformer = DualTailAlignment(
        lower_probability=0.34, upper_probability=0.34, tail_order="random"
    )
    with pytest.raises(
        ValueError, match=r"tail sizes overlap.*n_lower=2, n_upper=2, S=3"
    ):
        transformer.build_permutation(data, np.random.default_rng(2))
