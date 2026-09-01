# tests/dependence/test_kendall_rank_equivalence.py

"""
The hard invariant: Kendall tau-b on average ranks equals tau-b on raw data,
bitwise.

This is what licenses Kendall to consume ``RankRepresentation`` and never touch
the raw observations. It is an exact transformation, not an approximation, so
the assertion is ``array_equal`` and never ``allclose``.

Argument
--------
Average ranking satisfies, by construction,

    x_a = x_b  =>  r_a = r_b        (tie-block members share the block mean)
    x_a < x_b  =>  r_a < r_b        (distinct values occupy disjoint,
                                     order-respecting rank intervals)

hence ``sign(r_a - r_b) = sign(x_a - x_b)`` for every pair. tau-b is a function
of ``n0, n1, n2, n3`` and the discordant count, each determined purely by that
sign pattern, so all five counts are identical and the closing arithmetic is
applied to identical integers.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence.estimators.kendall import kendall_tau_b
from riskquant.observations.compile import compile_ranks, compile_sample
from riskquant.observations.orientation import ObservationOrientation
from tests.dependence.conftest import compile_both, continuous_sample

VARIABLES_ROWS = ObservationOrientation.VARIABLES_ON_ROWS


def _tau_from_ranks_of(matrix: np.ndarray) -> np.ndarray:
    """tau-b computed by feeding the average ranks back in as raw observations."""
    _, ranks = compile_both(matrix)
    rank_matrix = np.ascontiguousarray(ranks.ranks)
    rank_sample = compile_sample(rank_matrix, orientation=VARIABLES_ROWS)
    return kendall_tau_b(rank_sample, compile_ranks(rank_sample)).matrix


def _tau_of(matrix: np.ndarray) -> np.ndarray:
    """tau-b computed from the raw observations, via their rank representation."""
    sample, ranks = compile_both(matrix)
    return kendall_tau_b(sample, ranks).matrix


def _cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    base = continuous_sample(seed=5, n_variables=6, n_observations=200)
    return {
        "no_ties": base,
        "isolated_ties": np.ascontiguousarray(np.round(base, 2)),
        "large_tie_blocks": np.ascontiguousarray(
            np.where(rng.random((6, 200)) < 0.6, 0.0, base)
        ),
        "integer_valued": rng.integers(-2, 3, (6, 200)).astype(np.float64),
        "anti_monotone": np.ascontiguousarray(
            np.vstack([np.arange(200.0), -np.arange(200.0), base[2:]])
        ),
        "two_valued": np.ascontiguousarray(
            np.where(rng.random((6, 200)) < 0.5, 0.0, 1.0)
            + np.linspace(0.0, 1e-9, 200)
        ),
        "minimal": np.array([[1.0, 2.0], [2.0, 1.0]]),
    }


@pytest.mark.parametrize("case", list(_cases()))
def test_tau_b_from_ranks_is_bitwise_identical_to_tau_b_from_raw(case: str) -> None:
    matrix = _cases()[case]
    assert np.array_equal(_tau_of(matrix), _tau_from_ranks_of(matrix))


@pytest.mark.parametrize("case", list(_cases()))
def test_intermediate_integer_counts_are_identical(case: str) -> None:
    """
    The counts, not just the final value.

    ``tie_pairs`` is the only count materialised outside the kernel; the sort
    order determines every other one, so equality of both establishes that the
    kernel sees an identical problem in the two representations.
    """
    matrix = _cases()[case]

    _, raw_ranks = compile_both(matrix)
    rank_sample = compile_sample(
        np.ascontiguousarray(raw_ranks.ranks), orientation=VARIABLES_ROWS
    )
    of_ranks = compile_ranks(rank_sample)

    assert np.array_equal(raw_ranks.tie_pairs, of_ranks.tie_pairs)
    assert np.array_equal(raw_ranks.sort_order, of_ranks.sort_order)


@pytest.mark.parametrize("case", list(_cases()))
def test_the_sign_pattern_of_pairwise_differences_is_preserved(case: str) -> None:
    """The premise of the argument, checked directly on every pair."""
    matrix = _cases()[case]
    _, ranks = compile_both(matrix)

    for variable in range(matrix.shape[0]):
        raw = matrix[variable]
        rank = ranks.ranks[variable]
        assert np.array_equal(
            np.sign(raw[:, None] - raw[None, :]),
            np.sign(rank[:, None] - rank[None, :]),
        )


def test_a_complete_tie_is_rejected_identically_in_both_representations() -> None:
    matrix = np.array([[7.0, 7.0, 7.0, 7.0], [1.0, 2.0, 3.0, 4.0]])

    with pytest.raises(ValueError, match="Kendall tau-b is undefined"):
        _tau_of(matrix)
    with pytest.raises(ValueError, match="Kendall tau-b is undefined"):
        _tau_from_ranks_of(matrix)


def test_distinct_tie_blocks_never_collide_in_average_rank() -> None:
    """
    The one way the equivalence could fail.

    Consecutive blocks spanning ``[i+1..j]`` and ``[j+1..k]`` have means
    differing by ``(len_A + len_B)/2 >= 1``, and all such means are integers or
    half-integers, exactly representable while ``2S <= 2**53``.
    """
    rng = np.random.default_rng(7)
    for _ in range(30):
        values = rng.integers(0, 5, 400).astype(np.float64)
        sample = compile_sample(
            np.vstack([values, np.arange(400.0)]), orientation=VARIABLES_ROWS
        )
        ranks = compile_ranks(sample).ranks[0]
        for left in np.unique(values):
            for right in np.unique(values):
                if left != right:
                    assert ranks[values == left][0] != ranks[values == right][0]
