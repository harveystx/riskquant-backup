# tests/dependence_stress/test_tails.py

"""Lower-tail alignment and the comonotonic benchmark."""

from __future__ import annotations

import re

import numpy as np
import pytest

from riskquant.dependence_stress.diagnostics import (
    empirical_lower_tail_cooccurrence,
    pearson_correlation,
    spearman_correlation,
)
from riskquant.dependence_stress.permutations import transform_empirical_matrix
from riskquant.dependence_stress.tails import Comonotonic, LowerTailAlignment
from riskquant.dependence_stress.types import FloatArray

Q = 0.05


# ======================================================================
# Configuration contract
# ======================================================================


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.5, 2.0, np.nan, np.inf])
def test_invalid_tail_probability(bad_q: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        LowerTailAlignment(tail_probability=bad_q)


def test_invalid_tail_order() -> None:
    with pytest.raises(ValueError, match="unsupported tail_order"):
        LowerTailAlignment(tail_order="upper")  # type: ignore[arg-type]


def test_requires_rng_only_in_random_mode() -> None:
    assert not LowerTailAlignment(tail_order="comonotonic").requires_rng
    assert LowerTailAlignment(tail_order="random").requires_rng
    assert not Comonotonic().requires_rng


@pytest.mark.parametrize("bad_rng", [None, 42, "rng"])
def test_random_mode_rejects_malformed_rng(
    base: FloatArray,
    bad_rng: object,
) -> None:
    transformer = LowerTailAlignment(tail_order="random")

    with pytest.raises(TypeError, match=re.escape("numpy.random.Generator")):
        transformer.build_permutation(base, bad_rng)  # type: ignore[arg-type]


# ======================================================================
# Tail block
# ======================================================================


def test_exact_tail_count(base: FloatArray) -> None:
    s = base.shape[1]
    n_tail = int(np.ceil(Q * s))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    for row in range(base.shape[0]):
        expected = np.sort(base[row])[:n_tail]
        assert np.array_equal(np.sort(result.stressed[row, :n_tail]), expected)


def test_tail_count_floors_at_one() -> None:
    """A tiny q must still align at least one observation."""
    data = np.arange(20, dtype=np.float64).reshape(2, 10)

    result = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=1e-6),
    )

    assert result.stressed[0, 0] == data[0].min()
    assert result.stressed[1, 0] == data[1].min()


def test_tail_block_holds_the_true_lowest_observations(base: FloatArray) -> None:
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    for row in range(base.shape[0]):
        threshold = np.sort(base[row])[n_tail - 1]
        assert np.all(result.stressed[row, :n_tail] <= threshold)


def test_tail_alignment_maximizes_lower_tail_cooccurrence(
    base: FloatArray,
) -> None:
    """All rows share the same tail slots, so co-occurrence is exactly 1."""
    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    cooccurrence = empirical_lower_tail_cooccurrence(result.stressed, Q)

    assert np.allclose(cooccurrence, 1.0)


def test_comonotonic_tail_matches_severity(base: FloatArray) -> None:
    """Worst with worst: the tail block is ascending in every row."""
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    block = result.stressed[:, :n_tail]

    assert np.all(np.diff(block, axis=1) >= 0.0)


def test_random_mode_changes_only_the_tail_block(base: FloatArray) -> None:
    n_tail = int(np.ceil(Q * base.shape[1]))

    ordered = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )
    randomized = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q, tail_order="random"),
        np.random.default_rng(0),
    )

    assert np.array_equal(
        ordered.stressed[:, n_tail:],
        randomized.stressed[:, n_tail:],
    )
    assert np.array_equal(
        np.sort(ordered.stressed[:, :n_tail], axis=1),
        np.sort(randomized.stressed[:, :n_tail], axis=1),
    )
    assert not np.array_equal(
        ordered.stressed[:, :n_tail],
        randomized.stressed[:, :n_tail],
    )


def test_random_mode_still_occupies_the_same_tail_slots(base: FloatArray) -> None:
    """Randomization permutes severity pairing, not tail membership."""
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q, tail_order="random"),
        np.random.default_rng(3),
    )

    cooccurrence = empirical_lower_tail_cooccurrence(result.stressed, Q)

    assert np.allclose(cooccurrence, 1.0)
    assert n_tail >= 1


# ======================================================================
# Non-tail block
# ======================================================================


def test_non_tail_relative_order_is_preserved() -> None:
    data = np.array([[5.0, 1.0, 3.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]])

    result = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=0.2),
    )

    # Row 0: scenario 1 holds the minimum and moves into slot 0. The
    # remaining scenarios 0, 2, 3, 4 keep their original relative order.
    assert np.array_equal(result.permutations[0], [1, 0, 2, 3, 4])
    assert np.array_equal(result.stressed[0], [1.0, 5.0, 3.0, 2.0, 4.0])


def test_non_tail_block_is_not_sorted_when_preserved(base: FloatArray) -> None:
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    non_tail = result.stressed[:, n_tail:]

    assert not np.all(np.diff(non_tail, axis=1) >= 0.0)


def test_preserve_false_gives_full_comonotonic(base: FloatArray) -> None:
    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(
            tail_probability=Q,
            preserve_non_tail_order=False,
        ),
    )

    assert np.array_equal(result.stressed, np.sort(base, axis=1))


def test_q_near_one_leaves_no_non_tail_block() -> None:
    data = np.array([[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]])

    result = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=0.999),
    )

    assert np.array_equal(result.stressed, np.sort(data, axis=1))


# ======================================================================
# Ties
# ======================================================================


def test_tail_boundary_tie_is_broken_by_scenario_index() -> None:
    """Tied at the boundary: the lower scenario index enters the tail."""
    data = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0]])

    result = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=0.5),
    )

    # Row 0 tail is the two lowest: scenario 1 (0.0) then the first of the
    # tied 1.0 values, which is scenario 0.
    assert np.array_equal(result.permutations[0][:2], [1, 0])


def test_tie_selection_is_reproducible() -> None:
    data = np.array([[1.0, 1.0, 1.0, 2.0], [5.0, 5.0, 5.0, 5.0]])

    first = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=0.5),
    )
    second = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=0.5),
    )

    assert np.array_equal(first.permutations, second.permutations)


# ======================================================================
# Comonotonic
# ======================================================================


def test_comonotonic_exact_sorted_output(base: FloatArray) -> None:
    result = transform_empirical_matrix(base, Comonotonic())

    assert np.array_equal(result.stressed, np.sort(base, axis=1))


def test_comonotonic_idempotent(base: FloatArray) -> None:
    once = transform_empirical_matrix(base, Comonotonic()).stressed
    twice = transform_empirical_matrix(once, Comonotonic()).stressed

    assert np.array_equal(once, twice)


def test_comonotonic_unit_spearman(base: FloatArray) -> None:
    stressed = transform_empirical_matrix(base, Comonotonic()).stressed

    assert np.allclose(spearman_correlation(stressed), 1.0, atol=1e-12)


def test_comonotonic_pearson_is_maximal_not_unit(base: FloatArray) -> None:
    """
    Comonotonicity is a rank property.

    Pearson equals 1 only for affine-related rows, so the correct
    invariant is maximality over row-wise permutations (rearrangement
    inequality), not equality with 1.
    """
    comonotonic = pearson_correlation(
        transform_empirical_matrix(base, Comonotonic()).stressed
    )

    assert not np.allclose(comonotonic, 1.0)

    rng = np.random.default_rng(0)
    for _ in range(25):
        shuffled = np.take_along_axis(
            base,
            np.argsort(rng.random(base.shape), axis=1),
            axis=1,
        )
        assert np.all(pearson_correlation(shuffled) <= comonotonic + 1e-12)


def test_comonotonic_pearson_is_unit_for_affine_rows() -> None:
    """The Pearson=1 case, stated with its actual precondition."""
    rng = np.random.default_rng(1)
    row = rng.standard_normal(300)

    # Strictly increasing affine maps only. A decreasing map (-0.5 * row)
    # is comonotonic with the others after sorting, so its Pearson
    # correlation lands near +1 rather than -1 and would prove nothing.
    data = np.vstack([row, 2.0 * row + 5.0, 0.5 * row - 3.0])

    stressed = transform_empirical_matrix(data, Comonotonic()).stressed

    assert np.allclose(pearson_correlation(stressed), 1.0)


def test_comonotonic_ignores_rng(base: FloatArray) -> None:
    first = transform_empirical_matrix(base, Comonotonic())
    second = transform_empirical_matrix(base, Comonotonic(), np.random.default_rng(0))
    third = transform_empirical_matrix(base, Comonotonic(), np.random.default_rng(99))

    assert np.array_equal(first.permutations, second.permutations)
    assert np.array_equal(first.permutations, third.permutations)
