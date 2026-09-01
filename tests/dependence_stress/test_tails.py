# tests/dependence_stress/test_tails.py

"""Lower-tail alignment, upper/dual tail alignment, and the comonotonic benchmark."""

from __future__ import annotations

import re

import numpy as np
import pytest

from riskquant.dependence_stress.diagnostics import (
    empirical_lower_tail_cooccurrence,
    empirical_upper_tail_cooccurrence,
    pearson_correlation,
    spearman_correlation,
)
from riskquant.dependence_stress.permutations import (
    reconstruct_stressed,
    transform_empirical_matrix,
)
from riskquant.dependence_stress.tails import (
    Comonotonic,
    DualTailAlignment,
    LowerTailAlignment,
    UpperTailAlignment,
)
from riskquant.dependence_stress.types import DependenceStressResult, FloatArray


def stressed(base: FloatArray, result: DependenceStressResult) -> FloatArray:
    """
    Reconstruct the stressed K x S matrix for a result.

    The stressed matrix is no longer a result field. Tests that need
    row-level stressed observations reconstruct it explicitly, which is
    exactly what a production audit consumer does.
    """
    return reconstruct_stressed(base, result.permutations)


Q = 0.05


# ======================================================================
# Configuration contract
# ======================================================================


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.5, 2.0, np.nan, np.inf])
def test_invalid_tail_probability(bad_q: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        LowerTailAlignment(tail_probability=bad_q)

    with pytest.raises(ValueError, match="probability"):
        UpperTailAlignment(tail_probability=bad_q)

    with pytest.raises(ValueError, match="probability"):
        DualTailAlignment(lower_probability=bad_q)

    with pytest.raises(ValueError, match="probability"):
        DualTailAlignment(upper_probability=bad_q)


def test_dual_tail_probability_sum_limit() -> None:
    with pytest.raises(ValueError, match="strictly less than 1.0"):
        DualTailAlignment(lower_probability=0.6, upper_probability=0.5)


@pytest.mark.parametrize(
    "cls", [LowerTailAlignment, UpperTailAlignment, DualTailAlignment]
)
def test_invalid_tail_order(cls: type) -> None:
    with pytest.raises(ValueError, match="unsupported tail_order"):
        # The classes take kw_only config if we were using it,
        # but they accept standard dataclass init.
        cls(tail_order="invalid")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "cls", [LowerTailAlignment, UpperTailAlignment, DualTailAlignment]
)
def test_requires_rng_only_in_random_mode(cls: type) -> None:
    assert not cls(tail_order="comonotonic").requires_rng
    assert cls(tail_order="random").requires_rng


def test_comonotonic_requires_no_rng() -> None:
    assert not Comonotonic().requires_rng


@pytest.mark.parametrize(
    "cls", [LowerTailAlignment, UpperTailAlignment, DualTailAlignment]
)
@pytest.mark.parametrize("bad_rng", [None, 42, "rng"])
def test_random_mode_rejects_malformed_rng(
    base: FloatArray,
    cls: type,
    bad_rng: object,
) -> None:
    transformer = cls(tail_order="random")

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
        assert np.array_equal(np.sort(stressed(base, result)[row, :n_tail]), expected)


def test_tail_count_floors_at_one() -> None:
    """A tiny q must still align at least one observation."""
    data = np.arange(20, dtype=np.float64).reshape(2, 10)

    result = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=1e-6),
    )

    assert stressed(data, result)[0, 0] == data[0].min()
    assert stressed(data, result)[1, 0] == data[1].min()


def test_tail_block_holds_the_true_lowest_observations(base: FloatArray) -> None:
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    for row in range(base.shape[0]):
        threshold = np.sort(base[row])[n_tail - 1]
        assert np.all(stressed(base, result)[row, :n_tail] <= threshold)


def test_upper_tail_block_holds_the_true_highest_observations(base: FloatArray) -> None:
    s = base.shape[1]
    n_tail = int(np.ceil(Q * s))

    result = transform_empirical_matrix(
        base,
        UpperTailAlignment(tail_probability=Q),
    )

    for row in range(base.shape[0]):
        threshold = np.sort(base[row])[s - n_tail]
        assert np.all(stressed(base, result)[row, s - n_tail :] >= threshold)


def test_dual_tail_block_alignment(base: FloatArray) -> None:
    s = base.shape[1]
    n_tail = int(np.ceil(Q * s))

    result = transform_empirical_matrix(
        base,
        DualTailAlignment(lower_probability=Q, upper_probability=Q),
    )

    stressed_matrix = stressed(base, result)

    for row in range(base.shape[0]):
        sorted_row = np.sort(base[row])

        assert np.array_equal(
            np.sort(stressed_matrix[row, :n_tail]), sorted_row[:n_tail]
        )

        assert np.array_equal(
            np.sort(stressed_matrix[row, s - n_tail :]), sorted_row[s - n_tail :]
        )


def test_tail_alignment_maximizes_lower_tail_cooccurrence(
    base: FloatArray,
) -> None:
    """All rows share the same tail slots, so co-occurrence is exactly 1."""
    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    cooccurrence = empirical_lower_tail_cooccurrence(stressed(base, result), Q)

    assert np.allclose(cooccurrence, 1.0)


def test_upper_tail_alignment_maximizes_upper_tail_cooccurrence(
    base: FloatArray,
) -> None:
    result = transform_empirical_matrix(
        base,
        UpperTailAlignment(tail_probability=Q),
    )

    cooccurrence = empirical_upper_tail_cooccurrence(stressed(base, result), Q)

    assert np.allclose(cooccurrence, 1.0)


def test_comonotonic_tail_matches_severity(base: FloatArray) -> None:
    """Worst with worst: the tail block is ascending in every row."""
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    block = stressed(base, result)[:, :n_tail]

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
        stressed(base, ordered)[:, n_tail:],
        stressed(base, randomized)[:, n_tail:],
    )
    assert np.array_equal(
        np.sort(stressed(base, ordered)[:, :n_tail], axis=1),
        np.sort(stressed(base, randomized)[:, :n_tail], axis=1),
    )
    assert not np.array_equal(
        stressed(base, ordered)[:, :n_tail],
        stressed(base, randomized)[:, :n_tail],
    )


def test_random_mode_still_occupies_the_same_tail_slots(base: FloatArray) -> None:
    """Randomization permutes severity pairing, not tail membership."""
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q, tail_order="random"),
        np.random.default_rng(3),
    )

    cooccurrence = empirical_lower_tail_cooccurrence(stressed(base, result), Q)

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
    assert np.array_equal(stressed(data, result)[0], [1.0, 5.0, 3.0, 2.0, 4.0])


def test_dual_tail_non_tail_relative_order_is_preserved() -> None:
    data = np.array([[5.0, 1.0, 3.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]])

    result = transform_empirical_matrix(
        data,
        DualTailAlignment(lower_probability=0.2, upper_probability=0.2),
    )

    # Row 0: scenario 1 (1.0) goes to slot 0 (lower tail).
    #        scenario 0 (5.0) goes to slot 4 (upper tail).
    #        Remaining scenarios (2, 3, 4) keep relative order in middle.
    assert np.array_equal(result.permutations[0], [1, 2, 3, 4, 0])
    assert np.array_equal(stressed(data, result)[0], [1.0, 3.0, 2.0, 4.0, 5.0])


def test_non_tail_block_is_not_sorted_when_preserved(base: FloatArray) -> None:
    n_tail = int(np.ceil(Q * base.shape[1]))

    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(tail_probability=Q),
    )

    non_tail = stressed(base, result)[:, n_tail:]

    assert not np.all(np.diff(non_tail, axis=1) >= 0.0)


def test_preserve_false_gives_full_comonotonic(base: FloatArray) -> None:
    result = transform_empirical_matrix(
        base,
        LowerTailAlignment(
            tail_probability=Q,
            preserve_non_tail_order=False,
        ),
    )

    assert np.array_equal(stressed(base, result), np.sort(base, axis=1))


def test_q_near_one_leaves_no_non_tail_block() -> None:
    data = np.array([[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]])

    result = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=0.999),
    )

    assert np.array_equal(stressed(data, result), np.sort(data, axis=1))


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

    assert np.array_equal(stressed(base, result), np.sort(base, axis=1))


def test_comonotonic_idempotent(base: FloatArray) -> None:
    once = stressed(base, transform_empirical_matrix(base, Comonotonic()))
    twice = stressed(once, transform_empirical_matrix(once, Comonotonic()))

    assert np.array_equal(once, twice)


def test_comonotonic_unit_spearman(base: FloatArray) -> None:
    comonotonic = stressed(base, transform_empirical_matrix(base, Comonotonic()))

    assert np.allclose(spearman_correlation(comonotonic), 1.0, atol=1e-12)


def test_comonotonic_pearson_is_maximal_not_unit(base: FloatArray) -> None:
    """
    Comonotonicity is a rank property.

    Pearson equals 1 only for affine-related rows, so the correct
    invariant is maximality over row-wise permutations (rearrangement
    inequality), not equality with 1.
    """
    comonotonic = pearson_correlation(
        stressed(base, transform_empirical_matrix(base, Comonotonic()))
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

    comonotonic = stressed(data, transform_empirical_matrix(data, Comonotonic()))

    assert np.allclose(pearson_correlation(comonotonic), 1.0)


def test_comonotonic_ignores_rng(base: FloatArray) -> None:
    first = transform_empirical_matrix(base, Comonotonic())
    second = transform_empirical_matrix(base, Comonotonic(), np.random.default_rng(0))
    third = transform_empirical_matrix(base, Comonotonic(), np.random.default_rng(99))

    assert np.array_equal(first.permutations, second.permutations)
    assert np.array_equal(first.permutations, third.permutations)
