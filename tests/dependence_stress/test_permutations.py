# tests/dependence_stress/test_permutations.py

"""Permutation construction, application, and the public/trusted seam."""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.permutations import (
    DependenceTransformer,
    _build_rank_permutation_unchecked,
    apply_permutation,
    build_rank_permutation,
    comonotonic_permutation,
    reconstruct_stressed,
    transform_empirical_matrix,
)
from riskquant.dependence_stress.tails import Comonotonic
from riskquant.dependence_stress.types import DependenceStressResult, FloatArray
from riskquant.dependence_stress.validation import validate_permutation_matrix

SMALL = np.array([[5.0, 1.0, 3.0], [2.0, 4.0, 6.0]])


def stressed(base: FloatArray, result: DependenceStressResult) -> FloatArray:
    """
    Reconstruct the stressed K x S matrix for a result.

    The stressed matrix is no longer a result field. Tests that need
    row-level stressed observations reconstruct it explicitly, which is
    exactly what a production audit consumer does.
    """
    return reconstruct_stressed(base, result.permutations)


def test_build_rank_permutation_transfers_score_order() -> None:
    """Smallest score takes the smallest observation."""
    scores = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])

    permutations = build_rank_permutation(SMALL, scores)
    stressed = np.take_along_axis(SMALL, permutations, axis=1)

    assert np.array_equal(stressed, np.array([[5.0, 1.0, 3.0], [2.0, 4.0, 6.0]]))
    assert np.array_equal(np.argsort(scores, axis=1), np.argsort(stressed, axis=1))


def test_build_rank_permutation_is_valid_permutation(base: FloatArray) -> None:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(base.shape)

    permutations = build_rank_permutation(base, scores)

    validate_permutation_matrix(permutations, expected_shape=base.shape)
    assert permutations.dtype == np.int64
    assert permutations.flags.c_contiguous


def test_build_rank_permutation_preserves_marginals(base: FloatArray) -> None:
    rng = np.random.default_rng(1)
    permutations = build_rank_permutation(base, rng.standard_normal(base.shape))
    stressed = np.take_along_axis(base, permutations, axis=1)

    assert np.array_equal(np.sort(stressed, axis=1), np.sort(base, axis=1))


def test_build_rank_permutation_matches_row_loop_reference(
    base: FloatArray,
) -> None:
    """The vectorized construction equals the explicit per-row reference."""
    rng = np.random.default_rng(2)
    scores = rng.standard_normal(base.shape)

    reference = np.empty(base.shape, dtype=np.int64)
    for row in range(base.shape[0]):
        data_order = np.argsort(base[row], kind="stable")
        score_order = np.argsort(scores[row], kind="stable")
        reference[row, score_order] = data_order

    assert np.array_equal(
        _build_rank_permutation_unchecked(base, scores),
        reference,
    )


def test_score_ties_break_stably() -> None:
    """Tied scores keep ascending scenario order. No randomness."""
    scores = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    first = build_rank_permutation(SMALL, scores)
    second = build_rank_permutation(SMALL, scores)

    assert np.array_equal(first, second)
    # All scores tied: slot order is scenario order, so the row is sorted.
    assert np.array_equal(
        np.take_along_axis(SMALL, first, axis=1),
        np.sort(SMALL, axis=1),
    )


@pytest.mark.parametrize(
    "scores,match",
    [
        (np.zeros(3), "2-dimensional"),
        (np.zeros((2, 4)), "does not match data shape"),
        (np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]]), "NaN or infinite"),
    ],
)
def test_score_matrix_rejections(scores: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_rank_permutation(SMALL, scores)


def test_comonotonic_permutation_sorts_every_row(base: FloatArray) -> None:
    permutations = comonotonic_permutation(base)
    stressed = np.take_along_axis(base, permutations, axis=1)

    assert np.array_equal(stressed, np.sort(base, axis=1))


def test_apply_permutation_identity_returns_input(base: FloatArray) -> None:
    identity = np.broadcast_to(
        np.arange(base.shape[1], dtype=np.int64),
        base.shape,
    )

    assert np.array_equal(
        apply_permutation(base, np.ascontiguousarray(identity)),
        base,
    )


def test_apply_permutation_validates_its_inputs(base: FloatArray) -> None:
    broken = np.zeros(base.shape, dtype=np.int64)

    with pytest.raises(ValueError, match="exactly once"):
        apply_permutation(base, broken)


def test_transform_empirical_matrix_validates_at_the_edge() -> None:
    """The public entry point owns the boundary contract."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        transform_empirical_matrix(
            np.array([[1.0, np.nan], [1.0, 2.0]]),
            Comonotonic(),
        )


def test_no_public_trusted_escape_hatch() -> None:
    """A keyword that disables core validation must not be reachable."""
    import inspect

    signature = inspect.signature(transform_empirical_matrix)

    assert "_trusted" not in signature.parameters
    assert set(signature.parameters) == {"data", "transformer", "rng", "audit"}


def test_transform_accepts_non_canonical_input(base: FloatArray) -> None:
    """Fortran-ordered float32 input is canonicalized, not rejected."""
    awkward = np.asfortranarray(base.astype(np.float32))

    result = transform_empirical_matrix(awkward, Comonotonic())
    reconstructed = stressed(base, result)

    assert reconstructed.dtype == np.float64
    assert reconstructed.flags.c_contiguous
    assert result.portfolio_pnl.dtype == np.float64


def test_audit_mode_raises_on_a_lying_transformer(base: FloatArray) -> None:
    """A transformer returning a non-permutation is caught under audit."""

    class Liar:
        @property
        def requires_rng(self) -> bool:
            return False

        def build_permutation(
            self,
            data: FloatArray,
            rng: np.random.Generator | None = None,
        ) -> np.ndarray:
            return np.zeros(data.shape, dtype=np.int64)

    liar: DependenceTransformer = Liar()

    with pytest.raises(ValueError, match="exactly once"):
        transform_empirical_matrix(base, liar, audit=True)

    # Without audit the invalid permutation is applied silently. This is
    # the documented cost of the trusted path and the reason audit exists.
    unaudited = transform_empirical_matrix(base, liar)

    # take_along_axis directly, not reconstruct_stressed: the whole point
    # is that this permutation is invalid, so the validated reconstruction
    # path would (correctly) refuse it.
    reconstructed = np.take_along_axis(base, unaudited.permutations, axis=1)

    assert np.array_equal(reconstructed[:, 0], reconstructed[:, 1])
    assert np.array_equal(reconstructed.sum(axis=0), unaudited.portfolio_pnl)


def test_protocol_runtime_check() -> None:
    assert isinstance(Comonotonic(), DependenceTransformer)
    assert not isinstance(object(), DependenceTransformer)
