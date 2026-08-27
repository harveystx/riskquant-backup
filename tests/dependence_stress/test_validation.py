# tests/dependence_stress/test_validation.py

"""Boundary contract tests. Every rejection must name its contract."""

from __future__ import annotations

import re

import numpy as np
import pytest

from riskquant.dependence_stress.validation import (
    CORRELATION_MIN_EIGENVALUE,
    canonicalize_owned_matrix,
    validate_correlation,
    validate_marginal_preservation,
    validate_observation_matrix,
    validate_permutation_matrix,
    validate_probability,
    validate_rng,
    validate_transformation,
)

GOOD = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 5.0]], dtype=np.float64)


# ======================================================================
# Observation matrix
# ======================================================================


@pytest.mark.parametrize(
    "bad,match",
    [
        (np.arange(6, dtype=np.float64), "2-dimensional"),
        (np.zeros((2, 2, 2)), "2-dimensional"),
        (np.zeros((0, 0)), "at least 2 rows"),
        (np.ones((1, 5)), "at least 2 rows"),
        (np.ones((5, 1)), "at least 2 observations"),
        (np.array([[1.0, np.nan], [1.0, 2.0]]), "NaN or infinite"),
        (np.array([[1.0, np.inf], [1.0, 2.0]]), "NaN or infinite"),
        (np.array([[1.0, -np.inf], [1.0, 2.0]]), "NaN or infinite"),
    ],
)
def test_observation_matrix_rejections(bad: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_observation_matrix(bad)  # type: ignore[arg-type]


def test_observation_matrix_accepts_and_canonicalizes_integers() -> None:
    out = validate_observation_matrix(np.array([[1, 2, 3], [4, 5, 6]]))

    assert out.dtype == np.float64
    assert out.flags.c_contiguous


def test_observation_matrix_accepts_float32() -> None:
    out = validate_observation_matrix(GOOD.astype(np.float32))

    assert out.dtype == np.float64


def test_observation_matrix_canonicalizes_fortran_order() -> None:
    fortran = np.asfortranarray(GOOD)

    out = validate_observation_matrix(fortran)

    assert out.flags.c_contiguous
    assert np.array_equal(out, GOOD)


def test_observation_matrix_canonicalizes_non_contiguous_slice() -> None:
    strided = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]

    assert not strided.flags.c_contiguous

    out = validate_observation_matrix(strided)

    assert out.flags.c_contiguous
    assert np.array_equal(out, strided)


def test_observation_matrix_never_transposes() -> None:
    """A 2 x 500 input stays 2 x 500. The axis convention is not inferred."""
    wide = np.zeros((2, 500), dtype=np.float64)

    assert validate_observation_matrix(wide).shape == (2, 500)


# ======================================================================
# Ownership
# ======================================================================


def test_canonicalize_owned_matrix_decouples_from_source() -> None:
    source = GOOD.copy()

    owned = canonicalize_owned_matrix(source)

    source[0, 0] = -999.0

    assert owned[0, 0] == 1.0
    assert not owned.flags.writeable


def test_canonicalize_owned_matrix_is_read_only() -> None:
    owned = canonicalize_owned_matrix(GOOD)

    with pytest.raises(ValueError):
        owned[0, 0] = 0.0


# ======================================================================
# Correlation
# ======================================================================


def _equi(k: int, rho: float) -> np.ndarray:
    matrix = np.full((k, k), rho, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def test_correlation_accepts_valid() -> None:
    out = validate_correlation(_equi(4, 0.3))

    assert out.shape == (4, 4)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous


def test_correlation_canonicalizes_tolerable_asymmetry() -> None:
    """Accepted near-symmetric input is returned exactly symmetric."""
    matrix = _equi(3, 0.4)
    matrix[0, 1] += 1e-12

    out = validate_correlation(matrix)

    assert np.array_equal(out, out.T)
    assert out[0, 1] == pytest.approx(0.4 + 5e-13, abs=0.0, rel=1e-9)


@pytest.mark.parametrize(
    "bad,match",
    [
        (np.zeros((3, 4)), "square"),
        (np.zeros(3), "2-dimensional"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "NaN or infinite"),
        (np.array([[1.0, np.inf], [np.inf, 1.0]]), "NaN or infinite"),
        (np.array([[1.0, 0.9], [0.1, 1.0]]), "symmetric"),
        (np.array([[1.0, 0.5], [0.5, 0.9]]), "diagonal must equal 1"),
        (np.array([[1.0, 1.5], [1.5, 1.0]]), r"\[-1, 1\]"),
        (np.array([[1.0, -1.5], [-1.5, 1.0]]), r"\[-1, 1\]"),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), "positive definite"),
        (_equi(3, -0.5), "positive definite"),
    ],
)
def test_correlation_rejections(bad: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_correlation(bad)


def test_correlation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expected correlation shape"):
        validate_correlation(_equi(3, 0.2), 5)


def test_correlation_rejects_psd_not_pd() -> None:
    """Rank-deficient but positive semi-definite input is rejected."""
    matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="positive definite"):
        validate_correlation(matrix)


def test_correlation_threshold_boundary() -> None:
    """Just below the eigenvalue floor fails; comfortably above passes."""
    eps = 0.5 * CORRELATION_MIN_EIGENVALUE
    rho = 1.0 - eps

    with pytest.raises(ValueError, match="positive definite"):
        validate_correlation(np.array([[1.0, rho], [rho, 1.0]]))

    validate_correlation(np.array([[1.0, 1.0 - 1e-8], [1.0 - 1e-8, 1.0]]))


# ======================================================================
# Permutations
# ======================================================================


def test_permutation_accepts_valid() -> None:
    out = validate_permutation_matrix(np.array([[0, 1, 2], [2, 0, 1]]))

    assert out.dtype == np.int64
    assert out.flags.c_contiguous


@pytest.mark.parametrize(
    "bad,match",
    [
        (np.array([0, 1, 2]), "2-dimensional"),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), "integer indices"),
        (np.array([[0, 0], [0, 1]]), "exactly once"),
        (np.array([[0, 2], [1, 0]]), r"\[0, 1\]"),
        (np.array([[-1, 1], [1, 0]]), r"\[0, 1\]"),
    ],
)
def test_permutation_rejections(bad: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_permutation_matrix(bad)


def test_permutation_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected permutation shape"):
        validate_permutation_matrix(
            np.array([[0, 1], [1, 0]]),
            expected_shape=(2, 3),
        )


def test_permutation_missing_index_is_rejected() -> None:
    """A row that repeats one index and drops another must fail."""
    with pytest.raises(ValueError, match="exactly once"):
        validate_permutation_matrix(np.array([[0, 1, 1], [0, 1, 2]]))


# ======================================================================
# Probability and RNG
# ======================================================================


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, np.nan, np.inf])
def test_probability_rejections(bad: float) -> None:
    with pytest.raises(ValueError):
        validate_probability(bad)


def test_probability_returns_canonical_float() -> None:
    assert validate_probability(np.float32(0.05)) == pytest.approx(0.05)


@pytest.mark.parametrize("bad", [None, 42, "rng", np.random.RandomState(0)])
def test_rng_rejections(bad: object) -> None:
    with pytest.raises(TypeError, match=re.escape("numpy.random.Generator")):
        validate_rng(bad)


def test_rng_accepts_generator() -> None:
    generator = np.random.default_rng(0)

    assert validate_rng(generator) is generator


# ======================================================================
# Audit
# ======================================================================


def test_marginal_preservation_true_and_false() -> None:
    permuted = GOOD[:, ::-1].copy()

    assert validate_marginal_preservation(GOOD, permuted)
    assert not validate_marginal_preservation(GOOD, GOOD * 2.0)


def test_marginal_preservation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expected stressed shape"):
        validate_marginal_preservation(GOOD, GOOD[:, :2])


def test_validate_transformation_accepts_consistent_triple() -> None:
    permutations = np.array([[2, 1, 0], [0, 2, 1]], dtype=np.int64)
    stressed = np.take_along_axis(GOOD, permutations, axis=1)

    assert validate_transformation(GOOD, stressed, permutations)


def test_validate_transformation_detects_wrong_permutation() -> None:
    """Marginals can be preserved while the recorded permutation lies."""
    honest = np.array([[2, 1, 0], [0, 2, 1]], dtype=np.int64)
    stressed = np.take_along_axis(GOOD, honest, axis=1)
    liar = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)

    assert validate_marginal_preservation(GOOD, stressed)
    assert not validate_transformation(GOOD, stressed, liar)
