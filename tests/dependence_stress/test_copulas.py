# tests/dependence_stress/test_copulas.py

"""Copula configuration ownership, contracts, and dependence direction."""

from __future__ import annotations

import re

import numpy as np
import pytest

from riskquant.dependence_stress.copulas import GaussianCopula, TCopula
from riskquant.dependence_stress.diagnostics import (
    empirical_lower_tail_cooccurrence,
    spearman_correlation,
)
from riskquant.dependence_stress.permutations import (
    reconstruct_stressed,
    transform_empirical_matrix,
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



def _equi(k: int, rho: float) -> FloatArray:
    matrix = np.full((k, k), rho, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _mean_off_diagonal(matrix: FloatArray) -> float:
    k = matrix.shape[0]
    return float((matrix.sum() - np.trace(matrix)) / (k * (k - 1)))


class _ChiSquareSpy(np.random.Generator):
    """
    Generator subclass that records chisquare call shapes.

    Generator attributes are read-only, so monkeypatching an instance is
    impossible. Subclassing keeps ``isinstance(rng, Generator)`` true and
    therefore still satisfies the package RNG contract.
    """

    def __init__(self, seed: int) -> None:
        super().__init__(np.random.PCG64(seed))
        self.sizes: list[object] = []

    def chisquare(self, df: float, size: object = None) -> np.ndarray:
        self.sizes.append(size)
        return super().chisquare(df, size)


class _DegenerateChiSquare(np.random.Generator):
    """Generator whose chi-square draw underflows to zero."""

    def __init__(self, seed: int) -> None:
        super().__init__(np.random.PCG64(seed))

    def chisquare(self, df: float, size: object = None) -> np.ndarray:
        return np.zeros(size, dtype=np.float64)


# ======================================================================
# Configuration ownership
# ======================================================================


def test_correlation_input_mutation_does_not_change_copula() -> None:
    source = _equi(4, 0.5)
    copula = GaussianCopula(latent_correlation=source)

    source[0, 1] = 0.99
    source[1, 0] = 0.99

    assert copula.latent_correlation[0, 1] == pytest.approx(0.5)


def test_configuration_arrays_are_read_only() -> None:
    copula = GaussianCopula(latent_correlation=_equi(3, 0.2))

    assert not copula.latent_correlation.flags.writeable
    assert not copula._cholesky.flags.writeable

    with pytest.raises(ValueError):
        copula.latent_correlation[0, 0] = 0.0


def test_construction_does_not_freeze_the_callers_array() -> None:
    source = _equi(3, 0.2)

    GaussianCopula(latent_correlation=source)

    assert source.flags.writeable


def test_cholesky_computed_once_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    base: FloatArray,
) -> None:
    calls = {"n": 0}
    real = np.linalg.cholesky

    def counting(matrix: FloatArray) -> FloatArray:
        calls["n"] += 1
        return real(matrix)

    monkeypatch.setattr(np.linalg, "cholesky", counting)

    copula = GaussianCopula(latent_correlation=_equi(base.shape[0], 0.4))

    assert calls["n"] == 1

    for seed in range(5):
        transform_empirical_matrix(base, copula, np.random.default_rng(seed))

    assert calls["n"] == 1


def test_cholesky_reproduces_the_correlation() -> None:
    correlation = _equi(5, 0.35)
    copula = GaussianCopula(latent_correlation=correlation)

    assert np.allclose(
        copula._cholesky @ copula._cholesky.T,
        correlation,
    )


def test_equality_is_identity_based() -> None:
    """eq=False: array-bearing dataclasses must not autogenerate __eq__."""
    first = GaussianCopula(latent_correlation=_equi(3, 0.2))
    second = GaussianCopula(latent_correlation=_equi(3, 0.2))

    assert first == first
    assert first != second
    assert isinstance(first == second, bool)


# ======================================================================
# Contracts
# ======================================================================


def test_dimension_mismatch_raises_explicit_error(base: FloatArray) -> None:
    copula = GaussianCopula(latent_correlation=_equi(3, 0.3))

    with pytest.raises(ValueError, match=r"does not match K=6"):
        transform_empirical_matrix(base, copula, np.random.default_rng(0))


@pytest.mark.parametrize("bad_rng", [None, 42, "rng", np.random.RandomState(0)])
def test_malformed_rng_raises_type_error(
    base: FloatArray,
    equicorrelation: FloatArray,
    bad_rng: object,
) -> None:
    copula = GaussianCopula(latent_correlation=equicorrelation)

    with pytest.raises(TypeError, match=re.escape("numpy.random.Generator")):
        copula.build_permutation(base, bad_rng)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_nu", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_degrees_of_freedom(bad_nu: float) -> None:
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        TCopula(latent_correlation=_equi(3, 0.2), degrees_of_freedom=bad_nu)


@pytest.mark.parametrize("nu", [0.1, 0.5, 1.0, 2.0, 5.0, 30.0, 100.0])
def test_t_copula_across_representative_nu(
    base: FloatArray,
    equicorrelation: FloatArray,
    nu: float,
) -> None:
    copula = TCopula(latent_correlation=equicorrelation, degrees_of_freedom=nu)

    result = transform_empirical_matrix(base, copula, np.random.default_rng(7))

    assert np.array_equal(
        np.sort(stressed(base, result), axis=1),
        np.sort(base, axis=1),
    )


def test_t_copula_draws_one_scale_per_scenario(
    base: FloatArray,
    equicorrelation: FloatArray,
) -> None:
    """The chi-square scale is shared across K. size must be S, not (K, S)."""
    generator = _ChiSquareSpy(0)

    copula = TCopula(latent_correlation=equicorrelation, degrees_of_freedom=4.0)
    copula.build_permutation(base, generator)

    assert generator.sizes == [base.shape[1]]


# ======================================================================
# Reproducibility and realization
# ======================================================================


def test_identical_seed_gives_identical_permutation(
    base: FloatArray,
    equicorrelation: FloatArray,
) -> None:
    copula = GaussianCopula(latent_correlation=equicorrelation)

    first = transform_empirical_matrix(base, copula, np.random.default_rng(11))
    second = transform_empirical_matrix(base, copula, np.random.default_rng(11))

    assert np.array_equal(first.permutations, second.permutations)
    assert np.array_equal(stressed(base, first), stressed(base, second))


def test_different_seeds_give_different_realizations(
    base: FloatArray,
    equicorrelation: FloatArray,
) -> None:
    """Monte Carlo realization noise is expected, not a defect."""
    copula = GaussianCopula(latent_correlation=equicorrelation)

    first = transform_empirical_matrix(base, copula, np.random.default_rng(1))
    second = transform_empirical_matrix(base, copula, np.random.default_rng(2))

    assert not np.array_equal(first.permutations, second.permutations)
    assert np.array_equal(
        np.sort(stressed(base, first), axis=1),
        np.sort(stressed(base, second), axis=1),
    )


# ======================================================================
# Dependence direction (statistical, deliberately not brittle)
# ======================================================================


def test_positive_latent_correlation_creates_positive_rank_dependence(
    base: FloatArray,
) -> None:
    copula = GaussianCopula(latent_correlation=_equi(base.shape[0], 0.8))

    result = transform_empirical_matrix(base, copula, np.random.default_rng(3))

    assert _mean_off_diagonal(spearman_correlation(stressed(base, result))) > 0.6


def test_negative_latent_correlation_creates_negative_rank_dependence() -> None:
    rng = np.random.default_rng(5)
    data = rng.standard_normal((2, 2000))

    copula = GaussianCopula(latent_correlation=_equi(2, -0.8))
    result = transform_empirical_matrix(data, copula, np.random.default_rng(4))

    assert _mean_off_diagonal(spearman_correlation(stressed(data, result))) < -0.6


def test_identity_latent_correlation_creates_no_systematic_structure(
    base: FloatArray,
) -> None:
    """Only finite-sample realization noise should remain."""
    copula = GaussianCopula(latent_correlation=np.eye(base.shape[0]))

    values = [
        _mean_off_diagonal(
            spearman_correlation(
                stressed(
                    base,
                    transform_empirical_matrix(
                        base,
                        copula,
                        np.random.default_rng(seed),
                    ),
                )
            )
        )
        for seed in range(20)
    ]

    assert abs(float(np.mean(values))) < 0.05


def test_lower_nu_increases_symmetric_tail_cooccurrence() -> None:
    """Averaged over seeds so the assertion is statistical, not brittle."""
    rng = np.random.default_rng(9)
    data = rng.standard_normal((4, 3000))
    correlation = _equi(4, 0.5)

    def mean_tail(nu: float) -> float:
        copula = TCopula(latent_correlation=correlation, degrees_of_freedom=nu)
        return float(
            np.mean(
                [
                    _mean_off_diagonal(
                        empirical_lower_tail_cooccurrence(
                            stressed(
                                data,
                                transform_empirical_matrix(
                                    data,
                                    copula,
                                    np.random.default_rng(seed),
                                ),
                            ),
                            0.05,
                        )
                    )
                    for seed in range(15)
                ]
            )
        )

    assert mean_tail(1.5) > mean_tail(100.0)


def test_pathological_draw_is_rejected_loudly(
    base: FloatArray,
    equicorrelation: FloatArray,
) -> None:
    """A degenerate chi-square scale must raise, not silently sort infinities."""
    generator = _DegenerateChiSquare(0)

    copula = TCopula(latent_correlation=equicorrelation, degrees_of_freedom=0.1)

    with pytest.raises(ValueError, match="degenerate"):
        copula.build_permutation(base, generator)
