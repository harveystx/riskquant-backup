# tests/dependence/test_conformance.py

"""
Contract-generated conformance suite.

Every guarantee declared in ``ESTIMATOR_CONTRACT`` generates the test that
proves it, so an estimator cannot claim a property it does not have.

Non-circularity is the design requirement. A generated test must not reproduce
the implementation's own logic:

* ``psd_status``      -> LAPACK's symmetric eigensolver, a code path wholly
                         independent of every estimator here.
* ``invariances``     -> apply the transformation, re-run, compare.
* ``diagonal_value``  -> duplicate a variable and inspect the **off-diagonal**
                         cell between the original and the duplicate. Asserting
                         ``diag(M) == contract.diagonal_value`` would merely
                         verify that a constant was copied from the contract to
                         the matrix, which carries no information about the
                         mathematics and would pass for an estimator whose
                         entire off-diagonal computation was wrong.
* ``value_range``     -> the mathematical bound, checked against output. Honest
                         but weak: it catches gross violations only.
* symmetry            -> ``array_equal(M, M.T)``, weak on its own, which is why
                         variable-permutation equivariance is also generated:
                         it catches transposed-index bugs that a
                         post-computation mirror conceals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from riskquant.constants import NUMERICAL_EXCURSION_TOLERANCE
from riskquant.dependence.contracts import DependenceResult
from riskquant.dependence.diagnostics.spectrum import spectrum
from riskquant.dependence.estimators.kendall import kendall_tau_b
from riskquant.dependence.estimators.pearson import pearson
from riskquant.dependence.estimators.spearman import spearman
from riskquant.dependence.taxonomy import (
    ESTIMATOR_CONTRACT,
    DependenceEstimator,
    Invariance,
    PsdStatus,
)
from riskquant.observations.compile import CompiledSample, RankRepresentation
from tests.dependence.conftest import (
    compile_both,
    continuous_sample,
    zero_inflated_sample,
)

Estimate = Callable[[CompiledSample, RankRepresentation], DependenceResult]

#: The single dispatch point tying an enum member to its callable. Adding an
#: estimator without registering it here fails ``test_every_estimator_is_covered``.
ESTIMATE: dict[DependenceEstimator, Estimate] = {
    DependenceEstimator.PEARSON: lambda sample, ranks: pearson(sample),
    DependenceEstimator.SPEARMAN: spearman,
    DependenceEstimator.KENDALL_TAU_B: kendall_tau_b,
}

ESTIMATORS = list(DependenceEstimator)

#: Exercised under both regimes: continuous data, and the tie-heavy data on
#: which rank conventions actually bite.
PANELS = {
    "continuous": continuous_sample(seed=11, n_variables=6, n_observations=240),
    "zero_inflated": zero_inflated_sample(seed=12, n_variables=6, n_observations=240),
}


def _run(estimator: DependenceEstimator, matrix: np.ndarray) -> DependenceResult:
    sample, ranks = compile_both(matrix)
    return ESTIMATE[estimator](sample, ranks)


def test_every_estimator_is_covered_by_the_generator() -> None:
    assert set(ESTIMATE) == set(ESTIMATOR_CONTRACT)


# ---------------------------------------------------------------- structural


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_layout_and_immutability(estimator: DependenceEstimator, panel: str) -> None:
    result = _run(estimator, PANELS[panel])
    size = PANELS[panel].shape[0]

    assert result.matrix.shape == (size, size)
    assert result.matrix.dtype == np.float64
    assert result.matrix.flags.c_contiguous
    assert not result.matrix.flags.writeable
    assert result.contract is ESTIMATOR_CONTRACT[estimator]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_symmetry_is_bitwise(estimator: DependenceEstimator, panel: str) -> None:
    """No tolerance: the mirror makes both halves the same bit pattern."""
    result = _run(estimator, PANELS[panel])
    assert np.array_equal(result.matrix, result.matrix.T)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_declared_value_range_is_respected(
    estimator: DependenceEstimator, panel: str
) -> None:
    lower, upper = ESTIMATOR_CONTRACT[estimator].value_range
    matrix = _run(estimator, PANELS[panel]).matrix
    assert matrix.min() >= lower
    assert matrix.max() <= upper


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_declared_diagonal_via_a_duplicated_variable(
    estimator: DependenceEstimator, panel: str
) -> None:
    """
    Non-circular diagonal oracle.

    A variable duplicated at a distinct index goes through the full pairwise
    computation path, so its off-diagonal cell must equal the declared diagonal.
    Checking ``diag(M)`` instead would only verify a copy.
    """
    matrix = np.ascontiguousarray(np.vstack([PANELS[panel], PANELS[panel][0]]))
    result = _run(estimator, matrix)
    expected = ESTIMATOR_CONTRACT[estimator].diagonal_value

    assert result.matrix[0, -1] == pytest.approx(
        expected, abs=NUMERICAL_EXCURSION_TOLERANCE
    )
    assert np.array_equal(
        np.diag(result.matrix), np.full(matrix.shape[0], expected)
    )


# ----------------------------------------------------------------------- PSD


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_declared_psd_status_holds_numerically(
    estimator: DependenceEstimator, panel: str
) -> None:
    """
    Oracle is LAPACK, independent of every estimator in this package.

    ``PSD_BY_CONSTRUCTION`` asserts both that the estimand is PSD in exact
    arithmetic and that the returned floats satisfy
    ``lambda_min >= -K * eps * lambda_max``. An estimator declaring
    ``PSD_NOT_GUARANTEED`` is permitted to fail that bound.
    """
    contract = ESTIMATOR_CONTRACT[estimator]
    diagnostics = spectrum(_run(estimator, PANELS[panel]))

    if contract.psd_status is PsdStatus.PSD_BY_CONSTRUCTION:
        assert diagnostics.satisfies_numerical_psd, (
            f"{estimator.name} declares PSD_BY_CONSTRUCTION but returned "
            f"lambda_min={diagnostics.minimum_eigenvalue!r} against a tolerance "
            f"of {diagnostics.psd_tolerance!r}."
        )


def test_a_non_psd_estimator_exists_so_the_psd_test_is_not_vacuous() -> None:
    """Guards against the PSD test passing because nothing is ever non-PSD."""
    statuses = {c.psd_status for c in ESTIMATOR_CONTRACT.values()}
    assert PsdStatus.PSD_BY_CONSTRUCTION in statuses
    assert PsdStatus.PSD_NOT_GUARANTEED in statuses


def test_kendall_is_exempt_from_the_numerical_psd_requirement() -> None:
    """
    ``PSD_NOT_GUARANTEED`` is a conservative declaration, not a hedge.

    Kendall tau-b is assembled pair by pair and has no Gram representation, so
    positive semi-definiteness is not available as a theorem. Declaring
    ``PSD_BY_CONSTRUCTION`` would require a proof this package does not have,
    and the conservative declaration is the correct one.

    No indefinite tau-b matrix was found by a bounded search of this engine's
    own output: a hill-climb on the minimum eigenvalue over untied permutation
    panels at ``K`` in ``{4, 5, 6}`` with 25 restarts each reached only
    ``-8.2e-16``, which is rounding. That is a statement about the search
    budget, not about the mathematics, so it is recorded here rather than
    encoded as a skipping test that reports nothing in CI.

    What is asserted is the structural consequence: the conformance suite
    imposes the numerical PSD bound only where it was declared, so a future
    tau-b matrix that does violate it fails no test it was never promised to
    pass.
    """
    contract = ESTIMATOR_CONTRACT[DependenceEstimator.KENDALL_TAU_B]
    assert contract.psd_status is PsdStatus.PSD_NOT_GUARANTEED

    diagnostics = spectrum(_run(DependenceEstimator.KENDALL_TAU_B, PANELS["continuous"]))
    assert np.isfinite(diagnostics.minimum_eigenvalue)
    assert diagnostics.psd_tolerance > 0.0


# ---------------------------------------------------------------- invariances


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_joint_permutation_invariance(
    estimator: DependenceEstimator, panel: str
) -> None:
    if Invariance.JOINT_PERMUTATION not in ESTIMATOR_CONTRACT[estimator].invariances:
        pytest.skip("not declared")

    matrix = PANELS[panel]
    order = np.random.default_rng(3).permutation(matrix.shape[1])

    np.testing.assert_allclose(
        _run(estimator, np.ascontiguousarray(matrix[:, order])).matrix,
        _run(estimator, matrix).matrix,
        atol=NUMERICAL_EXCURSION_TOLERANCE,
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_variable_permutation_equivariance(
    estimator: DependenceEstimator, panel: str
) -> None:
    """
    Catches transposed-index bugs that a post-computation mirror conceals.

    Asserted within the numerical tolerance, not bitwise. Measured deviation for
    GEMM-derived estimators is 4.2e-17 at K=13, S=977: BLAS blocking depends on
    the matrix dimensions, so reordering rows can change the accumulation order
    inside a dot product. Equivariance is the mathematical claim; bitwise
    equality is not a mathematical property and is not claimed.
    """
    if (
        Invariance.VARIABLE_PERMUTATION_EQUIVARIANT
        not in ESTIMATOR_CONTRACT[estimator].invariances
    ):
        pytest.skip("not declared")

    matrix = PANELS[panel]
    order = np.random.default_rng(4).permutation(matrix.shape[0])

    permuted = _run(estimator, np.ascontiguousarray(matrix[order])).matrix
    expected = _run(estimator, matrix).matrix[np.ix_(order, order)]

    np.testing.assert_allclose(
        permuted, expected, atol=NUMERICAL_EXCURSION_TOLERANCE
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_sign_equivariance_under_reflection_is_bitwise(
    estimator: DependenceEstimator, panel: str
) -> None:
    """
    Bitwise: IEEE negation is exact, and the sum of negated terms is the exact
    negation of the sum in any accumulation order.
    """
    if (
        Invariance.SIGN_EQUIVARIANT_UNDER_REFLECTION
        not in ESTIMATOR_CONTRACT[estimator].invariances
    ):
        pytest.skip("not declared")

    matrix = PANELS[panel]
    reflected = matrix.copy()
    reflected[0] *= -1.0

    base = _run(estimator, matrix).matrix
    expected = base.copy()
    expected[0, 1:] *= -1.0
    expected[1:, 0] *= -1.0

    assert np.array_equal(_run(estimator, reflected).matrix, expected)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_positive_affine_marginal_invariance(
    estimator: DependenceEstimator, panel: str
) -> None:
    if (
        Invariance.POSITIVE_AFFINE_MARGINAL
        not in ESTIMATOR_CONTRACT[estimator].invariances
    ):
        pytest.skip("not declared")

    matrix = PANELS[panel]
    transformed = matrix.copy()
    transformed[0] = transformed[0] * 1e4 + 7.5
    transformed[1] = transformed[1] * 1e-4 - 3.0

    np.testing.assert_allclose(
        _run(estimator, transformed).matrix,
        _run(estimator, matrix).matrix,
        atol=1e-10,
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("panel", PANELS)
def test_strictly_increasing_marginal_invariance_is_bitwise(
    estimator: DependenceEstimator, panel: str
) -> None:
    """
    ``arcsinh`` is strictly increasing and does not underflow, so it stays
    injective in float64 even at denormal magnitudes -- unlike ``expm1(x/1e3)``,
    which merges distinct denormals into a tie.
    """
    if (
        Invariance.STRICTLY_INCREASING_MARGINAL
        not in ESTIMATOR_CONTRACT[estimator].invariances
    ):
        pytest.skip("not declared")

    matrix = PANELS[panel]
    transformed = matrix.copy()
    transformed[0] = np.arcsinh(transformed[0])
    transformed[1] = transformed[1] ** 3

    assert np.array_equal(
        _run(estimator, transformed).matrix, _run(estimator, matrix).matrix
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_estimators_never_mutate_the_compiled_sample(
    estimator: DependenceEstimator,
) -> None:
    matrix = PANELS["continuous"]
    sample, ranks = compile_both(matrix)

    observations_before = sample.observations.copy()
    ranks_before = ranks.ranks.copy()

    ESTIMATE[estimator](sample, ranks)

    assert np.array_equal(sample.observations, observations_before)
    assert np.array_equal(ranks.ranks, ranks_before)
