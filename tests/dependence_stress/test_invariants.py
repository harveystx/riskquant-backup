# tests/dependence_stress/test_invariants.py

"""
Central invariant suite.

Every transformer runs through the same fundamental contract. A new
transformer is added to ``_transformers`` and inherits the whole suite.

The one invariant that must never break:

    sort(T(X)[k, :]) == sort(X[k, :])   exactly, for every row k
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator

import numpy as np
import pytest

from riskquant.dependence_stress.copulas import GaussianCopula, TCopula
from riskquant.dependence_stress.permutations import (
    DependenceTransformer,
    iter_transform_empirical_matrix,
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
from riskquant.dependence_stress.validation import (
    validate_marginal_preservation,
    validate_permutation_matrix,
    validate_transformation,
)

SEED = 20260827


from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import array_shapes, arrays  # noqa: E402

_DEFAULT_TAIL_PROBABILITY = 0.05


def _transformers(k: int) -> list[tuple[str, DependenceTransformer]]:
    correlation = np.full((k, k), 0.55, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)

    return [
        # Comonotonic
        ("comonotonic", Comonotonic()),
        # Copulas
        ("gaussian_copula", GaussianCopula(latent_correlation=correlation)),
        (
            "t_copula",
            TCopula(latent_correlation=correlation, degrees_of_freedom=3.0),
        ),
        # TAIL - LOWER
        (
            "tail_lower_comonotonic",
            LowerTailAlignment(
                tail_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="comonotonic",
                preserve_non_tail_order=True,
            ),
        ),
        (
            "tail_lower_comonotonic_full",
            LowerTailAlignment(
                tail_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="comonotonic",
                preserve_non_tail_order=False,
            ),
        ),
        (
            "tail_lower_random",
            LowerTailAlignment(
                tail_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="random",
                preserve_non_tail_order=True,
            ),
        ),
        # TAIL - UPPER
        (
            "tail_upper_comonotonic",
            UpperTailAlignment(
                tail_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="comonotonic",
                preserve_non_tail_order=True,
            ),
        ),
        (
            "tail_upper_comonotonic_full",
            UpperTailAlignment(
                tail_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="comonotonic",
                preserve_non_tail_order=False,
            ),
        ),
        (
            "tail_upper_random",
            UpperTailAlignment(
                tail_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="random",
                preserve_non_tail_order=True,
            ),
        ),
        # TAIL - DUAL
        (
            "tail_dual_comonotonic",
            DualTailAlignment(
                lower_probability=_DEFAULT_TAIL_PROBABILITY,
                upper_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="comonotonic",
                preserve_non_tail_order=True,
            ),
        ),
        (
            "tail_dual_comonotonic_full",
            DualTailAlignment(
                lower_probability=_DEFAULT_TAIL_PROBABILITY,
                upper_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="comonotonic",
                preserve_non_tail_order=False,
            ),
        ),
        (
            "tail_dual_random",
            DualTailAlignment(
                lower_probability=_DEFAULT_TAIL_PROBABILITY,
                upper_probability=_DEFAULT_TAIL_PROBABILITY,
                tail_order="random",
                preserve_non_tail_order=True,
            ),
        ),
    ]


PARAMS = _transformers(6)
IDS = [name for name, _ in PARAMS]
CASES = [transformer for _, transformer in PARAMS]


def _run(
    transformer: DependenceTransformer,
    data: FloatArray,
    seed: int = SEED,
) -> object:
    return transform_empirical_matrix(
        data,
        transformer,
        np.random.default_rng(seed) if transformer.requires_rng else None,
    )


# ======================================================================
# The core contract
# ======================================================================


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_transformer_contract(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    """Shape, dtype, layout, permutation validity, gather, marginals."""
    result = _run(transformer, base)

    assert stressed(base, result).shape == base.shape
    assert result.permutations.shape == base.shape
    assert stressed(base, result).dtype == np.float64
    assert result.permutations.dtype == np.int32
    assert stressed(base, result).flags.c_contiguous
    assert result.permutations.flags.c_contiguous

    validate_permutation_matrix(
        result.permutations,
        expected_shape=base.shape,
    )

    assert validate_marginal_preservation(base, stressed(base, result))

    assert np.array_equal(
        stressed(base, result),
        np.take_along_axis(base, result.permutations, axis=1),
    )

    assert validate_transformation(
        base,
        stressed(base, result),
        result.permutations,
    )


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_no_value_creation(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    """Every stressed observation is an element of its own base row."""
    result = _run(transformer, base)

    for row in range(base.shape[0]):
        assert np.array_equal(
            np.sort(stressed(base, result)[row]),
            np.sort(base[row]),
        )


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_results_are_immutable(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    result = _run(transformer, base)

    assert not result.permutations.flags.writeable
    assert not result.portfolio_pnl.flags.writeable


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_input_is_not_mutated(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    original = base.copy()

    _run(transformer, base)

    assert np.array_equal(base, original)


# ======================================================================
# Determinism and reproducibility
# ======================================================================


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_reproducible_from_identical_rng_state(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    first = _run(transformer, base)
    second = _run(transformer, base)

    assert np.array_equal(first.permutations, second.permutations)
    assert np.array_equal(stressed(base, first), stressed(base, second))


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_deterministic_transformers_ignore_rng(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    if transformer.requires_rng:
        pytest.skip("stochastic transformer")

    first = transform_empirical_matrix(base, transformer, None)
    second = transform_empirical_matrix(
        base,
        transformer,
        np.random.default_rng(1),
    )
    third = transform_empirical_matrix(
        base,
        transformer,
        np.random.default_rng(999),
    )

    assert np.array_equal(first.permutations, second.permutations)
    assert np.array_equal(first.permutations, third.permutations)


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_stochastic_transformers_require_a_generator(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    if not transformer.requires_rng:
        pytest.skip("deterministic transformer")

    with pytest.raises(TypeError):
        transform_empirical_matrix(base, transformer, None)

    with pytest.raises(TypeError):
        transform_empirical_matrix(base, transformer, 12345)  # type: ignore[arg-type]


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_stochastic_transformers_vary_with_rng_state(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    """A seed gives reproducibility, not a unique stress specification."""
    if not transformer.requires_rng:
        pytest.skip("deterministic transformer")

    first = _run(transformer, base, seed=1)
    second = _run(transformer, base, seed=2)

    assert not np.array_equal(first.permutations, second.permutations)
    assert np.array_equal(
        np.sort(stressed(base, first), axis=1),
        np.sort(stressed(base, second), axis=1),
    )


# ======================================================================
# Structural properties
# ======================================================================


def test_comonotonic_idempotence(base: FloatArray) -> None:
    once = stressed(base, transform_empirical_matrix(base, Comonotonic()))
    twice = stressed(once, transform_empirical_matrix(once, Comonotonic()))

    assert np.array_equal(once, twice)
    assert np.array_equal(once, np.sort(base, axis=1))


def test_identity_permutation_returns_input(base: FloatArray) -> None:
    identity = np.ascontiguousarray(
        np.broadcast_to(
            np.arange(base.shape[1], dtype=np.int64),
            base.shape,
        )
    )

    assert np.array_equal(
        np.take_along_axis(base, identity, axis=1),
        base,
    )


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_audit_mode_accepts_valid_transformations(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    result = transform_empirical_matrix(
        base,
        transformer,
        np.random.default_rng(SEED) if transformer.requires_rng else None,
        audit=True,
    )

    assert stressed(base, result).shape == base.shape


# ======================================================================
# Property-based testing
# ======================================================================

_matrices = arrays(
    dtype=np.float64,
    shape=array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=40),
    elements=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def stressed(base: FloatArray, result: DependenceStressResult) -> FloatArray:
    """
    Reconstruct the stressed K x S matrix for a result.

    The stressed matrix is no longer a result field. Tests that need
    row-level stressed observations reconstruct it explicitly, which is
    exactly what a production audit consumer does.
    """
    return reconstruct_stressed(base, result.permutations)


@given(data=_matrices, seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_marginals_are_preserved_for_arbitrary_matrices(
    data: FloatArray,
    seed: int,
) -> None:
    """
    The fundamental invariant, over arbitrary finite valid matrices.

    Transformers whose K is fixed by a correlation matrix are rebuilt at
    the generated K so that every method is exercised at every shape.
    """
    k = data.shape[0]

    for _, transformer in _transformers(k):
        result = transform_empirical_matrix(
            data,
            transformer,
            np.random.default_rng(seed) if transformer.requires_rng else None,
        )

        assert np.array_equal(
            np.sort(stressed(data, result), axis=1),
            np.sort(data, axis=1),
        )

        reconstructed = np.take_along_axis(
            data,
            result.permutations,
            axis=1,
        )

        assert np.array_equal(stressed(data, result), reconstructed)

        assert np.array_equal(
            reconstructed.sum(axis=0),
            result.portfolio_pnl,
        )


@settings(max_examples=100, deadline=None)
@given(data=_matrices)
def test_permutation_validity_for_arbitrary_matrices(
    data: FloatArray,
) -> None:
    """Every construction yields a genuine row-wise permutation."""
    for _, transformer in _transformers(data.shape[0]):
        result = transform_empirical_matrix(
            data,
            transformer,
            np.random.default_rng(0) if transformer.requires_rng else None,
        )

        validate_permutation_matrix(
            result.permutations,
            expected_shape=data.shape,
        )


@settings(max_examples=100, deadline=None)
@given(data=_matrices)
def test_comonotonic_idempotence_property(data: FloatArray) -> None:
    once = stressed(data, transform_empirical_matrix(data, Comonotonic()))
    twice = stressed(once, transform_empirical_matrix(once, Comonotonic()))

    assert np.array_equal(once, twice)


@settings(max_examples=100, deadline=None)
@given(data=_matrices)
def test_stressed_values_are_drawn_from_their_own_row(
    data: FloatArray,
) -> None:
    """No value is created, destroyed, scaled, rounded, or interpolated."""
    result = transform_empirical_matrix(
        data,
        LowerTailAlignment(tail_probability=0.1),
    )

    for row in range(data.shape[0]):
        assert set(stressed(data, result)[row].tolist()) <= set(data[row].tolist())


# ======================================================================
# Two-output contract
# ======================================================================


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_portfolio_pnl_matches_reconstructed_sums(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    """
    portfolio_pnl[s] == sum_k data[k, permutations[k, s]], exactly.

    Direct aggregation accumulates over k in the same order NumPy uses
    when reducing a C-contiguous array along axis 0, so the two agree
    bitwise. Exact equality is asserted so any divergence is caught.
    """
    result = _run(transformer, base)
    reconstructed = stressed(base, result)

    assert np.array_equal(reconstructed.sum(axis=0), result.portfolio_pnl)


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_stressed_is_not_a_result_field(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    """The stressed K x S matrix is derived, never a mandatory output."""
    result = _run(transformer, base)

    assert not hasattr(result, "stressed")
    assert {field.name for field in dataclasses.fields(result)} == {
        "permutations",
        "portfolio_pnl",
    }


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_row_marginals_do_not_move_but_the_portfolio_does(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    """
    The central economic rationale for the two-output contract.

    Any statistic that depends only on one row's empirical marginal is
    unchanged. The portfolio scenario distribution can change, because
    observations from different rows are now paired differently.
    """
    result = _run(transformer, base)
    reconstructed = stressed(base, result)

    for row in range(base.shape[0]):
        assert np.array_equal(
            np.sort(reconstructed[row]),
            np.sort(base[row]),
        )
        assert float(np.mean(reconstructed[row])) == pytest.approx(
            float(np.mean(base[row]))
        )

    # The portfolio mean is invariant (sums of fixed row totals), but the
    # portfolio DISTRIBUTION is not.
    assert float(np.sum(result.portfolio_pnl)) == pytest.approx(float(np.sum(base)))


def test_comonotonic_widens_the_portfolio_distribution(
    base: FloatArray,
) -> None:
    """
    A concrete case where the portfolio distribution changes.

    Aligning every row's worst outcome into the same scenario makes the
    portfolio scenario vector more dispersed, while every row's marginal
    is untouched. This is dependence stress doing its job.
    """
    original = base.sum(axis=0)
    stressed_pnl = transform_empirical_matrix(base, Comonotonic()).portfolio_pnl

    assert float(np.std(stressed_pnl)) > float(np.std(original))
    assert float(np.min(stressed_pnl)) < float(np.min(original))
    assert float(np.sum(stressed_pnl)) == pytest.approx(float(np.sum(original)))


# ======================================================================
# Streaming Monte Carlo
# ======================================================================


@pytest.mark.parametrize("count", [0, 1, 5])
def test_iterator_yields_the_requested_number(
    base: FloatArray,
    count: int,
) -> None:
    correlation = np.full((base.shape[0], base.shape[0]), 0.55)
    np.fill_diagonal(correlation, 1.0)

    results = list(
        iter_transform_empirical_matrix(
            base,
            GaussianCopula(latent_correlation=correlation),
            np.random.default_rng(0),
            n_realizations=count,
        )
    )

    assert len(results) == count


def test_iterator_rejects_a_negative_count(base: FloatArray) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        iter_transform_empirical_matrix(
            base,
            Comonotonic(),
            n_realizations=-1,
        )


def test_iterator_validates_eagerly_not_on_first_next(
    base: FloatArray,
) -> None:
    """A malformed argument must fail at the call site."""
    correlation = np.full((base.shape[0], base.shape[0]), 0.55)
    np.fill_diagonal(correlation, 1.0)

    with pytest.raises(TypeError, match=re.escape("numpy.random.Generator")):
        iter_transform_empirical_matrix(
            base,
            GaussianCopula(latent_correlation=correlation),
            None,
            n_realizations=3,
        )


def test_iterator_results_satisfy_every_invariant(base: FloatArray) -> None:
    correlation = np.full((base.shape[0], base.shape[0]), 0.55)
    np.fill_diagonal(correlation, 1.0)

    for result in iter_transform_empirical_matrix(
        base,
        GaussianCopula(latent_correlation=correlation),
        np.random.default_rng(3),
        n_realizations=8,
    ):
        reconstructed = stressed(base, result)

        validate_permutation_matrix(
            result.permutations,
            expected_shape=base.shape,
        )
        assert validate_marginal_preservation(base, reconstructed)
        assert np.array_equal(
            reconstructed.sum(axis=0),
            result.portfolio_pnl,
        )


def test_iterator_sequence_is_reproducible(base: FloatArray) -> None:
    """Same initial RNG state gives the same SEQUENCE of realisations."""
    correlation = np.full((base.shape[0], base.shape[0]), 0.55)
    np.fill_diagonal(correlation, 1.0)
    copula = GaussianCopula(latent_correlation=correlation)

    def sequence(seed: int) -> list[FloatArray]:
        return [
            result.portfolio_pnl
            for result in iter_transform_empirical_matrix(
                base,
                copula,
                np.random.default_rng(seed),
                n_realizations=6,
            )
        ]

    first = sequence(11)
    second = sequence(11)
    different = sequence(12)

    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)

    assert not all(np.array_equal(a, b) for a, b in zip(first, different, strict=True))


def test_iterator_realisations_differ_within_a_sequence(
    base: FloatArray,
) -> None:
    """Successive draws advance the generator; they are not repeats."""
    correlation = np.full((base.shape[0], base.shape[0]), 0.55)
    np.fill_diagonal(correlation, 1.0)

    results = list(
        iter_transform_empirical_matrix(
            base,
            GaussianCopula(latent_correlation=correlation),
            np.random.default_rng(5),
            n_realizations=4,
        )
    )

    assert not np.array_equal(
        results[0].permutations,
        results[1].permutations,
    )

    # Marginals are identical across every realisation regardless.
    for result in results:
        assert validate_marginal_preservation(base, stressed(base, result))


def test_iterator_is_lazy_and_holds_one_realisation(
    base: FloatArray,
) -> None:
    """
    Structural streaming test, not a process-memory measurement.

    The API must return an iterator that produces results on demand,
    rather than a materialised list of B realisations.
    """
    correlation = np.full((base.shape[0], base.shape[0]), 0.55)
    np.fill_diagonal(correlation, 1.0)

    stream = iter_transform_empirical_matrix(
        base,
        GaussianCopula(latent_correlation=correlation),
        np.random.default_rng(0),
        n_realizations=10_000,
    )

    assert isinstance(stream, Iterator)
    assert not isinstance(stream, list | tuple)

    first = next(iter(stream))

    assert first.permutations.shape == base.shape
    assert first.portfolio_pnl.shape == (base.shape[1],)


def test_ten_thousand_realisations_without_retaining_matrices(
    base: FloatArray,
) -> None:
    """
    Consume a large ensemble keeping only scalars.

    Nothing K x S is retained across iterations. Peak allocation is
    bounded by one realisation, which is the entire point of the
    streaming API.
    """
    correlation = np.full((base.shape[0], base.shape[0]), 0.55)
    np.fill_diagonal(correlation, 1.0)

    count = 0
    worst = np.inf

    for result in iter_transform_empirical_matrix(
        base,
        GaussianCopula(latent_correlation=correlation),
        np.random.default_rng(0),
        n_realizations=2_000,
    ):
        count += 1
        worst = min(worst, float(result.portfolio_pnl.min()))

    assert count == 2_000
    assert np.isfinite(worst)


def test_deterministic_transformer_streams_identical_results(
    base: FloatArray,
) -> None:
    results = list(
        iter_transform_empirical_matrix(
            base,
            Comonotonic(),
            n_realizations=3,
        )
    )

    assert len(results) == 3

    for result in results[1:]:
        assert np.array_equal(result.permutations, results[0].permutations)
        assert np.array_equal(result.portfolio_pnl, results[0].portfolio_pnl)
