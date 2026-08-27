# tests/dependence_stress/test_invariants.py

"""
Central invariant suite.

Every transformer runs through the same fundamental contract. A new
transformer is added to ``_transformers`` and inherits the whole suite.

The one invariant that must never break:

    sort(T(X)[k, :]) == sort(X[k, :])   exactly, for every row k
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.dependence_stress.copulas import GaussianCopula, TCopula
from riskquant.dependence_stress.permutations import (
    DependenceTransformer,
    transform_empirical_matrix,
)
from riskquant.dependence_stress.tails import Comonotonic, LowerTailAlignment
from riskquant.dependence_stress.types import FloatArray
from riskquant.dependence_stress.validation import (
    validate_marginal_preservation,
    validate_permutation_matrix,
    validate_transformation,
)

SEED = 20260827

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import array_shapes, arrays  # noqa: E402


def _transformers(k: int) -> list[tuple[str, DependenceTransformer]]:
    correlation = np.full((k, k), 0.55, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)

    return [
        ("comonotonic", Comonotonic()),
        ("gaussian_copula", GaussianCopula(latent_correlation=correlation)),
        (
            "t_copula",
            TCopula(latent_correlation=correlation, degrees_of_freedom=3.0),
        ),
        ("tail_comonotonic", LowerTailAlignment(tail_probability=0.05)),
        (
            "tail_comonotonic_full",
            LowerTailAlignment(
                tail_probability=0.05,
                preserve_non_tail_order=False,
            ),
        ),
        (
            "tail_random",
            LowerTailAlignment(tail_probability=0.05, tail_order="random"),
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

    assert result.stressed.shape == base.shape
    assert result.permutations.shape == base.shape
    assert result.stressed.dtype == np.float64
    assert result.permutations.dtype == np.int64
    assert result.stressed.flags.c_contiguous
    assert result.permutations.flags.c_contiguous

    validate_permutation_matrix(
        result.permutations,
        expected_shape=base.shape,
    )

    assert validate_marginal_preservation(base, result.stressed)

    assert np.array_equal(
        result.stressed,
        np.take_along_axis(base, result.permutations, axis=1),
    )

    assert validate_transformation(
        base,
        result.stressed,
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
            np.sort(result.stressed[row]),
            np.sort(base[row]),
        )


@pytest.mark.parametrize("transformer", CASES, ids=IDS)
def test_results_are_immutable(
    transformer: DependenceTransformer,
    base: FloatArray,
) -> None:
    result = _run(transformer, base)

    assert not result.stressed.flags.writeable
    assert not result.permutations.flags.writeable


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
    assert np.array_equal(first.stressed, second.stressed)


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
        np.sort(first.stressed, axis=1),
        np.sort(second.stressed, axis=1),
    )


# ======================================================================
# Structural properties
# ======================================================================


def test_comonotonic_idempotence(base: FloatArray) -> None:
    once = transform_empirical_matrix(base, Comonotonic()).stressed
    twice = transform_empirical_matrix(once, Comonotonic()).stressed

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

    assert result.stressed.shape == base.shape


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
            np.sort(result.stressed, axis=1),
            np.sort(data, axis=1),
        )

        assert np.array_equal(
            result.stressed,
            np.take_along_axis(data, result.permutations, axis=1),
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
    once = transform_empirical_matrix(data, Comonotonic()).stressed
    twice = transform_empirical_matrix(once, Comonotonic()).stressed

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
        assert set(result.stressed[row].tolist()) <= set(data[row].tolist())
