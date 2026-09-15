# tests/dependence_stress/test_engine.py

"""Engine facade: ownership, RNG contract, delegation, thinness."""

from __future__ import annotations

import dataclasses
import inspect
import re

import numpy as np
import pytest

from riskquant.dependence_stress import engine as engine_module
from riskquant.dependence_stress.engine import DependenceStressEngine
from riskquant.dependence_stress.permutations import (
    aggregate_portfolio,
    reconstruct_stressed,
)
from riskquant.dependence_stress.tails import Comonotonic
from riskquant.dependence_stress.types import DependenceStressResult, FloatArray

FORBIDDEN = (
    "value_at_risk",
    "expected_shortfall",
    "calculate_var",
    "calculate_es",
    "calculate_csc",
    "confidence_level",
    "subportfolio",
    "capital",
    "pandas",
)


def stressed(base: FloatArray, result: DependenceStressResult) -> FloatArray:
    """
    Reconstruct the stressed K x S matrix for a result.

    The stressed matrix is no longer a result field. Tests that need
    row-level stressed observations reconstruct it explicitly, which is
    exactly what a production audit consumer does.
    """
    return reconstruct_stressed(base, result.permutations)


def _executable_source(path: str) -> str:
    """
    Return module source with comments and docstrings removed.

    Scope assertions must test what the code *does*, not what its prose
    says. The engine docstring legitimately contains the words
    "portfolio" and "default_rng" while promising never to do either;
    a raw substring scan would fail on its own disclaimer.
    """
    import ast

    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ):
            continue

        body = node.body

        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]

    return ast.unparse(ast.fix_missing_locations(tree)).lower()


# ======================================================================
# Ownership and immutability
# ======================================================================


def test_engine_owns_its_data(base: FloatArray) -> None:
    source = base.copy()
    stress_engine = DependenceStressEngine(data=source)

    source[0, 0] = -12345.0

    assert stress_engine.data[0, 0] == base[0, 0]


def test_engine_data_is_read_only(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    assert not stress_engine.data.flags.writeable

    with pytest.raises(ValueError):
        stress_engine.data[0, 0] = 0.0


def test_engine_is_frozen(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    with pytest.raises(dataclasses.FrozenInstanceError):
        stress_engine.data = base  # type: ignore[misc]


def test_engine_rejects_invalid_data() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        DependenceStressEngine(data=np.array([[1.0, np.nan], [1.0, 2.0]]))


def test_engine_equality_is_identity_based(base: FloatArray) -> None:
    first = DependenceStressEngine(data=base)
    second = DependenceStressEngine(data=base)

    assert first == first
    assert first != second
    assert isinstance(first == second, bool)


def test_shape_properties(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    assert stress_engine.shape == base.shape
    assert stress_engine.num_rows == base.shape[0]
    assert stress_engine.num_scenarios == base.shape[1]
    assert isinstance(stress_engine.num_rows, int)


# ======================================================================
# Result contract: two first-class outputs
# ======================================================================


def test_result_has_exactly_two_first_class_outputs(base: FloatArray) -> None:
    """permutations and portfolio_pnl; stressed is NOT a result field."""
    result = DependenceStressEngine(data=base).apply_comonotonic()

    assert hasattr(result, "permutations")
    assert hasattr(result, "portfolio_pnl")
    assert not hasattr(result, "stressed")

    assert {field.name for field in dataclasses.fields(result)} == {
        "permutations",
        "portfolio_pnl",
    }


def test_result_shapes_and_dtypes(base: FloatArray) -> None:
    k, s = base.shape
    result = DependenceStressEngine(data=base).apply_comonotonic()

    assert result.permutations.shape == (k, s)
    assert result.permutations.dtype == np.int32
    assert result.permutations.flags.c_contiguous

    assert result.portfolio_pnl.shape == (s,)
    assert result.portfolio_pnl.dtype == np.float64
    assert result.portfolio_pnl.flags.c_contiguous

    assert result.shape == (k, s)
    assert result.num_rows == k
    assert result.num_scenarios == s


def test_portfolio_pnl_is_a_flat_vector_not_a_row_matrix(
    base: FloatArray,
) -> None:
    """Shape is (S,), deliberately not (1, S)."""
    result = DependenceStressEngine(data=base).apply_comonotonic()

    assert result.portfolio_pnl.ndim == 1


def test_result_arrays_are_read_only(base: FloatArray) -> None:
    result = DependenceStressEngine(data=base).apply_comonotonic()

    assert not result.permutations.flags.writeable
    assert not result.portfolio_pnl.flags.writeable

    with pytest.raises((ValueError, TypeError)):
        result.permutations[0, 0] = 0

    with pytest.raises((ValueError, TypeError)):
        result.portfolio_pnl[0] = 0.0


def test_result_is_frozen(base: FloatArray) -> None:
    result = DependenceStressEngine(data=base).apply_comonotonic()

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.portfolio_pnl = base[0]  # type: ignore[misc]


def test_result_owns_its_buffers(base: FloatArray) -> None:
    """Constructing from writable arrays must copy, not alias."""
    permutations = np.ascontiguousarray(
        np.broadcast_to(
            np.arange(base.shape[1], dtype=np.int64),
            base.shape,
        )
    )
    portfolio = base.sum(axis=0).copy()

    result = DependenceStressResult(
        permutations=permutations,
        portfolio_pnl=portfolio,
    )

    permutations[0, 0] = 7
    portfolio[0] = -999.0

    assert result.permutations[0, 0] == 0
    assert result.portfolio_pnl[0] == base[:, 0].sum()
    assert permutations.flags.writeable
    assert portfolio.flags.writeable


def test_result_equality_is_identity_based(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    first = stress_engine.apply_comonotonic()
    second = stress_engine.apply_comonotonic()

    assert first == first
    assert first != second
    assert isinstance(first == second, bool)
    assert np.array_equal(first.permutations, second.permutations)


def test_result_rejects_inconsistent_shapes(base: FloatArray) -> None:
    permutations = np.ascontiguousarray(
        np.broadcast_to(
            np.arange(base.shape[1], dtype=np.int64),
            base.shape,
        )
    )

    with pytest.raises(ValueError, match="does not match the scenario count"):
        DependenceStressResult(
            permutations=permutations,
            portfolio_pnl=np.zeros(3),
        )

    with pytest.raises(ValueError, match="1-dimensional S-vector"):
        DependenceStressResult(
            permutations=permutations,
            portfolio_pnl=np.zeros((1, base.shape[1])),
        )

    with pytest.raises(ValueError, match="2-dimensional K x S"):
        DependenceStressResult(
            permutations=np.arange(4, dtype=np.int64),
            portfolio_pnl=np.zeros(4),
        )


# ======================================================================
# Portfolio aggregation
# ======================================================================


def test_portfolio_pnl_equals_reconstructed_scenario_sums(
    base: FloatArray,
) -> None:
    """
    The exact aggregation contract.

    Direct aggregation accumulates over k in the same order as NumPy's
    sum(axis=0) over a C-contiguous array, so the agreement is bitwise,
    not merely close. Asserting exact equality keeps that guarantee
    under test instead of assumed.
    """
    stress_engine = DependenceStressEngine(data=base)
    result = stress_engine.apply_comonotonic()

    reconstructed = stress_engine.stressed_matrix(result)

    assert np.array_equal(
        reconstructed.sum(axis=0),
        result.portfolio_pnl,
    )


@pytest.mark.parametrize(
    "k,s",
    [(2, 2), (3, 7), (8, 64), (17, 251), (64, 1000)],
)
def test_aggregation_is_bitwise_exact_across_shapes(k: int, s: int) -> None:
    """Wide magnitude spread, to expose any summation-order divergence."""
    rng = np.random.default_rng(k * 1000 + s)
    scale = 10.0 ** rng.integers(-8, 8, (k, 1))
    data = np.ascontiguousarray(rng.standard_normal((k, s)) * scale)

    stress_engine = DependenceStressEngine(data=data)
    result = stress_engine.apply_comonotonic()

    assert np.array_equal(
        stress_engine.stressed_matrix(result).sum(axis=0),
        result.portfolio_pnl,
    )


def test_aggregate_portfolio_public_entry_point(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)
    result = stress_engine.apply_comonotonic()

    assert np.array_equal(
        aggregate_portfolio(base, result.permutations),
        result.portfolio_pnl,
    )


def test_hand_verifiable_portfolio_aggregation() -> None:
    """
    K = 3, S = 5, verifiable with a calculator.

        A = [2, 5, 1, 7, 4]   permutation [1, 3, 0, 2, 4]
        B = [3, 4, 8, 2, 6]   permutation [3, 0, 2, 4, 1]
        C = [1, 6, 2, 5, 9]   permutation [2, 1, 3, 0, 4]

    Scenario 0: A[1]=5, B[3]=2, C[2]=2  ->  9
    Scenario 1: A[3]=7, B[0]=3, C[1]=6  ->  16
    Scenario 2: A[0]=2, B[2]=8, C[3]=5  ->  15
    Scenario 3: A[2]=1, B[4]=6, C[0]=1  ->  8
    Scenario 4: A[4]=4, B[1]=4, C[4]=9  ->  17
    """
    data = np.array([
        [2.0, 5.0, 1.0, 7.0, 4.0],
        [3.0, 4.0, 8.0, 2.0, 6.0],
        [1.0, 6.0, 2.0, 5.0, 9.0],
    ])
    permutations = np.array(
        [
            [1, 3, 0, 2, 4],
            [3, 0, 2, 4, 1],
            [2, 1, 3, 0, 4],
        ],
        dtype=np.int64,
    )

    expected = np.array([9.0, 16.0, 15.0, 8.0, 17.0])

    assert np.array_equal(aggregate_portfolio(data, permutations), expected)

    reconstructed = reconstruct_stressed(data, permutations)

    assert np.array_equal(reconstructed.sum(axis=0), expected)
    assert np.array_equal(
        np.sort(reconstructed, axis=1),
        np.sort(data, axis=1),
    )


def test_stressed_matrix_is_an_optional_derived_representation(
    base: FloatArray,
) -> None:
    stress_engine = DependenceStressEngine(data=base)
    result = stress_engine.apply_comonotonic()

    reconstructed = stress_engine.stressed_matrix(result)

    assert reconstructed.shape == base.shape
    assert np.array_equal(
        reconstructed,
        np.take_along_axis(base, result.permutations, axis=1),
    )
    assert np.array_equal(
        np.sort(reconstructed, axis=1),
        np.sort(base, axis=1),
    )


# ======================================================================
# RNG contract
# ======================================================================


def test_engine_has_no_rng_field(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    assert not hasattr(stress_engine, "rng")
    assert "rng" not in inspect.signature(DependenceStressEngine).parameters


def test_comonotonic_needs_no_rng(base: FloatArray) -> None:
    parameters = inspect.signature(DependenceStressEngine.apply_comonotonic).parameters

    assert "rng" not in parameters


def test_copula_methods_require_rng(
    base: FloatArray,
    equicorrelation: FloatArray,
) -> None:
    stress_engine = DependenceStressEngine(data=base)

    with pytest.raises(TypeError):
        stress_engine.apply_gaussian_copula(equicorrelation)  # type: ignore[call-arg]

    with pytest.raises(TypeError, match=re.escape("numpy.random.Generator")):
        stress_engine.apply_gaussian_copula(equicorrelation, rng=None)  # type: ignore[arg-type]


def test_engine_never_creates_a_default_generator() -> None:
    """No implicit np.random.default_rng() and no global random state."""
    text = _executable_source(engine_module.__file__)

    assert "default_rng(" not in text
    assert "np.random.seed" not in text
    assert "np.random.rand" not in text


def test_tail_alignment_rng_only_needed_in_random_mode(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    stress_engine.apply_lower_tail_alignment(0.05)

    with pytest.raises(TypeError, match=re.escape("numpy.random.Generator")):
        stress_engine.apply_lower_tail_alignment(0.05, tail_order="random")


# ======================================================================
# Delegation and audit
# ======================================================================


def test_all_methods_return_results(
    base: FloatArray,
    equicorrelation: FloatArray,
) -> None:
    stress_engine = DependenceStressEngine(data=base)

    results = [
        stress_engine.apply_comonotonic(),
        stress_engine.apply_lower_tail_alignment(0.05),
        stress_engine.apply_lower_tail_alignment(
            0.05,
            tail_order="random",
            rng=np.random.default_rng(0),
        ),
        stress_engine.apply_gaussian_copula(
            equicorrelation,
            rng=np.random.default_rng(0),
        ),
        stress_engine.apply_t_copula(
            equicorrelation,
            degrees_of_freedom=4.0,
            rng=np.random.default_rng(0),
        ),
        stress_engine.apply_transformer(Comonotonic()),
    ]

    for result in results:
        assert isinstance(result, DependenceStressResult)
        assert stress_engine.validate_result(result)


def test_apply_transformer_rejects_non_transformer(base: FloatArray) -> None:
    with pytest.raises(TypeError, match="DependenceTransformer"):
        DependenceStressEngine(data=base).apply_transformer(object())  # type: ignore[arg-type]


def test_audit_flag_is_off_by_default(base: FloatArray) -> None:
    for name in (
        "apply_comonotonic",
        "apply_gaussian_copula",
        "apply_t_copula",
        "apply_lower_tail_alignment",
        "apply_transformer",
    ):
        parameter = inspect.signature(getattr(DependenceStressEngine, name)).parameters[
            "audit"
        ]

        assert parameter.default is False


def test_engine_public_surface_is_transformation_only() -> None:
    """
    Phase 3: measurement is owned by riskquant.dependence. The engine exposes
    no correlation / tail-dependence methods; it is a transformation and
    orchestration facade only.
    """
    removed = (
        "pearson_correlation",
        "spearman_correlation",
        "empirical_lower_tail_cooccurrence",
        "empirical_upper_tail_cooccurrence",
    )
    for name in removed:
        assert not hasattr(DependenceStressEngine, name), name


def test_measurement_runs_through_the_canonical_estate_estimator(
    base: FloatArray,
) -> None:
    """
    A comonotonic stress has unit Spearman rank correlation, measured with the
    canonical riskquant.dependence estimator on the engine's compiled sample --
    not with any package-local statistic.
    """
    from canonical_measures import spearman_matrix

    stress_engine = DependenceStressEngine(data=base)
    comonotonic = stress_engine.stressed_matrix(stress_engine.apply_comonotonic())

    assert np.allclose(spearman_matrix(comonotonic), 1.0)


def test_marginal_preservation_helper(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)
    reconstructed = stress_engine.stressed_matrix(stress_engine.apply_comonotonic())

    assert stress_engine.validate_marginal_preservation(reconstructed)
    assert not stress_engine.validate_marginal_preservation(reconstructed * 2.0)


# ======================================================================
# Scope
# ======================================================================


def test_package_contains_no_downstream_risk_logic() -> None:
    """No VaR, ES, CSC, portfolio aggregation, or Pandas anywhere."""
    import pathlib

    package = pathlib.Path(engine_module.__file__).parent

    for path in package.glob("*.py"):
        text = _executable_source(str(path))

        for token in FORBIDDEN:
            assert token not in text, f"{path.name} references {token}"


def test_engine_is_thin() -> None:
    """The facade must not implement copula or sorting mathematics."""
    text = _executable_source(engine_module.__file__)

    for token in ("cholesky", "argsort", "rankdata", "take_along_axis"):
        assert token not in text
