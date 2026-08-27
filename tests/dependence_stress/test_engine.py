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
from riskquant.dependence_stress.tails import Comonotonic
from riskquant.dependence_stress.types import DependenceStressResult, FloatArray

FORBIDDEN = (
    "value_at_risk",
    "expected_shortfall",
    "calculate_var",
    "calculate_es",
    "calculate_csc",
    "portfolio",
    "capital",
    "pandas",
)


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
    with pytest.raises(ValueError, match="NaN or infinite"):
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
# Result immutability
# ======================================================================


def test_result_arrays_are_read_only(base: FloatArray) -> None:
    result = DependenceStressEngine(data=base).apply_comonotonic()

    assert not result.stressed.flags.writeable
    assert not result.permutations.flags.writeable

    with pytest.raises(ValueError):
        result.stressed[0, 0] = 0.0

    with pytest.raises(ValueError):
        result.permutations[0, 0] = 0


def test_result_is_frozen(base: FloatArray) -> None:
    result = DependenceStressEngine(data=base).apply_comonotonic()

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.stressed = base  # type: ignore[misc]


def test_result_owns_its_buffers(base: FloatArray) -> None:
    """Constructing from a writable array must copy, not alias."""
    stressed = base.copy()
    permutations = np.broadcast_to(
        np.arange(base.shape[1], dtype=np.int64),
        base.shape,
    ).copy()

    result = DependenceStressResult(
        stressed=stressed,
        permutations=permutations,
    )

    stressed[0, 0] = -999.0

    assert result.stressed[0, 0] == base[0, 0]
    assert stressed.flags.writeable


def test_result_dtypes_and_layout(base: FloatArray) -> None:
    result = DependenceStressEngine(data=base).apply_comonotonic()

    assert result.stressed.dtype == np.float64
    assert result.permutations.dtype == np.int64
    assert result.stressed.flags.c_contiguous
    assert result.permutations.flags.c_contiguous
    assert result.shape == base.shape


def test_result_equality_is_identity_based(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    first = stress_engine.apply_comonotonic()
    second = stress_engine.apply_comonotonic()

    assert first == first
    assert first != second
    assert isinstance(first == second, bool)
    assert np.array_equal(first.stressed, second.stressed)


def test_result_rejects_shape_mismatch(base: FloatArray) -> None:
    with pytest.raises(ValueError, match="does not match"):
        DependenceStressResult(
            stressed=base,
            permutations=np.zeros((base.shape[0], 3), dtype=np.int64),
        )


# ======================================================================
# RNG contract
# ======================================================================


def test_engine_has_no_rng_field(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)

    assert not hasattr(stress_engine, "rng")
    assert "rng" not in inspect.signature(DependenceStressEngine).parameters


def test_comonotonic_needs_no_rng(base: FloatArray) -> None:
    parameters = inspect.signature(
        DependenceStressEngine.apply_comonotonic
    ).parameters

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
        parameter = inspect.signature(
            getattr(DependenceStressEngine, name)
        ).parameters["audit"]

        assert parameter.default is False


def test_diagnostics_delegate(base: FloatArray) -> None:
    from riskquant.dependence_stress import diagnostics as diag

    stress_engine = DependenceStressEngine(data=base)

    assert np.array_equal(
        stress_engine.pearson_correlation(),
        diag.pearson_correlation(base),
    )
    assert np.array_equal(
        stress_engine.spearman_correlation(),
        diag.spearman_correlation(base),
    )
    assert np.array_equal(
        stress_engine.empirical_lower_tail_cooccurrence(threshold=0.05),
        diag.empirical_lower_tail_cooccurrence(base, 0.05),
    )


def test_diagnostics_accept_an_explicit_matrix(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)
    stressed = stress_engine.apply_comonotonic().stressed

    assert np.allclose(stress_engine.spearman_correlation(stressed), 1.0)


def test_marginal_preservation_helper(base: FloatArray) -> None:
    stress_engine = DependenceStressEngine(data=base)
    result = stress_engine.apply_comonotonic()

    assert stress_engine.validate_marginal_preservation(result.stressed)
    assert not stress_engine.validate_marginal_preservation(result.stressed * 2.0)


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
