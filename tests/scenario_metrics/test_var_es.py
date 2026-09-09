# tests/scenario_metrics/test_var_es.py

"""
Production regression suite for riskquant.scenario_metrics.

Test philosophy
---------------
The suite protects four contracts:

1. Mathematical correctness
   - empirical lower-tail VaR
   - exact fractional-tail ES
   - integral-tail handling
   - floating-point integer-boundary handling

2. Numerical / structural invariants
   - translation invariance
   - positive homogeneity
   - monotonicity
   - scenario permutation invariance
   - ES <= VaR for lower-tail P&L

3. API / memory contracts
   - finite float64 input
   - C-contiguous input
   - read-only input accepted
   - input never mutated
   - read-only output arrays
   - correct output dtype / shape

4. Compiler / kernel seam
   - exact tail-parameter compilation
   - multiple groups
   - ties
   - pathological ordering
   - randomized agreement with an independent reference

The production implementation uses quickselect rather than a full sort.
The reference implementation intentionally uses a full sort so the two paths
are algorithmically independent.

Numerical-testing convention
----------------------------
The production implementation is expected to use IEEE-754 float64 arithmetic.
Tests therefore do not demand impossible decimal exactness from ordinary
floating-point operations.

Where a result is mathematically exact but passes through floating-point
arithmetic, the tolerance is deliberately tied to float64 machine precision
rather than being an arbitrary large decimal tolerance.

This is especially important for:
    - fractional tail-mass arithmetic
    - ES calculations involving non-binary decimal literals
"""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

from riskquant.scenario_metrics._computer import (
    _compute_lower_tail_parameters,
)
from riskquant.scenario_metrics._result import VarEsResult
from riskquant.scenario_metrics.api.var_es import compute_var_es

# ============================================================================
# numerical test conventions
# ============================================================================

_FLOAT64_EPS: float = np.finfo(np.float64).eps

# A deliberately tight floating-point tolerance.
#
# 16 eps is still extremely strict at float64 precision, while allowing for
# the small amount of arithmetic noise expected from basic scalar operations.
_FLOAT64_ABS_TOLERANCE: float = 16.0 * _FLOAT64_EPS

# Some aggregate floating-point operations, especially tail summation, can
# accumulate more than a handful of ulps. Keep this separate from the scalar
# compiler tolerance so the reason for the tolerance is explicit.
_FLOAT64_AGGREGATE_ABS_TOLERANCE: float = 64.0 * _FLOAT64_EPS


# ============================================================================
# independent mathematical reference
# ============================================================================


def _reference_tail_mass(
    scenario_count: int,
    confidence_level: float,
) -> float:
    """
    Independent reference for the empirical lower-tail probability mass.

    The production compiler deliberately snaps only floating-point noise around
    an integer tail mass. The same mathematical convention is reproduced here,
    but the actual metric calculation uses a full NumPy sort and therefore does
    not share the production quickselect implementation.
    """

    alpha = 1.0 - confidence_level
    tail_mass = alpha * float(scenario_count)

    nearest_integer = float(np.rint(tail_mass))
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(tail_mass))

    if abs(tail_mass - nearest_integer) <= tolerance:
        return nearest_integer

    return tail_mass


def _reference_var_es(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Independent scalar reference implementation.

    Convention
    ----------
    Input values are P&Ls, with more negative values more adverse.

    Let:

        alpha = 1 - confidence_level
        m     = alpha * S

    VaR is x_(ceil(m)).

    ES integrates exactly m observations' worth of empirical probability mass.
    """

    ordered = np.sort(
        np.asarray(values, dtype=np.float64),
    )

    scenario_count = ordered.size

    tail_mass = _reference_tail_mass(
        scenario_count=scenario_count,
        confidence_level=confidence_level,
    )

    full_tail_count = math.floor(tail_mass)
    fractional_weight = tail_mass - float(full_tail_count)

    if fractional_weight == 0.0:
        var_index = max(0, full_tail_count - 1)
    else:
        var_index = full_tail_count

    var = float(ordered[var_index])

    tail_sum = float(
        ordered[:full_tail_count].sum(
            dtype=np.float64,
        )
    )

    if fractional_weight > 0.0:
        tail_sum += fractional_weight * var

    es = tail_sum / tail_mass

    return var, es


# ============================================================================
# parameter compiler
# ============================================================================


@pytest.mark.parametrize(
    (
        "scenario_count",
        "confidence_level",
        "expected_var_index",
        "expected_full_tail_count",
        "expected_fractional_weight",
        "expected_tail_mass",
    ),
    [
        # alpha*S = 0.01*100 = 1
        (100, 0.99, 0, 1, 0.0, 1.0),
        # alpha*S = 0.01*200 = 2
        (200, 0.99, 1, 2, 0.0, 2.0),
        # alpha*S = 0.01*250 = 2.5
        (250, 0.99, 2, 2, 0.5, 2.5),
        # alpha*S = 0.025*200 = 5
        (200, 0.975, 4, 5, 0.0, 5.0),
        # alpha*S = 0.001*1000 = 1
        (1000, 0.999, 0, 1, 0.0, 1.0),
        # alpha*S < 1
        (10, 0.999, 0, 0, 0.01, 0.01),
        # alpha*S = 1.5
        (10, 0.85, 1, 1, 0.5, 1.5),
        # alpha*S = 2.5
        (10, 0.75, 2, 2, 0.5, 2.5),
    ],
)
def test_lower_tail_parameter_compiler(
    scenario_count: int,
    confidence_level: float,
    expected_var_index: int,
    expected_full_tail_count: int,
    expected_fractional_weight: float,
    expected_tail_mass: float,
) -> None:
    (
        var_index,
        full_tail_count,
        fractional_weight,
        tail_mass,
    ) = _compute_lower_tail_parameters(
        scenario_count=scenario_count,
        confidence_level=confidence_level,
    )

    assert var_index == expected_var_index
    assert full_tail_count == expected_full_tail_count

    assert fractional_weight == pytest.approx(
        expected_fractional_weight,
        rel=0.0,
        abs=_FLOAT64_ABS_TOLERANCE,
    )

    assert tail_mass == pytest.approx(
        expected_tail_mass,
        rel=0.0,
        abs=_FLOAT64_ABS_TOLERANCE,
    )


def test_integral_tail_mass_is_normalized() -> None:
    """
    Protect against binary floating-point noise moving an intended integer
    tail mass into the next fractional bucket.
    """

    (
        var_index,
        full_tail_count,
        fractional_weight,
        tail_mass,
    ) = _compute_lower_tail_parameters(
        scenario_count=100,
        confidence_level=0.99,
    )

    assert var_index == 0
    assert full_tail_count == 1

    assert fractional_weight == pytest.approx(
        0.0,
        rel=0.0,
        abs=_FLOAT64_ABS_TOLERANCE,
    )

    assert tail_mass == pytest.approx(
        1.0,
        rel=0.0,
        abs=_FLOAT64_ABS_TOLERANCE,
    )


def test_integral_tail_mass_with_decimal_input_is_canonicalized() -> None:
    """
    Decimal is accepted by the public scalar canonicalization path and should
    yield the same numerical convention as the equivalent float.
    """

    result_float = compute_var_es(
        np.ascontiguousarray(
            [[-5.0, -4.0, -3.0, -2.0, -1.0]],
            dtype=np.float64,
        ),
        0.80,
    )

    result_decimal = compute_var_es(
        np.ascontiguousarray(
            [[-5.0, -4.0, -3.0, -2.0, -1.0]],
            dtype=np.float64,
        ),
        Decimal("0.80"),  # pyright: ignore[reportArgumentType]
    )

    np.testing.assert_array_equal(
        result_float.var,
        result_decimal.var,
    )

    np.testing.assert_array_equal(
        result_float.es,
        result_decimal.es,
    )


# ============================================================================
# exact deterministic mathematical cases
# ============================================================================


@pytest.mark.parametrize(
    (
        "confidence_level",
        "expected_var",
        "expected_es",
    ),
    [
        # S=5, alpha*S=1
        (0.80, -5.0, -5.0),
        # S=5, alpha*S=1.5
        (
            0.70,
            -4.0,
            (-5.0 + 0.5 * -4.0) / 1.5,
        ),
        # S=5, alpha*S=2
        (
            0.60,
            -4.0,
            (-5.0 + -4.0) / 2.0,
        ),
        # S=5, alpha*S=2.5
        (
            0.50,
            -3.0,
            (-5.0 + -4.0 + 0.5 * -3.0) / 2.5,
        ),
        # S=5, alpha*S=3
        (
            0.40,
            -3.0,
            (-5.0 + -4.0 + -3.0) / 3.0,
        ),
        # S=5, alpha*S=4
        (
            0.20,
            -2.0,
            (-5.0 + -4.0 + -3.0 + -2.0) / 4.0,
        ),
    ],
)
def test_exact_empirical_tail_geometry(
    confidence_level: float,
    expected_var: float,
    expected_es: float,
) -> None:
    matrix = np.ascontiguousarray(
        [[-5.0, -4.0, -3.0, -2.0, -1.0]],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        confidence_level,
    )

    assert result.var[0] == expected_var

    assert result.es[0] == pytest.approx(
        expected_es,
        rel=0.0,
        abs=_FLOAT64_ABS_TOLERANCE,
    )


def test_single_scenario_group() -> None:
    """
    With one scenario, any non-empty lower tail must select that scenario.

    VaR is the selected input observation and should be exact.

    ES passes through the general tail-mass arithmetic path, so the result is
    numerically equal to the input value up to ordinary float64 rounding.
    """
    matrix = np.ascontiguousarray(
        [[-123.456789]],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        0.99,
    )

    expected = matrix[0, 0]

    # VaR is the selected scenario itself.
    assert result.var[0] == expected

    # ES follows the general floating-point tail arithmetic path.
    assert result.es[0] == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-13,
    )


def test_all_scenarios_equal() -> None:
    matrix = np.ascontiguousarray(
        np.full(
            (8, 1000),
            -7.25,
            dtype=np.float64,
        ),
    )

    result = compute_var_es(
        matrix,
        0.99,
    )

    np.testing.assert_array_equal(
        result.var,
        np.full(
            8,
            -7.25,
            dtype=np.float64,
        ),
    )

    np.testing.assert_array_equal(
        result.es,
        np.full(
            8,
            -7.25,
            dtype=np.float64,
        ),
    )


# ============================================================================
# ties / selection robustness
# ============================================================================


@pytest.mark.parametrize(
    "values",
    [
        [-5.0, -5.0, -5.0, -2.0, -2.0, 10.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        [10.0, -5.0, 10.0, -5.0, -5.0, 10.0],
        [-100.0, -100.0, 0.0, 0.0, 100.0, 100.0],
        [-3.0, -2.0, -2.0, -2.0, 9.0, 11.0],
    ],
)
@pytest.mark.parametrize(
    "confidence_level",
    [0.50, 0.60, 0.75, 0.80, 0.90],
)
def test_quickselect_matches_reference_with_ties(
    values: list[float],
    confidence_level: float,
) -> None:
    matrix = np.ascontiguousarray(
        [values],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        confidence_level,
    )

    expected_var, expected_es = _reference_var_es(
        np.asarray(
            values,
            dtype=np.float64,
        ),
        confidence_level,
    )

    assert result.var[0] == expected_var

    assert result.es[0] == pytest.approx(
        expected_es,
        rel=0.0,
        abs=1e-14,
    )


# ============================================================================
# independent randomized reference testing
# ============================================================================


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.975,
        0.99,
        0.995,
        0.999,
    ],
)
@pytest.mark.parametrize(
    ("n_groups", "n_scenarios"),
    [
        (1, 1),
        (1, 2),
        (1, 7),
        (1, 10),
        (2, 11),
        (8, 25),
        (16, 101),
        (8, 250),
        (4, 1000),
    ],
)
def test_randomized_against_independent_sorted_reference(
    confidence_level: float,
    n_groups: int,
    n_scenarios: int,
) -> None:
    seed = n_groups * 1_000_003 + n_scenarios * 97 + int(confidence_level * 1_000_000)

    rng = np.random.default_rng(seed)

    matrix = np.ascontiguousarray(
        rng.standard_normal(
            (n_groups, n_scenarios),
        ),
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        confidence_level,
    )

    expected_var = np.empty(
        n_groups,
        dtype=np.float64,
    )

    expected_es = np.empty(
        n_groups,
        dtype=np.float64,
    )

    for group in range(n_groups):
        (
            expected_var[group],
            expected_es[group],
        ) = _reference_var_es(
            matrix[group],
            confidence_level,
        )

    np.testing.assert_array_equal(
        result.var,
        expected_var,
    )

    np.testing.assert_allclose(
        result.es,
        expected_es,
        rtol=0.0,
        atol=1e-12,
    )


# ============================================================================
# multiple groups / SoA contract
# ============================================================================


def test_multiple_groups_are_computed_independently() -> None:
    matrix = np.ascontiguousarray(
        [
            [-10.0, -9.0, -8.0, 1.0, 2.0],
            [100.0, 50.0, 25.0, -5.0, -10.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        ],
        dtype=np.float64,
    )

    confidence_level = 0.80

    result = compute_var_es(
        matrix,
        confidence_level,
    )

    for group in range(matrix.shape[0]):
        expected_var, expected_es = _reference_var_es(
            matrix[group],
            confidence_level,
        )

        assert result.var[group] == expected_var

        assert result.es[group] == pytest.approx(
            expected_es,
            rel=0.0,
            abs=1e-14,
        )


def test_output_shape_and_dtype() -> None:
    matrix = np.ascontiguousarray(
        np.arange(
            24,
            dtype=np.float64,
        ).reshape(4, 6),
    )

    result = compute_var_es(
        matrix,
        0.99,
    )

    assert isinstance(
        result,
        VarEsResult,
    )

    assert result.var.shape == (4,)
    assert result.es.shape == (4,)

    assert result.var.dtype == np.dtype(np.float64)
    assert result.es.dtype == np.dtype(np.float64)

    assert result.var.flags.c_contiguous
    assert result.es.flags.c_contiguous


# ============================================================================
# input contract
# ============================================================================


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([], dtype=np.float64),
        np.empty((0, 5), dtype=np.float64),
        np.empty((5, 0), dtype=np.float64),
        np.ones((5, 5, 1), dtype=np.float64),
    ],
)
def test_invalid_matrix_shape_is_rejected(
    matrix: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        compute_var_es(
            matrix,
            0.99,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.bool_,
    ],
)
def test_non_float64_input_is_rejected(
    dtype: np.dtype,
) -> None:
    matrix = np.ascontiguousarray(
        np.ones(
            (2, 10),
            dtype=dtype,
        ),
    )

    with pytest.raises(TypeError):
        compute_var_es(
            matrix,
            0.99,
        )


def test_non_contiguous_input_is_rejected() -> None:
    base = np.ascontiguousarray(
        np.arange(
            40,
            dtype=np.float64,
        ).reshape(4, 10),
    )

    matrix = base[:, ::2]

    assert not matrix.flags.c_contiguous

    with pytest.raises(ValueError):
        compute_var_es(
            matrix,
            0.99,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_input_is_rejected(
    bad_value: float,
) -> None:
    matrix = np.ascontiguousarray(
        np.ones(
            (2, 10),
            dtype=np.float64,
        ),
    )

    matrix[0, 0] = bad_value

    with pytest.raises(ValueError):
        compute_var_es(
            matrix,
            0.99,
        )


# ============================================================================
# confidence-level contract
# ============================================================================


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
        -1.0,
        2.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_confidence_level_is_rejected(
    confidence_level: float,
) -> None:
    matrix = np.ascontiguousarray(
        [[-1.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        compute_var_es(
            matrix,
            confidence_level,
        )


@pytest.mark.parametrize(
    "confidence_level",
    [
        1e-15,
        1e-6,
        0.50,
        0.975,
        0.99,
        0.999,
        1.0 - 1e-15,
    ],
)
def test_valid_confidence_levels_are_accepted(
    confidence_level: float,
) -> None:
    matrix = np.ascontiguousarray(
        [[-3.0, -2.0, -1.0]],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        confidence_level,
    )

    assert result.var.shape == (1,)
    assert result.es.shape == (1,)


def test_confidence_level_is_canonicalized() -> None:
    matrix = np.ascontiguousarray(
        [[-3.0, -2.0, -1.0]],
        dtype=np.float64,
    )

    result_float = compute_var_es(
        matrix,
        0.99,
    )

    result_numpy_scalar = compute_var_es(
        matrix,
        np.float64(0.99),
    )

    np.testing.assert_array_equal(
        result_float.var,
        result_numpy_scalar.var,
    )

    np.testing.assert_array_equal(
        result_float.es,
        result_numpy_scalar.es,
    )


# ============================================================================
# input immutability / readonly contracts
# ============================================================================


def test_compute_var_es_does_not_mutate_input() -> None:
    matrix = np.ascontiguousarray(
        [
            [-5.0, -2.0, 4.0, 7.0],
            [10.0, -8.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    original = matrix.copy()

    compute_var_es(
        matrix,
        0.75,
    )

    np.testing.assert_array_equal(
        matrix,
        original,
    )


def test_compute_var_es_accepts_read_only_input() -> None:
    matrix = np.ascontiguousarray(
        [
            [-5.0, -4.0, -3.0, -2.0, -1.0],
        ],
        dtype=np.float64,
    )

    matrix.setflags(write=False)

    result = compute_var_es(
        matrix,
        0.80,
    )

    assert matrix.flags.writeable is False
    assert result.var[0] == -5.0
    assert result.es[0] == -5.0


def test_result_arrays_are_read_only() -> None:
    matrix = np.ascontiguousarray(
        [[-5.0, -4.0, -3.0]],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        0.80,
    )

    assert result.var.flags.writeable is False
    assert result.es.flags.writeable is False

    with pytest.raises(ValueError):
        result.var[0] = 123.0

    with pytest.raises(ValueError):
        result.es[0] = 123.0


def test_result_container_is_structurally_frozen() -> None:
    matrix = np.ascontiguousarray(
        [[-5.0, -4.0, -3.0]],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        0.80,
    )

    with pytest.raises(AttributeError):
        result.var = result.es  # type: ignore


# ============================================================================
# mathematical invariants
# ============================================================================


def test_es_is_not_above_var_for_lower_tail_pnl() -> None:
    rng = np.random.default_rng(42)

    matrix = np.ascontiguousarray(
        rng.standard_normal(
            (32, 5000),
        ),
        dtype=np.float64,
    )

    for confidence_level in (
        0.975,
        0.99,
        0.999,
    ):
        result = compute_var_es(
            matrix,
            confidence_level,
        )

        assert np.all(
            result.es <= result.var + _FLOAT64_AGGREGATE_ABS_TOLERANCE,
        )


def test_translation_invariance() -> None:
    rng = np.random.default_rng(42)

    matrix = np.ascontiguousarray(
        rng.standard_normal(
            (12, 500),
        ),
        dtype=np.float64,
    )

    shift = 37.5

    base = compute_var_es(
        matrix,
        0.99,
    )

    shifted = compute_var_es(
        np.ascontiguousarray(
            matrix + shift,
        ),
        0.99,
    )

    np.testing.assert_allclose(
        shifted.var,
        base.var + shift,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        shifted.es,
        base.es + shift,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "scale",
    [
        0.1,
        1.0,
        2.5,
        1000.0,
    ],
)
def test_positive_homogeneity(
    scale: float,
) -> None:
    rng = np.random.default_rng(123)

    matrix = np.ascontiguousarray(
        rng.standard_normal(
            (12, 500),
        ),
        dtype=np.float64,
    )

    base = compute_var_es(
        matrix,
        0.99,
    )

    scaled = compute_var_es(
        np.ascontiguousarray(
            matrix * scale,
        ),
        0.99,
    )

    np.testing.assert_allclose(
        scaled.var,
        base.var * scale,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        scaled.es,
        base.es * scale,
        rtol=0.0,
        atol=1e-12,
    )


def test_monotonicity_under_elementwise_increase() -> None:
    rng = np.random.default_rng(7)

    lower = np.ascontiguousarray(
        rng.standard_normal(
            (16, 1000),
        ),
        dtype=np.float64,
    )

    increment = np.ascontiguousarray(
        rng.uniform(
            0.0,
            5.0,
            size=lower.shape,
        ),
        dtype=np.float64,
    )

    upper = np.ascontiguousarray(
        lower + increment,
        dtype=np.float64,
    )

    lower_result = compute_var_es(
        lower,
        0.99,
    )

    upper_result = compute_var_es(
        upper,
        0.99,
    )

    assert np.all(
        lower_result.var <= upper_result.var + _FLOAT64_AGGREGATE_ABS_TOLERANCE,
    )

    assert np.all(
        lower_result.es <= upper_result.es + _FLOAT64_AGGREGATE_ABS_TOLERANCE,
    )


def test_scenario_order_invariance() -> None:
    rng = np.random.default_rng(99)

    matrix = np.ascontiguousarray(
        rng.standard_normal(
            (16, 1001),
        ),
        dtype=np.float64,
    )

    permutation = rng.permutation(
        matrix.shape[1],
    )

    base = compute_var_es(
        matrix,
        0.99,
    )

    shuffled = compute_var_es(
        np.ascontiguousarray(
            matrix[:, permutation],
        ),
        0.99,
    )

    np.testing.assert_array_equal(
        base.var,
        shuffled.var,
    )

    np.testing.assert_allclose(
        base.es,
        shuffled.es,
        rtol=0.0,
        atol=1e-12,
    )


# ============================================================================
# adversarial ordering / quickselect robustness
# ============================================================================


@pytest.mark.parametrize(
    "matrix_builder",
    [
        lambda: np.ascontiguousarray(
            np.arange(
                5000,
                dtype=np.float64,
            ).reshape(1, -1),
        ),
        lambda: np.ascontiguousarray(
            np.arange(
                4999,
                -1,
                -1,
                dtype=np.float64,
            ).reshape(1, -1),
        ),
        lambda: np.ascontiguousarray(
            np.tile(
                np.array(
                    [
                        -10.0,
                        -5.0,
                        0.0,
                        5.0,
                        10.0,
                    ],
                    dtype=np.float64,
                ),
                1000,
            ).reshape(1, -1),
        ),
        lambda: np.ascontiguousarray(
            np.zeros(
                (1, 5000),
                dtype=np.float64,
            ),
        ),
    ],
)
def test_adversarial_orderings_match_reference(
    matrix_builder,
) -> None:
    matrix = matrix_builder()

    result = compute_var_es(
        matrix,
        0.99,
    )

    expected_var, expected_es = _reference_var_es(
        matrix[0],
        0.99,
    )

    assert result.var[0] == expected_var

    assert result.es[0] == pytest.approx(
        expected_es,
        rel=0.0,
        abs=1e-12,
    )


# ============================================================================
# repeated execution / determinism
# ============================================================================


def test_repeated_execution_is_bitwise_deterministic() -> None:
    rng = np.random.default_rng(2026)

    matrix = np.ascontiguousarray(
        rng.standard_normal(
            (32, 5000),
        ),
        dtype=np.float64,
    )

    first = compute_var_es(
        matrix,
        0.99,
    )

    for _ in range(5):
        result = compute_var_es(
            matrix,
            0.99,
        )

        np.testing.assert_array_equal(
            result.var,
            first.var,
        )

        np.testing.assert_array_equal(
            result.es,
            first.es,
        )


# ============================================================================
# fractional / integral tail geometry
# ============================================================================


def test_fractional_es_uses_only_required_boundary_mass() -> None:
    """
    Explicitly prove that the implementation does not average an extra whole
    observation when alpha*S is fractional.

    S=10, confidence=0.85 -> alpha*S=1.5.

    Therefore:

        VaR = x_(2)
        ES  = (x_(1) + 0.5*x_(2)) / 1.5
    """

    matrix = np.ascontiguousarray(
        [
            [
                -100.0,
                -20.0,
                -10.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]
        ],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        0.85,
    )

    assert result.var[0] == -20.0

    expected_es = (-100.0 + 0.5 * -20.0) / 1.5

    assert result.es[0] == pytest.approx(
        expected_es,
        rel=0.0,
        abs=_FLOAT64_ABS_TOLERANCE,
    )


def test_integral_es_does_not_include_boundary_outside_tail() -> None:
    """
    S=10, confidence=0.80 -> alpha*S=2.

    ES must be exactly the average of the two worst observations.
    """

    matrix = np.ascontiguousarray(
        [
            [
                -100.0,
                -50.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]
        ],
        dtype=np.float64,
    )

    result = compute_var_es(
        matrix,
        0.80,
    )

    assert result.var[0] == -50.0
    assert result.es[0] == -75.0
