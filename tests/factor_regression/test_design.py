# tests/factor_regression/test_design.py
"""
S3.T1 to S3.T5 and S3.T7: design assembly, scaling, rank gate and condition
number (A.2, C.8, C.9, C.10, C.19).

Every oracle is independent of ``design.py`` (A.7): the closed form
``sqrt((1 + rho) / (1 - rho))``, ``numpy.linalg.matrix_rank`` with ``tol=``
on a design that the test scales with ``numpy.linalg.norm``, ``eigvalsh`` of
the Gram matrix, ``math.hypot``, and hand relations such as ``(1, 1, -1, 0)``.
The return designs of S3.T3, S3.T4 and S3.T7 are built here with plain numpy
in the A.1 ratio form; the test does not import ``rows.py``.

Tolerances (owner decision 6): S3.T1 kappa equal to 1e-12 relative; S3.T5
1e-12 relative.

Sandbox evidence (Python 3.12.3, numpy 2.4.4, Linux), generator of this file:
S3.T3 detected 200 of 200 seeds at vol 1% and 0.5% with ``rank_rtol=1e-10``;
S3.T4 ``kappa`` 4.6e12 to 6.5e12 at vol 0.02% and 2.3e12 to 3.3e12 at vol
0.01% over seeds 0 to 199, no raise; S3.T7 ``kappa * vol`` 2.37 to 3.60 and
``kappa(0.5%) / kappa(1%)`` 1.996 to 2.004 over seeds 0 to 199.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.design import (
    CoreScaledSvd,
    assemble_design,
    rank_gate,
)
from riskquant.factor_regression.diagnostics import diagnose
from riskquant.factor_regression.errors import (
    CoreInputError,
    CoreParameterError,
    CoreRankError,
    CoreSampleSizeError,
)

EPS = float(np.finfo(np.float64).eps)
ROWS = 250


# ------------------------------------------------------------------ helpers


def _random_design(seed: int, n_rows: int, n_columns: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    scales = 10.0 ** rng.uniform(-3.0, 3.0, n_columns)
    return np.ascontiguousarray(rng.standard_normal((n_rows, n_columns)) * scales)


def _unit_columns(design: NDArray[np.float64]) -> NDArray[np.float64]:
    """Oracle scaling: numpy.linalg.norm, not the code under test."""
    norms = np.linalg.norm(design, axis=0)
    return np.asarray(design / norms, dtype=np.float64)


def _gram_kappa(design: NDArray[np.float64]) -> float:
    """Oracle kappa: square root of the eigenvalue ratio of the unit Gram."""
    unit = _unit_columns(design)
    eigenvalues = np.linalg.eigvalsh(unit.T @ unit)
    return math.sqrt(float(eigenvalues[-1]) / float(eigenvalues[0]))


def _rank_error(design: NDArray[np.float64], rank_rtol: float) -> CoreRankError:
    with pytest.raises(CoreRankError) as info:
        rank_gate(design, rank_rtol=rank_rtol)
    return info.value


def _levels(
    seed: int, vol: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Z (USD) and S (EUR per USD) as log random walks; Q = Z * S rounded."""
    rng = np.random.default_rng(seed)
    z = 100.0 * np.exp(np.cumsum(rng.normal(0.0, vol, ROWS + 1)))
    s = 1.1 * np.exp(np.cumsum(rng.normal(0.0, vol, ROWS + 1)))
    return z, s, z * s


def _log_design(seed: int, vol: float) -> NDArray[np.float64]:
    # A.1 ratio form: the ratio is formed before the logarithm.
    columns = [np.log(v[1:] / v[:-1]) for v in _levels(seed, vol)]
    return np.ascontiguousarray(np.column_stack(columns))


def _simple_design(seed: int, vol: float) -> NDArray[np.float64]:
    columns = [v[1:] / v[:-1] - 1.0 for v in _levels(seed, vol)]
    return np.ascontiguousarray(np.column_stack(columns))


def _neutral_diagnose(condition_number: float, threshold: float) -> NDArray[np.int8]:
    """Only the condition rule can fire: no LEVEL term, q = 0."""
    return diagnose(
        term_transforms=np.array([codes.LOG_RETURN] * 4, dtype=np.int8),
        condition_number=condition_number,
        condition_threshold=threshold,
        apply_condition_threshold=True,
        r_squared=0.5,
        durbin_watson_value=2.0,
        overlap_order=0,
        covariance_kind=int(codes.IID),
        covariance_lag=0,
    ).codes


def _no_svd(*args: object, **kwargs: object) -> object:
    raise AssertionError("numpy.linalg.svd must not run")


# ------------------------------------------------------------------- C.8


@pytest.mark.parametrize("intercept", [True, False, np.True_, np.False_])
def test_the_intercept_comes_first_then_the_regressors_in_declared_order(
    intercept: bool,
) -> None:
    rng = np.random.default_rng(1)
    regressors = rng.standard_normal((3, 7))
    before = regressors.copy()
    design = assemble_design(regressors, intercept=intercept)

    offset = 1 if intercept else 0
    assert design.shape == (7, 3 + offset)
    assert design.dtype == np.float64
    assert design.flags.c_contiguous
    assert not design.flags.writeable
    if offset:
        assert design[:, 0].tolist() == [1.0] * 7
    for t in range(3):
        # Bit for bit: a copy, no arithmetic.
        assert np.array_equal(
            design[:, t + offset].view(np.int64), regressors[t].view(np.int64)
        )
    assert np.array_equal(regressors, before)
    assert regressors.flags.writeable
    assert not np.shares_memory(design, regressors)


def test_an_intercept_only_design_is_one_column_of_ones() -> None:
    design = assemble_design(np.empty((0, 4), dtype=np.float64), intercept=True)
    assert design.shape == (4, 1)
    assert design[:, 0].tolist() == [1.0] * 4


def test_no_regressor_and_no_intercept_raises_empty_design() -> None:
    with pytest.raises(CoreParameterError) as info:
        assemble_design(np.empty((0, 4), dtype=np.float64), intercept=False)
    assert info.value.reason == codes.PARAM_EMPTY_DESIGN
    assert info.value.value == 0.0


@pytest.mark.parametrize("intercept", [1, 0, None, "yes"])
def test_a_non_bool_intercept_raises_type_error(intercept: object) -> None:
    with pytest.raises(TypeError):
        assemble_design(np.ones((1, 4)), intercept=intercept)  # type: ignore[arg-type]


def test_an_assembled_design_passes_the_gate_with_the_intercept_first() -> None:
    rng = np.random.default_rng(2)
    regressors = rng.standard_normal((2, 30)) + 5.0
    gate = rank_gate(assemble_design(regressors, intercept=True), rank_rtol=1e-10)
    assert gate.n_columns == 3
    assert gate.design[:, 0].tolist() == [1.0] * 30
    assert gate.column_norms[0] == math.sqrt(30.0)


# ------------------------------------------------------------------ S3.T1


def test_a_zero_column_raises_with_its_index_and_no_svd_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _random_design(3, 12, 4)
    design[:, 2] = 0.0
    monkeypatch.setattr(np.linalg, "svd", _no_svd)
    error = _rank_error(design, 1e-10)
    assert error.reason == codes.RANK_ZERO_COLUMN
    assert error.n_columns == 4
    # Owner decision OD-1 A: the rank is an upper bound, k minus the zeros.
    assert error.rank == 3
    assert error.singular_values.shape == (0,)
    assert error.participation.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert error.involved.tolist() == [2]
    assert error.involved.dtype == np.int32
    assert error.relations.tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_two_zero_columns_give_one_unit_relation_each() -> None:
    design = _random_design(4, 12, 5)
    design[:, 1] = -0.0
    design[:, 4] = 0.0
    error = _rank_error(design, 1e-10)
    assert error.reason == codes.RANK_ZERO_COLUMN
    assert error.rank == 3
    assert error.involved.tolist() == [1, 4]
    assert error.participation.tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert error.relations.tolist() == [
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("column", [0, 2])
def test_a_column_multiplied_by_1e9_leaves_kappa_unchanged(
    seed: int, column: int
) -> None:
    design = _random_design(seed, 40, 3)
    base = rank_gate(design, rank_rtol=1e-10)
    stretched = design.copy()
    stretched[:, column] *= 1e9
    gate = rank_gate(stretched, rank_rtol=1e-10)
    assert abs(gate.condition_number / base.condition_number - 1.0) <= 1e-12
    # Independent oracle: the eigenvalues of the Gram of the unit columns.
    oracle = _gram_kappa(design)
    assert abs(gate.condition_number / oracle - 1.0) <= 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_column_norms_match_math_hypot_and_scaled_columns_are_unit(seed: int) -> None:
    design = _random_design(seed, 25, 4)
    gate = rank_gate(design, rank_rtol=1e-10)
    for j in range(4):
        oracle = math.hypot(*design[:, j].tolist())
        assert abs(float(gate.column_norms[j]) / oracle - 1.0) <= 1e-14
        assert abs(float(np.linalg.norm(gate.scaled_design[:, j])) - 1.0) <= 1e-14
    assert np.array_equal(gate.scaled_design, design / gate.column_norms)


@pytest.mark.parametrize("magnitude", [1e-170, 1e-300, 1e160, 1e300])
def test_extreme_column_magnitudes_give_the_kappa_of_moderate_ones(
    magnitude: float,
) -> None:
    # An unscaled sum of squares underflows to 0 below 1e-162 (a false zero
    # column) and overflows to inf above 1e154.
    design = _random_design(5, 30, 3)
    base = rank_gate(design, rank_rtol=1e-10)
    extreme = design.copy()
    extreme[:, 1] = design[:, 1] / float(np.max(np.abs(design[:, 1]))) * magnitude
    gate = rank_gate(extreme, rank_rtol=1e-10)
    assert np.isfinite(gate.column_norms).all()
    assert abs(gate.condition_number / base.condition_number - 1.0) <= 1e-12


def test_a_column_norm_above_the_float64_range_raises_non_finite() -> None:
    design = _random_design(6, 9, 3)
    design[:, 1] = 1e308
    design[4, 1] = -1.5e308
    with pytest.raises(CoreInputError) as info:
        rank_gate(design, rank_rtol=1e-10)
    assert info.value.reason == codes.INPUT_NON_FINITE
    # Flat C-order index of the largest magnitude of column 1: row 4.
    assert info.value.positions == (4 * 3 + 1,)


# ------------------------------------------------------------------ S3.T2


def _dependent_design(seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    c1 = rng.standard_normal(20)
    c2 = 10.0 * rng.standard_normal(20)
    c4 = 0.01 * rng.standard_normal(20)
    return np.ascontiguousarray(np.column_stack([c1, c2, c1 + c2, c4]))


@pytest.mark.parametrize("seed", range(10))
def test_a_sum_column_gives_rank_three_and_the_relation_one_one_minus_one(
    seed: int,
) -> None:
    design = _dependent_design(seed)
    rank_rtol = 1e-10
    error = _rank_error(design, rank_rtol)
    assert error.reason == codes.RANK_DEFICIENT
    assert (error.rank, error.n_columns) == (3, 4)
    assert error.involved.tolist() == [0, 1, 2]
    assert error.singular_values.shape == (4,)
    assert np.all(np.diff(error.singular_values) <= 0.0)
    assert abs(float(error.participation.sum()) - 1.0) <= 1e-14

    assert error.relations.shape == (1, 4)
    relation = error.relations[0]
    # The entry of largest magnitude is exactly 1.0 (A.2 step 4).
    assert float(np.max(np.abs(relation))) == 1.0
    assert float(relation[int(np.argmax(np.abs(relation)))]) == 1.0
    # Original units: proportional to (1, 1, -1, 0) whatever the column scales.
    proportional = relation / relation[0]
    assert np.allclose(proportional, [1.0, 1.0, -1.0, 0.0], rtol=0.0, atol=1e-9)

    unit = _unit_columns(design)
    largest = float(np.linalg.svd(unit, compute_uv=False)[0])
    assert np.linalg.matrix_rank(unit, tol=rank_rtol * largest) == 3


def test_two_null_directions_give_a_basis_of_two_relations() -> None:
    rng = np.random.default_rng(8)
    c1 = rng.standard_normal(15)
    c2 = 3.0 * rng.standard_normal(15)
    design = np.ascontiguousarray(np.column_stack([c1, c2, c1 + c2, 2.0 * c1 - c2]))
    error = _rank_error(design, 1e-10)
    assert (error.rank, error.n_columns) == (2, 4)
    assert error.involved.tolist() == [0, 1, 2, 3]
    assert abs(float(error.participation.sum()) - 2.0) <= 1e-14
    assert error.relations.shape == (2, 4)
    scale = np.abs(design).sum(axis=0)
    for relation in error.relations:
        assert float(np.max(np.abs(relation))) == 1.0
        # sum_j w_j X[:, j] is 0 in original units, to rounding.
        residual = float(np.max(np.abs(design @ relation)))
        assert residual <= 1e-12 * float(np.abs(relation) @ scale)


# ------------------------------------------------------------------ S3.T3


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("vol", [0.01, 0.005])
def test_the_exact_log_identity_raises_with_all_three_columns(
    vol: float, seed: int
) -> None:
    design = _log_design(seed, vol)
    rank_rtol = 1e-10
    error = _rank_error(design, rank_rtol)
    assert error.reason == codes.RANK_DEFICIENT
    assert error.rank == 2
    assert error.involved.tolist() == [0, 1, 2]
    unit = _unit_columns(design)
    largest = float(np.linalg.svd(unit, compute_uv=False)[0])
    assert np.linalg.matrix_rank(unit, tol=rank_rtol * largest) == 2


# ------------------------------------------------------------------ S3.T4


@pytest.mark.parametrize("vol", [0.0002, 0.0001])
def test_a_near_identity_below_the_tolerance_passes_with_kappa_reported(
    vol: float,
) -> None:
    # F-12: at these vols the rounding residual of the identity is above
    # 250 * eps of s[0], so the gate does not raise. kappa reports it.
    for seed in range(200):
        gate = rank_gate(_log_design(seed, vol), rank_rtol=ROWS * EPS)
        assert gate.rank == 3, seed
        assert gate.condition_number >= 1e12, seed


# ------------------------------------------------------------------ S3.T5


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rho", [0.99, 0.5, 0.0, -0.9])
def test_kappa_of_two_unit_columns_equals_the_closed_form(
    rho: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((50, 2)))
    first = basis[:, 0]
    second = rho * basis[:, 0] + math.sqrt(1.0 - rho * rho) * basis[:, 1]
    design = np.ascontiguousarray(np.column_stack([first, 1e6 * second]))
    gate = rank_gate(design, rank_rtol=1e-10)
    # The singular values are sqrt(1 + rho) and sqrt(1 - rho); A.7 states the
    # ratio for rho >= 0, and abs(rho) orders them for a negative rho.
    oracle = math.sqrt((1.0 + abs(rho)) / (1.0 - abs(rho)))
    assert abs(gate.condition_number / oracle - 1.0) <= 1e-12


def test_a_single_column_has_kappa_one() -> None:
    gate = rank_gate(np.array([[3.0], [-4.0], [12.0]]), rank_rtol=1e-10)
    assert gate.condition_number == 1.0
    assert gate.column_norms.tolist() == [13.0]


# ------------------------------------------------------------------ S3.T7


@pytest.mark.parametrize("seed", range(20))
def test_the_simple_return_design_has_full_rank_and_a_vol_driven_kappa(
    seed: int,
) -> None:
    kappas = {}
    for vol in (0.01, 0.005):
        gate = rank_gate(_simple_design(seed, vol), rank_rtol=1e-10)
        assert gate.rank == 3
        assert 1.5 <= gate.condition_number * vol <= 5.0, (vol, gate.condition_number)
        flagged = _neutral_diagnose(gate.condition_number, 100.0)
        assert flagged.tolist() == [int(codes.CONDITION_ABOVE_THRESHOLD)]
        kappas[vol] = gate.condition_number
    assert 1.9 <= kappas[0.005] / kappas[0.01] <= 2.1


# --------------------------------------------- rank_rtol (owner decision 3)


@pytest.mark.parametrize(
    "rank_rtol",
    [0.0, 1.0, -1e-10, 1.5, math.nan, math.inf, -math.inf, 0, 1, np.float64(2.0)],
)
def test_a_rank_rtol_outside_the_open_unit_interval_raises(rank_rtol: float) -> None:
    with pytest.raises(CoreParameterError) as info:
        rank_gate(_random_design(0, 10, 2), rank_rtol=rank_rtol)
    assert info.value.reason == codes.PARAM_RANK_RTOL_OUT_OF_RANGE
    expected = float(rank_rtol)
    assert (math.isnan(expected) and math.isnan(info.value.value)) or (
        info.value.value == expected
    )


@pytest.mark.parametrize("rank_rtol", [True, np.True_, "1e-10", None])
def test_a_rank_rtol_of_the_wrong_type_raises_type_error(rank_rtol: object) -> None:
    with pytest.raises(TypeError):
        rank_gate(_random_design(0, 10, 2), rank_rtol=rank_rtol)  # type: ignore[arg-type]


def test_a_numpy_float_rank_rtol_gives_the_float_result() -> None:
    design = _random_design(1, 10, 2)
    a = rank_gate(design, rank_rtol=1e-10)
    b = rank_gate(design, rank_rtol=np.float64(1e-10))
    assert a.condition_number == b.condition_number


# --------------------------------------------------------- sample size


@pytest.mark.parametrize(("n_rows", "n_columns"), [(2, 2), (3, 3), (2, 5)])
def test_rows_not_above_columns_raise_before_the_svd(
    n_rows: int, n_columns: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    design = _random_design(2, n_rows, n_columns)
    monkeypatch.setattr(np.linalg, "svd", _no_svd)
    with pytest.raises(CoreSampleSizeError) as info:
        rank_gate(design, rank_rtol=1e-10)
    assert info.value.reason == codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS
    assert (info.value.n_available, info.value.n_required) == (n_rows, n_columns + 1)


def test_one_row_above_the_column_count_is_enough() -> None:
    gate = rank_gate(_random_design(3, 4, 3), rank_rtol=1e-10)
    assert (gate.n_rows, gate.n_columns, gate.rank) == (4, 3, 3)


# ----------------------------------------- non-finite input (owner decision 2)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("order", ["C", "F"])
def test_a_non_finite_design_raises_with_the_flat_c_order_index(
    bad: float, order: Literal["C", "F"]
) -> None:
    # The first non-finite element in C order is (1, 2); in column order it
    # is (4, 1). The flat C-order index of (1, 2) is 1 * 3 + 2.
    design = _random_design(4, 6, 3)
    design[4, 1] = bad
    design[1, 2] = bad
    laid_out = np.asarray(design, order=order)
    with pytest.raises(CoreInputError) as info:
        rank_gate(laid_out, rank_rtol=1e-10)
    assert info.value.reason == codes.INPUT_NON_FINITE
    assert info.value.operand == 0
    assert info.value.positions == (1 * 3 + 2,)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_regressors_raise_with_the_flat_c_order_index(bad: float) -> None:
    regressors = np.ones((2, 6))
    regressors[1, 4] = bad
    with pytest.raises(CoreInputError) as info:
        assemble_design(regressors, intercept=True)
    assert info.value.reason == codes.INPUT_NON_FINITE
    assert info.value.positions == (1 * 6 + 4,)


# ------------------------------------------------------- input contract


class _TrivialSubclass(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """An ndarray subclass with no override: only the exact type differs."""


def _gate_input_error(design: object) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        rank_gate(design, rank_rtol=1e-10)  # type: ignore[arg-type]
    return info.value


def _assembly_input_error(regressors: object) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        assemble_design(regressors, intercept=True)  # type: ignore[arg-type]
    return info.value


def test_a_masked_design_with_nan_under_the_mask_is_rejected() -> None:
    raw = _random_design(5, 6, 2)
    raw[3, 1] = math.nan
    hidden = np.zeros((6, 2), dtype=np.bool_)
    hidden[3, 1] = True
    masked: NDArray[Any] = np.ma.masked_array(raw, mask=hidden)
    # The mask hides the NaN from a reduction.
    assert bool(np.isfinite(masked).all())
    error = _gate_input_error(masked)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        0,
        (),
    )


def test_masked_regressors_with_nan_under_the_mask_are_rejected() -> None:
    raw = np.ones((2, 5))
    raw[0, 2] = math.nan
    hidden = np.zeros((2, 5), dtype=np.bool_)
    hidden[0, 2] = True
    masked: NDArray[Any] = np.ma.masked_array(raw, mask=hidden)
    error = _assembly_input_error(masked)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        0,
        (),
    )


def test_a_view_of_a_trivial_ndarray_subclass_is_rejected() -> None:
    design = _random_design(6, 6, 2)
    view = design.view(_TrivialSubclass)
    assert isinstance(view, np.ndarray)
    assert _gate_input_error(view).reason == codes.INPUT_NOT_NDARRAY
    assert _assembly_input_error(design.T.view(_TrivialSubclass)).reason == (
        codes.INPUT_NOT_NDARRAY
    )


def test_the_input_contract_rejects_without_repair() -> None:
    good = _random_design(7, 6, 2)
    cases: list[tuple[object, int]] = [
        (good.tolist(), int(codes.INPUT_NOT_NDARRAY)),
        (good.astype(np.float32), int(codes.INPUT_WRONG_DTYPE)),
        (good.astype(np.int64), int(codes.INPUT_WRONG_DTYPE)),
        (good[:, 0].copy(), int(codes.INPUT_WRONG_NDIM)),
        (good.reshape(1, 6, 2), int(codes.INPUT_WRONG_NDIM)),
        (np.empty((0, 2), dtype=np.float64), int(codes.INPUT_EMPTY)),
        (np.empty((6, 0), dtype=np.float64), int(codes.INPUT_EMPTY)),
    ]
    for design, reason in cases:
        error = _gate_input_error(design)
        assert (error.reason, error.operand, error.positions) == (reason, 0, ())

    regressors = good.T.copy()
    assembly_cases: list[tuple[object, int]] = [
        (regressors.tolist(), int(codes.INPUT_NOT_NDARRAY)),
        (regressors.astype(np.float32), int(codes.INPUT_WRONG_DTYPE)),
        (regressors[0].copy(), int(codes.INPUT_WRONG_NDIM)),
        (np.empty((2, 0), dtype=np.float64), int(codes.INPUT_EMPTY)),
    ]
    for array, reason in assembly_cases:
        error = _assembly_input_error(array)
        assert (error.reason, error.operand, error.positions) == (reason, 0, ())


# -------------------------------------------------- result contract (C.19)


def _arrays(gate: CoreScaledSvd) -> list[NDArray[np.float64]]:
    return [
        gate.design,
        gate.column_norms,
        gate.scaled_design,
        gate.u,
        gate.singular_values,
        gate.vh,
    ]


def test_the_result_holds_a_consistent_read_only_svd_and_no_caller_array() -> None:
    design = np.asfortranarray(_random_design(9, 30, 4))
    before = design.copy()
    gate = rank_gate(design, rank_rtol=1e-10)

    assert np.array_equal(design, before)
    assert design.flags.writeable
    assert np.array_equal(gate.design, before)
    assert gate.design.flags.c_contiguous
    assert gate.scaled_design.flags.c_contiguous
    for array in _arrays(gate):
        assert array.dtype == np.float64
        assert not array.flags.writeable
        assert not np.shares_memory(array, design)
    assert gate.u.shape == (30, 4)
    assert gate.singular_values.shape == (4,)
    assert gate.vh.shape == (4, 4)
    assert gate.rank == 4
    assert type(gate.rank) is int
    assert type(gate.condition_number) is float

    s = gate.singular_values
    assert gate.condition_number == float(s[0] / s[-1])
    rebuilt = gate.u @ np.diag(s) @ gate.vh
    assert float(np.max(np.abs(rebuilt - gate.scaled_design))) <= 1e-14
    assert float(np.max(np.abs(gate.vh @ gate.vh.T - np.eye(4)))) <= 1e-14

    design[0, 0] = 99.0
    assert np.array_equal(gate.design, before)


def test_the_result_is_frozen() -> None:
    gate = rank_gate(_random_design(1, 5, 2), rank_rtol=1e-10)
    with pytest.raises(AttributeError):
        gate.rank = 1  # type: ignore[misc]
