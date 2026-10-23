# tests/factor_regression/test_combination.py
"""
S1.T1 to S1.T4 and S1.T7: linear combination of aligned levels and the LOG
domain (WS_FACTOR_REGRESSION_COINTEGRATION C.2, C.3, C.11, C.13, C.14, A.8).

Oracles share no code with ``combination.py`` (A.7): ``fractions.Fraction``
gives the exact sums, ``math.log`` of the exact inputs gives the LOG value
case, and the identity ``ln(Z * M) - ln Z - ln M = 0`` gives S1.T2.

Tolerances.
- S1.T1 dyadic case: every level is a multiple of 0.25 and every weight is
  dyadic, so every product and partial sum is exact in float64. The result is
  compared with the ``Fraction`` value bit for bit.
- S1.T1 non-dyadic case: three terms with positive weights and positive
  levels, so no cancellation occurs. Each product and each addition is one
  correctly rounded operation (unit roundoff ``u = 2**-53``), so the error is
  below ``5 u |exact| < 5.6e-16 |exact|``. The test allows
  ``1e-15 * max_t |exact[t]|`` per date.
- S1.T2: ``ln Q``, ``ln Z`` and ``ln M`` are each below 6 in magnitude here;
  the rounding of ``Q = Z * M`` and of the three logarithms and two additions
  gives an error of a few times ``6 * 2**-52``, below ``1e-14``.
- LOG value case: ``numpy.log`` and ``math.log`` are separate implementations
  and can differ in the last bit, so the test allows
  ``8 u (|w_0 ln a| + |w_1 ln b|)``.
- S1.T5, S1.T6 and S1.T8 need no code here: they run the existing tests of
  ``tests/test_architecture.py``, ``tests/solutions/factor_regression/
  test_digest.py`` and ``tests/factor_regression/test_errors.py``.
"""

from __future__ import annotations

import ast
import inspect
import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes, combination
from riskquant.factor_regression.combination import (
    OPERAND_LEVELS,
    OPERAND_TERM_SERIES,
    OPERAND_WEIGHTS,
    combine_core,
)
from riskquant.factor_regression.errors import CoreInputError, CoreScaleDomainError

SCALE_LEVEL = int(codes.SCALE_LEVEL)
SCALE_LOG = int(codes.SCALE_LOG)
_U = 2.0**-53


# ------------------------------------------------------------------ helpers


def _levels(rows: list[list[float]]) -> NDArray[np.float64]:
    return np.array(rows, dtype=np.float64)


def _series(indices: list[int]) -> NDArray[np.int32]:
    return np.array(indices, dtype=np.int32)


def _weights(values: list[float]) -> NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _exact(
    rows: list[list[float]], indices: list[int], weights: list[float]
) -> list[Fraction]:
    # Fraction(float) is the exact binary value of the float.
    m = len(rows[0])
    return [
        sum(
            (
                Fraction(w) * Fraction(rows[s][t])
                for s, w in zip(indices, weights, strict=True)
            ),
            Fraction(0),
        )
        for t in range(m)
    ]


def _input_error(
    levels: Any, term_series: Any, weights: Any, scale: Any = SCALE_LEVEL
) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        combine_core(levels, term_series, weights, scale)
    return info.value


def _good() -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    return (
        _levels([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        _series([0, 1]),
        _weights([1.0, -2.5]),
    )


# ===================================================================== S1.T1


def test_t1_level_dyadic_equals_the_fraction_oracle_exactly() -> None:
    rows = [
        [1.25, -3.5, 100.75, 0.0, 7.0, -0.25],
        [2.5, 4.0, -8.75, 12.5, 0.25, 1000.5],
    ]
    indices = [0, 1]
    weights = [1.0, -2.5]
    z = combine_core(_levels(rows), _series(indices), _weights(weights), SCALE_LEVEL)
    exact = _exact(rows, indices, weights)
    assert z.dtype == np.float64
    assert z.shape == (6,)
    assert [Fraction(float(v)) for v in z] == exact


def test_t1_weight_order_follows_the_declared_terms() -> None:
    # Swapped weights give another value on every date, so a weight applied
    # to the wrong term fails the oracle.
    rows = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    z = combine_core(_levels(rows), _series([0, 1]), _weights([1.0, -2.5]), SCALE_LEVEL)
    swapped = _exact(rows, [0, 1], [-2.5, 1.0])
    assert [Fraction(float(v)) for v in z] == _exact(rows, [0, 1], [1.0, -2.5])
    assert all(Fraction(float(v)) != s for v, s in zip(z, swapped, strict=True))


def test_t1_term_series_selects_the_rows_in_declared_order() -> None:
    # Term order differs from series order, and one series appears twice.
    rows = [[0.5, 1.5, 2.5], [4.0, -4.0, 8.0], [0.25, 0.75, -1.25]]
    indices = [2, 0, 2, 1]
    weights = [1.0, -2.5, 0.5, 3.0]
    z = combine_core(_levels(rows), _series(indices), _weights(weights), SCALE_LEVEL)
    assert [Fraction(float(v)) for v in z] == _exact(rows, indices, weights)


def test_t1_single_term() -> None:
    rows = [[1.0, 2.0], [3.0, 4.0]]
    z = combine_core(_levels(rows), _series([1]), _weights([-2.5]), SCALE_LEVEL)
    assert z.tolist() == [-7.5, -10.0]


def test_t1_level_non_dyadic_within_the_stated_tolerance() -> None:
    rows = [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [0.7, 0.3, 0.9, 1.7, 2.9],
        [10.1, 20.3, 30.7, 40.9, 50.3],
    ]
    indices = [0, 1, 2]
    weights = [0.1, 0.3, 0.7]
    z = combine_core(_levels(rows), _series(indices), _weights(weights), SCALE_LEVEL)
    exact = _exact(rows, indices, weights)
    bound = Fraction(1e-15) * max(abs(e) for e in exact)
    for value, e in zip(z, exact, strict=True):
        assert abs(Fraction(float(value)) - e) <= bound


def test_t1_terms_are_added_left_to_right_in_declared_order() -> None:
    # Float addition is not associative. Left to right: (1e16 + 1) rounds to
    # 1e16, then - 1e16 gives 0. Any other order gives 1.
    rows = [[1e16], [1.0], [-1e16]]
    z = combine_core(
        _levels(rows), _series([0, 1, 2]), _weights([1.0, 1.0, 1.0]), SCALE_LEVEL
    )
    assert z.tolist() == [0.0]
    z = combine_core(
        _levels(rows), _series([0, 2, 1]), _weights([1.0, 1.0, 1.0]), SCALE_LEVEL
    )
    assert z.tolist() == [1.0]


def test_a_zero_weight_is_accepted_by_the_core() -> None:
    # C.1: the facade rejects a zero weight; the core does not.
    rows = [[1.0, 2.0], [3.0, 4.0]]
    z = combine_core(_levels(rows), _series([0, 1]), _weights([0.0, 1.0]), SCALE_LEVEL)
    assert z.tolist() == [3.0, 4.0]


# ===================================================================== S1.T2


def test_t2_log_of_a_product_cancels_to_zero() -> None:
    generator = np.random.default_rng(20261019)
    m = 500
    z_level = 100.0 * np.exp(np.cumsum(generator.normal(0.0, 0.01, m)))
    m_level = 1.0 * np.exp(np.cumsum(generator.normal(0.0, 0.005, m)))
    q_level = z_level * m_level
    levels = np.ascontiguousarray(np.vstack([q_level, z_level, m_level]))
    assert float(np.abs(np.log(levels)).max()) < 6.0
    combined = combine_core(
        levels, _series([0, 1, 2]), _weights([1.0, -1.0, -1.0]), SCALE_LOG
    )
    assert float(np.abs(combined).max()) < 1e-14
    # A product in place of the log sum gives Q / (Z * M) = 1, not 0.
    assert float(np.abs(combined - 1.0).min()) > 0.5


def test_log_value_against_math_log() -> None:
    rows = [[1.5, 2.0, 7.25, 0.125], [3.0, 0.5, 11.0, 9.75]]
    weights = [1.0, -2.0]
    z = combine_core(_levels(rows), _series([0, 1]), _weights(weights), SCALE_LOG)
    for t in range(4):
        a = weights[0] * math.log(rows[0][t])
        b = weights[1] * math.log(rows[1][t])
        oracle = a + b
        assert abs(float(z[t]) - oracle) <= 8.0 * _U * (abs(a) + abs(b))


# ===================================================================== S1.T3


def _spec_case() -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    return (
        _levels([[1.0, -1.0, 2.0, 0.0, 3.0], [-2.0, 1.0, 1.0, 1.0, 1.0]]),
        _series([0, 1]),
        _weights([1.0, -1.0]),
    )


def test_t3_first_offending_level_is_term_major_ascending_date() -> None:
    levels, term_series, weights = _spec_case()
    with pytest.raises(CoreScaleDomainError) as info:
        combine_core(levels, term_series, weights, SCALE_LOG)
    error = info.value
    assert (error.reason, error.term, error.aligned_index, error.level) == (
        int(codes.SCALE_LOG_LEVEL_NOT_POSITIVE),
        0,
        1,
        -1.0,
    )


def test_t3_no_logarithm_of_a_non_positive_level_is_computed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels, term_series, weights = _spec_case()
    real_log = np.log
    seen: list[NDArray[np.float64]] = []

    def spy(x: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(np.array(x, dtype=np.float64, copy=True))
        return real_log(x, *args, **kwargs)

    monkeypatch.setattr(np, "log", spy)
    with (
        np.errstate(divide="raise", invalid="raise"),
        pytest.raises(CoreScaleDomainError),
    ):
        combine_core(levels, term_series, weights, SCALE_LOG)
    assert seen == []


def test_t3_the_term_is_the_declared_index_not_the_series_index() -> None:
    # Series 0 is bad at date 0; series 1 is bad at date 2. Term 0 reads
    # series 1, so the first offending pair is (term 0, date 2).
    levels = _levels([[0.0, 1.0, 1.0], [1.0, 1.0, -3.0]])
    with pytest.raises(CoreScaleDomainError) as info:
        combine_core(levels, _series([1, 0]), _weights([1.0, 1.0]), SCALE_LOG)
    assert (info.value.term, info.value.aligned_index, info.value.level) == (
        0,
        2,
        -3.0,
    )


def test_t3_a_bad_level_in_a_later_term_names_that_term() -> None:
    levels = _levels([[1.0, 2.0, 3.0], [1.0, 0.0, 2.0]])
    with pytest.raises(CoreScaleDomainError) as info:
        combine_core(levels, _series([0, 1]), _weights([1.0, 1.0]), SCALE_LOG)
    assert (info.value.term, info.value.aligned_index, info.value.level) == (
        1,
        1,
        0.0,
    )


def test_t3_a_series_that_no_term_reads_is_not_domain_checked() -> None:
    levels = _levels([[1.0, 2.0], [-1.0, 0.0], [4.0, 8.0]])
    z = combine_core(levels, _series([0, 2]), _weights([1.0, -1.0]), SCALE_LOG)
    assert z.shape == (2,)


def test_t3_log_with_every_level_above_zero_logs_positive_values_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_log = np.log
    seen: list[NDArray[np.float64]] = []

    def spy(x: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(np.array(x, dtype=np.float64, copy=True))
        return real_log(x, *args, **kwargs)

    monkeypatch.setattr(np, "log", spy)
    levels = _levels([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with np.errstate(divide="raise", invalid="raise"):
        z = combine_core(levels, _series([0, 1]), _weights([1.0, 1.0]), SCALE_LOG)
    assert seen
    assert all(bool((values > 0.0).all()) for values in seen)
    assert np.isfinite(z).all()


def test_t3_level_scale_accepts_non_positive_levels() -> None:
    levels, term_series, weights = _spec_case()
    with np.errstate(divide="raise", invalid="raise"):
        z = combine_core(levels, term_series, weights, SCALE_LEVEL)
    assert z.tolist() == [3.0, -2.0, 1.0, -1.0, 2.0]


# ===================================================================== S1.T4


def test_t4_read_only_inputs_are_unchanged_and_the_result_is_read_only() -> None:
    levels, term_series, weights = _good()
    copies = (levels.copy(), term_series.copy(), weights.copy())
    for array in (levels, term_series, weights):
        array.setflags(write=False)
    for scale in (SCALE_LEVEL, SCALE_LOG):
        z = combine_core(levels, term_series, weights, scale)
        assert not z.flags.writeable
        assert z.flags.c_contiguous
        for array in (levels, term_series, weights):
            assert not np.shares_memory(z, array)
        np.testing.assert_array_equal(levels, copies[0])
        np.testing.assert_array_equal(term_series, copies[1])
        np.testing.assert_array_equal(weights, copies[2])


def test_t4_the_result_does_not_follow_a_later_change_of_the_inputs() -> None:
    levels, term_series, weights = _good()
    z = combine_core(levels, term_series, weights, SCALE_LEVEL)
    before = z.copy()
    levels[:] = 99.0
    weights[:] = 99.0
    term_series[:] = 1
    np.testing.assert_array_equal(z, before)


# ===================================================================== S1.T7

_EXISTING_CONSTANTS = frozenset({
    "LEVEL",
    "LOG_RETURN",
    "SIMPLE_RETURN",
    "DIFFERENCE",
    "IID",
    "HC0",
    "HAC_BARTLETT",
    "OUTSIDE_WINDOW",
    "NOT_COMMON_DATE",
    "NOT_USED_BY_ROWS",
    "CONDITION_ABOVE_THRESHOLD",
    "LEVEL_TERM_PRESENT",
    "R2_ABOVE_DURBIN_WATSON",
    "OVERLAP_NOT_CORRECTED",
    "INPUT_NO_SERIES",
    "INPUT_COUNT_MISMATCH",
    "INPUT_NOT_NDARRAY",
    "INPUT_WRONG_DTYPE",
    "INPUT_WRONG_NDIM",
    "INPUT_WRONG_SHAPE",
    "INPUT_EMPTY",
    "INPUT_TOO_LONG",
    "INPUT_NOT_STRICTLY_INCREASING",
    "INPUT_NON_FINITE",
    "ALIGN_EMPTY_WINDOW",
    "ALIGN_WINDOW_OUT_OF_RANGE",
    "ALIGN_EMPTY_INTERSECTION",
    "PARAM_EMPTY_DESIGN",
    "PARAM_UNKNOWN_TRANSFORM",
    "PARAM_NEGATIVE_STEP",
    "PARAM_STRIDE_BELOW_ONE",
    "PARAM_ZERO_STEP_WITH_RETURN",
    "PARAM_POSITIVE_STEP_LEVEL_ONLY",
    "PARAM_RANK_RTOL_OUT_OF_RANGE",
    "PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE",
    "PARAM_UNKNOWN_COVARIANCE",
    "PARAM_LAG_NOT_ALLOWED",
    "PARAM_NEGATIVE_LAG",
    "PARAM_LAG_ABOVE_LIMIT",
    "PARAM_LAG_OUTSIDE_WINDOW",
    "RANK_ZERO_COLUMN",
    "RANK_DEFICIENT",
    "SAMPLE_STEP_NOT_BELOW_COUNT",
    "SAMPLE_ROWS_NOT_ABOVE_COLUMNS",
    "DEGENERATE_EXACT_FIT",
    "DEGENERATE_ZERO_RESPONSE_VARIANCE",
    "DEGENERATE_NON_POSITIVE_VARIANCE",
    "RESTRICTION_WRONG_SHAPE",
    "RESTRICTION_NON_FINITE",
    "RESTRICTION_RANK_DEFICIENT",
    "RESTRICTION_NOT_POSITIVE_DEFINITE",
})

# A.8: group tuple name -> ordered (constant name, value) pairs.
_NEW_GROUPS: dict[str, tuple[tuple[str, int], ...]] = {
    "SCALE_CODES": (("SCALE_LEVEL", 0), ("SCALE_LOG", 1)),
    "DETERMINISTIC_CODES": (("DET_CONSTANT", 0), ("DET_CONSTANT_TREND", 1)),
    "TEST_KIND_CODES": (("TEST_ADF", 0),),
    "SCALE_DOMAIN_ERROR_CODES": (("SCALE_LOG_LEVEL_NOT_POSITIVE", 0),),
}
_APPENDED = ("SAMPLE_ROWS_BELOW_TABLE_MINIMUM", 2)


def test_t7_the_existing_set_holds_fifty_constants() -> None:
    assert len(_EXISTING_CONSTANTS) == 51


@pytest.mark.parametrize("tuple_name", sorted(_NEW_GROUPS))
def test_t7_new_group_holds_the_a8_values_as_int8(tuple_name: str) -> None:
    group = getattr(codes, tuple_name)
    assert type(group) is tuple
    expected = _NEW_GROUPS[tuple_name]
    assert len(group) == len(expected)
    for code, (name, value) in zip(group, expected, strict=True):
        assert type(code) is np.int8
        assert getattr(codes, name) is code
        assert int(code) == value
    assert len({int(code) for code in group}) == len(group)


def test_t7_sample_rows_below_table_minimum_is_appended_last() -> None:
    name, value = _APPENDED
    code = getattr(codes, name)
    assert type(code) is np.int8
    assert int(code) == value
    group = codes.SAMPLE_SIZE_ERROR_CODES
    assert group[-1] is code
    assert group[0] is codes.SAMPLE_STEP_NOT_BELOW_COUNT
    assert group[1] is codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS
    assert [int(c) for c in group] == [0, 1, 2]
    assert all(type(c) is np.int8 for c in group)


def test_t7_no_new_constant_reuses_an_existing_name() -> None:
    new_names = {name for pairs in _NEW_GROUPS.values() for name, _ in pairs}
    new_names.add(_APPENDED[0])
    assert len(new_names) == 7
    assert not new_names & _EXISTING_CONSTANTS
    declared = {name for name, value in vars(codes).items() if type(value) is np.int8}
    # A clash would rebind an existing name, so the union would shrink.
    assert declared == _EXISTING_CONSTANTS | new_names
    assert len(declared) == 58


# ============================================== checks (owner decision 2)


@pytest.mark.parametrize(
    ("operand", "make"),
    [
        (OPERAND_LEVELS, lambda a: a.tolist()),
        (OPERAND_TERM_SERIES, lambda a: a.tolist()),
        (OPERAND_WEIGHTS, lambda a: a.tolist()),
    ],
)
def test_an_operand_that_is_not_exactly_ndarray_is_rejected(
    operand: int, make: Callable[[Any], Any]
) -> None:
    arrays: list[Any] = list(_good())
    arrays[operand] = make(arrays[operand])
    error = _input_error(*arrays)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        operand,
        (),
    )


def test_masked_levels_with_a_zero_under_the_mask_are_rejected() -> None:
    levels, term_series, weights = _good()
    raw = levels.copy()
    raw[0, 1] = 0.0
    hidden = np.zeros(raw.shape, dtype=np.bool_)
    hidden[0, 1] = True
    masked: NDArray[Any] = np.ma.masked_array(raw, mask=hidden)
    # The mask hides the zero from a reduction.
    assert bool((masked > 0.0).all())
    error = _input_error(masked, term_series, weights, SCALE_LOG)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_LEVELS,
    )


def test_masked_term_series_is_rejected() -> None:
    levels, term_series, weights = _good()
    masked: NDArray[Any] = np.ma.masked_array(
        term_series, mask=np.zeros(2, dtype=np.bool_)
    )
    error = _input_error(levels, masked, weights)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_TERM_SERIES,
    )


def test_masked_weights_with_a_nan_under_the_mask_are_rejected() -> None:
    levels, term_series, _ = _good()
    masked: NDArray[Any] = np.ma.masked_array(
        np.array([1.0, np.nan]), mask=np.array([False, True])
    )
    assert bool(np.isfinite(masked).all())
    error = _input_error(levels, term_series, masked)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_WEIGHTS,
    )


@pytest.mark.parametrize(
    ("operand", "dtype"),
    [
        (OPERAND_LEVELS, np.float32),
        (OPERAND_LEVELS, np.int64),
        (OPERAND_TERM_SERIES, np.int64),
        (OPERAND_TERM_SERIES, np.int8),
        (OPERAND_WEIGHTS, np.float32),
        (OPERAND_WEIGHTS, np.int32),
    ],
)
def test_a_wrong_dtype_is_rejected(operand: int, dtype: type[np.generic]) -> None:
    arrays: list[Any] = list(_good())
    arrays[operand] = arrays[operand].astype(dtype)
    error = _input_error(*arrays)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_WRONG_DTYPE),
        operand,
        (),
    )


@pytest.mark.parametrize(
    ("operand", "reshape"),
    [
        (OPERAND_LEVELS, lambda a: a.reshape(-1)),
        (OPERAND_LEVELS, lambda a: a.reshape(1, 2, 4)),
        (OPERAND_TERM_SERIES, lambda a: a.reshape(1, 2)),
        (OPERAND_WEIGHTS, lambda a: a.reshape(2, 1)),
    ],
)
def test_a_wrong_ndim_is_rejected(operand: int, reshape: Callable[[Any], Any]) -> None:
    arrays: list[Any] = list(_good())
    arrays[operand] = reshape(arrays[operand])
    error = _input_error(*arrays)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_WRONG_NDIM),
        operand,
        (),
    )


def test_every_dtype_check_runs_before_the_first_ndim_check() -> None:
    levels, term_series, weights = _good()
    error = _input_error(levels.reshape(-1), term_series, weights.astype(np.float32))
    assert (error.reason, error.operand) == (
        int(codes.INPUT_WRONG_DTYPE),
        OPERAND_WEIGHTS,
    )


@pytest.mark.parametrize(
    ("arrays", "operand"),
    [
        (
            (np.empty((2, 0)), _series([0, 1]), _weights([1.0, 1.0])),
            OPERAND_LEVELS,
        ),
        (
            (np.empty((0, 3)), _series([0]), _weights([1.0])),
            OPERAND_LEVELS,
        ),
        (
            (_levels([[1.0, 2.0]]), _series([]), _weights([])),
            OPERAND_TERM_SERIES,
        ),
        (
            (_levels([[1.0, 2.0]]), _series([0]), _weights([])),
            OPERAND_WEIGHTS,
        ),
    ],
)
def test_an_empty_array_is_rejected(arrays: tuple[Any, Any, Any], operand: int) -> None:
    error = _input_error(*arrays)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_EMPTY),
        operand,
        (),
    )


@pytest.mark.parametrize("n_weights", [1, 3])
def test_a_weight_count_different_from_the_term_count_is_rejected(
    n_weights: int,
) -> None:
    levels, term_series, _ = _good()
    error = _input_error(levels, term_series, np.ones(n_weights))
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_COUNT_MISMATCH),
        OPERAND_WEIGHTS,
        (),
    )


@pytest.mark.parametrize(
    ("indices", "first"),
    [([0, -1], 1), ([2, 0], 0), ([1, 0, 5, -3], 2)],
)
def test_a_series_index_outside_the_levels_rows_is_rejected(
    indices: list[int], first: int
) -> None:
    levels, _, _ = _good()
    error = _input_error(levels, _series(indices), np.ones(len(indices)))
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_WRONG_SHAPE),
        OPERAND_TERM_SERIES,
        (first,),
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_level_is_rejected_with_its_flat_index(bad: float) -> None:
    levels, term_series, weights = _good()
    levels[1, 2] = bad
    levels[1, 3] = bad
    error = _input_error(levels, term_series, weights)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        OPERAND_LEVELS,
        (1 * 4 + 2,),
    )


def test_a_non_finite_level_in_a_series_that_no_term_reads_is_rejected() -> None:
    levels = _levels([[1.0, 2.0], [np.nan, 3.0]])
    error = _input_error(levels, _series([0]), _weights([1.0]))
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        OPERAND_LEVELS,
        (2,),
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_weight_is_rejected_with_its_index(bad: float) -> None:
    levels, term_series, _ = _good()
    error = _input_error(levels, term_series, _weights([1.0, bad]))
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        OPERAND_WEIGHTS,
        (1,),
    )


def test_levels_are_checked_for_finiteness_before_weights() -> None:
    levels, term_series, _ = _good()
    levels[0, 0] = np.nan
    error = _input_error(levels, term_series, _weights([np.nan, 1.0]))
    assert error.operand == OPERAND_LEVELS


def test_the_count_check_runs_before_the_index_check() -> None:
    levels, _, _ = _good()
    error = _input_error(levels, _series([0, 9]), _weights([1.0]))
    assert error.reason == int(codes.INPUT_COUNT_MISMATCH)


def test_the_index_check_runs_before_the_finiteness_check() -> None:
    levels, _, weights = _good()
    levels[0, 0] = np.nan
    error = _input_error(levels, _series([0, 9]), weights)
    assert error.reason == int(codes.INPUT_WRONG_SHAPE)


def test_a_non_finite_level_wins_over_the_log_domain() -> None:
    levels, term_series, weights = _good()
    levels[0, 0] = -1.0
    levels[1, 0] = np.nan
    error = _input_error(levels, term_series, weights, SCALE_LOG)
    assert error.reason == int(codes.INPUT_NON_FINITE)


@pytest.mark.parametrize(
    "scale",
    [True, False, np.True_, np.False_, 1.0, np.float64(0.0), "SCALE_LOG", None],
)
def test_a_scale_that_is_not_an_integer_raises_type_error(scale: Any) -> None:
    levels, term_series, weights = _good()
    with pytest.raises(TypeError):
        combine_core(levels, term_series, weights, scale)


@pytest.mark.parametrize("scale", [2, -1, np.int8(5), np.int64(127)])
def test_an_unknown_integer_scale_raises_value_error(scale: Any) -> None:
    levels, term_series, weights = _good()
    with pytest.raises(ValueError, match="unknown scale code"):
        combine_core(levels, term_series, weights, scale)


@pytest.mark.parametrize(
    "scale",
    [0, 1, codes.SCALE_LEVEL, codes.SCALE_LOG, np.int64(1), np.int32(0)],
)
def test_every_integer_form_of_a_known_scale_is_accepted(scale: Any) -> None:
    levels, term_series, weights = _good()
    z = combine_core(levels, term_series, weights, scale)
    assert z.shape == (4,)


def test_array_checks_run_before_the_scale_check() -> None:
    levels, term_series, weights = _good()
    error = _input_error(levels, term_series, weights.astype(np.float32), "bad")
    assert error.reason == int(codes.INPUT_WRONG_DTYPE)


def test_the_scale_check_runs_before_the_log_domain() -> None:
    levels, term_series, weights = _spec_case()
    float_scale: Any = 1.0
    with pytest.raises(TypeError):
        combine_core(levels, term_series, weights, float_scale)


def test_an_overflow_gives_inf_with_a_runtime_warning() -> None:
    # Owner decision 4: no overflow guard in the core; S3 rejects a
    # non-finite z.
    levels = _levels([[1e308], [1e308]])
    with pytest.warns(RuntimeWarning):
        z = combine_core(levels, _series([0, 1]), _weights([10.0, 1.0]), SCALE_LEVEL)
    assert z.tolist() == [math.inf]


# ================================================================ C.11 scan


def test_combination_imports_only_the_standard_library_numpy_and_its_package() -> None:
    tree = ast.parse(inspect.getsource(combination))
    allowed_roots = {"__future__", "operator", "typing", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert (
                name.split(".")[0] in allowed_roots
                or name == "riskquant.factor_regression"
                or name.startswith("riskquant.factor_regression.")
            ), name
