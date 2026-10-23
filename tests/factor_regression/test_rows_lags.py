# tests/factor_regression/test_rows_lags.py
"""
WS_FACTOR_REGRESSION_LAGS S1.T3, S1.T4 and S1.T8: lagged terms in build_rows.

Every oracle is plain Python: ``math.log``, float division, ``while`` loops over
aligned indices and set logic. It shares no code with ``build_rows``.

S1.T3 (C.3, C.4): row indices, row count, the reads of a term with lag ``l``,
the sample-size error and the NOT_USED_BY_ROWS exclusions.
S1.T4 (C.5): with every lag 0 and ``max_lag = 0`` the rows equal an in-test
reconstruction of the A.1 formula bit for bit, with the lag arguments given
and omitted.
S1.T8 (C.9): the ``term_lags`` array contract, ``max_lag`` as a declared
integer and the new reason ``PARAM_LAG_OUTSIDE_WINDOW``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from riskquant.factor_regression import codes
from riskquant.factor_regression.align import CoreAlignedLevels, align_core
from riskquant.factor_regression.errors import (
    CoreInputError,
    CoreParameterError,
    CoreSampleSizeError,
)
from riskquant.factor_regression.rows import (
    OPERAND_TERM_LAGS,
    OPERAND_TERM_TRANSFORMS,
    CoreRows,
    build_rows,
)

LEVEL = int(codes.LEVEL)
LOG = int(codes.LOG_RETURN)
SIMPLE = int(codes.SIMPLE_RETURN)
DIFF = int(codes.DIFFERENCE)

_ORACLE: dict[int, Callable[[float, float], float]] = {
    LEVEL: lambda start, end: end,
    LOG: lambda start, end: math.log(end / start),
    SIMPLE: lambda start, end: end / start - 1.0,
    DIFF: lambda start, end: end - start,
}


# ------------------------------------------------------------------ helpers


def _aligned(levels: list[list[float]]) -> CoreAlignedLevels:
    """Series on one calendar 100, 101, ... through align_core."""
    m = len(levels[0])
    dates = np.arange(100, 100 + m, dtype=np.int32)
    return align_core(
        [dates.copy() for _ in levels],
        [np.array(row, dtype=np.float64) for row in levels],
        start_ord=0,
        end_ord=1_000,
    )


def _walks(seed: int, n_series: int, m: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    out: list[list[float]] = []
    for _ in range(n_series):
        path = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, m)))
        out.append([float(x) for x in path])
    return out


def _rows(
    aligned: CoreAlignedLevels,
    series: list[int],
    transforms: list[int],
    lags: list[int] | None,
    *,
    step: int,
    stride: int,
    max_lag: int,
) -> CoreRows:
    return build_rows(
        aligned,
        term_series=np.array(series, dtype=np.int32),
        term_transforms=np.array(transforms, dtype=np.int8),
        term_lags=None if lags is None else np.array(lags, dtype=np.int32),
        step=step,
        stride=stride,
        max_lag=max_lag,
    )


def _oracle_ends(m: int, h: int, s: int, lag_window: int) -> list[int]:
    """Row end indices of C.3 by an explicit loop."""
    ends: list[int] = []
    e = h + lag_window
    while e <= m - 1 - lag_window:
        ends.append(e)
        e += s
    return ends


def _oracle_unread(
    m: int, ends: list[int], h: int, transforms: list[int], lags: list[int]
) -> list[int]:
    """Aligned indices that no term reads on any row (C.4), by set logic."""
    read: set[int] = set()
    for e in ends:
        for transform, lag in zip(transforms, lags, strict=True):
            read.add(e - lag)
            if transform != LEVEL:
                read.add(e - lag - h)
    return sorted(set(range(m)) - read)


# ================================================================ S1.T3


@pytest.mark.parametrize(
    ("m", "h", "s", "lag_window"),
    [
        (12, 1, 1, 1),
        (12, 1, 1, 2),
        (20, 3, 2, 2),
        (20, 5, 4, 3),
        (9, 1, 1, 3),
        (30, 2, 7, 4),
    ],
)
def test_t3_row_indices_and_count_follow_c3(
    m: int, h: int, s: int, lag_window: int
) -> None:
    aligned = _aligned(_walks(1, 2, m))
    rows = _rows(
        aligned,
        [0, 1, 1],
        [LOG, LOG, LOG],
        [0, lag_window, -lag_window],
        step=h,
        stride=s,
        max_lag=lag_window,
    )
    ends = _oracle_ends(m, h, s, lag_window)
    expected_count = (m - 1 - h - 2 * lag_window) // s + 1
    assert len(ends) == expected_count
    assert rows.n_rows == expected_count
    assert rows.row_end_index.tolist() == ends
    assert rows.row_start_index.tolist() == [e - h for e in ends]
    assert rows.row_end_date_ord.tolist() == [100 + e for e in ends]
    assert rows.row_start_date_ord.tolist() == [100 + e - h for e in ends]
    assert rows.max_lag == lag_window
    assert rows.term_lags.tolist() == [0, lag_window, -lag_window]
    assert rows.step == h
    assert rows.stride == s


@pytest.mark.parametrize("transform", [LOG, SIMPLE, DIFF])
@pytest.mark.parametrize("h", [1, 3])
def test_t3_a_lagged_term_reads_its_shifted_interval(transform: int, h: int) -> None:
    m = 25
    levels = _walks(7, 3, m)
    aligned = _aligned(levels)
    series = [0, 1, 1, 2, 2, 1]
    transforms = [transform] * 6
    lags = [0, 0, 2, -1, 1, -2]
    rows = _rows(aligned, series, transforms, lags, step=h, stride=2, max_lag=2)
    ends = _oracle_ends(m, h, 2, 2)
    assert rows.n_rows == len(ends)
    for t, (j, lag) in enumerate(zip(series, lags, strict=True)):
        for k, e in enumerate(ends):
            v = levels[j]
            expected = _ORACLE[transform](v[e - lag - h], v[e - lag])
            got = float(rows.values[t, k])
            if transform == LOG:
                assert got == pytest.approx(
                    expected, rel=0.0, abs=2.0 * np.spacing(abs(expected))
                )
            else:
                assert got == expected


def test_t3_a_level_term_with_a_lag_reads_its_shifted_end_only() -> None:
    m = 15
    levels = _walks(8, 2, m)
    aligned = _aligned(levels)
    rows = _rows(
        aligned, [0, 1, 1], [LOG, LEVEL, LEVEL], [0, 2, -1], step=1, stride=1, max_lag=2
    )
    ends = _oracle_ends(m, 1, 1, 2)
    assert rows.values[1].tolist() == [levels[1][e - 2] for e in ends]
    assert rows.values[2].tolist() == [levels[1][e + 1] for e in ends]


def test_t3_the_common_sample_depends_on_max_lag_not_on_the_declared_lags() -> None:
    aligned = _aligned(_walks(2, 2, 18))
    base = _rows(aligned, [0, 1], [LOG, LOG], [0, 0], step=1, stride=1, max_lag=3)
    shifted = _rows(aligned, [0, 1], [LOG, LOG], [0, 3], step=1, stride=1, max_lag=3)
    lead = _rows(aligned, [0, 1], [LOG, LOG], [0, -3], step=1, stride=1, max_lag=3)
    for other in (shifted, lead):
        assert other.row_end_index.tolist() == base.row_end_index.tolist()
        assert other.row_start_index.tolist() == base.row_start_index.tolist()
        # Term 0 has lag 0 in all three fits: identical response rows.
        np.testing.assert_array_equal(other.values[0], base.values[0])


def test_t3_a_lag_shifts_by_aligned_dates_independently_of_the_step() -> None:
    m = 20
    levels = _walks(5, 1, m)
    aligned = _aligned(levels)
    h = 4
    rows = _rows(aligned, [0, 0], [DIFF, DIFF], [0, 1], step=h, stride=1, max_lag=1)
    ends = _oracle_ends(m, h, 1, 1)
    v = levels[0]
    # Lag 1 reads [e - 1 - h, e - 1]: one aligned date earlier, not h earlier.
    assert rows.values[1].tolist() == [v[e - 1] - v[e - 1 - h] for e in ends]


@pytest.mark.parametrize(
    ("m", "h", "lag_window"),
    [
        (5, 1, 2),
        (4, 2, 1),
        (2, 0, 1),
        (6, 2, 2),
    ],
)
def test_t3_no_row_raises_the_sample_size_error_with_the_c3_counts(
    m: int, h: int, lag_window: int
) -> None:
    assert m - 1 < h + 2 * lag_window
    aligned = _aligned(_walks(3, 2, m))
    transforms = [LEVEL, LEVEL] if h == 0 else [LOG, LOG]
    with pytest.raises(CoreSampleSizeError) as info:
        _rows(aligned, [0, 1], transforms, [0, 1], step=h, stride=1, max_lag=lag_window)
    assert info.value.reason == int(codes.SAMPLE_STEP_NOT_BELOW_COUNT)
    assert info.value.n_available == m
    assert info.value.n_required == h + 2 * lag_window + 1


def test_t3_the_smallest_sample_gives_exactly_one_row() -> None:
    h, lag_window = 2, 3
    m = h + 2 * lag_window + 1
    aligned = _aligned(_walks(4, 2, m))
    rows = _rows(
        aligned,
        [0, 1, 1],
        [LOG, LOG, LOG],
        [0, 3, -3],
        step=h,
        stride=5,
        max_lag=lag_window,
    )
    assert rows.n_rows == 1
    assert rows.row_end_index.tolist() == [h + lag_window]
    assert rows.row_start_index.tolist() == [lag_window]


# ---- C.4: NOT_USED_BY_ROWS lists exactly the aligned dates no term reads


@pytest.mark.parametrize(
    ("m", "h", "s", "lag_window", "transforms", "lags"),
    [
        (14, 1, 1, 1, [LOG, LOG, LOG], [0, 1, -1]),
        (14, 1, 1, 2, [LOG, LOG], [0, 0]),
        (20, 2, 3, 2, [LOG, LOG, LEVEL], [0, 2, -2]),
        (20, 3, 4, 1, [DIFF, LEVEL, SIMPLE], [0, 1, -1]),
        (16, 0, 2, 2, [LEVEL, LEVEL, LEVEL], [0, 2, -1]),
        (25, 1, 5, 3, [LOG, LOG, LOG, LOG], [0, 3, 1, -2]),
    ],
)
def test_t3_exclusions_are_exactly_the_unread_aligned_dates(
    m: int,
    h: int,
    s: int,
    lag_window: int,
    transforms: list[int],
    lags: list[int],
) -> None:
    aligned = _aligned(_walks(6, 2, m))
    series = [0] + [1] * (len(transforms) - 1)
    rows = _rows(
        aligned, series, transforms, lags, step=h, stride=s, max_lag=lag_window
    )
    ends = _oracle_ends(m, h, s, lag_window)
    unread = _oracle_unread(m, ends, h, transforms, lags)
    assert rows.excluded_index.tolist() == unread
    assert rows.excluded_date_ord.tolist() == [100 + i for i in unread]
    assert rows.excluded_reason.tolist() == [int(codes.NOT_USED_BY_ROWS)] * len(unread)
    assert rows.excluded_index.dtype == np.int32
    assert rows.excluded_reason.dtype == np.int8


def test_t3_a_level_term_does_not_mark_its_start_index() -> None:
    # Row 0 of h=1, L=1, s=1 on m=6 ends at 2. The LEVEL term with lag 1
    # reads index 1 only; the return response reads 1 and 2. With lag -1
    # the LEVEL term reads 3 only, so index 0 is read by no term.
    aligned = _aligned(_walks(9, 2, 6))
    rows = _rows(aligned, [0, 1], [LOG, LEVEL], [0, -1], step=1, stride=10, max_lag=1)
    assert rows.n_rows == 1
    assert rows.excluded_index.tolist() == [0, 4, 5]


# ================================================================ S1.T4
# Every lag 0 and max_lag 0: the A.1 rows, bit for bit, by reconstruction.


def _a1_rows(
    levels: list[list[float]], series: list[int], transforms: list[int], h: int, s: int
) -> tuple[list[int], list[int], list[list[float]], list[int]]:
    """The A.1 formula: e_r = h + r*s, b_r = e_r - h, values by the oracle."""
    m = len(levels[0])
    ends: list[int] = []
    e = h
    while e <= m - 1:
        ends.append(e)
        e += s
    starts = [e - h for e in ends]
    values = [
        [
            _ORACLE[t](levels[j][b], levels[j][e])
            for b, e in zip(starts, ends, strict=True)
        ]
        for j, t in zip(series, transforms, strict=True)
    ]
    used = set(starts) | set(ends)
    unused = sorted(set(range(m)) - used)
    return starts, ends, values, unused


@pytest.mark.parametrize(
    ("series", "transforms", "h", "s"),
    [
        ([0, 1], [LOG, LOG], 1, 1),
        ([0, 1, 2], [SIMPLE, DIFF, LEVEL], 3, 2),
        ([0, 0, 1], [DIFF, DIFF, DIFF], 5, 3),
        ([0, 1, 2], [LEVEL, LEVEL, LEVEL], 0, 2),
    ],
)
@pytest.mark.parametrize("give_lags", [True, False])
def test_t4_lag_zero_rows_equal_the_a1_reconstruction_bit_for_bit(
    series: list[int], transforms: list[int], h: int, s: int, give_lags: bool
) -> None:
    levels = _walks(10, 3, 40)
    aligned = _aligned(levels)
    rows = _rows(
        aligned,
        series,
        transforms,
        [0] * len(series) if give_lags else None,
        step=h,
        stride=s,
        max_lag=0,
    )
    starts, ends, values, unused = _a1_rows(levels, series, transforms, h, s)
    assert rows.row_start_index.tolist() == starts
    assert rows.row_end_index.tolist() == ends
    assert rows.excluded_index.tolist() == unused
    assert rows.max_lag == 0
    assert rows.term_lags.tolist() == [0] * len(series)
    for t, transform in enumerate(transforms):
        got = rows.values[t].tolist()
        if transform == LOG:
            for g, v in zip(got, values[t], strict=True):
                assert abs(g - v) <= 2.0 * np.spacing(abs(v))
        else:
            assert got == values[t]


def test_t4_explicit_zero_lags_and_omitted_lags_give_identical_results() -> None:
    aligned = _aligned(_walks(11, 2, 30))
    with_lags = _rows(
        aligned, [0, 1], [LOG, SIMPLE], [0, 0], step=2, stride=3, max_lag=0
    )
    without = build_rows(
        aligned,
        term_series=np.array([0, 1], dtype=np.int32),
        term_transforms=np.array([LOG, SIMPLE], dtype=np.int8),
        step=2,
        stride=3,
    )
    np.testing.assert_array_equal(with_lags.values, without.values)
    np.testing.assert_array_equal(with_lags.row_start_index, without.row_start_index)
    np.testing.assert_array_equal(with_lags.row_end_index, without.row_end_index)
    np.testing.assert_array_equal(with_lags.excluded_index, without.excluded_index)
    assert (with_lags.step, with_lags.stride, with_lags.overlap_order) == (
        without.step,
        without.stride,
        without.overlap_order,
    )
    assert without.max_lag == 0
    assert without.term_lags.tolist() == [0, 0]
    assert without.term_lags.dtype == np.int32
    assert not without.term_lags.flags.writeable


# ================================================================ S1.T8


def test_t8_the_new_reason_is_appended_last_to_its_group() -> None:
    group = codes.PARAMETER_ERROR_CODES
    assert group[-1] is codes.PARAM_LAG_OUTSIDE_WINDOW
    assert type(codes.PARAM_LAG_OUTSIDE_WINDOW) is np.int8
    assert int(codes.PARAM_LAG_OUTSIDE_WINDOW) == 12
    assert group[-2] is codes.PARAM_LAG_ABOVE_LIMIT
    assert len({int(c) for c in group}) == len(group)


@pytest.mark.parametrize(
    ("lags", "lag_window", "offending"),
    [
        ([0, 2], 1, 2),
        ([0, -2], 1, -2),
        ([0, 1, 3, 2], 2, 3),
        ([0, 1], 0, 1),
        ([0, 0], -1, 0),
    ],
)
def test_t8_a_lag_outside_the_window_raises_the_new_reason(
    lags: list[int], lag_window: int, offending: int
) -> None:
    aligned = _aligned(_walks(12, 2, 30))
    series = [0] + [1] * (len(lags) - 1)
    with pytest.raises(CoreParameterError) as info:
        _rows(
            aligned,
            series,
            [LOG] * len(lags),
            lags,
            step=1,
            stride=1,
            max_lag=lag_window,
        )
    assert info.value.reason == int(codes.PARAM_LAG_OUTSIDE_WINDOW)
    assert info.value.value == float(offending)


def test_t8_the_lag_check_runs_after_the_existing_parameter_checks() -> None:
    aligned = _aligned(_walks(13, 2, 30))
    # A negative step is reported before the lag outside the window.
    with pytest.raises(CoreParameterError) as info:
        _rows(aligned, [0, 1], [LOG, LOG], [0, 5], step=-1, stride=1, max_lag=1)
    assert info.value.reason == int(codes.PARAM_NEGATIVE_STEP)
    # A lag outside the window is reported before the sample-size error.
    with pytest.raises(CoreParameterError) as info:
        _rows(aligned, [0, 1], [LOG, LOG], [0, 40], step=1, stride=1, max_lag=30)
    assert info.value.reason == int(codes.PARAM_LAG_OUTSIDE_WINDOW)


def test_t8_an_extreme_int32_lag_is_compared_without_overflow() -> None:
    aligned = _aligned(_walks(14, 2, 30))
    lags = [0, int(np.iinfo(np.int32).min)]
    with pytest.raises(CoreParameterError) as info:
        _rows(aligned, [0, 1], [LOG, LOG], lags, step=1, stride=1, max_lag=5)
    assert info.value.reason == int(codes.PARAM_LAG_OUTSIDE_WINDOW)
    assert info.value.value == float(lags[1])


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: [0, 1], codes.INPUT_NOT_NDARRAY),
        (lambda: np.array([0, 1], dtype=np.int64), codes.INPUT_WRONG_DTYPE),
        (lambda: np.array([0, 1], dtype=np.int8), codes.INPUT_WRONG_DTYPE),
        (lambda: np.array([[0, 1]], dtype=np.int32), codes.INPUT_WRONG_NDIM),
        (lambda: np.array([0, 1, 0], dtype=np.int32), codes.INPUT_WRONG_SHAPE),
        (lambda: np.array([0], dtype=np.int32), codes.INPUT_WRONG_SHAPE),
    ],
)
def test_t8_term_lags_is_the_fifth_operand_of_the_array_contract(
    make: Callable[[], Any], reason: np.int8
) -> None:
    aligned = _aligned(_walks(15, 2, 30))
    with pytest.raises(CoreInputError) as info:
        build_rows(
            aligned,
            term_series=np.array([0, 1], dtype=np.int32),
            term_transforms=np.array([LOG, LOG], dtype=np.int8),
            term_lags=make(),
            step=1,
            stride=1,
            max_lag=1,
        )
    assert info.value.reason == int(reason)
    assert info.value.operand == OPERAND_TERM_LAGS
    assert OPERAND_TERM_LAGS == OPERAND_TERM_TRANSFORMS + 1


def test_t8_a_masked_lag_array_is_rejected_before_any_reduction() -> None:
    aligned = _aligned(_walks(16, 2, 30))
    masked = np.ma.MaskedArray(np.array([0, 99], dtype=np.int32), mask=[False, True])
    with pytest.raises(CoreInputError) as info:
        build_rows(
            aligned,
            term_series=np.array([0, 1], dtype=np.int32),
            term_transforms=np.array([LOG, LOG], dtype=np.int8),
            term_lags=masked,
            step=1,
            stride=1,
            max_lag=1,
        )
    assert info.value.reason == int(codes.INPUT_NOT_NDARRAY)
    assert info.value.operand == OPERAND_TERM_LAGS


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.0, "1", None, 1.5])
def test_t8_max_lag_must_be_a_declared_integer(bad: object) -> None:
    aligned = _aligned(_walks(17, 2, 30))
    with pytest.raises(TypeError):
        build_rows(
            aligned,
            term_series=np.array([0, 1], dtype=np.int32),
            term_transforms=np.array([LOG, LOG], dtype=np.int8),
            term_lags=np.array([0, 0], dtype=np.int32),
            step=1,
            stride=1,
            max_lag=bad,  # type: ignore[arg-type]
        )


def test_t8_a_numpy_integer_max_lag_is_stored_as_int() -> None:
    aligned = _aligned(_walks(18, 2, 30))
    rows = build_rows(
        aligned,
        term_series=np.array([0, 1], dtype=np.int32),
        term_transforms=np.array([LOG, LOG], dtype=np.int8),
        term_lags=np.array([0, 1], dtype=np.int32),
        step=1,
        stride=1,
        max_lag=np.int64(1),  # type: ignore[arg-type]
    )
    assert type(rows.max_lag) is int
    assert rows.max_lag == 1


def test_t8_no_caller_array_is_mutated_or_retained() -> None:
    aligned = _aligned(_walks(19, 2, 30))
    lags = np.array([0, 1], dtype=np.int32)
    before = lags.copy()
    rows = build_rows(
        aligned,
        term_series=np.array([0, 1], dtype=np.int32),
        term_transforms=np.array([LOG, LOG], dtype=np.int8),
        term_lags=lags,
        step=1,
        stride=1,
        max_lag=1,
    )
    np.testing.assert_array_equal(lags, before)
    assert not np.shares_memory(rows.term_lags, lags)
    assert not rows.term_lags.flags.writeable
    lags[1] = 7
    assert rows.term_lags.tolist() == [0, 1]
