# tests/factor_regression/test_rows.py
"""
S2.T1 to S2.T9: rows and transforms on aligned levels (A.1, C.4, C.6, C.7).

Every oracle is plain Python: ``math.log``, float division and subtraction,
``while`` loops over aligned indices, set logic and ``fractions.Fraction``. It
shares no code with ``build_rows``.

Tolerances. LEVEL, SIMPLE_RETURN and DIFFERENCE are compared bit for bit: each
is a copy or one or two correctly rounded IEEE-754 operations in both the
oracle and the core. LOG_RETURN is compared to two units in the last place of
the oracle value: ``numpy.log`` and ``math.log`` are separate implementations
of the logarithm and can differ in the last bit. Sandbox evidence (numpy
2.4.4, Linux): on 200 000 ratios the largest difference was 1 ulp.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.align import CoreAlignedLevels, align_core
from riskquant.factor_regression.errors import (
    CoreDomainError,
    CoreInputError,
    CoreParameterError,
    CoreSampleSizeError,
)
from riskquant.factor_regression.rows import (
    OPERAND_DATE_ORD,
    OPERAND_LEVELS,
    OPERAND_TERM_SERIES,
    OPERAND_TERM_TRANSFORMS,
    CoreRows,
    build_rows,
)

LEVEL = int(codes.LEVEL)
LOG = int(codes.LOG_RETURN)
SIMPLE = int(codes.SIMPLE_RETURN)
DIFF = int(codes.DIFFERENCE)

# Oracle per transform: (start level, end level) -> row value (A.1).
_ORACLE: dict[int, Callable[[float, float], float]] = {
    LEVEL: lambda start, end: end,
    LOG: lambda start, end: math.log(end / start),
    SIMPLE: lambda start, end: end / start - 1.0,
    DIFF: lambda start, end: end - start,
}


# ------------------------------------------------------------------ helpers


def _terms(
    series: list[int], transforms: list[int]
) -> tuple[NDArray[np.int32], NDArray[np.int8]]:
    return (
        np.array(series, dtype=np.int32),
        np.array(transforms, dtype=np.int8),
    )


def _rows(
    aligned: CoreAlignedLevels,
    series: list[int],
    transforms: list[int],
    *,
    step: int,
    stride: int,
) -> CoreRows:
    term_series, term_transforms = _terms(series, transforms)
    return build_rows(
        aligned,
        term_series=term_series,
        term_transforms=term_transforms,
        step=step,
        stride=stride,
    )


def _hand_aligned(
    date_ord: NDArray[np.int32], levels: NDArray[np.float64]
) -> CoreAlignedLevels:
    """
    Aligned levels built by hand: the arrays pass through unchanged.

    ``build_rows`` reads ``date_ord`` and ``levels`` only. ``positions`` is the
    identity map when ``levels`` is 2-D, so a good input stays consistent.
    """
    positions = (
        np.tile(np.arange(levels.shape[1], dtype=np.int32), (levels.shape[0], 1))
        if levels.ndim == 2
        else np.empty((0, 0), dtype=np.int32)
    )
    return CoreAlignedLevels(
        start_ord=-1_000_000,
        end_ord=1_000_000,
        date_ord=date_ord,
        positions=positions,
        levels=levels,
        excluded_series=np.empty(0, dtype=np.int32),
        excluded_date_ord=np.empty(0, dtype=np.int32),
        excluded_reason=np.empty(0, dtype=np.int8),
    )


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


def _walk(rng: np.random.Generator, start: float, vol: float, n: int) -> list[float]:
    path = start * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
    return [float(x) for x in path]


def _oracle_indices(m: int, h: int, s: int) -> tuple[list[int], list[int], list[int]]:
    """Row start and end indices by an explicit loop, and the unused indices."""
    starts: list[int] = []
    ends: list[int] = []
    e = h
    while e <= m - 1:
        starts.append(e - h)
        ends.append(e)
        e += s
    unused = sorted(set(range(m)) - set(starts) - set(ends))
    return starts, ends, unused


def _assert_values(
    rows: CoreRows,
    levels: list[list[float]],
    series: list[int],
    transforms: list[int],
    starts: list[int],
    ends: list[int],
) -> None:
    assert rows.values.shape == (len(series), len(ends))
    for t, (j, code) in enumerate(zip(series, transforms, strict=True)):
        for r, (b, e) in enumerate(zip(starts, ends, strict=True)):
            expected = _ORACLE[code](levels[j][b], levels[j][e])
            got = float(rows.values[t, r])
            if code == LOG:
                assert abs(got - expected) <= 2.0 * math.ulp(expected), (t, r)
            else:
                assert got == expected, (t, r, got, expected)
                assert math.copysign(1.0, got) == math.copysign(1.0, expected)


def _domain_error(
    aligned: CoreAlignedLevels,
    series: list[int],
    transforms: list[int],
    *,
    step: int,
    stride: int = 1,
) -> CoreDomainError:
    with pytest.raises(CoreDomainError) as info:
        _rows(aligned, series, transforms, step=step, stride=stride)
    return info.value


def _parameter_error(
    aligned: CoreAlignedLevels,
    series: list[int],
    transforms: list[int],
    *,
    step: int,
    stride: int,
) -> CoreParameterError:
    with pytest.raises(CoreParameterError) as info:
        _rows(aligned, series, transforms, step=step, stride=stride)
    return info.value


# ------------------------------------------------------------------ S2.T1


def _aligned_first_oracle(
    dates: list[NDArray[np.int32]],
    values: list[NDArray[np.float64]],
    *,
    start: int,
    end: int,
    series: list[int],
    transforms: list[int],
) -> tuple[list[int], list[int], list[list[float]]]:
    """Common dates by set logic, then one-step rows on the common dates."""
    common = sorted(
        set.intersection(
            *({int(d) for d in ds if start <= int(d) < end} for ds in dates)
        )
    )
    lookups = [
        dict(zip(ds.tolist(), vs.tolist(), strict=True))
        for ds, vs in zip(dates, values, strict=True)
    ]
    start_dates: list[int] = []
    end_dates: list[int] = []
    by_term: list[list[float]] = [[] for _ in series]
    e = 1
    while e < len(common):
        b_date, e_date = common[e - 1], common[e]
        start_dates.append(b_date)
        end_dates.append(e_date)
        for t, (j, code) in enumerate(zip(series, transforms, strict=True)):
            by_term[t].append(_ORACLE[code](lookups[j][b_date], lookups[j][e_date]))
        e += 1
    return start_dates, end_dates, by_term


def _own_calendar_log_returns(
    dates: NDArray[np.int32], values: NDArray[np.float64]
) -> dict[int, tuple[int, float]]:
    """Per-series-first: end date -> (own previous date, log return)."""
    d = dates.tolist()
    v = values.tolist()
    return {d[k]: (d[k - 1], math.log(v[k] / v[k - 1])) for k in range(1, len(d))}


def _check_aligned_first(
    dates: list[NDArray[np.int32]],
    values: list[NDArray[np.float64]],
    *,
    start: int,
    end: int,
) -> list[int]:
    """
    Compare the core with the aligned-first oracle. Return the row positions
    where a per-series-first return covers a different interval.
    """
    series = [0, 1, 0, 1]
    transforms = [LOG, LOG, DIFF, SIMPLE]
    aligned = align_core(dates, values, start_ord=start, end_ord=end)
    rows = _rows(aligned, series, transforms, step=1, stride=1)

    start_dates, end_dates, by_term = _aligned_first_oracle(
        dates, values, start=start, end=end, series=series, transforms=transforms
    )
    assert rows.row_start_date_ord.tolist() == start_dates
    assert rows.row_end_date_ord.tolist() == end_dates
    for t, code in enumerate(transforms):
        for r, expected in enumerate(by_term[t]):
            got = float(rows.values[t, r])
            if code == LOG:
                assert abs(got - expected) <= 2.0 * math.ulp(expected)
            else:
                assert got == expected

    own = [_own_calendar_log_returns(d, v) for d, v in zip(dates, values, strict=True)]
    shifted: list[int] = []
    for r, (b_date, e_date) in enumerate(zip(start_dates, end_dates, strict=True)):
        own_starts = [own[j][e_date][0] for j in (0, 1)]
        if own_starts != [b_date, b_date]:
            shifted.append(r)
            # The per-series-first pair covers two different intervals, and
            # its value differs from the aligned-first value of the core.
            mismatched = [j for j in (0, 1) if own_starts[j] != b_date]
            for j in mismatched:
                assert abs(own[j][e_date][1] - float(rows.values[j, r])) > 1e-12
    return shifted


def test_rows_are_built_on_the_aligned_calendar_on_a_hand_case() -> None:
    a_dates = np.arange(0, 8, dtype=np.int32)
    b_dates = np.array([0, 1, 3, 4, 6, 7], dtype=np.int32)
    a_values = np.array([10.0, 11.0, 9.5, 10.5, 12.0, 11.0, 13.0, 12.5])
    b_values = np.array([2.0, 2.2, 2.1, 1.9, 2.4, 2.3])
    shifted = _check_aligned_first(
        [a_dates, b_dates], [a_values, b_values], start=0, end=8
    )
    # Common dates 0, 1, 3, 4, 6, 7. Series A has dates 2 and 5, so its own
    # returns ending at 3 and 6 start at 2 and 5, not at 1 and 4.
    assert shifted == [1, 3]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rows_are_built_on_the_aligned_calendar_on_random_calendars(seed: int) -> None:
    rng = np.random.default_rng(seed)
    dates = []
    for p in (0.85, 0.7):
        days = np.arange(19_000, 19_300, dtype=np.int32)
        dates.append(np.ascontiguousarray(days[rng.random(300) < p], dtype=np.int32))
    values = [
        np.array(_walk(rng, 100.0, 0.01, int(d.shape[0])), dtype=np.float64)
        for d in dates
    ]
    shifted = _check_aligned_first(dates, values, start=19_010, end=19_290)
    assert shifted, "the calendars must differ for the test to discriminate"


# ------------------------------------------------------------------ S2.T2


def test_one_step_returns_equal_the_endpoint_formula_per_element() -> None:
    rng = np.random.default_rng(11)
    levels = [_walk(rng, 100.0, 0.02, 60), _walk(rng, 1.1, 0.01, 60)]
    levels.append(_walk(rng, 0.5, 0.05, 60))
    series = [0, 1, 2, 0, 0]
    transforms = [LOG, SIMPLE, DIFF, SIMPLE, DIFF]
    aligned = _aligned(levels)
    rows = _rows(aligned, series, transforms, step=1, stride=1)

    starts, ends, unused = _oracle_indices(60, 1, 1)
    _assert_values(rows, levels, series, transforms, starts, ends)
    dates = aligned.date_ord.tolist()
    assert rows.row_start_date_ord.tolist() == [dates[b] for b in starts]
    assert rows.row_end_date_ord.tolist() == [dates[e] for e in ends]
    assert rows.n_rows == 59
    assert rows.overlap_order == 0
    assert unused == []
    assert rows.excluded_index.shape == (0,)


# ------------------------------------------------------------------ S2.T3

_STEP_STRIDE_Q = [(3, 1, 2), (3, 3, 0), (1, 2, 0), (5, 1, 4), (5, 5, 0)]
_INDEX_CASES = [
    pytest.param(h, s, q, m, id=f"h{h}-s{s}-m{m}")
    for h, s, q in _STEP_STRIDE_Q
    for m in (5, 6, 10, 11)
    if h < m
]


@pytest.mark.parametrize(("h", "s", "q", "m"), _INDEX_CASES)
def test_row_count_indices_overlap_and_unused_dates(
    h: int, s: int, q: int, m: int
) -> None:
    rng = np.random.default_rng(h * 100 + s * 10 + m)
    levels = [_walk(rng, 50.0, 0.02, m), _walk(rng, 3.0, 0.02, m)]
    series = [0, 1]
    transforms = [LOG, DIFF]
    aligned = _aligned(levels)
    rows = _rows(aligned, series, transforms, step=h, stride=s)

    starts, ends, unused = _oracle_indices(m, h, s)
    assert rows.n_rows == (m - 1 - h) // s + 1 == len(ends)
    assert rows.row_start_index.tolist() == starts
    assert rows.row_end_index.tolist() == ends
    assert rows.overlap_order == q
    assert (rows.step, rows.stride) == (h, s)
    dates = aligned.date_ord.tolist()
    assert rows.row_start_date_ord.tolist() == [dates[b] for b in starts]
    assert rows.row_end_date_ord.tolist() == [dates[e] for e in ends]
    _assert_values(rows, levels, series, transforms, starts, ends)

    assert rows.excluded_index.tolist() == unused
    assert rows.excluded_date_ord.tolist() == [dates[i] for i in unused]
    assert rows.excluded_reason.tolist() == [int(codes.NOT_USED_BY_ROWS)] * len(unused)
    # Every aligned date is used by a row or recorded, never both.
    used = set(starts) | set(ends)
    assert used.isdisjoint(unused)
    assert used | set(unused) == set(range(m))


def test_the_table_cases_give_the_stated_overlap_order() -> None:
    aligned = _aligned([[float(x) for x in range(1, 13)]])
    got = [
        _rows(aligned, [0], [LOG], step=h, stride=s).overlap_order
        for h, s in ((3, 1), (3, 3), (1, 2))
    ]
    assert got == [2, 0, 0]


@pytest.mark.parametrize("h", [1, 2, 3, 4, 5])
def test_unit_stride_leaves_no_unused_date_unless_m_is_below_2h(h: int) -> None:
    for m in range(h + 1, 2 * h + 3):
        aligned = _aligned([[float(x) for x in range(1, m + 1)]])
        rows = _rows(aligned, [0], [LOG], step=h, stride=1)
        expected = list(range(m - h, h)) if m < 2 * h else []
        assert rows.excluded_index.tolist() == expected, (h, m)


# ------------------------------------------------------------------ S2.T4


@pytest.mark.parametrize("s", [1, 2, 3])
def test_level_rows_read_the_end_level(s: int) -> None:
    levels = [
        [1.5, -2.0, 0.0, 4.25, -0.0, 6.0, 7.5],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    ]
    series = [0, 1]
    transforms = [LEVEL, LEVEL]
    aligned = _aligned(levels)
    rows = _rows(aligned, series, transforms, step=0, stride=s)

    starts, ends, unused = _oracle_indices(7, 0, s)
    assert starts == ends
    _assert_values(rows, levels, series, transforms, starts, ends)
    assert rows.row_start_date_ord.tolist() == rows.row_end_date_ord.tolist()
    assert rows.row_start_index.tolist() == ends
    assert rows.overlap_order == 0
    assert rows.excluded_index.tolist() == unused


def test_a_level_term_beside_a_return_term_reads_the_end_level() -> None:
    levels = [[-1.0, -2.0, 3.0, 0.0, 5.0], [1.0, 1.1, 1.2, 1.3, 1.4]]
    series = [0, 1]
    transforms = [LEVEL, LOG]
    rows = _rows(_aligned(levels), series, transforms, step=2, stride=1)
    starts, ends, _ = _oracle_indices(5, 2, 1)
    _assert_values(rows, levels, series, transforms, starts, ends)
    assert rows.values[0].tolist() == [3.0, 0.0, 5.0]


@pytest.mark.parametrize(
    "transforms",
    [[LOG], [SIMPLE], [DIFF], [LEVEL, LOG], [DIFF, LEVEL]],
    ids=["log", "simple", "difference", "level-and-log", "difference-and-level"],
)
def test_a_zero_step_with_a_return_term_raises(transforms: list[int]) -> None:
    aligned = _aligned([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    series = [j % 2 for j in range(len(transforms))]
    error = _parameter_error(aligned, series, transforms, step=0, stride=1)
    assert error.reason == codes.PARAM_ZERO_STEP_WITH_RETURN
    assert error.value == 0.0


@pytest.mark.parametrize("h", [1, 3])
def test_a_positive_step_with_level_terms_only_raises(h: int) -> None:
    aligned = _aligned([[1.0, 2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0, 8.0]])
    error = _parameter_error(aligned, [0, 1], [LEVEL, LEVEL], step=h, stride=1)
    assert error.reason == codes.PARAM_POSITIVE_STEP_LEVEL_ONLY
    assert error.value == float(h)


@pytest.mark.parametrize("s", [0, -1])
def test_a_stride_below_one_raises(s: int) -> None:
    aligned = _aligned([[1.0, 2.0, 3.0]])
    error = _parameter_error(aligned, [0], [LOG], step=1, stride=s)
    assert error.reason == codes.PARAM_STRIDE_BELOW_ONE
    assert error.value == float(s)


def test_a_negative_step_raises() -> None:
    aligned = _aligned([[1.0, 2.0, 3.0]])
    error = _parameter_error(aligned, [0], [LOG], step=-1, stride=1)
    assert error.reason == codes.PARAM_NEGATIVE_STEP
    assert error.value == -1.0


@pytest.mark.parametrize("code", [4, -1, 127])
def test_an_unknown_transform_code_raises(code: int) -> None:
    aligned = _aligned([[1.0, 2.0, 3.0]])
    error = _parameter_error(aligned, [0, 0], [LOG, code], step=1, stride=1)
    assert error.reason == codes.PARAM_UNKNOWN_TRANSFORM
    assert error.value == float(code)


# ------------------------------------------------------------------ S2.T5


@pytest.mark.parametrize("bad", [0.0, -0.0, -1.5])
def test_a_log_term_rejects_a_non_positive_level(bad: float) -> None:
    rng = np.random.default_rng(5)
    levels = [_walk(rng, 10.0, 0.01, 8) for _ in range(3)]
    levels[2][5] = bad
    # Term 1 reads series 2: the payload names the term, not the series.
    error = _domain_error(_aligned(levels), [0, 2, 1], [LOG, LOG, DIFF], step=1)
    assert (error.column, error.transform, error.aligned_index) == (1, LOG, 5)
    assert error.level == bad
    assert math.copysign(1.0, error.level) == math.copysign(1.0, bad)
    assert type(error.column) is int
    assert type(error.transform) is int
    assert type(error.aligned_index) is int
    assert type(error.level) is float


def test_the_domain_check_covers_levels_that_no_row_reads() -> None:
    levels = [[float(x) for x in range(1, 11)]]
    clean = _rows(_aligned(levels), [0], [LOG], step=1, stride=3)
    assert 2 in clean.excluded_index.tolist()
    levels[0][2] = 0.0
    error = _domain_error(_aligned(levels), [0], [LOG], step=1, stride=3)
    assert (error.column, error.aligned_index, error.level) == (0, 2, 0.0)


def test_the_lowest_term_and_its_first_level_are_reported() -> None:
    levels = [
        [1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 0.0, 8.0],
        [1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ]
    # Term 0 is DIFFERENCE, so its negative level is admissible. Term 2 has an
    # earlier offending index than term 1, but term 1 comes first.
    error = _domain_error(_aligned(levels), [0, 1, 2], [DIFF, LOG, SIMPLE], step=1)
    assert (error.column, error.transform, error.aligned_index) == (1, LOG, 3)
    assert error.level == -4.0


def test_a_level_term_accepts_negative_and_zero_levels() -> None:
    levels = [[-3.0, 0.0, -1.0, 2.0]]
    rows = _rows(_aligned(levels), [0], [LEVEL], step=0, stride=1)
    assert rows.values.tolist() == levels


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(("code", "h"), [(LEVEL, 0), (LOG, 1), (SIMPLE, 1), (DIFF, 1)])
def test_every_transform_rejects_a_non_finite_level(
    bad: float, code: int, h: int
) -> None:
    # align_core keeps finite levels only, so only a hand-built input reaches
    # this rule (C.7: LEVEL and DIFFERENCE need finite levels).
    levels = np.array([[1.0, 2.0, 3.0, 4.0]])
    levels[0, 2] = bad
    aligned = _hand_aligned(np.arange(4, dtype=np.int32), levels)
    error = _domain_error(aligned, [0], [code], step=h)
    assert (error.column, error.transform, error.aligned_index) == (0, code, 2)
    assert (math.isnan(bad) and math.isnan(error.level)) or error.level == bad


# ------------------------------------------------------------------ S2.T6


@pytest.mark.parametrize("bad", [0.0, -0.0])
def test_a_simple_term_rejects_a_zero_level(bad: float) -> None:
    levels = [[2.0, 3.0, bad, 5.0]]
    error = _domain_error(_aligned(levels), [0], [SIMPLE], step=1)
    assert (error.column, error.transform, error.aligned_index) == (0, SIMPLE, 2)
    assert error.level == 0.0


def test_a_simple_term_accepts_negative_levels() -> None:
    levels = [[-2.0, -3.0, 4.0, -0.5, -0.25]]
    rows = _rows(_aligned(levels), [0], [SIMPLE], step=1, stride=1)
    starts, ends, _ = _oracle_indices(5, 1, 1)
    _assert_values(rows, levels, [0], [SIMPLE], starts, ends)
    # A sign change between start and end gives a return below -1.
    assert rows.values[0].tolist()[0] == 0.5
    assert float(rows.values[0, 1]) < -1.0


# ------------------------------------------------------------------ S2.T7


def test_a_difference_term_accepts_zero_and_negative_levels() -> None:
    levels = [[0.0, -1.25, 0.0, -0.0, 3.5, -2.0]]
    series = [0, 0]
    transforms = [DIFF, DIFF]
    for h in (1, 2):
        rows = _rows(_aligned(levels), series, transforms, step=h, stride=1)
        starts, ends, _ = _oracle_indices(6, h, 1)
        _assert_values(rows, levels, series, transforms, starts, ends)


# ------------------------------------------------------------------ S2.T8


@pytest.mark.parametrize("excess", [0, 3])
def test_a_step_at_or_above_the_aligned_count_raises(excess: int) -> None:
    m = 6
    aligned = _aligned([[float(x) for x in range(1, m + 1)]])
    h = m + excess
    with pytest.raises(CoreSampleSizeError) as info:
        _rows(aligned, [0], [LOG], step=h, stride=1)
    error = info.value
    assert error.reason == codes.SAMPLE_STEP_NOT_BELOW_COUNT
    assert (error.n_available, error.n_required) == (m, h + 1)
    assert type(error.n_available) is int
    assert type(error.n_required) is int


@pytest.mark.parametrize("s", [1, 5, 10**30])
def test_a_step_one_below_the_aligned_count_gives_one_row(s: int) -> None:
    m = 6
    aligned = _aligned([[float(x) for x in range(1, m + 1)]])
    rows = _rows(aligned, [0], [LOG], step=m - 1, stride=s)
    assert rows.n_rows == 1
    assert (rows.row_start_index.tolist(), rows.row_end_index.tolist()) == ([0], [5])
    assert abs(float(rows.values[0, 0]) - math.log(6.0)) <= 2.0 * math.ulp(
        math.log(6.0)
    )
    assert rows.stride == s


def test_a_single_aligned_date_gives_one_level_row_and_no_return_row() -> None:
    aligned = _aligned([[4.0]])
    rows = _rows(aligned, [0], [LEVEL], step=0, stride=1)
    assert rows.values.tolist() == [[4.0]]
    with pytest.raises(CoreSampleSizeError) as info:
        _rows(aligned, [0], [LOG], step=1, stride=1)
    assert (info.value.n_available, info.value.n_required) == (1, 2)


# ------------------------------------------------------------------ S2.T9

# Volatility of the case (backlog R-7): daily log volatility 1% for Z and for
# S. The 1e-15 SIMPLE bound depends on it. Sandbox evidence (seeds 0 to 199,
# 250 rows, Q rounded): largest LOG residual 4.2e-16, largest SIMPLE residual
# 4.3e-16, smallest per-seed max |sr_Q - sr_Z - sr_S| 2.8e-4.
_T9_VOL = 0.01
_T9_ROWS = 250


def _identity_levels(
    seed: int, *, exact_product: bool
) -> tuple[list[float], list[float], list[float]]:
    rng = np.random.default_rng(seed)
    z = _walk(rng, 100.0, _T9_VOL, _T9_ROWS + 1)
    fx = _walk(rng, 1.1, _T9_VOL, _T9_ROWS + 1)
    if exact_product:
        # On a 2^-20 grid, z has at most 28 and fx at most 22 significant bits,
        # so the float64 product z * fx is exact.
        grid = 2.0**20
        z = [round(x * grid) / grid for x in z]
        fx = [round(x * grid) / grid for x in fx]
    q = [a * b for a, b in zip(z, fx, strict=True)]
    return z, fx, q


@pytest.mark.parametrize("exact_product", [False, True], ids=["rounded", "exact"])
@pytest.mark.parametrize("seed", range(10))
def test_log_returns_add_and_simple_returns_do_not(
    seed: int, exact_product: bool
) -> None:
    z, fx, q = _identity_levels(seed, exact_product=exact_product)
    rows = _rows(
        _aligned([z, fx, q]),
        [0, 1, 2, 0, 1, 2],
        [LOG, LOG, LOG, SIMPLE, SIMPLE, SIMPLE],
        step=1,
        stride=1,
    )
    assert rows.n_rows == _T9_ROWS
    r_z, r_s, r_q, sr_z, sr_s, sr_q = (rows.values[t] for t in range(6))

    assert float(np.max(np.abs(r_q - r_z - r_s))) <= 1e-15
    cross = sr_q - sr_z - sr_s
    assert float(np.max(np.abs(cross - sr_z * sr_s))) <= 1e-15
    # The additivity claim is wrong for SIMPLE by many orders above 1e-15.
    assert float(np.max(np.abs(cross))) > 1e-8


@pytest.mark.parametrize("seed", range(3))
def test_the_simple_identity_holds_exactly_and_additivity_fails_exactly(
    seed: int,
) -> None:
    # A.7 oracle on the exact-product levels: Fraction arithmetic, no float.
    z, fx, q = _identity_levels(seed, exact_product=True)
    exact = [[Fraction(x) for x in series] for series in (z, fx, q)]
    assert all(c == a * b for a, b, c in zip(exact[0], exact[1], exact[2], strict=True))
    for k in range(1, _T9_ROWS + 1):
        sr_z, sr_s, sr_q = (series[k] / series[k - 1] - 1 for series in exact)
        assert sr_q == sr_z + sr_s + sr_z * sr_s
        assert (sr_q != sr_z + sr_s) == (sr_z * sr_s != 0)


# ---------------------------------------------------- input contract (note 4)


class _TrivialSubclass(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """An ndarray subclass with no override: only the exact type differs."""


def _input_error(
    aligned: CoreAlignedLevels,
    term_series: NDArray[np.int32],
    term_transforms: NDArray[np.int8],
) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        build_rows(
            aligned,
            term_series=term_series,
            term_transforms=term_transforms,
            step=1,
            stride=1,
        )
    return info.value


def _good() -> tuple[CoreAlignedLevels, NDArray[np.int32], NDArray[np.int8]]:
    aligned = _hand_aligned(
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
    )
    term_series, term_transforms = _terms([0, 1], [LOG, DIFF])
    return aligned, term_series, term_transforms


def test_the_good_input_passes() -> None:
    aligned, term_series, term_transforms = _good()
    rows = build_rows(
        aligned,
        term_series=term_series,
        term_transforms=term_transforms,
        step=1,
        stride=1,
    )
    assert rows.n_terms == 2
    assert rows.n_rows == 3


def test_masked_levels_with_a_zero_under_the_mask_are_rejected() -> None:
    _, term_series, term_transforms = _good()
    hidden = np.array([[False, True, False, False], [False, False, False, False]])
    raw = np.array([[1.0, 0.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    masked: NDArray[Any] = np.ma.masked_array(raw, mask=hidden)
    # The mask hides the zero from a reduction: a LOG domain check on the
    # masked array alone passes.
    assert bool((masked > 0.0).all())
    aligned = _hand_aligned(np.array([1, 2, 3, 4], dtype=np.int32), masked)
    error = _input_error(aligned, term_series, term_transforms)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_LEVELS,
        (),
    )


def test_masked_dates_with_a_duplicate_under_the_mask_are_rejected() -> None:
    good, term_series, term_transforms = _good()
    hidden = np.array([False, False, True, False])
    masked: NDArray[Any] = np.ma.masked_array(
        np.array([1, 2, 2, 4], dtype=np.int32), mask=hidden
    )
    assert bool((masked[1:] > masked[:-1]).all())
    aligned = _hand_aligned(masked, np.array(good.levels))
    error = _input_error(aligned, term_series, term_transforms)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_DATE_ORD,
    )


def test_masked_term_arrays_are_rejected() -> None:
    aligned, term_series, term_transforms = _good()
    unmasked = np.zeros(2, dtype=np.bool_)
    masked_series: NDArray[Any] = np.ma.masked_array(term_series, mask=unmasked)
    masked_transforms: NDArray[Any] = np.ma.masked_array(term_transforms, mask=unmasked)
    error = _input_error(aligned, masked_series, term_transforms)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_TERM_SERIES,
    )
    error = _input_error(aligned, term_series, masked_transforms)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_TERM_TRANSFORMS,
    )


def test_a_view_of_a_trivial_ndarray_subclass_is_rejected() -> None:
    good, term_series, term_transforms = _good()
    view = np.array(good.levels).view(_TrivialSubclass)
    assert isinstance(view, np.ndarray)
    aligned = _hand_aligned(np.array(good.date_ord), view)
    error = _input_error(aligned, term_series, term_transforms)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_LEVELS,
    )


def test_input_contract_rejects_without_repair() -> None:
    good, term_series, term_transforms = _good()
    dates = np.array(good.date_ord)
    levels = np.array(good.levels)

    def check(
        aligned: CoreAlignedLevels,
        series: NDArray[np.int32],
        transforms: NDArray[np.int8],
        expected: tuple[int, int, tuple[int, ...]],
    ) -> None:
        error = _input_error(aligned, series, transforms)
        assert (error.reason, error.operand, error.positions) == expected

    # NDArray[Any] carries a wrong dtype past the static types to the check.
    wrong_dtype = int(codes.INPUT_WRONG_DTYPE)
    int64_dates: NDArray[Any] = dates.astype(np.int64)
    float32_levels: NDArray[Any] = levels.astype(np.float32)
    int64_series: NDArray[Any] = term_series.astype(np.int64)
    int32_transforms: NDArray[Any] = term_transforms.astype(np.int32)
    check(
        _hand_aligned(int64_dates, levels),
        term_series,
        term_transforms,
        (wrong_dtype, OPERAND_DATE_ORD, ()),
    )
    check(
        _hand_aligned(dates, float32_levels),
        term_series,
        term_transforms,
        (wrong_dtype, OPERAND_LEVELS, ()),
    )
    check(good, int64_series, term_transforms, (wrong_dtype, OPERAND_TERM_SERIES, ()))
    check(
        good,
        term_series,
        int32_transforms,
        (wrong_dtype, OPERAND_TERM_TRANSFORMS, ()),
    )

    wrong_ndim = int(codes.INPUT_WRONG_NDIM)
    check(
        _hand_aligned(dates, levels[0].copy()),
        term_series,
        term_transforms,
        (wrong_ndim, OPERAND_LEVELS, ()),
    )
    check(
        good,
        term_series.reshape(1, 2),
        term_transforms,
        (wrong_ndim, OPERAND_TERM_SERIES, ()),
    )
    check(
        good,
        term_series,
        term_transforms.reshape(2, 1),
        (wrong_ndim, OPERAND_TERM_TRANSFORMS, ()),
    )

    wrong_shape = int(codes.INPUT_WRONG_SHAPE)
    check(
        _hand_aligned(dates, levels[:, :3].copy()),
        term_series,
        term_transforms,
        (wrong_shape, OPERAND_LEVELS, ()),
    )
    check(
        good,
        term_series,
        term_transforms[:1].copy(),
        (wrong_shape, OPERAND_TERM_TRANSFORMS, ()),
    )
    check(
        good,
        term_series[:0].copy(),
        term_transforms[:0].copy(),
        (int(codes.INPUT_EMPTY), OPERAND_TERM_SERIES, ()),
    )

    not_increasing = int(codes.INPUT_NOT_STRICTLY_INCREASING)
    for bad_dates, first in (([1, 2, 2, 4], 2), ([1, 3, 2, 4], 2), ([4, 3, 2, 1], 1)):
        check(
            _hand_aligned(np.array(bad_dates, dtype=np.int32), levels),
            term_series,
            term_transforms,
            (not_increasing, OPERAND_DATE_ORD, (first,)),
        )

    # A negative series index would wrap silently to the last series.
    for bad_series, first in (([0, 2], 1), ([-1, 0], 0), ([1, -2], 1)):
        check(
            good,
            np.array(bad_series, dtype=np.int32),
            term_transforms,
            (wrong_shape, OPERAND_TERM_SERIES, (first,)),
        )


def test_the_dates_must_be_one_dimensional() -> None:
    good, term_series, term_transforms = _good()
    aligned = _hand_aligned(
        np.array(good.date_ord).reshape(4, 1), np.array(good.levels)
    )
    error = _input_error(aligned, term_series, term_transforms)
    assert (error.reason, error.operand) == (
        int(codes.INPUT_WRONG_NDIM),
        OPERAND_DATE_ORD,
    )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(True, id="True"),
        pytest.param(np.True_, id="numpy-True"),
        pytest.param(1.0, id="float"),
        pytest.param("1", id="str"),
    ],
)
def test_a_bool_float_or_str_step_or_stride_raises_type_error(bad: object) -> None:
    aligned, term_series, term_transforms = _good()
    with pytest.raises(TypeError):
        build_rows(
            aligned,
            term_series=term_series,
            term_transforms=term_transforms,
            step=bad,  # type: ignore[arg-type]
            stride=1,
        )
    with pytest.raises(TypeError):
        build_rows(
            aligned,
            term_series=term_series,
            term_transforms=term_transforms,
            step=1,
            stride=bad,  # type: ignore[arg-type]
        )


def test_numpy_integer_step_and_stride_give_the_int_result() -> None:
    aligned, term_series, term_transforms = _good()
    expected = build_rows(
        aligned,
        term_series=term_series,
        term_transforms=term_transforms,
        step=1,
        stride=2,
    )
    for step, stride in ((np.int32(1), np.int32(2)), (np.int64(1), np.int64(2))):
        got = build_rows(
            aligned,
            term_series=term_series,
            term_transforms=term_transforms,
            step=step,  # type: ignore[arg-type]
            stride=stride,  # type: ignore[arg-type]
        )
        assert (type(got.step), type(got.stride)) == (int, int)
        assert (got.step, got.stride) == (1, 2)
        assert got.values.tolist() == expected.values.tolist()


# ------------------------------------------------------------------- C.19


def test_no_caller_array_is_mutated_or_retained() -> None:
    rng = np.random.default_rng(3)
    aligned = _aligned([_walk(rng, 10.0, 0.01, 12), _walk(rng, 2.0, 0.01, 12)])
    term_series, term_transforms = _terms([0, 1, 0], [LOG, SIMPLE, LEVEL])
    inputs: list[NDArray[np.generic]] = [
        aligned.date_ord,
        aligned.levels,
        term_series,
        term_transforms,
    ]
    before = [x.copy() for x in inputs]
    rows = build_rows(
        aligned,
        term_series=term_series,
        term_transforms=term_transforms,
        step=2,
        stride=3,
    )
    for x, x_before in zip(inputs, before, strict=True):
        assert np.array_equal(x, x_before)
    assert term_series.flags.writeable
    assert term_transforms.flags.writeable

    outputs: list[NDArray[np.generic]] = [
        rows.term_series,
        rows.term_transforms,
        rows.row_start_index,
        rows.row_end_index,
        rows.row_start_date_ord,
        rows.row_end_date_ord,
        rows.values,
        rows.excluded_index,
        rows.excluded_date_ord,
        rows.excluded_reason,
    ]
    for out in outputs:
        assert not out.flags.writeable
        for source in inputs:
            assert not np.shares_memory(out, source)
    assert rows.values.flags.c_contiguous
    assert rows.values.dtype == np.float64
    int32_outputs = [
        rows.term_series,
        rows.row_start_index,
        rows.row_end_index,
        rows.row_start_date_ord,
        rows.row_end_date_ord,
        rows.excluded_index,
        rows.excluded_date_ord,
    ]
    assert all(out.dtype == np.int32 for out in int32_outputs)
    assert rows.term_transforms.dtype == np.int8
    assert rows.excluded_reason.dtype == np.int8


def test_the_result_is_frozen() -> None:
    aligned, term_series, term_transforms = _good()
    rows = build_rows(
        aligned,
        term_series=term_series,
        term_transforms=term_transforms,
        step=1,
        stride=1,
    )
    with pytest.raises(AttributeError):
        rows.step = 2  # type: ignore[misc]
