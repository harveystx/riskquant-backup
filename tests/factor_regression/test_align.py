# tests/factor_regression/test_align.py
"""
S1.T5 to S1.T7: integer alignment (C.2, C.3, C.19).

The oracle is plain Python set logic on the input dates. It shares no code
with ``align_core``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.align import CoreAlignedLevels, align_core
from riskquant.factor_regression.errors import CoreAlignmentError, CoreInputError


def _calendar(
    rng: np.random.Generator, start: int, n: int, p: float
) -> NDArray[np.int32]:
    days = np.arange(start, start + n, dtype=np.int32)
    return np.ascontiguousarray(days[rng.random(n) < p], dtype=np.int32)


def _three_series(
    seed: int,
) -> tuple[list[NDArray[np.int32]], list[NDArray[np.float64]]]:
    rng = np.random.default_rng(seed)
    dates = [
        _calendar(rng, 19_000, 400, 0.9),
        _calendar(rng, 19_020, 400, 0.8),
        _calendar(rng, 18_990, 380, 0.7),
    ]
    values = [
        np.asarray(rng.standard_normal(int(d.shape[0])), dtype=np.float64)
        for d in dates
    ]
    return dates, values


def _oracle(
    dates: list[NDArray[np.int32]], start: int, end: int
) -> tuple[list[int], list[set[tuple[int, int]]], list[set[tuple[int, int]]]]:
    in_window = [{int(x) for x in d if start <= int(x) < end} for d in dates]
    common = set.intersection(*in_window)
    outside: list[set[tuple[int, int]]] = []
    not_common: list[set[tuple[int, int]]] = []
    for j, d in enumerate(dates):
        outside.append({(j, int(x)) for x in d if not start <= int(x) < end})
        not_common.append({
            (j, int(x)) for x in d if start <= int(x) < end and int(x) not in common
        })
    return sorted(common), outside, not_common


# ------------------------------------------------------------------ S1.T5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kept_dates_equal_the_set_intersection_and_positions_index_inputs(
    seed: int,
) -> None:
    dates, values = _three_series(seed)
    start, end = 19_050, 19_330
    aligned = align_core(dates, values, start_ord=start, end_ord=end)

    common, _, _ = _oracle(dates, start, end)
    assert aligned.date_ord.dtype == np.int32
    assert aligned.date_ord.tolist() == common
    assert aligned.positions.shape == (3, len(common))
    assert aligned.levels.shape == (3, len(common))
    assert aligned.positions.flags.c_contiguous
    assert aligned.levels.flags.c_contiguous
    for j in range(3):
        kept = aligned.positions[j]
        assert np.array_equal(dates[j][kept], aligned.date_ord)
        # Bit-for-bit: no value change on the way through.
        assert np.array_equal(
            values[j][kept].view(np.int64), aligned.levels[j].view(np.int64)
        )


def test_a_single_series_keeps_its_window_slice() -> None:
    dates = np.array([3, 5, 8, 13], dtype=np.int32)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    aligned = align_core([dates], [values], start_ord=5, end_ord=13)
    assert aligned.date_ord.tolist() == [5, 8]
    assert aligned.positions.tolist() == [[1, 2]]
    assert aligned.levels.tolist() == [[2.0, 3.0]]
    assert aligned.excluded_date_ord.tolist() == [3, 13]
    assert not np.shares_memory(aligned.date_ord, dates)


# ------------------------------------------------------------------ S1.T6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_unkept_observation_has_exactly_one_correct_record(seed: int) -> None:
    dates, values = _three_series(seed)
    start, end = 19_050, 19_330
    aligned = align_core(dates, values, start_ord=start, end_ord=end)
    _, outside, not_common = _oracle(dates, start, end)

    records = list(
        zip(
            aligned.excluded_series.tolist(),
            aligned.excluded_date_ord.tolist(),
            aligned.excluded_reason.tolist(),
            strict=True,
        )
    )
    keys = [(j, d) for j, d, _ in records]
    assert len(keys) == len(set(keys)), "duplicate exclusion record"

    for j in range(3):
        n_excluded = sum(1 for s, _, _ in records if s == j)
        assert aligned.n_dates + n_excluded == dates[j].shape[0]
        got_outside = {
            (s, d) for s, d, r in records if s == j and r == codes.OUTSIDE_WINDOW
        }
        got_not_common = {
            (s, d) for s, d, r in records if s == j and r == codes.NOT_COMMON_DATE
        }
        assert got_outside == outside[j]
        assert got_not_common == not_common[j]
        kept = {(j, int(d)) for d in aligned.date_ord}
        assert kept | got_outside | got_not_common == {(j, int(x)) for x in dates[j]}

    assert set(aligned.excluded_reason.tolist()) <= {
        int(codes.OUTSIDE_WINDOW),
        int(codes.NOT_COMMON_DATE),
    }
    assert aligned.excluded_series.dtype == np.int32
    assert aligned.excluded_date_ord.dtype == np.int32
    assert aligned.excluded_reason.dtype == np.int8


def test_window_cut_and_missing_date_on_a_hand_case() -> None:
    a = np.array([1, 2, 3, 5], dtype=np.int32)
    b = np.array([2, 3, 4, 5, 9], dtype=np.int32)
    aligned = align_core(
        [a, b], [np.arange(4.0), 10.0 * np.arange(5.0)], start_ord=2, end_ord=9
    )
    assert aligned.date_ord.tolist() == [2, 3, 5]
    assert aligned.levels.tolist() == [[1.0, 2.0, 3.0], [0.0, 10.0, 30.0]]
    # 1 is before the window; 9 equals end_ord, which the half-open window
    # excludes; 4 is inside the window but absent from series 0.
    assert aligned.excluded_series.tolist() == [0, 1, 1]
    assert aligned.excluded_date_ord.tolist() == [1, 4, 9]
    assert aligned.excluded_reason.tolist() == [
        int(codes.OUTSIDE_WINDOW),
        int(codes.NOT_COMMON_DATE),
        int(codes.OUTSIDE_WINDOW),
    ]


def test_full_overlap_records_no_exclusion() -> None:
    d = np.array([10, 11, 12], dtype=np.int32)
    aligned = align_core([d, d], [np.ones(3), np.zeros(3)], start_ord=0, end_ord=100)
    assert aligned.n_dates == 3
    assert aligned.excluded_series.shape == (0,)
    assert aligned.excluded_date_ord.shape == (0,)
    assert aligned.excluded_reason.shape == (0,)


# ------------------------------------------------------------------ S1.T7


def test_an_empty_intersection_raises_with_an_integer_payload() -> None:
    a = np.array([1, 3, 5], dtype=np.int32)
    b = np.array([2, 4, 6], dtype=np.int32)
    with pytest.raises(CoreAlignmentError) as info:
        align_core([a, b], [np.ones(3), np.ones(3)], start_ord=0, end_ord=10)
    assert info.value.reason == codes.ALIGN_EMPTY_INTERSECTION
    assert (info.value.start_ord, info.value.end_ord) == (0, 10)
    assert type(info.value.start_ord) is int
    assert type(info.value.end_ord) is int


def test_a_window_with_no_dates_raises_empty_intersection() -> None:
    a = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(CoreAlignmentError) as info:
        align_core([a], [np.ones(3)], start_ord=50, end_ord=60)
    assert info.value.reason == codes.ALIGN_EMPTY_INTERSECTION


@pytest.mark.parametrize(("start", "end"), [(5, 5), (6, 5)])
def test_a_window_with_start_not_below_end_raises(start: int, end: int) -> None:
    a = np.array([1, 5, 9], dtype=np.int32)
    with pytest.raises(CoreAlignmentError) as info:
        align_core([a], [np.ones(3)], start_ord=start, end_ord=end)
    assert info.value.reason == codes.ALIGN_EMPTY_WINDOW
    assert (info.value.start_ord, info.value.end_ord) == (start, end)


@pytest.mark.parametrize(("start", "end"), [(-(2**31) - 1, 0), (0, 2**31)])
def test_a_window_bound_outside_int32_raises(start: int, end: int) -> None:
    a = np.array([1, 5, 9], dtype=np.int32)
    with pytest.raises(CoreAlignmentError) as info:
        align_core([a], [np.ones(3)], start_ord=start, end_ord=end)
    assert info.value.reason == codes.ALIGN_WINDOW_OUT_OF_RANGE


def test_a_float_window_bound_is_not_rounded() -> None:
    a = np.array([1, 5, 9], dtype=np.int32)
    with pytest.raises(TypeError):
        align_core([a], [np.ones(3)], start_ord=0, end_ord=5.5)  # type: ignore[arg-type]


# ------------------------------------------------------------ input contract


def _input_error(
    dates: list[NDArray[np.int32]], values: list[NDArray[np.float64]]
) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        align_core(dates, values, start_ord=0, end_ord=100)
    return info.value


def test_input_contract_rejects_without_repair() -> None:
    good_d = np.array([1, 2, 3], dtype=np.int32)
    good_v = np.array([1.0, 2.0, 3.0])

    assert _input_error([], []).reason == codes.INPUT_NO_SERIES
    assert _input_error([good_d], []).reason == codes.INPUT_COUNT_MISMATCH

    wrong_dtype = _input_error([good_d, good_d.astype(np.int64)], [good_v, good_v])
    assert (wrong_dtype.reason, wrong_dtype.operand) == (
        int(codes.INPUT_WRONG_DTYPE),
        1,
    )
    float32 = _input_error([good_d], [good_v.astype(np.float32)])
    assert float32.reason == codes.INPUT_WRONG_DTYPE

    two_d = _input_error([good_d.reshape(1, 3)], [good_v.reshape(1, 3)])
    assert two_d.reason == codes.INPUT_WRONG_NDIM
    assert _input_error([good_d], [good_v[:2]]).reason == codes.INPUT_WRONG_SHAPE
    empty = _input_error([good_d[:0]], [good_v[:0]])
    assert empty.reason == codes.INPUT_EMPTY

    unsorted = _input_error([np.array([1, 3, 2], dtype=np.int32)], [good_v])
    assert (unsorted.reason, unsorted.positions) == (
        int(codes.INPUT_NOT_STRICTLY_INCREASING),
        (2,),
    )
    duplicate = _input_error([np.array([1, 2, 2], dtype=np.int32)], [good_v])
    assert (duplicate.reason, duplicate.positions) == (
        int(codes.INPUT_NOT_STRICTLY_INCREASING),
        (2,),
    )
    for bad in (np.nan, np.inf, -np.inf):
        v = good_v.copy()
        v[1] = bad
        non_finite = _input_error([good_d, good_d], [good_v, v])
        assert (non_finite.reason, non_finite.operand, non_finite.positions) == (
            int(codes.INPUT_NON_FINITE),
            1,
            (1,),
        )
    not_array = _input_error([[1, 2, 3]], [good_v])  # type: ignore[list-item]
    assert not_array.reason == codes.INPUT_NOT_NDARRAY


# ------------------------------------------------------------------- C.19


def test_no_caller_array_is_mutated_or_retained() -> None:
    dates, values = _three_series(7)
    dates_before = [d.copy() for d in dates]
    values_before = [v.copy() for v in values]
    aligned = align_core(dates, values, start_ord=19_050, end_ord=19_330)

    for j in range(3):
        assert np.array_equal(dates[j], dates_before[j])
        assert np.array_equal(values[j], values_before[j])
        assert dates[j].flags.writeable
        assert values[j].flags.writeable
    outputs = _outputs(aligned)
    for out in outputs:
        assert not out.flags.writeable
        for source in (*dates, *values):
            assert not np.shares_memory(out, source)


def _outputs(aligned: CoreAlignedLevels) -> list[NDArray[np.generic]]:
    return [
        aligned.date_ord,
        aligned.positions,
        aligned.levels,
        aligned.excluded_series,
        aligned.excluded_date_ord,
        aligned.excluded_reason,
    ]


def test_the_result_is_frozen() -> None:
    d = np.array([1, 2], dtype=np.int32)
    aligned = align_core([d], [np.ones(2)], start_ord=0, end_ord=10)
    with pytest.raises(AttributeError):
        aligned.start_ord = 5  # type: ignore[misc]
