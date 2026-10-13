# src/riskquant/factor_regression/align.py
"""
Integer alignment of observed level series. PRIVATE.

The core keeps exactly the dates ``d`` with ``start_ord <= d < end_ord`` that
occur in every series (C.2). It does no fill, interpolation, as-of join, sort,
cast or value change. Every input observation that is not kept gets one
exclusion record (C.3). Dates are ``int32`` ordinals; the core sees no name
and no calendar text (C.22).

Layout. ``levels`` and ``positions`` are ``(n_series, m)`` C-order: one
contiguous row per series. Row construction (A.1) and the domain checks (C.7)
read one series along the date axis, so that axis is the contiguous axis.

Cost. The loop runs over the series, never over the dates. Each series costs
two ``searchsorted`` calls on its window bounds, one ``intersect1d`` on sorted
unique arrays and one ``searchsorted`` for the kept positions.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.errors import CoreAlignmentError, CoreInputError

_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)

_ScalarT = TypeVar("_ScalarT", bound=np.generic)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CoreAlignedLevels:
    """
    Aligned levels on the common date axis, with every exclusion recorded.

    Attributes
    ----------
    start_ord, end_ord
        The declared half-open window ``[start_ord, end_ord)``.
    date_ord
        Kept dates, ``int32`` ``(m,)``, strictly increasing, ``m >= 1``.
    positions
        ``int32`` ``(n_series, m)``. ``positions[j, i]`` is the input position
        of ``date_ord[i]`` in series ``j``.
    levels
        ``float64`` ``(n_series, m)``. ``levels[j, i]`` equals
        ``values[j][positions[j, i]]`` bit for bit.
    excluded_series
        ``int32`` ``(E,)``: series index of each excluded observation.
    excluded_date_ord
        ``int32`` ``(E,)``: date ordinal of each excluded observation.
    excluded_reason
        ``int8`` ``(E,)``: ``codes.OUTSIDE_WINDOW`` or ``codes.NOT_COMMON_DATE``.

    Notes
    -----
    Exclusion records are ordered by series index, then by input position.
    For each series ``j``: ``m`` plus the count of ``excluded_series == j``
    equals the input length. ``(series, date_ord)`` identifies one input
    observation, because each input date axis is strictly increasing.
    ``OUTSIDE_WINDOW`` takes precedence: a date outside the window is not
    tested for membership of the common set. All arrays are read-only.
    ``eq=False`` follows the D064 precedent for array-bearing containers.
    """

    start_ord: int
    end_ord: int
    date_ord: NDArray[np.int32]
    positions: NDArray[np.int32]
    levels: NDArray[np.float64]
    excluded_series: NDArray[np.int32]
    excluded_date_ord: NDArray[np.int32]
    excluded_reason: NDArray[np.int8]

    @property
    def n_series(self) -> int:
        return int(self.levels.shape[0])

    @property
    def n_dates(self) -> int:
        return int(self.date_ord.shape[0])


def _validate_series(
    j: int, date_ord: NDArray[np.int32], values: NDArray[np.float64]
) -> None:
    # Exact type, not isinstance: an ndarray subclass can redefine the
    # comparisons and reductions below. numpy.ma.MaskedArray skips masked
    # elements in .all(), so a NaN or a duplicate date under the mask passes
    # both checks. The core converts nothing, so it rejects every subclass
    # before the first reduction on the operand.
    for operand in (date_ord, values):
        if type(operand) is not np.ndarray:
            raise CoreInputError(
                reason=codes.INPUT_NOT_NDARRAY, operand=j, positions=()
            )
    if date_ord.dtype != np.int32 or values.dtype != np.float64:
        raise CoreInputError(reason=codes.INPUT_WRONG_DTYPE, operand=j, positions=())
    if date_ord.ndim != 1 or values.ndim != 1:
        raise CoreInputError(reason=codes.INPUT_WRONG_NDIM, operand=j, positions=())
    n = int(date_ord.shape[0])
    if int(values.shape[0]) != n:
        raise CoreInputError(reason=codes.INPUT_WRONG_SHAPE, operand=j, positions=())
    if n == 0:
        raise CoreInputError(reason=codes.INPUT_EMPTY, operand=j, positions=())
    if n > _INT32_MAX:
        # Positions are int32; a longer series would wrap silently.
        raise CoreInputError(reason=codes.INPUT_TOO_LONG, operand=j, positions=())
    steps_up = date_ord[1:] > date_ord[:-1]
    if not bool(steps_up.all()):
        first = int(np.flatnonzero(~steps_up)[0]) + 1
        raise CoreInputError(
            reason=codes.INPUT_NOT_STRICTLY_INCREASING, operand=j, positions=(first,)
        )
    finite = np.isfinite(values)
    if not bool(finite.all()):
        first = int(np.flatnonzero(~finite)[0])
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE, operand=j, positions=(first,)
        )


def _read_only(array: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
    array.setflags(write=False)
    return array


def align_core(
    date_ords: Sequence[NDArray[np.int32]],
    values: Sequence[NDArray[np.float64]],
    *,
    start_ord: int,
    end_ord: int,
) -> CoreAlignedLevels:
    """
    Align level series on the dates common to all series inside the window.

    Parameters
    ----------
    date_ords
        One ``int32`` 1-D array per series, strictly increasing, length >= 1.
    values
        One ``float64`` 1-D array per series, all finite, same length as the
        matching ``date_ords`` entry.
    start_ord, end_ord
        Half-open window ``[start_ord, end_ord)`` in ``int32`` ordinal range.
        Each bound is an ``int``. ``operator.index`` also accepts a NumPy
        integer. A ``bool``, ``numpy.bool_``, float or str bound raises
        ``TypeError``: the core never rounds or reinterprets a bound.

    Returns
    -------
    CoreAlignedLevels
        New read-only arrays. No caller array is changed or retained (C.19).

    Raises
    ------
    TypeError
        A window bound is not an integer, or is a ``bool`` or ``numpy.bool_``.
    CoreInputError
        An array breaks the contract above. ``operand`` is the series index.
    CoreAlignmentError
        ``start_ord >= end_ord`` (``ALIGN_EMPTY_WINDOW``), a bound outside the
        ``int32`` range (``ALIGN_WINDOW_OUT_OF_RANGE``), or no date is common
        to every series inside the window (``ALIGN_EMPTY_INTERSECTION``).

    Notes
    -----
    Every array must be exactly ``numpy.ndarray``. ``numpy.ma.MaskedArray``
    and every other ndarray subclass raise ``CoreInputError`` with
    ``INPUT_NOT_NDARRAY``, before any check reads the array. A subclass can
    redefine the comparisons and reductions of the checks: a mask hides a
    NaN level or a duplicate date from them. The core does not convert the
    input, so it rejects it.
    """
    n_series = len(date_ords)
    if n_series == 0:
        raise CoreInputError(reason=codes.INPUT_NO_SERIES, operand=0, positions=())
    if len(values) != n_series:
        raise CoreInputError(
            reason=codes.INPUT_COUNT_MISMATCH,
            operand=min(n_series, len(values)),
            positions=(),
        )
    for j in range(n_series):
        _validate_series(j, date_ords[j], values[j])

    # operator.index rejects a float or str bound with TypeError: the core never
    # rounds a window bound. bool is an int subclass, so operator.index reads
    # True as 1; a flag is not a date ordinal. numpy.bool_ is rejected by the
    # same check, so the outcome does not depend on the NumPy version.
    for bound in (start_ord, end_ord):
        if isinstance(bound, (bool, np.bool_)):
            raise TypeError("a window bound must be an int, not a bool")
    start = operator.index(start_ord)
    end = operator.index(end_ord)
    if not (_INT32_MIN <= start <= _INT32_MAX and _INT32_MIN <= end <= _INT32_MAX):
        raise CoreAlignmentError(
            reason=codes.ALIGN_WINDOW_OUT_OF_RANGE, start_ord=start, end_ord=end
        )
    if start >= end:
        raise CoreAlignmentError(
            reason=codes.ALIGN_EMPTY_WINDOW, start_ord=start, end_ord=end
        )

    # Window bounds per series. The axes are strictly increasing, so the
    # window is one slice [lo, hi) of each axis.
    bounds = np.array([start, end], dtype=np.int32)
    lo_hi = [np.searchsorted(d, bounds, side="left") for d in date_ords]

    common: NDArray[np.int32] = np.array(
        date_ords[0][int(lo_hi[0][0]) : int(lo_hi[0][1])], dtype=np.int32, copy=True
    )
    for j in range(1, n_series):
        lo, hi = int(lo_hi[j][0]), int(lo_hi[j][1])
        common = np.intersect1d(common, date_ords[j][lo:hi], assume_unique=True)
    if common.shape[0] == 0:
        raise CoreAlignmentError(
            reason=codes.ALIGN_EMPTY_INTERSECTION, start_ord=start, end_ord=end
        )
    common = np.ascontiguousarray(common, dtype=np.int32)

    m = int(common.shape[0])
    positions = np.empty((n_series, m), dtype=np.int32)
    levels = np.empty((n_series, m), dtype=np.float64)
    excluded_series: list[NDArray[np.int32]] = []
    excluded_date_ord: list[NDArray[np.int32]] = []
    excluded_reason: list[NDArray[np.int8]] = []
    for j in range(n_series):
        d = date_ords[j]
        lo, hi = int(lo_hi[j][0]), int(lo_hi[j][1])
        # common is a subset of d[lo:hi], so each search hits an exact match.
        kept = lo + np.searchsorted(d[lo:hi], common, side="left")
        positions[j, :] = kept
        levels[j, :] = values[j][kept]

        n = int(d.shape[0])
        is_kept = np.zeros(n, dtype=np.bool_)
        is_kept[kept] = True
        dropped = np.flatnonzero(~is_kept)
        outside = (dropped < lo) | (dropped >= hi)
        excluded_series.append(np.full(dropped.shape[0], j, dtype=np.int32))
        excluded_date_ord.append(d[dropped].astype(np.int32, copy=True))
        excluded_reason.append(
            np.where(outside, codes.OUTSIDE_WINDOW, codes.NOT_COMMON_DATE).astype(
                np.int8
            )
        )

    return CoreAlignedLevels(
        start_ord=start,
        end_ord=end,
        date_ord=_read_only(common),
        positions=_read_only(positions),
        levels=_read_only(levels),
        excluded_series=_read_only(np.concatenate(excluded_series)),
        excluded_date_ord=_read_only(np.concatenate(excluded_date_ord)),
        excluded_reason=_read_only(np.concatenate(excluded_reason)),
    )
