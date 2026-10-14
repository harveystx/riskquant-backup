# src/riskquant/factor_regression/rows.py
"""
Row construction and transform domain checks on aligned levels. PRIVATE.

Rows are built after alignment, from aligned levels only (C.4). Every term of
one row covers the same aligned interval: row ``r`` reads the aligned indices
``b_r = e_r - h`` and ``e_r = h + r*s`` of every term series (A.1). The core
never forms a return on the calendar of one series and aligns afterwards,
because the pair ``(r_P[t], r_Z[t])`` would then cover different intervals.

Transforms are integer codes of :mod:`riskquant.factor_regression.codes`. The
core sees no name, unit or label (C.22).

Domain rules (C.7) follow the D025 convention and run over every aligned level
of a term series, before any return exists: LOG_RETURN needs every level
finite and above 0, SIMPLE_RETURN finite and non-zero, LEVEL and DIFFERENCE
finite. A level that no row reads is checked too, so the outcome does not
depend on the declared step and stride.

Layout. ``values`` is ``(n_terms, R)`` C-order: one contiguous row per term,
as ``CoreAlignedLevels.levels`` holds one contiguous row per series. Each term
is a gather along the date axis of one source row into one output row, so both
the reads and the writes run along the contiguous axis. The design module
assembles the ``(R, k)`` design with the intercept column itself (A.2).

Cost. The loop runs over the terms, never over the rows. Each term costs one
gather for LEVEL and two gathers and one or two in-place ufunc passes for a
return. One ``(R,)`` scratch buffer serves every term.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.align import CoreAlignedLevels
from riskquant.factor_regression.errors import (
    CoreDomainError,
    CoreInputError,
    CoreParameterError,
    CoreSampleSizeError,
)

# Operand positions of a CoreInputError raised by build_rows: the order in
# which build_rows reads its four arrays.
OPERAND_DATE_ORD: Final = 0
OPERAND_LEVELS: Final = 1
OPERAND_TERM_SERIES: Final = 2
OPERAND_TERM_TRANSFORMS: Final = 3

_TRANSFORM_CODES: Final = np.array(codes.TRANSFORM_CODES, dtype=np.int8)

_ScalarT = TypeVar("_ScalarT", bound=np.generic)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CoreRows:
    """
    Rows of every term on the aligned date axis (A.1).

    Attributes
    ----------
    step, stride
        The declared ``h >= 0`` and ``s >= 1``.
    overlap_order
        ``q = ceil(h / s) - 1`` for ``h >= 1``, ``0`` for ``h = 0``. Adjacent
        rows share increments up to lag ``q``, so the row errors can carry an
        MA(``q``) structure.
    term_series
        ``int32`` ``(n_terms,)``: the series index of each term.
    term_transforms
        ``int8`` ``(n_terms,)``: the transform code of each term.
    row_start_index, row_end_index
        ``int32`` ``(R,)``: the aligned indices ``b_r`` and ``e_r``.
    row_start_date_ord, row_end_date_ord
        ``int32`` ``(R,)``: ``date_ord[b_r]`` and ``date_ord[e_r]``. Equal for
        ``h = 0``.
    values
        ``float64`` ``(n_terms, R)`` C-order. ``values[t, r]`` is term ``t``
        on row ``r``: LEVEL ``v[e_r]``, LOG_RETURN ``ln(v[e_r] / v[b_r])``,
        SIMPLE_RETURN ``v[e_r] / v[b_r] - 1``, DIFFERENCE ``v[e_r] - v[b_r]``.
    excluded_index, excluded_date_ord
        ``int32`` ``(U,)``: aligned index and date of each aligned date that is
        neither a start nor an end index of any row, ascending.
    excluded_reason
        ``int8`` ``(U,)``: ``codes.NOT_USED_BY_ROWS`` for every record.

    Notes
    -----
    All arrays are new and read-only. ``eq=False`` follows the D064 precedent
    for array-bearing containers.
    """

    step: int
    stride: int
    overlap_order: int
    term_series: NDArray[np.int32]
    term_transforms: NDArray[np.int8]
    row_start_index: NDArray[np.int32]
    row_end_index: NDArray[np.int32]
    row_start_date_ord: NDArray[np.int32]
    row_end_date_ord: NDArray[np.int32]
    values: NDArray[np.float64]
    excluded_index: NDArray[np.int32]
    excluded_date_ord: NDArray[np.int32]
    excluded_reason: NDArray[np.int8]

    @property
    def n_terms(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[1])


def _read_only(array: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
    array.setflags(write=False)
    return array


def _declared_integer(value: int) -> int:
    # Same rule as the window bounds of align_core: operator.index rejects a
    # float or str with TypeError, so the core never rounds a step or stride.
    # bool is an int subclass, so operator.index reads True as 1; a flag is
    # not a step. numpy.bool_ is rejected by the same check, so the outcome
    # does not depend on the NumPy version.
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("a step or stride must be an int, not a bool")
    return operator.index(value)


def _validate_arrays(
    date_ord: NDArray[np.int32],
    levels: NDArray[np.float64],
    term_series: NDArray[np.int32],
    term_transforms: NDArray[np.int8],
) -> None:
    operands: tuple[NDArray[np.generic], ...] = (
        date_ord,
        levels,
        term_series,
        term_transforms,
    )
    # Exact type, not isinstance, before the first reduction on any operand.
    # numpy.ma.MaskedArray skips masked elements in .all(): a zero level under
    # the mask passes the LOG domain check, and a duplicate date passes the
    # order check. The core converts nothing, so it rejects every subclass.
    for k, operand in enumerate(operands):
        if type(operand) is not np.ndarray:
            raise CoreInputError(
                reason=codes.INPUT_NOT_NDARRAY, operand=k, positions=()
            )
    expected: tuple[tuple[type[np.generic], int], ...] = (
        (np.int32, 1),
        (np.float64, 2),
        (np.int32, 1),
        (np.int8, 1),
    )
    for k, (operand, (dtype, ndim)) in enumerate(zip(operands, expected, strict=True)):
        if operand.dtype != dtype:
            raise CoreInputError(
                reason=codes.INPUT_WRONG_DTYPE, operand=k, positions=()
            )
        if operand.ndim != ndim:
            raise CoreInputError(reason=codes.INPUT_WRONG_NDIM, operand=k, positions=())
    if int(levels.shape[1]) != int(date_ord.shape[0]):
        raise CoreInputError(
            reason=codes.INPUT_WRONG_SHAPE, operand=OPERAND_LEVELS, positions=()
        )
    n_terms = int(term_series.shape[0])
    if int(term_transforms.shape[0]) != n_terms:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_SHAPE,
            operand=OPERAND_TERM_TRANSFORMS,
            positions=(),
        )
    if n_terms == 0:
        raise CoreInputError(
            reason=codes.INPUT_EMPTY, operand=OPERAND_TERM_SERIES, positions=()
        )
    steps_up = date_ord[1:] > date_ord[:-1]
    if not bool(steps_up.all()):
        first = int(np.flatnonzero(~steps_up)[0]) + 1
        raise CoreInputError(
            reason=codes.INPUT_NOT_STRICTLY_INCREASING,
            operand=OPERAND_DATE_ORD,
            positions=(first,),
        )
    # A negative index would wrap silently to another series; an index at or
    # above n_series names no series. Both mean that term_series does not fit
    # axis 0 of levels.
    in_range = (term_series >= 0) & (term_series < levels.shape[0])
    if not bool(in_range.all()):
        first = int(np.flatnonzero(~in_range)[0])
        raise CoreInputError(
            reason=codes.INPUT_WRONG_SHAPE,
            operand=OPERAND_TERM_SERIES,
            positions=(first,),
        )


def _validate_parameters(
    term_transforms: NDArray[np.int8], *, step: int, stride: int
) -> None:
    if step < 0:
        raise CoreParameterError(reason=codes.PARAM_NEGATIVE_STEP, value=float(step))
    if stride < 1:
        raise CoreParameterError(
            reason=codes.PARAM_STRIDE_BELOW_ONE, value=float(stride)
        )
    known = np.isin(term_transforms, _TRANSFORM_CODES)
    if not bool(known.all()):
        first = int(np.flatnonzero(~known)[0])
        raise CoreParameterError(
            reason=codes.PARAM_UNKNOWN_TRANSFORM, value=float(term_transforms[first])
        )
    level_only = bool((term_transforms == codes.LEVEL).all())
    if step == 0 and not level_only:
        # A return over the interval [d[e], d[e]] is identically zero.
        raise CoreParameterError(
            reason=codes.PARAM_ZERO_STEP_WITH_RETURN, value=float(step)
        )
    if step > 0 and level_only:
        # LEVEL reads v[e_r] only, so a positive step would drop the first h
        # aligned dates without a return that needs them.
        raise CoreParameterError(
            reason=codes.PARAM_POSITIVE_STEP_LEVEL_ONLY, value=float(step)
        )


def _validate_domains(
    levels: NDArray[np.float64],
    term_series: NDArray[np.int32],
    term_transforms: NDArray[np.int8],
) -> None:
    # The comparisons meet NaN and inf only when a caller builds the aligned
    # levels by hand; align_core keeps finite values only. errstate keeps the
    # check silent so that the raise below is the only signal.
    with np.errstate(all="ignore"):
        for t in range(int(term_series.shape[0])):
            transform = term_transforms[t]
            series = levels[term_series[t]]
            admissible = np.isfinite(series)
            if transform == codes.LOG_RETURN:
                admissible &= series > 0.0
            elif transform == codes.SIMPLE_RETURN:
                admissible &= series != 0.0
            if not bool(admissible.all()):
                first = int(np.flatnonzero(~admissible)[0])
                raise CoreDomainError(
                    column=t,
                    transform=transform,
                    aligned_index=first,
                    level=float(series[first]),
                )


def build_rows(
    aligned: CoreAlignedLevels,
    *,
    term_series: NDArray[np.int32],
    term_transforms: NDArray[np.int8],
    step: int,
    stride: int,
) -> CoreRows:
    """
    Build the rows of every term from the aligned levels (A.1, C.4, C.6, C.7).

    Parameters
    ----------
    aligned
        Output of ``align_core``. ``date_ord`` and ``levels`` are read.
    term_series
        ``int32`` ``(n_terms,)``, ``n_terms >= 1``: the series index of each
        term in ``aligned.levels``. A series may appear in several terms.
    term_transforms
        ``int8`` ``(n_terms,)``: a code of ``codes.TRANSFORM_CODES`` per term.
    step
        ``h >= 0``. ``h = 0`` needs LEVEL terms only. ``h > 0`` needs at least
        one return term.
    stride
        ``s >= 1``.

    Returns
    -------
    CoreRows
        New read-only arrays. No caller array is changed or retained (C.19).

    Raises
    ------
    TypeError
        ``step`` or ``stride`` is not an integer, or is a ``bool`` or
        ``numpy.bool_``.
    CoreInputError
        An array breaks the contract above. ``operand`` is one of
        ``OPERAND_DATE_ORD``, ``OPERAND_LEVELS``, ``OPERAND_TERM_SERIES`` and
        ``OPERAND_TERM_TRANSFORMS``. ``INPUT_WRONG_SHAPE`` on
        ``OPERAND_TERM_SERIES`` means a series index outside
        ``[0, n_series)``; ``positions`` holds the first such term.
    CoreParameterError
        ``PARAM_NEGATIVE_STEP``, ``PARAM_STRIDE_BELOW_ONE``,
        ``PARAM_UNKNOWN_TRANSFORM`` (``value`` is the code),
        ``PARAM_ZERO_STEP_WITH_RETURN`` or ``PARAM_POSITIVE_STEP_LEVEL_ONLY``.
    CoreSampleSizeError
        ``SAMPLE_STEP_NOT_BELOW_COUNT``: ``h >= m``, so no row exists.
        ``n_available`` is ``m`` and ``n_required`` is ``h + 1``.
    CoreDomainError
        A level of a term series is outside the domain of its transform.
        ``column`` is the lowest such term index; ``aligned_index`` is the
        first offending aligned index of that term.

    Notes
    -----
    The checks run in this order: arrays, parameters, sample size, domains.
    Every check runs before the first value is computed.

    The LOG ratio is formed before the logarithm, as in
    ``ar1_filter._ret_log`` (A.1). SIMPLE_RETURN is ``v_end / v_start - 1``
    in two correctly rounded float64 operations, DIFFERENCE one subtraction
    and LEVEL a copy.
    """
    date_ord = aligned.date_ord
    levels = aligned.levels
    _validate_arrays(date_ord, levels, term_series, term_transforms)
    h = _declared_integer(step)
    s = _declared_integer(stride)
    m = int(date_ord.shape[0])
    _validate_parameters(term_transforms, step=h, stride=s)
    if h >= m:
        # A.1: R = floor((m - 1 - h) / s) + 1 needs m - 1 >= h.
        raise CoreSampleSizeError(
            reason=codes.SAMPLE_STEP_NOT_BELOW_COUNT, n_available=m, n_required=h + 1
        )
    _validate_domains(levels, term_series, term_transforms)

    n_rows = (m - 1 - h) // s + 1
    # For R >= 2, (R - 1) * s <= m - 1 - h < m, so the clamp changes no index.
    # It keeps a very large declared stride out of the int64 product.
    reach = min(s, m)
    end_index = (h + reach * np.arange(n_rows, dtype=np.int64)).astype(np.int32)
    start_index = (end_index - np.int32(h)).astype(np.int32)

    used = np.zeros(m, dtype=np.bool_)
    used[start_index] = True
    used[end_index] = True
    unused = np.flatnonzero(~used).astype(np.int32)

    n_terms = int(term_series.shape[0])
    values = np.empty((n_terms, n_rows), dtype=np.float64)
    start_level = np.empty(n_rows, dtype=np.float64)
    for t in range(n_terms):
        transform = term_transforms[t]
        series = levels[term_series[t]]
        out = values[t]
        np.take(series, end_index, out=out)
        if transform == codes.LEVEL:
            continue
        np.take(series, start_index, out=start_level)
        if transform == codes.DIFFERENCE:
            np.subtract(out, start_level, out=out)
            continue
        np.divide(out, start_level, out=out)
        if transform == codes.LOG_RETURN:
            np.log(out, out=out)
        else:
            np.subtract(out, 1.0, out=out)

    overlap_order = 0 if h == 0 else -(-h // s) - 1
    return CoreRows(
        step=h,
        stride=s,
        overlap_order=overlap_order,
        term_series=_read_only(np.array(term_series, dtype=np.int32, copy=True)),
        term_transforms=_read_only(np.array(term_transforms, dtype=np.int8, copy=True)),
        row_start_index=_read_only(start_index),
        row_end_index=_read_only(end_index),
        row_start_date_ord=_read_only(date_ord[start_index]),
        row_end_date_ord=_read_only(date_ord[end_index]),
        values=_read_only(values),
        excluded_index=_read_only(unused),
        excluded_date_ord=_read_only(date_ord[unused]),
        excluded_reason=_read_only(
            np.full(unused.shape[0], codes.NOT_USED_BY_ROWS, dtype=np.int8)
        ),
    )
