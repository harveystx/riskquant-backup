# src/riskquant/factor_regression/rows.py
"""
Row construction and transform domain checks on aligned levels. PRIVATE.

Rows are built after alignment, from aligned levels only (C.4). Row ``r``
ends at the aligned index ``e_r = h + L + r*s`` and starts at ``b_r = e_r - h``
(A.1 with the lag window ``L`` of WS_FACTOR_REGRESSION_LAGS C.3). A term with
lag ``l`` reads the interval ``[e_r - l - h, e_r - l]``: the same aligned
interval shifted by ``l`` aligned dates, independently of ``h``. A positive
lag reads an earlier interval; a negative lag reads a later one (a lead). The
core never forms a return on the calendar of one series and aligns afterwards,
because the pair ``(r_P[t], r_Z[t])`` would then cover different intervals.

The declared ``L = max_lag`` fixes the common sample: every row reads inside
``[0, m - 1]`` for every lag in ``[-L, L]``, so two specs with the same ``L``,
``h`` and ``s`` use the same rows whatever lags they declare. With ``L = 0``
the rows are the A.1 rows unchanged.

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
return; a lagged term adds two in-place index shifts. Three ``(R,)`` scratch
buffers serve every term: the start levels and the two shifted index arrays.
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
# which build_rows reads its five arrays.
OPERAND_DATE_ORD: Final = 0
OPERAND_LEVELS: Final = 1
OPERAND_TERM_SERIES: Final = 2
OPERAND_TERM_TRANSFORMS: Final = 3
OPERAND_TERM_LAGS: Final = 4

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
    max_lag
        The declared ``L >= 0``.
    overlap_order
        ``q = ceil(h / s) - 1`` for ``h >= 1``, ``0`` for ``h = 0``. Adjacent
        rows share increments up to lag ``q``, so the row errors can carry an
        MA(``q``) structure.
    term_series
        ``int32`` ``(n_terms,)``: the series index of each term.
    term_transforms
        ``int8`` ``(n_terms,)``: the transform code of each term.
    term_lags
        ``int32`` ``(n_terms,)``: the lag ``l`` of each term, in ``[-L, L]``.
    row_start_index, row_end_index
        ``int32`` ``(R,)``: the aligned indices ``b_r`` and ``e_r`` of the
        lag-0 interval. Term 0 (the response of the facade) has lag 0, so
        these are the response indices.
    row_start_date_ord, row_end_date_ord
        ``int32`` ``(R,)``: ``date_ord[b_r]`` and ``date_ord[e_r]``. Equal for
        ``h = 0``.
    values
        ``float64`` ``(n_terms, R)`` C-order. ``values[t, r]`` is term ``t``
        on row ``r`` with ``e = e_r - l_t`` and ``b = b_r - l_t``: LEVEL
        ``v[e]``, LOG_RETURN ``ln(v[e] / v[b])``, SIMPLE_RETURN
        ``v[e] / v[b] - 1``, DIFFERENCE ``v[e] - v[b]``.
    excluded_index, excluded_date_ord
        ``int32`` ``(U,)``: aligned index and date of each aligned date that
        no term reads on any row, ascending. A LEVEL term reads its end index
        only; a return term reads its start and end index.
    excluded_reason
        ``int8`` ``(U,)``: ``codes.NOT_USED_BY_ROWS`` for every record.

    Notes
    -----
    All arrays are new and read-only. ``eq=False`` follows the D064 precedent
    for array-bearing containers.
    """

    step: int
    stride: int
    max_lag: int
    overlap_order: int
    term_series: NDArray[np.int32]
    term_transforms: NDArray[np.int8]
    term_lags: NDArray[np.int32]
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
    # float or str with TypeError, so the core never rounds a step, stride or
    # lag window. bool is an int subclass, so operator.index reads True as 1;
    # a flag is not a step. numpy.bool_ is rejected by the same check, so the
    # outcome does not depend on the NumPy version.
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("a step, stride or max_lag must be an int, not a bool")
    return operator.index(value)


def _validate_arrays(
    date_ord: NDArray[np.int32],
    levels: NDArray[np.float64],
    term_series: NDArray[np.int32],
    term_transforms: NDArray[np.int8],
    term_lags: NDArray[np.int32] | None,
) -> None:
    # term_lags is None when the caller declared no lags: the zero lags are
    # built after this check, so there is no fifth operand to check.
    operands: tuple[NDArray[np.generic], ...] = (
        date_ord,
        levels,
        term_series,
        term_transforms,
    )
    if term_lags is not None:
        operands = (*operands, term_lags)
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
        (np.int32, 1),
    )[: len(operands)]
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
    if term_lags is not None and int(term_lags.shape[0]) != n_terms:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_SHAPE,
            operand=OPERAND_TERM_LAGS,
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
    term_transforms: NDArray[np.int8],
    term_lags: NDArray[np.int32],
    *,
    step: int,
    stride: int,
    max_lag: int,
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
    # Every lag must lie in [-L, L], so that every row of the common sample
    # reads inside the aligned axis (C.3, C.9). A negative L fails here on the
    # first term, because no lag satisfies |lag| <= L < 0. The comparison runs
    # in int64: abs(-2**31) does not fit int32.
    inside = np.abs(term_lags.astype(np.int64)) <= max_lag
    if not bool(inside.all()):
        first = int(np.flatnonzero(~inside)[0])
        raise CoreParameterError(
            reason=codes.PARAM_LAG_OUTSIDE_WINDOW, value=float(term_lags[first])
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
    term_lags: NDArray[np.int32] | None = None,
    max_lag: int = 0,
) -> CoreRows:
    """
    Build the rows of every term from the aligned levels (A.1, C.4, C.6, C.7).

    Row ``k`` (``k = 0 .. R-1``) ends at the aligned index
    ``e_k = h + L + k*s`` with ``R = floor((m - 1 - h - 2L) / s) + 1``. Term
    ``t`` with lag ``l`` reads ``[e_k - l - h, e_k - l]``
    (WS_FACTOR_REGRESSION_LAGS C.3).

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
    term_lags
        ``int32`` ``(n_terms,)``: the lag of each term, each in
        ``[-max_lag, max_lag]``. ``None`` (the default) means lag 0 for every
        term, so a call without lags builds the A.1 rows. The facade passes
        lag 0 for term 0 (the response), so the row dates are the response
        interval; the core does not check term 0.
    max_lag
        ``L >= 0``, default ``0``: the lag window that fixes the common
        sample.

    Returns
    -------
    CoreRows
        New read-only arrays. No caller array is changed or retained (C.19).

    Raises
    ------
    TypeError
        ``step``, ``stride`` or ``max_lag`` is not an integer, or is a
        ``bool`` or ``numpy.bool_``.
    CoreInputError
        An array breaks the contract above. ``operand`` is one of
        ``OPERAND_DATE_ORD``, ``OPERAND_LEVELS``, ``OPERAND_TERM_SERIES``,
        ``OPERAND_TERM_TRANSFORMS`` and ``OPERAND_TERM_LAGS``.
        ``INPUT_WRONG_SHAPE`` on ``OPERAND_TERM_SERIES`` means a series index
        outside ``[0, n_series)``; ``positions`` holds the first such term.
    CoreParameterError
        ``PARAM_NEGATIVE_STEP``, ``PARAM_STRIDE_BELOW_ONE``,
        ``PARAM_UNKNOWN_TRANSFORM`` (``value`` is the code),
        ``PARAM_ZERO_STEP_WITH_RETURN``, ``PARAM_POSITIVE_STEP_LEVEL_ONLY``
        or ``PARAM_LAG_OUTSIDE_WINDOW`` (``value`` is the first lag with
        ``|lag| > max_lag``; a negative ``max_lag`` fails on term 0).
    CoreSampleSizeError
        ``SAMPLE_STEP_NOT_BELOW_COUNT``: ``m - 1 < h + 2L``, so no row
        exists. ``n_available`` is ``m`` and ``n_required`` is ``h + 2L + 1``.
    CoreDomainError
        A level of a term series is outside the domain of its transform.
        ``column`` is the lowest such term index; ``aligned_index`` is the
        first offending aligned index of that term.

    Notes
    -----
    The checks run in this order: arrays, parameters (the lag window last),
    sample size, domains. Every check runs before the first value is computed.

    With every lag 0 and ``L = 0`` the rows, the indices and the exclusions
    equal the A.1 rows bit for bit (C.5): the index arrays are the same
    arrays, and each term gathers with them unchanged.

    The LOG ratio is formed before the logarithm, as in
    ``ar1_filter._ret_log`` (A.1). SIMPLE_RETURN is ``v_end / v_start - 1``
    in two correctly rounded float64 operations, DIFFERENCE one subtraction
    and LEVEL a copy.
    """
    date_ord = aligned.date_ord
    levels = aligned.levels
    _validate_arrays(date_ord, levels, term_series, term_transforms, term_lags)
    lags = (
        np.zeros(int(term_series.shape[0]), dtype=np.int32)
        if term_lags is None
        else term_lags
    )
    h = _declared_integer(step)
    s = _declared_integer(stride)
    lag_window = _declared_integer(max_lag)
    m = int(date_ord.shape[0])
    _validate_parameters(term_transforms, lags, step=h, stride=s, max_lag=lag_window)
    if m - 1 < h + 2 * lag_window:
        # C.3: R = floor((m - 1 - h - 2L) / s) + 1 needs m - 1 >= h + 2L.
        raise CoreSampleSizeError(
            reason=codes.SAMPLE_STEP_NOT_BELOW_COUNT,
            n_available=m,
            n_required=h + 2 * lag_window + 1,
        )
    _validate_domains(levels, term_series, term_transforms)

    n_rows = (m - 1 - h - 2 * lag_window) // s + 1
    # For R >= 2, (R - 1) * s <= m - 1 - h - 2L < m, so the clamp changes no
    # index. It keeps a very large declared stride out of the int64 product.
    reach = min(s, m)
    end_index = (h + lag_window + reach * np.arange(n_rows, dtype=np.int64)).astype(
        np.int32
    )
    start_index = (end_index - np.int32(h)).astype(np.int32)

    # The lag-0 reads of every row, then each term marks the indices it reads:
    # a LEVEL term its end index, a return term its start and end index (C.4).
    # e_k - l stays in [0, m - 1] for every lag in [-L, L] (C.3), so no mark
    # wraps. Lag 0 marks the same indices as the A.1 rows did.
    n_terms = int(term_series.shape[0])
    used = np.zeros(m, dtype=np.bool_)
    for t in range(n_terms):
        lag = int(lags[t])
        used[end_index - lag] = True
        if term_transforms[t] != codes.LEVEL:
            used[start_index - lag] = True
    unused = np.flatnonzero(~used).astype(np.int32)

    values = np.empty((n_terms, n_rows), dtype=np.float64)
    start_level = np.empty(n_rows, dtype=np.float64)
    shifted_end = np.empty(n_rows, dtype=np.int32)
    shifted_start = np.empty(n_rows, dtype=np.int32)
    for t in range(n_terms):
        transform = term_transforms[t]
        series = levels[term_series[t]]
        lag = int(lags[t])
        if lag == 0:
            # The unshifted index arrays: the lag-0 gather is the A.1 gather.
            term_end = end_index
            term_start = start_index
        else:
            np.subtract(end_index, np.int32(lag), out=shifted_end)
            np.subtract(start_index, np.int32(lag), out=shifted_start)
            term_end = shifted_end
            term_start = shifted_start
        out = values[t]
        np.take(series, term_end, out=out)
        if transform == codes.LEVEL:
            continue
        np.take(series, term_start, out=start_level)
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
        max_lag=lag_window,
        overlap_order=overlap_order,
        term_series=_read_only(np.array(term_series, dtype=np.int32, copy=True)),
        term_transforms=_read_only(np.array(term_transforms, dtype=np.int8, copy=True)),
        term_lags=_read_only(np.array(lags, dtype=np.int32, copy=True)),
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
