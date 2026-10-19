# src/riskquant/factor_regression/combination.py
"""
Linear combination of aligned levels on a declared scale. PRIVATE.

The combination value on the aligned date axis is
(WS_FACTOR_REGRESSION_COINTEGRATION C.2)::

    z[t] = sum_i weights[i] * g(levels[term_series[i], t])

with ``g`` the identity (``codes.SCALE_LEVEL``) or the natural logarithm
(``codes.SCALE_LOG``). The core sees integer series indices, float64 weights
and an integer scale code only (C.11). Names, units and the facade ``Scale``
member are resolved at the facade.

Domain rule (C.3). ``SCALE_LOG`` needs every level of every term above 0. The
check is term-major, as in ``rows._validate_domains``, and runs completely
before the first logarithm, so no logarithm of a non-positive level exists.

Layout. ``levels`` is ``(n_series, m)`` C-order, as ``CoreAlignedLevels.levels``:
one contiguous row per series. Each term reads one row along the contiguous
date axis, and the accumulator ``z`` is written along the same axis. The loop
runs over the terms, never over the dates. One ``(m,)`` scratch buffer serves
every term after the first.

Summation order. Floating-point addition is not associative, so the order is
part of the contract: the terms are added from left to right in declared order,
``((w_0 g_0 + w_1 g_1) + w_2 g_2) + ...``. Each product ``w_i * g_i`` is one
correctly rounded float64 operation.
"""

from __future__ import annotations

import operator
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.errors import CoreInputError, CoreScaleDomainError

# Operand positions of a CoreInputError raised by combine_core: the order of
# the array parameters.
OPERAND_LEVELS: Final = 0
OPERAND_TERM_SERIES: Final = 1
OPERAND_WEIGHTS: Final = 2

_SCALE_VALUES: Final = frozenset(int(code) for code in codes.SCALE_CODES)


def _validate_arrays(
    levels: NDArray[np.float64],
    term_series: NDArray[np.int32],
    weights: NDArray[np.float64],
) -> None:
    operands: tuple[NDArray[np.generic], ...] = (levels, term_series, weights)
    # Exact type, not isinstance, before the first reduction on any operand.
    # numpy.ma.MaskedArray skips masked elements in .all(): a zero level under
    # the mask would pass the LOG domain check. The core converts nothing, so
    # it rejects every subclass.
    for k, operand in enumerate(operands):
        if type(operand) is not np.ndarray:
            raise CoreInputError(
                reason=codes.INPUT_NOT_NDARRAY, operand=k, positions=()
            )
    expected: tuple[tuple[type[np.generic], int], ...] = (
        (np.float64, 2),
        (np.int32, 1),
        (np.float64, 1),
    )
    for k, (operand, (dtype, _)) in enumerate(zip(operands, expected, strict=True)):
        if operand.dtype != dtype:
            raise CoreInputError(
                reason=codes.INPUT_WRONG_DTYPE, operand=k, positions=()
            )
    for k, (operand, (_, ndim)) in enumerate(zip(operands, expected, strict=True)):
        if operand.ndim != ndim:
            raise CoreInputError(reason=codes.INPUT_WRONG_NDIM, operand=k, positions=())
    for k, operand in enumerate(operands):
        if int(operand.size) == 0:
            raise CoreInputError(reason=codes.INPUT_EMPTY, operand=k, positions=())
    if int(weights.shape[0]) != int(term_series.shape[0]):
        raise CoreInputError(
            reason=codes.INPUT_COUNT_MISMATCH, operand=OPERAND_WEIGHTS, positions=()
        )
    # A negative index would wrap silently to another series; an index at or
    # above n_series names no series. Both mean that term_series does not fit
    # axis 0 of levels (same reason and operand rule as rows.py).
    in_range = (term_series >= 0) & (term_series < levels.shape[0])
    if not bool(in_range.all()):
        first = int(np.flatnonzero(~in_range)[0])
        raise CoreInputError(
            reason=codes.INPUT_WRONG_SHAPE,
            operand=OPERAND_TERM_SERIES,
            positions=(first,),
        )
    # The first non-finite element as a flat C-order index (np.flatnonzero
    # reads in C order whatever the memory layout), as in design.py. Every
    # level is checked, also a series that no term reads.
    for k, operand in ((OPERAND_LEVELS, levels), (OPERAND_WEIGHTS, weights)):
        finite = np.isfinite(operand)
        if not bool(finite.all()):
            first = int(np.flatnonzero(~finite)[0])
            raise CoreInputError(
                reason=codes.INPUT_NON_FINITE, operand=k, positions=(first,)
            )


def _declared_scale(scale: int | np.integer[Any]) -> int:
    # bool is an int subclass, so operator.index reads True as 1; a flag is
    # not a scale code. numpy.bool_ is rejected by the same check, so the
    # outcome does not depend on the NumPy version. operator.index rejects a
    # float or str with TypeError, so the core never rounds a code.
    if isinstance(scale, (bool, np.bool_)):
        raise TypeError("a scale code must be an integer, not a bool")
    code = operator.index(scale)
    if code not in _SCALE_VALUES:
        # No reason code exists: the facade passes Scale member values only,
        # so this is an internal contract violation.
        raise ValueError(f"unknown scale code {code}")
    return code


def _validate_log_domain(
    levels: NDArray[np.float64], term_series: NDArray[np.int32]
) -> None:
    # Term-major scan (C.3): every level of term 0 in ascending date order,
    # then term 1. The levels are finite here, so the comparison is exact and
    # silent.
    for i in range(int(term_series.shape[0])):
        series = levels[term_series[i]]
        not_positive = series <= 0.0
        if bool(not_positive.any()):
            first = int(np.flatnonzero(not_positive)[0])
            raise CoreScaleDomainError(
                reason=codes.SCALE_LOG_LEVEL_NOT_POSITIVE,
                term=i,
                aligned_index=first,
                level=float(series[first]),
            )


def _weighted_term(
    series: NDArray[np.float64],
    weight: np.float64,
    is_log: bool,
    out: NDArray[np.float64],
) -> None:
    # Writes w * g(series) into out; reads series only.
    if is_log:
        np.log(series, out=out)
        np.multiply(out, weight, out=out)
    else:
        np.multiply(series, weight, out=out)


def combine_core(
    levels: NDArray[np.float64],
    term_series: NDArray[np.int32],
    weights: NDArray[np.float64],
    scale: int | np.integer[Any],
) -> NDArray[np.float64]:
    """
    Build the combination value ``z`` on the aligned date axis (C.2, C.3).

    Parameters
    ----------
    levels
        Exactly ``numpy.ndarray``, ``float64``, ``(n_series, m)``, C-order:
        ``CoreAlignedLevels.levels``.
    term_series
        Exactly ``numpy.ndarray``, ``int32``, ``(n_terms,)``, ``n_terms >= 1``:
        the row of ``levels`` that each term reads. A series may appear in
        several terms.
    weights
        Exactly ``numpy.ndarray``, ``float64``, ``(n_terms,)``: the declared
        weight of each term. The core does not reject a zero weight; the
        facade does (C.1).
    scale
        ``codes.SCALE_LEVEL`` or ``codes.SCALE_LOG``, as ``int`` or a numpy
        integer.

    Returns
    -------
    numpy.ndarray
        New read-only ``float64`` ``(m,)``. ``z[t]`` is the left-to-right sum
        over the terms, in declared order, of ``weights[i] * g(levels[
        term_series[i], t])``. No caller array is changed or retained (C.13).

    Raises
    ------
    CoreInputError
        ``operand`` is ``OPERAND_LEVELS``, ``OPERAND_TERM_SERIES`` or
        ``OPERAND_WEIGHTS``. ``INPUT_NOT_NDARRAY`` (any ndarray subclass,
        ``numpy.ma`` included), ``INPUT_WRONG_DTYPE``, ``INPUT_WRONG_NDIM``,
        ``INPUT_EMPTY`` and ``INPUT_COUNT_MISMATCH`` (on ``OPERAND_WEIGHTS``)
        have empty ``positions``. ``INPUT_WRONG_SHAPE`` on
        ``OPERAND_TERM_SERIES``: a series index outside ``[0, n_series)``;
        ``positions`` holds the first such term. ``INPUT_NON_FINITE`` on
        ``OPERAND_LEVELS`` or ``OPERAND_WEIGHTS``: ``positions`` holds the flat
        C-order index of the first NaN or inf.
    TypeError
        ``scale`` is a ``bool``, ``numpy.bool_``, or a type without an integer
        index (``float``, ``str``, ``None``).
    ValueError
        ``scale`` is an integer that is not in ``codes.SCALE_CODES``.
    CoreScaleDomainError
        ``SCALE_LOG_LEVEL_NOT_POSITIVE``: a level of a term is at or below 0.
        ``term`` is the lowest such term index in declared order;
        ``aligned_index`` is the first offending date index of that term.

    Notes
    -----
    The checks run in this order, and each check runs over all three arrays
    before the next check starts:

    1. type is exactly ``numpy.ndarray`` (operands 0, 1, 2);
    2. dtype (operands 0, 1, 2);
    3. ndim (operands 0, 1, 2);
    4. not empty (operands 0, 1, 2);
    5. ``len(weights) == len(term_series)``;
    6. every ``term_series`` index in ``[0, n_series)``;
    7. every level finite, then every weight finite;
    8. ``scale`` type and value;
    9. ``SCALE_LOG`` only: every level of every term above 0, term-major.

    Every check runs before the first value is computed. The core has no
    overflow guard, as ``rows.py``: a product or sum outside the float64 range
    gives ``inf`` with a numpy ``RuntimeWarning``, and the ADF stage rejects a
    non-finite ``z`` (S3).
    """
    _validate_arrays(levels, term_series, weights)
    is_log = _declared_scale(scale) == int(codes.SCALE_LOG)
    if is_log:
        _validate_log_domain(levels, term_series)

    m = int(levels.shape[1])
    n_terms = int(term_series.shape[0])
    z = np.empty(m, dtype=np.float64)
    _weighted_term(levels[term_series[0]], weights[0], is_log, z)
    if n_terms > 1:
        scratch = np.empty(m, dtype=np.float64)
        for i in range(1, n_terms):
            _weighted_term(levels[term_series[i]], weights[i], is_log, scratch)
            np.add(z, scratch, out=z)
    z.setflags(write=False)
    return z
