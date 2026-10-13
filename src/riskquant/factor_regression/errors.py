# src/riskquant/factor_regression/errors.py
"""
Exceptions of the factor regression core. PRIVATE.

Every class subclasses :class:`riskquant.exceptions.RiskQuantError` (C.20).
A core exception carries integer indices, integer codes and numeric values
only: no series name, unit, label or date text (C.22). The ``reason`` field is
a code of the matching ``*_ERROR_CODES`` group in
:mod:`riskquant.factor_regression.codes`. The message is built from those
numbers; it is the only text on the instance.

The facade catches each core exception and raises its own exception with names
and dates, chained as ``__cause__`` (C.20). The facade can do that only from
the payload, so each payload holds every index that the facade message needs.
Array payloads are private read-only copies, so a caller cannot change the
evidence after the raise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from riskquant.exceptions import RiskQuantError


def _read_only_float64(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.array(values, dtype=np.float64, copy=True, order="C")
    out.setflags(write=False)
    return out


def _read_only_int32(values: NDArray[np.int32]) -> NDArray[np.int32]:
    out = np.array(values, dtype=np.int32, copy=True, order="C")
    out.setflags(write=False)
    return out


class CoreInputError(RiskQuantError):
    """
    An input array breaks the core array contract.

    Parameters
    ----------
    reason
        Code of ``codes.INPUT_ERROR_CODES``.
    operand
        Position of the offending array in the input sequence (for alignment:
        the series index).
    positions
        Element positions that break the contract, in ascending order. Empty
        when the reason concerns the whole array (dtype, ndim, shape, length).
    """

    reason: int
    operand: int
    positions: tuple[int, ...]

    def __init__(
        self, *, reason: int | np.int8, operand: int, positions: tuple[int, ...]
    ) -> None:
        self.reason = int(reason)
        self.operand = int(operand)
        self.positions = tuple(int(p) for p in positions)
        super().__init__(
            f"core input error: reason={self.reason} operand={self.operand} "
            f"positions={self.positions}"
        )


class CoreAlignmentError(RiskQuantError):
    """
    The alignment window or the common date set is empty or invalid.

    Parameters
    ----------
    reason
        Code of ``codes.ALIGNMENT_ERROR_CODES``.
    start_ord, end_ord
        The declared half-open window ``[start_ord, end_ord)`` as received.
    """

    reason: int
    start_ord: int
    end_ord: int

    def __init__(self, *, reason: int | np.int8, start_ord: int, end_ord: int) -> None:
        self.reason = int(reason)
        self.start_ord = int(start_ord)
        self.end_ord = int(end_ord)
        super().__init__(
            f"core alignment error: reason={self.reason} "
            f"start_ord={self.start_ord} end_ord={self.end_ord}"
        )


class CoreParameterError(RiskQuantError):
    """
    A declared parameter is outside its domain.

    Parameters
    ----------
    reason
        Code of ``codes.PARAMETER_ERROR_CODES``.
    value
        The offending declared value, as float64 (step, stride, lag,
        ``rank_rtol``, threshold or code value).
    """

    reason: int
    value: float

    def __init__(self, *, reason: int | np.int8, value: float) -> None:
        self.reason = int(reason)
        self.value = float(value)
        super().__init__(
            f"core parameter error: reason={self.reason} value={self.value!r}"
        )


class CoreDomainError(RiskQuantError):
    """
    An aligned level is outside the domain of its transform (C.7).

    Parameters
    ----------
    column
        Index of the offending term in the declared term order.
    transform
        Code of ``codes.TRANSFORM_CODES``.
    aligned_index
        Index of the first offending level on the aligned date axis.
    level
        The offending level.
    """

    column: int
    transform: int
    aligned_index: int
    level: float

    def __init__(
        self, *, column: int, transform: int | np.int8, aligned_index: int, level: float
    ) -> None:
        self.column = int(column)
        self.transform = int(transform)
        self.aligned_index = int(aligned_index)
        self.level = float(level)
        super().__init__(
            f"core domain error: column={self.column} transform={self.transform} "
            f"aligned_index={self.aligned_index} level={self.level!r}"
        )


class CoreRankError(RiskQuantError):
    """
    The design has a zero column or is rank deficient (A.2).

    Parameters
    ----------
    reason
        Code of ``codes.RANK_ERROR_CODES``.
    n_columns
        Column count ``k`` of the design.
    rank
        ``RANK_DEFICIENT``: the count of singular values above the tolerance.
        ``RANK_ZERO_COLUMN``: the SVD does not run; the value is ``k`` minus
        the zero-column count, an upper bound of the rank.
    singular_values
        Singular values of the scaled design, shape ``(k,)``. Shape ``(0,)``
        for ``RANK_ZERO_COLUMN``.
    participation
        Participation score per column, shape ``(k,)`` (A.2 step 4). For
        ``RANK_ZERO_COLUMN``: 1.0 for a zero column, else 0.0.
    involved
        Involved column indices, ascending, ``int32``.
    relations
        Null relations in original units, shape ``(nullity, k)``. For
        ``RANK_ZERO_COLUMN``: one unit row per zero column.
    """

    reason: int
    n_columns: int
    rank: int
    singular_values: NDArray[np.float64]
    participation: NDArray[np.float64]
    involved: NDArray[np.int32]
    relations: NDArray[np.float64]

    def __init__(
        self,
        *,
        reason: int | np.int8,
        n_columns: int,
        rank: int,
        singular_values: NDArray[np.float64],
        participation: NDArray[np.float64],
        involved: NDArray[np.int32],
        relations: NDArray[np.float64],
    ) -> None:
        self.reason = int(reason)
        self.n_columns = int(n_columns)
        self.rank = int(rank)
        self.singular_values = _read_only_float64(singular_values)
        self.participation = _read_only_float64(participation)
        self.involved = _read_only_int32(involved)
        self.relations = _read_only_float64(relations)
        super().__init__(
            f"core rank error: reason={self.reason} rank={self.rank} "
            f"n_columns={self.n_columns} involved={self.involved.tolist()}"
        )


class CoreSampleSizeError(RiskQuantError):
    """
    The sample is too small for the declared rows or design.

    Parameters
    ----------
    reason
        Code of ``codes.SAMPLE_SIZE_ERROR_CODES``.
    n_available
        ``SAMPLE_STEP_NOT_BELOW_COUNT``: aligned date count ``m``.
        ``SAMPLE_ROWS_NOT_ABOVE_COLUMNS``: row count ``R``.
    n_required
        ``SAMPLE_STEP_NOT_BELOW_COUNT``: ``h + 1``.
        ``SAMPLE_ROWS_NOT_ABOVE_COLUMNS``: ``k + 1``, so ``k = n_required - 1``.
    """

    reason: int
    n_available: int
    n_required: int

    def __init__(
        self, *, reason: int | np.int8, n_available: int, n_required: int
    ) -> None:
        self.reason = int(reason)
        self.n_available = int(n_available)
        self.n_required = int(n_required)
        super().__init__(
            f"core sample size error: reason={self.reason} "
            f"n_available={self.n_available} n_required={self.n_required}"
        )


class CoreDegenerateFitError(RiskQuantError):
    """
    The fit is degenerate (C.14).

    Parameters
    ----------
    reason
        Code of ``codes.DEGENERATE_FIT_ERROR_CODES``.
    value
        The tested quantity: ``rss``, the centred response sum of squares, or
        the smallest covariance diagonal.
    bound
        The bound that ``value`` did not exceed.
    columns
        ``DEGENERATE_NON_POSITIVE_VARIANCE``: column indices with a diagonal
        at or below zero, ascending. Empty for the other reasons.
    """

    reason: int
    value: float
    bound: float
    columns: tuple[int, ...]

    def __init__(
        self,
        *,
        reason: int | np.int8,
        value: float,
        bound: float,
        columns: tuple[int, ...],
    ) -> None:
        self.reason = int(reason)
        self.value = float(value)
        self.bound = float(bound)
        self.columns = tuple(int(c) for c in columns)
        super().__init__(
            f"core degenerate fit error: reason={self.reason} value={self.value!r} "
            f"bound={self.bound!r} columns={self.columns}"
        )


class CoreRestrictionError(RiskQuantError):
    """
    The Wald restriction ``(R, r)`` is invalid (A.5).

    Parameters
    ----------
    reason
        Code of ``codes.RESTRICTION_ERROR_CODES``.
    shape
        Shape of the restriction matrix as received.
    """

    reason: int
    shape: tuple[int, ...]

    def __init__(self, *, reason: int | np.int8, shape: tuple[int, ...]) -> None:
        self.reason = int(reason)
        self.shape = tuple(int(s) for s in shape)
        super().__init__(
            f"core restriction error: reason={self.reason} shape={self.shape}"
        )
