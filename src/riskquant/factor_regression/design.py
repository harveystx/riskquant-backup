# src/riskquant/factor_regression/design.py
"""
Design assembly, column scaling, rank gate and condition number. PRIVATE.

The design ``X`` has shape ``(R, k)``: the intercept column of ones first if
declared, then the regressors in declared order (C.8). Every column is divided
by its 2-norm ``d_j``. Rank, conditioning and the solve of S4 use the scaled
design ``Xs = X / d`` only (A.2). Scaling makes the gate independent of the
units of each column: an unscaled least-squares solve loses a rank at a mean
to sd ratio of 1e7 (F-11), the scaled SVD does not.

The rank gate is ``numpy.linalg.svd`` of ``Xs`` with tolerance
``rank_rtol * s[0]`` (C.9). A rank below ``k`` raises ``CoreRankError`` before
any solve. A full-rank design returns a frozen, read-only ``CoreScaledSvd``
that holds every array the S4 solve needs (A.3): S4 solves from this SVD and
computes no second one.

The core sees no name, unit or label (C.22). Errors carry column indices and
numbers only; the facade adds the names.

Layout. ``X`` and ``Xs`` are ``(R, k)`` C-order: one contiguous row per
observation, as A.6 digests ``X``. The regressor input is ``(n_regressors, R)``
C-order, as ``CoreRows.values`` holds one contiguous row per term, so assembly
is one transposing copy. ``numpy.linalg.svd`` copies its operand into the LAPACK
layout itself, so no layout of ``Xs`` saves a copy there.

Cost. Assembly: one ``(R, k)`` write. Gate: one ``(R, k)`` copy of the caller
design, one power-of-two scaled ``(R, k)`` temporary for the norms, one
``(R, k)`` scaled design, and the thin SVD with ``u`` of shape ``(R, k)``. No
loop runs over the rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.errors import (
    CoreInputError,
    CoreParameterError,
    CoreRankError,
    CoreSampleSizeError,
)

# A.2 step 4: a column with a null-space participation at or below this score
# takes no part in a null relation.
_PARTICIPATION_FLOOR: Final = 1e-12

_ScalarT = TypeVar("_ScalarT", bound=np.generic)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CoreScaledSvd:
    """
    The scaled design and its thin SVD for a full-rank design (A.2, A.3).

    Attributes
    ----------
    design
        ``float64`` ``(R, k)`` C-order: a copy of the design before scaling.
    column_norms
        ``float64`` ``(k,)``: ``d_j = ||X[:, j]||_2``, all finite and above 0.
    scaled_design
        ``float64`` ``(R, k)`` C-order: ``Xs = X / d``.
    u
        ``float64`` ``(R, k)``: left singular vectors of ``Xs``.
    singular_values
        ``float64`` ``(k,)``: singular values of ``Xs``, descending, all above
        ``rank_rtol * singular_values[0]``.
    vh
        ``float64`` ``(k, k)``: right singular vectors of ``Xs``, one per row.
    rank
        Equal to ``k``: the gate returns a result for a full-rank design only.
    condition_number
        ``kappa = s[0] / s[k-1]`` of ``Xs`` (C.10).

    Notes
    -----
    ``Xs = u @ diag(singular_values) @ vh`` to rounding. S4 forms
    ``gamma = vh.T @ ((u.T @ y) / s)`` and ``coef = gamma / d`` from these
    arrays (A.3). All arrays are new and read-only and share no memory with
    the caller design (C.19). ``eq=False`` follows the D064 precedent for
    array-bearing containers.
    """

    design: NDArray[np.float64]
    column_norms: NDArray[np.float64]
    scaled_design: NDArray[np.float64]
    u: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    vh: NDArray[np.float64]
    rank: int
    condition_number: float

    @property
    def n_rows(self) -> int:
        return int(self.design.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self.design.shape[1])


def _read_only(array: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
    array.setflags(write=False)
    return array


def _validate_matrix(operand: NDArray[np.float64]) -> None:
    # Exact type, not isinstance, before the first reduction. numpy.ma skips
    # masked elements in .all(): a NaN under the mask passes the finite check.
    # The core converts nothing, so it rejects every ndarray subclass.
    if type(operand) is not np.ndarray:
        raise CoreInputError(reason=codes.INPUT_NOT_NDARRAY, operand=0, positions=())
    if operand.dtype != np.float64:
        raise CoreInputError(reason=codes.INPUT_WRONG_DTYPE, operand=0, positions=())
    if operand.ndim != 2:
        raise CoreInputError(reason=codes.INPUT_WRONG_NDIM, operand=0, positions=())


def _reject_non_finite(operand: NDArray[np.float64]) -> None:
    # Owner decision OD-6 B: the first non-finite element, as a flat C-order
    # index (np.flatnonzero reads in C order whatever the memory layout).
    finite = np.isfinite(operand)
    if not bool(finite.all()):
        first = int(np.flatnonzero(~finite)[0])
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE, operand=0, positions=(first,)
        )


def _declared_flag(value: bool) -> bool:
    # A flag is a bool or numpy.bool_. An int 0 or 1 is not read as a flag,
    # so the outcome does not depend on a silent truth-value conversion.
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("intercept must be a bool")
    return bool(value)


def _declared_real(value: float) -> float:
    # bool is an int subclass, so float(True) reads as 1.0; a flag is not a
    # tolerance. A str, None or complex value raises: the core never parses.
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("rank_rtol must be a real number")
    return float(value)


def assemble_design(
    regressors: NDArray[np.float64], *, intercept: bool
) -> NDArray[np.float64]:
    """
    Assemble the design ``X`` from regressor rows (C.8).

    Parameters
    ----------
    regressors
        ``float64`` ``(n_regressors, R)``, ``R >= 1``, all finite: one row per
        regressor term in declared order, as ``CoreRows.values`` holds one row
        per term. ``n_regressors`` may be 0 when ``intercept`` is True.
    intercept
        ``True`` puts a column of ones first. A ``bool`` or ``numpy.bool_``.

    Returns
    -------
    NDArray[np.float64]
        New read-only ``(R, k)`` C-order design, ``k = n_regressors +
        intercept``. Column ``0`` is the intercept if declared; regressor
        ``t`` is column ``t + intercept``. Every regressor value is copied bit
        for bit. No caller array is changed or retained (C.19).

    Raises
    ------
    TypeError
        ``intercept`` is not a ``bool`` or ``numpy.bool_``.
    CoreInputError
        ``operand`` is ``0``. ``INPUT_NOT_NDARRAY`` (any ndarray subclass,
        ``numpy.ma`` included), ``INPUT_WRONG_DTYPE``, ``INPUT_WRONG_NDIM``,
        ``INPUT_EMPTY`` (``R == 0``) with empty ``positions``.
        ``INPUT_NON_FINITE``: ``positions`` holds the flat C-order index
        ``t * R + r`` of the first NaN or inf in ``regressors``.
    CoreParameterError
        ``PARAM_EMPTY_DESIGN``: no regressor and no intercept, so ``k == 0``.
        ``value`` is ``0.0``.
    """
    _validate_matrix(regressors)
    n_regressors = int(regressors.shape[0])
    n_rows = int(regressors.shape[1])
    if n_rows == 0:
        raise CoreInputError(reason=codes.INPUT_EMPTY, operand=0, positions=())
    _reject_non_finite(regressors)
    offset = int(_declared_flag(intercept))
    n_columns = n_regressors + offset
    if n_columns == 0:
        raise CoreParameterError(reason=codes.PARAM_EMPTY_DESIGN, value=0.0)

    design = np.empty((n_rows, n_columns), dtype=np.float64)
    if offset:
        design[:, 0] = 1.0
    design[:, offset:] = regressors.T
    return _read_only(design)


def _zero_column_error(zero: NDArray[np.bool_]) -> CoreRankError:
    # Owner decision OD-1 A: no SVD runs. The payload follows the
    # RANK_ZERO_COLUMN rows of the CoreRankError docstring.
    n_columns = int(zero.shape[0])
    involved = np.flatnonzero(zero).astype(np.int32)
    n_zero = int(involved.shape[0])
    relations = np.zeros((n_zero, n_columns), dtype=np.float64)
    relations[np.arange(n_zero), involved] = 1.0
    return CoreRankError(
        reason=codes.RANK_ZERO_COLUMN,
        n_columns=n_columns,
        rank=n_columns - n_zero,
        singular_values=np.empty(0, dtype=np.float64),
        participation=zero.astype(np.float64),
        involved=involved,
        relations=relations,
    )


def _column_norms(design: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    ``d_j = ||X[:, j]||_2`` without overflow or underflow of the squares.

    Each column is scaled by a power of two ``2**-e_j`` with ``e_j`` the
    binary exponent of its largest magnitude. A power-of-two scaling is
    exact, so ``d`` equals ``sqrt(sum(X[:, j]**2))`` bit for bit when that
    expression neither overflows nor underflows, and stays exact when it
    would. An unscaled sum of squares reads a column of values near 1e-170
    as a zero column and a column near 1e160 as an infinite norm.
    """
    peak = np.max(np.abs(design), axis=0)
    zero = peak == 0.0
    if bool(zero.any()):
        raise _zero_column_error(zero)
    _, exponent = np.frexp(peak)
    unit = np.ldexp(design, -exponent)
    # errstate keeps an overflow of the last ldexp silent, so that the raise
    # below is the only signal.
    with np.errstate(over="ignore"):
        norms: NDArray[np.float64] = np.ldexp(
            np.sqrt(np.sum(unit * unit, axis=0)), exponent
        )
    finite = np.isfinite(norms)
    if not bool(finite.all()):
        # Every element is finite, but the 2-norm of the column exceeds the
        # float64 range, so X / d and coef = gamma / d are not representable.
        column = int(np.flatnonzero(~finite)[0])
        row = int(np.argmax(np.abs(design[:, column])))
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE,
            operand=0,
            positions=(row * int(design.shape[1]) + column,),
        )
    return norms


def _rank_deficiency_error(
    *,
    singular_values: NDArray[np.float64],
    vh: NDArray[np.float64],
    column_norms: NDArray[np.float64],
    rank: int,
) -> CoreRankError:
    # numpy.linalg.svd returns s in descending order, so the rows of vh with
    # s[i] <= tol are exactly the rows rank .. k-1 (A.2 step 4).
    null = vh[rank:]
    nullity = int(null.shape[0])
    participation = np.sum(null * null, axis=0)
    involved = np.flatnonzero(participation > _PARTICIPATION_FLOOR).astype(np.int32)
    # Original units: sum_j w_j X[:, j] = Xs @ vh[i] approximately 0. Each row
    # is divided by its entry of largest magnitude; np.argmax takes the lowest
    # index on a tie. A row of vh has unit norm, so that entry is not zero.
    weights = null / column_norms
    pivot = np.argmax(np.abs(weights), axis=1)
    relations = weights / weights[np.arange(nullity), pivot][:, np.newaxis]
    return CoreRankError(
        reason=codes.RANK_DEFICIENT,
        n_columns=int(vh.shape[1]),
        rank=rank,
        singular_values=singular_values,
        participation=participation,
        involved=involved,
        relations=relations,
    )


def rank_gate(design: NDArray[np.float64], *, rank_rtol: float) -> CoreScaledSvd:
    """
    Scale the design, run the SVD rank gate and report the condition number.

    Parameters
    ----------
    design
        ``float64`` ``(R, k)``, ``k >= 1``, ``R >= k + 1``, all finite. Any
        memory layout. Use ``assemble_design`` for the C.8 column order.
    rank_rtol
        Relative rank tolerance, declared by the caller (O-7), in the open
        interval ``(0, 1)``. No default.

    Returns
    -------
    CoreScaledSvd
        New read-only arrays: the design copy, ``d``, ``Xs``, ``u``, ``s``,
        ``vh``, ``rank == k`` and ``kappa``. No caller array is changed or
        retained (C.19).

    Raises
    ------
    TypeError
        ``rank_rtol`` is not a real number, or is a ``bool`` or
        ``numpy.bool_``.
    CoreInputError
        ``operand`` is ``0``. ``INPUT_NOT_NDARRAY`` (any ndarray subclass,
        ``numpy.ma`` included), ``INPUT_WRONG_DTYPE``, ``INPUT_WRONG_NDIM``,
        ``INPUT_EMPTY`` (``R == 0`` or ``k == 0``) with empty ``positions``.
        ``INPUT_NON_FINITE``: ``positions`` holds the flat C-order index
        ``r * k + j`` of the first NaN or inf; or, when every element is
        finite but the 2-norm of column ``j`` exceeds the float64 range, the
        flat index of the largest magnitude of the first such column.
    CoreParameterError
        ``PARAM_RANK_RTOL_OUT_OF_RANGE``: ``rank_rtol`` is not in ``(0, 1)``
        (NaN included). ``value`` is ``rank_rtol``.
    CoreSampleSizeError
        ``SAMPLE_ROWS_NOT_ABOVE_COLUMNS``: ``R < k + 1``. ``n_available`` is
        ``R``, ``n_required`` is ``k + 1``. Raised before the SVD.
    CoreRankError
        ``RANK_ZERO_COLUMN``: a column is all zeros. No SVD runs. ``rank`` is
        ``k`` minus the zero-column count, an upper bound of the rank.
        ``singular_values`` has shape ``(0,)``. ``participation`` is 1.0 for a
        zero column, else 0.0. ``involved`` holds the zero columns.
        ``relations`` holds one unit row per zero column.
        ``RANK_DEFICIENT``: ``rank = count(s > rank_rtol * s[0]) < k``. The
        payload follows A.2 step 4: ``singular_values`` of ``Xs``,
        ``participation`` per column, ``involved`` (participation above
        1e-12, ascending) and one relation row per null direction, in
        original units, scaled so that its entry of largest magnitude is 1.0.

    Notes
    -----
    The checks run in this order: array, parameter, sample size, zero column,
    rank. Every check runs before the SVD, except the rank check.

    ``numpy.linalg.matrix_rank(Xs, tol=rank_rtol * s[0])`` gives the same
    rank. The tests use ``tol``: numpy 2.0 added ``rtol`` (A.2 step 6).
    """
    _validate_matrix(design)
    n_rows = int(design.shape[0])
    n_columns = int(design.shape[1])
    if n_rows == 0 or n_columns == 0:
        raise CoreInputError(reason=codes.INPUT_EMPTY, operand=0, positions=())
    _reject_non_finite(design)
    rtol = _declared_real(rank_rtol)
    if not 0.0 < rtol < 1.0:
        raise CoreParameterError(reason=codes.PARAM_RANK_RTOL_OUT_OF_RANGE, value=rtol)
    if n_rows < n_columns + 1:
        # A.2: with R >= k + 1, vh has shape (k, k) and the null space of Xs is
        # spanned by rows of vh. S4 needs the same bound for dof >= 1 (C.12).
        raise CoreSampleSizeError(
            reason=codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS,
            n_available=n_rows,
            n_required=n_columns + 1,
        )

    copy = np.array(design, dtype=np.float64, copy=True, order="C")
    norms = _column_norms(copy)
    scaled = np.ascontiguousarray(copy / norms)
    u, singular_values, vh = np.linalg.svd(scaled, full_matrices=False)
    tolerance = rtol * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < n_columns:
        raise _rank_deficiency_error(
            singular_values=singular_values, vh=vh, column_norms=norms, rank=rank
        )

    return CoreScaledSvd(
        design=_read_only(copy),
        column_norms=_read_only(norms),
        scaled_design=_read_only(scaled),
        u=_read_only(u),
        singular_values=_read_only(singular_values),
        vh=_read_only(vh),
        rank=rank,
        condition_number=float(singular_values[0] / singular_values[-1]),
    )
