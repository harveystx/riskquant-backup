# src/riskquant/factor_regression/fit.py
"""
Core OLS fit on the scaled design (A.3, C.12, C.13, C.14, C.19). PRIVATE.

``fit_core`` runs the rank gate of ``design.py`` once and solves from that
SVD (C.12): ``gamma = vh.T @ ((u.T @ y) / s)`` and ``coef = gamma / d``. It
runs no second SVD, no least-squares routine and no matrix inverse. An
unscaled least-squares solve loses a rank at a mean to sd ratio of 1e7
(F-11); the scaled SVD solve keeps the slope to about 1e-9 relative there.
Sandbox evidence (Python 3.12.3, numpy 2.4.4), intercept and one regressor,
``x = 1e7 + N(0, 1)``, seeds 1 to 20, against an exact ``fractions.Fraction``
solution: slope 7.0e-10 relative at 250 rows and 5.2e-10 at 2500 rows;
centred intercept and fitted values 2.1e-15 relative or better.

Residuals are ``e = y - X @ coef`` with the design copy of the gate (A.3).
The covariance is in ``covariance.py``. p-values are two-sided Student t with
``dof = R - k`` (C.13); for HC0 and HAC_BARTLETT that reference distribution
is an approximation (A.3).

Sums of squares. ``sum(y**2)``, ``rss`` and the centred sum of squares are
computed after a scaling by the power of two ``2**-e``, with ``e`` the binary
exponent of the largest magnitude of the vector, as ``design.py`` scales the
columns for their norms. The degenerate-fit rule and ``r_squared`` compare the
scaled sums, so no square overflows or underflows on the way. The scaling is
exact, so each sum equals the unscaled formula bit for bit when that formula
neither overflows nor underflows.

The core sees no name, unit or label (C.22). Errors carry indices, codes and
numbers only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from riskquant.factor_regression import codes, diagnostics
from riskquant.factor_regression.covariance import (
    check_covariance_parameters,
    coefficient_covariance,
)
from riskquant.factor_regression.design import CoreScaledSvd, rank_gate
from riskquant.factor_regression.errors import CoreDegenerateFitError, CoreInputError

# Operand positions of a CoreInputError raised by fit_core.
OPERAND_DESIGN: Final = 0
OPERAND_RESPONSE: Final = 1

_EPS: Final = float(np.finfo(np.float64).eps)

_ScalarT = TypeVar("_ScalarT", bound=np.generic)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CoreFit:
    """
    The result of one core OLS fit (A.3).

    Attributes
    ----------
    gate
        The ``CoreScaledSvd`` of the rank gate: design copy, column norms,
        scaled design, SVD, rank and condition number.
    coef
        ``float64`` ``(k,)``: coefficients in original units, intercept first
        if declared (C.8 order).
    se, t_stats, p_values
        ``float64`` ``(k,)``: ``sqrt(diag(cov))``, ``coef / se`` and the
        two-sided Student t p-value with ``dof`` degrees of freedom.
    cov
        ``float64`` ``(k, k)``: coefficient covariance in original units,
        exactly symmetric, of the declared covariance kind.
    resid, fitted
        ``float64`` ``(R,)``: ``y - X @ coef`` and ``X @ coef``.
    rss
        ``resid @ resid`` as float64, above zero and finite.
    r_squared
        ``1 - rss / sum((y - mean(y))**2)`` with an intercept, else
        ``1 - rss / sum(y**2)`` (A.3).
    dof
        ``R - k >= 1``.
    n_obs
        ``R``.
    durbin_watson
        ``diagnostics.durbin_watson(resid)`` (A.4).

    Notes
    -----
    All arrays are new and read-only and share no memory with a caller array
    (C.19). Every float is finite. The core approves no assumption: no field
    says that a fit is accepted, approved or valid. ``eq=False`` follows the
    D064 precedent for array-bearing containers.
    """

    gate: CoreScaledSvd
    coef: NDArray[np.float64]
    se: NDArray[np.float64]
    t_stats: NDArray[np.float64]
    p_values: NDArray[np.float64]
    cov: NDArray[np.float64]
    resid: NDArray[np.float64]
    fitted: NDArray[np.float64]
    rss: float
    r_squared: float
    dof: int
    n_obs: int
    durbin_watson: float


def _read_only(array: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
    array.setflags(write=False)
    return array


def _declared_flag(value: bool) -> bool:
    # A flag is a bool or numpy.bool_. An int 0 or 1 is not read as a flag,
    # so the outcome does not depend on a silent truth-value conversion.
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("intercept must be a bool")
    return bool(value)


def _ldexp(value: float, exponent: int) -> float:
    # math.ldexp raises OverflowError; the comparisons below need inf.
    with np.errstate(over="ignore", under="ignore"):
        return float(np.ldexp(value, exponent))


def _square_sum(values: NDArray[np.float64]) -> tuple[float, int]:
    """
    ``sum(values**2)`` as ``(unit, e)`` with ``sum = unit * 4**e``.

    ``e`` is the binary exponent of the largest magnitude, so ``unit`` is in
    ``[0.25, R)``, or ``0.0`` with ``e == 0`` for an all-zero vector.
    """
    _, exponent = math.frexp(float(np.max(np.abs(values))))
    unit = np.ldexp(values, -exponent)
    return float(np.dot(unit, unit)), exponent


def _reject_non_finite(operand: NDArray[np.float64], index: int) -> None:
    # Owner decision OD-6 B: the first non-finite element, as a flat C-order
    # index (np.flatnonzero reads in C order whatever the memory layout).
    finite = np.isfinite(operand)
    if not bool(finite.all()):
        first = int(np.flatnonzero(~finite)[0])
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE, operand=index, positions=(first,)
        )


def _validate_design(design: NDArray[np.float64]) -> None:
    # Exact type, not isinstance, before the first reduction. numpy.ma skips
    # masked elements in .all(): a NaN under the mask passes the finite check.
    # rank_gate repeats these checks; fit_core runs them first so that the
    # design (operand 0) is checked before the response (operand 1).
    if type(design) is not np.ndarray:
        raise CoreInputError(
            reason=codes.INPUT_NOT_NDARRAY, operand=OPERAND_DESIGN, positions=()
        )
    if design.dtype != np.float64:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_DTYPE, operand=OPERAND_DESIGN, positions=()
        )
    if design.ndim != 2:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_NDIM, operand=OPERAND_DESIGN, positions=()
        )
    if int(design.shape[0]) == 0 or int(design.shape[1]) == 0:
        raise CoreInputError(
            reason=codes.INPUT_EMPTY, operand=OPERAND_DESIGN, positions=()
        )
    _reject_non_finite(design, OPERAND_DESIGN)


def _validate_response(y: NDArray[np.float64], n_rows: int) -> None:
    # Same exact-type rule as the design: a mask hides a NaN response from
    # the finite check.
    if type(y) is not np.ndarray:
        raise CoreInputError(
            reason=codes.INPUT_NOT_NDARRAY, operand=OPERAND_RESPONSE, positions=()
        )
    if y.dtype != np.float64:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_DTYPE, operand=OPERAND_RESPONSE, positions=()
        )
    if y.ndim != 1:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_NDIM, operand=OPERAND_RESPONSE, positions=()
        )
    if int(y.shape[0]) == 0:
        raise CoreInputError(
            reason=codes.INPUT_EMPTY, operand=OPERAND_RESPONSE, positions=()
        )
    if int(y.shape[0]) != n_rows:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_SHAPE, operand=OPERAND_RESPONSE, positions=()
        )
    _reject_non_finite(y, OPERAND_RESPONSE)


def _response_out_of_range(y: NDArray[np.float64]) -> CoreInputError:
    # Every element of y is finite, but a quantity in the units of y (fitted,
    # resid) or of y squared (rss) exceeds the float64 range. design.py
    # reports a column 2-norm above the range the same way: INPUT_NON_FINITE
    # at the largest magnitude.
    return CoreInputError(
        reason=codes.INPUT_NON_FINITE,
        operand=OPERAND_RESPONSE,
        positions=(int(np.argmax(np.abs(y))),),
    )


def _non_positive_variance(
    diagonal: NDArray[np.float64], columns: NDArray[np.intp]
) -> CoreDegenerateFitError:
    return CoreDegenerateFitError(
        reason=codes.DEGENERATE_NON_POSITIVE_VARIANCE,
        value=float(np.min(diagonal[columns])),
        bound=0.0,
        columns=tuple(int(c) for c in columns),
    )


def fit_core(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    intercept: bool,
    rank_rtol: float,
    covariance_kind: int,
    covariance_lag: int,
) -> CoreFit:
    """
    Fit ``y`` on ``design`` by OLS on the scaled design (A.3, C.12 to C.14).

    Parameters
    ----------
    design
        ``float64`` ``(R, k)``, all finite, any memory layout: the output of
        ``design.assemble_design``. When ``intercept`` is True, column ``0``
        must be the intercept column of ones (C.8 order). ``fit_core`` does
        not check this column; the facade passes the same flag to
        ``assemble_design`` and to ``fit_core``.
    y
        ``float64`` ``(R,)``, all finite: the response, rows in date order.
    intercept
        ``True`` when column ``0`` is the intercept. It selects the centred
        ``r_squared`` and the zero-variance rule. A ``bool`` or
        ``numpy.bool_``.
    rank_rtol
        Relative rank tolerance of the gate, in ``(0, 1)``. No default.
    covariance_kind
        ``codes.IID``, ``codes.HC0`` or ``codes.HAC_BARTLETT``. No default.
    covariance_lag
        ``0`` for IID and HC0; the Bartlett lag ``0 <= L <= R - 1`` for
        HAC_BARTLETT. No default.

    Returns
    -------
    CoreFit
        New read-only arrays. No caller array is changed or retained (C.19).

    Raises
    ------
    TypeError
        ``intercept`` is not a ``bool`` or ``numpy.bool_``; the covariance
        kind or lag is not an integer or is a ``bool``; ``rank_rtol`` is not
        a real number or is a ``bool``.
    CoreInputError
        ``operand`` is ``OPERAND_DESIGN`` (0) or ``OPERAND_RESPONSE`` (1).
        ``INPUT_NOT_NDARRAY`` (any ndarray subclass, ``numpy.ma`` included),
        ``INPUT_WRONG_DTYPE``, ``INPUT_WRONG_NDIM``, ``INPUT_EMPTY`` and, for
        ``y``, ``INPUT_WRONG_SHAPE`` (length is not ``R``), with empty
        ``positions``. ``INPUT_NON_FINITE``: ``positions`` holds the flat
        C-order index of the first NaN or inf. ``INPUT_NON_FINITE`` on
        ``OPERAND_RESPONSE`` also when every element of ``y`` is finite but
        ``fitted``, ``resid`` or ``rss`` exceeds the float64 range;
        ``positions`` then holds the index of the largest ``abs(y)``.
    CoreParameterError
        ``PARAM_UNKNOWN_COVARIANCE``, ``PARAM_LAG_NOT_ALLOWED``,
        ``PARAM_NEGATIVE_LAG``, ``PARAM_LAG_ABOVE_LIMIT`` (see
        ``covariance.check_covariance_parameters``) and
        ``PARAM_RANK_RTOL_OUT_OF_RANGE`` (see ``design.rank_gate``).
    CoreSampleSizeError
        ``SAMPLE_ROWS_NOT_ABOVE_COLUMNS``: ``R < k + 1``. ``n_available`` is
        ``R``, ``n_required`` is ``k + 1``. Raised by the gate before the SVD.
    CoreRankError
        A zero column (before the SVD) or ``rank < k`` (see
        ``design.rank_gate``).
    CoreDegenerateFitError
        ``DEGENERATE_ZERO_RESPONSE_VARIANCE``: ``intercept`` is True and every
        element of ``y`` equals ``y[0]``. ``value`` and ``bound`` are ``0.0``.
        ``DEGENERATE_EXACT_FIT``: ``rss <= (R * eps)**2 * sum(y**2)``, with
        ``rss`` as float64 (an ``rss`` that underflows to ``0.0`` counts as
        ``rss == 0.0``). ``value`` is ``rss``, ``bound`` is the right side.
        ``DEGENERATE_NON_POSITIVE_VARIANCE``: a diagonal entry of ``cov`` is
        not above zero or not finite; or ``coef``, ``se``, ``t_stats``,
        ``p_values`` or ``cov`` holds a non-finite value. ``columns`` holds
        the affected column indices, ascending (both indices for an entry of
        ``cov``); ``value`` is the smallest diagonal entry of ``cov`` over
        those columns; ``bound`` is ``0.0``.

    Notes
    -----
    The checks run in this order:

    1. ``design`` (operand 0): type, dtype, ndim, empty, finite.
    2. ``y`` (operand 1): type, dtype, ndim, empty, length ``R``, finite.
    3. ``intercept`` type.
    4. Covariance kind and lag (``check_covariance_parameters``).
    5. ``design.rank_gate``: ``rank_rtol``, ``R >= k + 1``, zero column, then
       the SVD, then the rank.
    6. ``DEGENERATE_ZERO_RESPONSE_VARIANCE``, before the solve.
    7. The solve, then the float64 range of ``fitted``, ``resid`` and
       ``rss``.
    8. ``DEGENERATE_EXACT_FIT``.
    9. The covariance, then ``DEGENERATE_NON_POSITIVE_VARIANCE`` on its
       diagonal.
    10. ``DEGENERATE_NON_POSITIVE_VARIANCE`` on a non-finite ``coef``,
        ``se``, ``t_stats``, ``p_values`` or ``cov``.

    Every check of steps 1 to 5 except the rank runs before the SVD.

    Zero variance is tested exactly: every element equals ``y[0]``. That is
    ``sum((y - mean(y))**2) == 0`` of A.3 in exact arithmetic. The float64
    evaluation of the A.3 expression is not zero for some constant ``y``,
    because ``mean(y)`` rounds (sandbox evidence: 15 of 30 constant vectors,
    for example ``[0.1] * 3`` gives 5.8e-34).

    For a non-constant ``y`` the centred sum is above zero after the scaling,
    so ``r_squared`` is finite. An all-zero ``y`` without an intercept is an
    exact fit: ``rss == 0.0``.
    """
    _validate_design(design)
    n_rows = int(design.shape[0])
    _validate_response(y, n_rows)
    has_intercept = _declared_flag(intercept)
    kind, lag = check_covariance_parameters(
        covariance_kind=covariance_kind, covariance_lag=covariance_lag, n_rows=n_rows
    )
    gate = rank_gate(design, rank_rtol=rank_rtol)
    n_columns = gate.n_columns
    dof = n_rows - n_columns

    if has_intercept and bool(np.all(y == y[0])):
        raise CoreDegenerateFitError(
            reason=codes.DEGENERATE_ZERO_RESPONSE_VARIANCE,
            value=0.0,
            bound=0.0,
            columns=(),
        )

    # A.3 solve from the gate SVD. errstate keeps an overflow silent: the
    # range check below is the only signal.
    with np.errstate(over="ignore", under="ignore"):
        gamma = gate.vh.T @ ((gate.u.T @ y) / gate.singular_values)
        coef = gamma / gate.column_norms
        fitted = gate.design @ coef
        resid = y - fitted
    if not (bool(np.isfinite(fitted).all()) and bool(np.isfinite(resid).all())):
        raise _response_out_of_range(y)
    rss_unit, rss_exponent = _square_sum(resid)
    rss = _ldexp(rss_unit, 2 * rss_exponent)
    if not math.isfinite(rss):
        raise _response_out_of_range(y)

    # C.14: rss <= (R * eps)**2 * sum(y**2), compared on the scaled sums.
    y_unit, y_exponent = _square_sum(y)
    factor = (n_rows * _EPS) ** 2
    if rss == 0.0 or rss_unit <= _ldexp(
        factor * y_unit, 2 * (y_exponent - rss_exponent)
    ):
        raise CoreDegenerateFitError(
            reason=codes.DEGENERATE_EXACT_FIT,
            value=rss,
            bound=_ldexp(factor * y_unit, 2 * y_exponent),
            columns=(),
        )

    cov = coefficient_covariance(gate, resid, covariance_kind=kind, covariance_lag=lag)
    diagonal = np.diagonal(cov)
    admissible = np.isfinite(diagonal) & (diagonal > 0.0)
    if not bool(admissible.all()):
        raise _non_positive_variance(diagonal, np.flatnonzero(~admissible))

    with np.errstate(over="ignore", under="ignore"):
        se = np.sqrt(diagonal)
        t_stats = coef / se
    p_values = 2.0 * np.asarray(stats.t.sf(np.abs(t_stats), dof), dtype=np.float64)
    cov_finite = np.isfinite(cov)
    affected = ~(
        np.isfinite(coef)
        & np.isfinite(se)
        & np.isfinite(t_stats)
        & np.isfinite(p_values)
        & cov_finite.all(axis=0)
        & cov_finite.all(axis=1)
    )
    if bool(affected.any()):
        raise _non_positive_variance(diagonal, np.flatnonzero(affected))

    if has_intercept:
        base_unit, base_exponent = _square_sum(y - np.mean(y))
    else:
        base_unit, base_exponent = y_unit, y_exponent
    r_squared = 1.0 - _ldexp(rss_unit / base_unit, 2 * (rss_exponent - base_exponent))

    return CoreFit(
        gate=gate,
        coef=_read_only(coef),
        se=_read_only(se),
        t_stats=_read_only(t_stats),
        p_values=_read_only(p_values),
        cov=cov,
        resid=_read_only(resid),
        fitted=_read_only(fitted),
        rss=rss,
        r_squared=r_squared,
        dof=dof,
        n_obs=n_rows,
        durbin_watson=diagnostics.durbin_watson(resid),
    )
