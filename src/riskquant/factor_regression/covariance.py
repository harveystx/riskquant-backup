# src/riskquant/factor_regression/covariance.py
"""
Coefficient covariance: IID, HC0 and HAC_BARTLETT (A.3, C.13). PRIVATE.

The formulas are those of A.3, on the scaled design ``Xs = X / d`` with the
thin SVD ``Xs = u @ diag(s) @ vh`` of the rank gate:

* bread ``B = vh.T @ diag(1 / s**2) @ vh = (Xs.T @ Xs)^-1``;
* IID: ``C = (rss / dof) * B``;
* HC0: ``C = B @ M0 @ B`` with ``M0 = Xs.T @ diag(e**2) @ Xs``;
* HAC_BARTLETT(L): ``C = B @ M @ B`` with ``M = M0 + sum_{j=1..L} w_j
  (G_j + G_j.T)``, ``G_j = sum_{t=j..R-1} e_t e_{t-j} x_t x_{t-j}.T`` and
  ``w_j = 1 - j / (L + 1)``. No small-sample factor. ``L = 0`` gives HC0;
* ``cov = C / outer(d, d)``.

Evaluation form. With ``P = diag(1 / s) @ vh``, ``B = P.T @ P`` and
``Xs @ B = u @ P``. So ``B @ M0 @ B = Q.T @ Q`` with ``Q = diag(e) @ u @ P``:
row ``t`` of ``Q`` is ``e_t * (B @ x_t).T``. The same substitution gives
``B @ G_j @ B = Q[j:].T @ Q[:R-j]``. The module evaluates ``Q`` and never
forms ``B @ M @ B``. Both forms are the same matrix in exact arithmetic. The
literal form multiplies the rounding error of ``M0`` by ``B`` twice, so its
relative error grows as ``kappa**2 * eps``; the ``Q`` form grows as
``kappa * eps``. Sandbox evidence (Python 3.12.3, numpy 2.4.4), HC0, intercept
and one regressor at mean to sd ratio 1e7, 250 rows, seeds 1 to 5, against a
``fractions.Fraction`` sandwich: literal form 3.6e-2, ``Q`` form 1.5e-9 (max
abs error over max abs exact).

Scaling. ``C`` is quadratic in the residuals. The residuals are scaled by the
power of two ``2**-e`` with ``e`` the binary exponent of their largest
magnitude, as ``design.py`` scales the columns for their norms. Each column
norm is split as ``d_j = m_j * 2**p_j`` with ``m_j`` in ``[0.5, 1)``. The
result is ``ldexp(C_unit / outer(m, m), 2*e - p_j - p_l)``: one rounding to
the float64 range at the end, and no intermediate overflow or underflow.
Power-of-two scaling is exact, so the result equals the unscaled formula bit
for bit when that formula neither overflows nor underflows.

Layout. ``u`` is ``(R, k)`` C-order from the gate, so ``Q`` is ``(R, k)``
C-order and each lag reads two row blocks of it. The loop runs over the lags,
never over the rows (owner decision 6). Cost: one ``(R, k)`` scaled copy of
``u``, one ``(R, k)`` product for ``Q`` and one ``(k, R - j) @ (R - j, k)``
product per lag.
"""

from __future__ import annotations

import math
import operator
from typing import Final

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.design import CoreScaledSvd
from riskquant.factor_regression.errors import CoreInputError, CoreParameterError

_IID: Final = int(codes.IID)
_HAC_BARTLETT: Final = int(codes.HAC_BARTLETT)
_COVARIANCE_CODES: Final = frozenset(int(code) for code in codes.COVARIANCE_CODES)


def _declared_integer(value: int) -> int:
    # operator.index rejects a float or str with TypeError, so the core never
    # rounds a lag. bool is an int subclass and operator.index reads True as 1;
    # a flag is not a covariance kind or a lag.
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("a covariance kind or lag must be an int, not a bool")
    return operator.index(value)


def check_covariance_parameters(
    *, covariance_kind: int, covariance_lag: int, n_rows: int
) -> tuple[int, int]:
    """
    Check the covariance kind and lag against the row count (C.13).

    Parameters
    ----------
    covariance_kind
        A code of ``codes.COVARIANCE_CODES``. An ``int`` or a NumPy integer.
    covariance_lag
        ``0`` for IID and HC0. ``0 <= L <= n_rows - 1`` for HAC_BARTLETT. The
        facade passes ``0`` when its spec has no lag.
    n_rows
        Row count ``R`` of the design.

    Returns
    -------
    tuple[int, int]
        The kind and the lag as ``int``.

    Raises
    ------
    TypeError
        The kind or the lag is not an integer, or is a ``bool`` or
        ``numpy.bool_``. The kind is checked first.
    CoreParameterError
        ``PARAM_UNKNOWN_COVARIANCE`` (``value`` is the kind),
        ``PARAM_LAG_NOT_ALLOWED`` (IID or HC0 with a non-zero lag),
        ``PARAM_NEGATIVE_LAG`` (HAC_BARTLETT with ``L < 0``) or
        ``PARAM_LAG_ABOVE_LIMIT`` (HAC_BARTLETT with ``L > R - 1``). For the
        lag reasons, ``value`` is the lag.

    Notes
    -----
    The checks run in this order: kind type, lag type, kind, lag. The lag
    limit is ``R - 1`` because ``G_j`` sums over ``t = j .. R - 1``: a lag of
    ``R`` or more adds an empty sum and hides a wrong declaration.
    """
    kind = _declared_integer(covariance_kind)
    lag = _declared_integer(covariance_lag)
    if kind not in _COVARIANCE_CODES:
        raise CoreParameterError(
            reason=codes.PARAM_UNKNOWN_COVARIANCE, value=float(kind)
        )
    if kind != _HAC_BARTLETT:
        if lag != 0:
            raise CoreParameterError(
                reason=codes.PARAM_LAG_NOT_ALLOWED, value=float(lag)
            )
    elif lag < 0:
        raise CoreParameterError(reason=codes.PARAM_NEGATIVE_LAG, value=float(lag))
    elif lag > n_rows - 1:
        raise CoreParameterError(reason=codes.PARAM_LAG_ABOVE_LIMIT, value=float(lag))
    return kind, lag


def _validate_residuals(resid: NDArray[np.float64], n_rows: int) -> None:
    # Exact type, not isinstance, before the first reduction. numpy.ma skips
    # masked elements in .all(): a NaN residual under the mask passes the
    # finite check. The core converts nothing, so it rejects every subclass.
    if type(resid) is not np.ndarray:
        raise CoreInputError(reason=codes.INPUT_NOT_NDARRAY, operand=0, positions=())
    if resid.dtype != np.float64:
        raise CoreInputError(reason=codes.INPUT_WRONG_DTYPE, operand=0, positions=())
    if resid.ndim != 1:
        raise CoreInputError(reason=codes.INPUT_WRONG_NDIM, operand=0, positions=())
    if int(resid.shape[0]) == 0:
        raise CoreInputError(reason=codes.INPUT_EMPTY, operand=0, positions=())
    if int(resid.shape[0]) != n_rows:
        raise CoreInputError(reason=codes.INPUT_WRONG_SHAPE, operand=0, positions=())
    finite = np.isfinite(resid)
    if not bool(finite.all()):
        first = int(np.flatnonzero(~finite)[0])
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE, operand=0, positions=(first,)
        )


def coefficient_covariance(
    gate: CoreScaledSvd,
    resid: NDArray[np.float64],
    *,
    covariance_kind: int,
    covariance_lag: int,
) -> NDArray[np.float64]:
    """
    Covariance of the coefficients in original units (A.3).

    Parameters
    ----------
    gate
        Output of ``design.rank_gate`` for the design of the fit: ``u``,
        ``singular_values``, ``vh`` and ``column_norms`` are read.
        ``R >= k + 1``, so ``dof = R - k >= 1``.
    resid
        ``float64`` ``(R,)``, all finite: the residuals ``e = y - X @ coef``
        in row (date) order. HAC_BARTLETT reads the order.
    covariance_kind
        ``codes.IID``, ``codes.HC0`` or ``codes.HAC_BARTLETT``.
    covariance_lag
        ``0`` for IID and HC0; the Bartlett lag ``L`` for HAC_BARTLETT.

    Returns
    -------
    NDArray[np.float64]
        New read-only ``(k, k)`` C-order matrix ``cov = C / outer(d, d)``,
        exactly symmetric. Entries outside the float64 range round to
        ``inf`` or to zero; the caller checks the diagonal (C.14). No caller
        array is changed or retained (C.19).

    Raises
    ------
    TypeError
        See ``check_covariance_parameters``.
    CoreInputError
        ``operand`` is ``0``. ``INPUT_NOT_NDARRAY`` (any ndarray subclass,
        ``numpy.ma`` included), ``INPUT_WRONG_DTYPE``, ``INPUT_WRONG_NDIM``,
        ``INPUT_EMPTY``, ``INPUT_WRONG_SHAPE`` (length is not ``R``) with
        empty ``positions``. ``INPUT_NON_FINITE``: ``positions`` holds the
        index of the first NaN or inf.
    CoreParameterError
        See ``check_covariance_parameters``.

    Notes
    -----
    The checks run in this order: residual array, parameters. The module
    docstring gives the evaluation form and the scaling.
    """
    n_rows = gate.n_rows
    n_columns = gate.n_columns
    _validate_residuals(resid, n_rows)
    kind, lag = check_covariance_parameters(
        covariance_kind=covariance_kind, covariance_lag=covariance_lag, n_rows=n_rows
    )

    _, exponent = math.frexp(float(np.max(np.abs(resid))))
    unit = np.ldexp(resid, -exponent)
    # P = diag(1 / s) @ vh, so B = P.T @ P. The gate keeps s above
    # rank_rtol * s[0] > 0.
    p = gate.vh / gate.singular_values[:, np.newaxis]
    if kind == _IID:
        dof = n_rows - n_columns
        unit_cov = (float(np.dot(unit, unit)) / dof) * (p.T @ p)
    else:
        q = (gate.u * unit[:, np.newaxis]) @ p
        unit_cov = q.T @ q
        for j in range(1, lag + 1):
            cross = q[j:].T @ q[: n_rows - j]
            unit_cov += (1.0 - j / (lag + 1)) * (cross + cross.T)
    # A product of the form A.T @ A is not always symmetric bit for bit. The
    # average is exact for a symmetric input: 2 * C and 0.5 * (2 * C) round
    # to nothing.
    unit_cov = 0.5 * (unit_cov + unit_cov.T)

    mantissa, power = np.frexp(gate.column_norms)
    shift = (2 * exponent - power[:, np.newaxis] - power[np.newaxis, :]).astype(
        np.int32
    )
    # errstate keeps an overflow or underflow of the last rounding silent:
    # the diagonal check of the caller is the only signal (C.14).
    with np.errstate(over="ignore", under="ignore"):
        cov: NDArray[np.float64] = np.ldexp(
            unit_cov / np.outer(mantissa, mantissa), shift
        )
    cov.setflags(write=False)
    return cov
