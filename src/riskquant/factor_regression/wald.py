# src/riskquant/factor_regression/wald.py
"""
Core joint Wald test for a linear restriction on OLS coefficients (A.5, C.18). PRIVATE.

Given a fitted coefficient vector ``coef`` of length ``K``, its ``(K, K)``
covariance matrix ``cov``, the integer degrees of freedom ``dof``, a
``(J, K)`` restriction matrix and a ``(J,)`` right-hand-side vector, this
module computes the joint Wald statistic:

    gap = restriction @ coef - rhs                               (J,)
    m   = restriction @ cov @ restriction.T                      (J, J)
    W   = gap^T  m^{-1}  gap

``m^{-1}`` is never formed explicitly. The Cholesky factor ``L`` of ``m``
(so that ``m = L @ L.T``) is computed with ``numpy.linalg.cholesky``. Then
``z = L^{-1} gap`` is found by ``numpy.linalg.solve(L, gap)`` and
``W = z @ z``. This is algebraically equal to the Mahalanobis-distance form
of A.5, avoids inverting ``m`` and preserves the full precision of the
Cholesky path.

``F = W / J``.  ``chi2_pvalue = chi2.sf(W, J)`` and
``f_pvalue = f.sf(F, J, dof)`` from ``scipy.stats`` (one import, C.21).

The core sees no name, unit or label (C.22). Errors carry integer codes,
shape tuples and numeric values only; the facade adds names and dates.

Check order (wald_core):
    1. coef (operand 0): type, dtype, ndim, empty, finite.
    2. cov  (operand 1): type, dtype, ndim, square, consistent K, finite.
    3. dof: TypeError for bool or non-integer; CoreParameterError for dof < 1.
    4. restriction (operand 2): type, dtype.
    5. rhs         (operand 3): type, dtype.
    6. RESTRICTION_WRONG_SHAPE: restriction not 2-D; J is 0; K column count
       differs from len(coef); rhs length differs from J.
    7. RESTRICTION_NON_FINITE: a NaN or inf in restriction or rhs.
    8. RESTRICTION_RANK_DEFICIENT: row rank of restriction is below J.
    9. RESTRICTION_NOT_POSITIVE_DEFINITE: Cholesky of m fails.
   10. RESTRICTION_NON_FINITE: W or F is not finite after the solve.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from riskquant.factor_regression import codes
from riskquant.factor_regression.errors import (
    CoreInputError,
    CoreParameterError,
    CoreRestrictionError,
)

# A.5: row-rank tolerance for the restriction matrix SVD.
_RANK_TOL: Final = 1e-12


@dataclass(frozen=True, slots=True, kw_only=True)
class CoreWald:
    """
    Result of one joint Wald test (A.5, C.18).

    Attributes
    ----------
    w
        Wald statistic ``W``: the Mahalanobis distance of ``restriction @
        coef - rhs`` under ``restriction @ cov @ restriction.T``.
    f_stat
        ``F = W / j``.
    chi2_pvalue
        Two-tail p-value: ``chi2.sf(W, j)``.  Asymptotically chi-squared;
        exact under normality with IID covariance.
    f_pvalue
        Two-tail p-value: ``f.sf(F, j, dof)``.  Exact under normality and
        IID covariance; approximation for HC0 or HAC (A.5).
    j
        Number of restrictions ``J >= 1``.
    dof
        Degrees of freedom ``R - k >= 1`` from the OLS fit.

    Notes
    -----
    ``CoreWald`` holds Python scalars only (no arrays), so ``eq=True`` is
    correct and no ``eq=False`` override is needed.  All fields are finite
    because ``wald_core`` raises before returning a non-finite statistic.
    """

    w: float
    f_stat: float
    chi2_pvalue: float
    f_pvalue: float
    j: int
    dof: int


# ------------------------------------------------------------------ helpers


def _check_ndarray_float64(arr: NDArray[np.float64], operand: int, ndim: int) -> None:
    """Type, dtype and ndim gate for coef and cov (steps 1-2 of check order)."""
    # Exact type before the first reduction. numpy.ma.MaskedArray and other
    # subclasses can hide NaN from .all(): reject them before any check reads
    # the array. The core converts nothing (C.19).
    if type(arr) is not np.ndarray:
        raise CoreInputError(
            reason=codes.INPUT_NOT_NDARRAY, operand=operand, positions=()
        )
    if arr.dtype != np.float64:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_DTYPE, operand=operand, positions=()
        )
    if arr.ndim != ndim:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_NDIM, operand=operand, positions=()
        )


def _reject_non_finite(arr: NDArray[np.float64], operand: int) -> None:
    """Raise INPUT_NON_FINITE at the first non-finite flat index."""
    # np.errstate is not needed here: arr is already confirmed finite-range
    # by dtype (float64), and isfinite handles ±inf and NaN cleanly.
    finite = np.isfinite(arr)
    if not bool(finite.all()):
        first = int(np.flatnonzero(~finite)[0])
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE, operand=operand, positions=(first,)
        )


# ------------------------------------------------------------------ public


def wald_core(
    coef: NDArray[np.float64],
    cov: NDArray[np.float64],
    dof: int,
    restriction: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> CoreWald:
    """
    Joint Wald statistic for ``restriction @ coef = rhs`` (A.5, C.18).

    Parameters
    ----------
    coef
        ``float64`` ``(K,)``, all finite: the OLS coefficient vector from
        a ``CoreFit`` or equivalent, intercept first if declared (C.8).
    cov
        ``float64`` ``(K, K)``, symmetric, all finite: the coefficient
        covariance from the same ``CoreFit``.  IID, HC0 or HAC_BARTLETT.
    dof
        Degrees of freedom ``R - k >= 1`` from the OLS fit.  A ``bool``,
        float or str raises ``TypeError``.  A plain int below 1 raises
        ``CoreParameterError(PARAM_STRIDE_BELOW_ONE)``.
    restriction
        ``float64`` ``(J, K)``: the linear restriction matrix.  Must have
        full row rank ``J`` (checked by SVD, tol = 1e-12 * s[0]).
    rhs
        ``float64`` ``(J,)``: the right-hand side of ``restriction @
        coef = rhs``.

    Returns
    -------
    CoreWald
        Frozen result with ``W``, ``F = W/J``, both p-values, ``J`` and
        ``dof``.  All fields are Python scalars; no caller array is changed
        or retained (C.19).

    Raises
    ------
    TypeError
        ``dof`` is a ``bool``, ``numpy.bool_``, float or str (steps 3).
        ``restriction`` or ``rhs`` is not exactly ``numpy.ndarray``
        (INPUT_NOT_NDARRAY at operand 2 or 3).
    CoreInputError
        Steps 1-2 and 4-5: ``INPUT_NOT_NDARRAY``, ``INPUT_WRONG_DTYPE``,
        ``INPUT_WRONG_NDIM``, ``INPUT_EMPTY``, ``INPUT_WRONG_SHAPE``,
        ``INPUT_NON_FINITE`` (flat index in ``positions``).
    CoreParameterError
        Step 3: ``PARAM_STRIDE_BELOW_ONE`` when ``dof < 1``.
    CoreRestrictionError
        Steps 6-10: ``RESTRICTION_WRONG_SHAPE``, ``RESTRICTION_NON_FINITE``,
        ``RESTRICTION_RANK_DEFICIENT``, ``RESTRICTION_NOT_POSITIVE_DEFINITE``
        or ``RESTRICTION_NON_FINITE`` (non-finite ``W`` or ``F``).  The
        ``shape`` field is the shape of ``restriction`` as received.

    Notes
    -----
    Check order:

    1. ``coef`` (operand 0): type, dtype, ndim (1-D), empty, finite.
    2. ``cov``  (operand 1): type, dtype, ndim (2-D), square, consistent
       ``K``, finite.
    3. ``dof``: ``TypeError`` for bool or non-integer;
       ``CoreParameterError(PARAM_STRIDE_BELOW_ONE)`` for ``dof < 1``.
    4. ``restriction`` (operand 2): type (exact ``ndarray``), dtype.
    5. ``rhs``         (operand 3): type (exact ``ndarray``), dtype.
    6. ``RESTRICTION_WRONG_SHAPE``: not 2-D; ``J == 0``; ``K`` column count
       differs from ``len(coef)``; ``rhs`` length differs from ``J``.
    7. ``RESTRICTION_NON_FINITE``: NaN or inf in ``restriction`` or ``rhs``.
    8. ``RESTRICTION_RANK_DEFICIENT``: row rank < ``J`` (SVD, tol
       ``= 1e-12 * s[0]``).  ``J > K`` is always rank-deficient.
    9. ``RESTRICTION_NOT_POSITIVE_DEFINITE``: Cholesky of ``m`` fails.
    10. ``RESTRICTION_NON_FINITE``: ``W`` or ``F`` is not finite.

    The Cholesky path guarantees that ``m = L @ L.T`` is positive definite
    before any solve, so ``L`` is non-singular and
    ``numpy.linalg.solve(L, gap)`` is well-conditioned.  ``scipy.linalg``
    is not used (C.21).
    """
    # ---- 1. coef ----------------------------------------------------------------
    _check_ndarray_float64(coef, 0, 1)
    k = int(coef.shape[0])
    if k == 0:
        raise CoreInputError(reason=codes.INPUT_EMPTY, operand=0, positions=())
    _reject_non_finite(coef, 0)

    # ---- 2. cov -----------------------------------------------------------------
    _check_ndarray_float64(cov, 1, 2)
    if int(cov.shape[0]) != int(cov.shape[1]):
        # Non-square matrix cannot be a covariance.
        raise CoreInputError(reason=codes.INPUT_WRONG_SHAPE, operand=1, positions=())
    if int(cov.shape[0]) != k:
        raise CoreInputError(reason=codes.INPUT_WRONG_SHAPE, operand=1, positions=())
    _reject_non_finite(cov, 1)

    # ---- 3. dof -----------------------------------------------------------------
    # bool is an int subclass; operator.index(True) == 1, which is wrong.
    # numpy.bool_ is also rejected so the result does not depend on NumPy version.
    if isinstance(dof, (bool, np.bool_)):
        raise TypeError("dof must be an int, not a bool")
    try:
        dof_int = operator.index(dof)
    except TypeError as exc:
        raise TypeError("dof must be an int") from exc
    if dof_int < 1:
        # PARAM_STRIDE_BELOW_ONE: "integer parameter must be at least 1".
        # The stride and dof share the same lower bound and the same reason
        # code; codes.py has no dedicated dof code (STOP AND ASK not needed
        # because PARAM_STRIDE_BELOW_ONE is the closest existing match).
        raise CoreParameterError(
            reason=codes.PARAM_STRIDE_BELOW_ONE, value=float(dof_int)
        )

    # ---- 4. restriction (type and dtype only; shape checked in step 6) ----------
    if type(restriction) is not np.ndarray:
        raise CoreInputError(reason=codes.INPUT_NOT_NDARRAY, operand=2, positions=())
    if restriction.dtype != np.float64:
        raise CoreInputError(reason=codes.INPUT_WRONG_DTYPE, operand=2, positions=())

    # ---- 5. rhs (type and dtype only) -------------------------------------------
    if type(rhs) is not np.ndarray:
        raise CoreInputError(reason=codes.INPUT_NOT_NDARRAY, operand=3, positions=())
    if rhs.dtype != np.float64:
        raise CoreInputError(reason=codes.INPUT_WRONG_DTYPE, operand=3, positions=())

    # The shape carried in every CoreRestrictionError is the shape of restriction
    # as received by this function (owner decision 3).
    rest_shape = tuple(int(s) for s in restriction.shape)

    # ---- 6. RESTRICTION_WRONG_SHAPE ---------------------------------------------
    if restriction.ndim != 2:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_WRONG_SHAPE, shape=rest_shape
        )
    j = int(restriction.shape[0])
    if j == 0:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_WRONG_SHAPE, shape=rest_shape
        )
    if int(restriction.shape[1]) != k:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_WRONG_SHAPE, shape=rest_shape
        )
    if rhs.ndim != 1 or int(rhs.shape[0]) != j:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_WRONG_SHAPE, shape=rest_shape
        )

    # ---- 7. RESTRICTION_NON_FINITE ----------------------------------------------
    if not bool(np.isfinite(restriction).all()) or not bool(np.isfinite(rhs).all()):
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_NON_FINITE, shape=rest_shape
        )

    # ---- 8. RESTRICTION_RANK_DEFICIENT ------------------------------------------
    # Thin SVD of restriction: s has shape (min(J, K),).
    # J > K gives at most K singular values; rank <= K < J, so rank < j.
    _, sv, _ = np.linalg.svd(restriction, full_matrices=False)
    rank_tol = _RANK_TOL * float(sv[0])
    rank = int(np.sum(sv > rank_tol))
    if rank < j:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_RANK_DEFICIENT, shape=rest_shape
        )

    # ---- 9. RESTRICTION_NOT_POSITIVE_DEFINITE + solve ---------------------------
    gap = restriction @ coef - rhs
    m = restriction @ cov @ restriction.T
    # Symmetrise to guard bit-for-bit asymmetry from floating-point arithmetic:
    # 2*C and 0.5*(2*C) round to C exactly, so the average introduces no error
    # on a symmetric matrix.
    m = 0.5 * (m + m.T)
    try:
        chol = np.linalg.cholesky(m)  # lower triangular, m = chol @ chol.T
    except np.linalg.LinAlgError as exc:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_NOT_POSITIVE_DEFINITE, shape=rest_shape
        ) from exc

    # z solves chol @ z = gap.  Then W = z @ z = gap.T @ m^{-1} @ gap.
    # numpy.linalg.solve is used instead of a forward substitution because
    # scipy.linalg is prohibited (C.21).  The triangular L is non-singular
    # (positive definite Cholesky), so the solve is numerically stable.
    z = np.linalg.solve(chol, gap)
    w = float(np.dot(z, z))
    f_stat = w / j

    # ---- 10. non-finite W or F --------------------------------------------------
    if not (np.isfinite(w) and np.isfinite(f_stat)):
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_NON_FINITE, shape=rest_shape
        )

    chi2_pvalue = float(stats.chi2.sf(w, j))
    f_pvalue = float(stats.f.sf(f_stat, j, dof_int))

    return CoreWald(
        w=w,
        f_stat=f_stat,
        chi2_pvalue=chi2_pvalue,
        f_pvalue=f_pvalue,
        j=j,
        dof=dof_int,
    )
