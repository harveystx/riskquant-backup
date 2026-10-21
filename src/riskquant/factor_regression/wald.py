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
    8. RESTRICTION_RANK_DEFICIENT: a zero row, J above K, or s_min <= tau * s_max
       of the row-normalised restriction (OD-10, amendment to A.5).
    9. RESTRICTION_NON_FINITE: a non-finite entry in gap or in m (the matrix
       passed to numpy.linalg.cholesky), checked before the Cholesky step;
       numpy.linalg.cholesky is not called. This also applies where m is not
       positive definite: that input class raised
       RESTRICTION_NOT_POSITIVE_DEFINITE before this check existed.
       RESTRICTION_NOT_POSITIVE_DEFINITE: Cholesky of the finite m fails.
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

# OD-10, option B (amendment to the rank rule of A.5). The rank test runs on the
# row-normalised restriction: RESTRICTION_RANK_DEFICIENT when s_min <= tau * s_max.
# For unit rows, m = restriction @ cov @ restriction.T has a condition number of
# about 1 / ratio**2 with ratio = s_min / s_max. The relative error of W grows as
# about eps / ratio**2: 1.8e-8 at ratio 1.1e-4 and 5.5e-9 at ratio 2e-4.
# Sandbox evidence (numpy 2.4.4, Fraction oracle, 2000 random draws per cell):
# at ratio 1.1e-4 the median error of W is 3e-9 and the worst is 9e-8; at ratio
# 2e-4 the median is 9e-10 and the worst is 2.4e-8. An error of about 1e-8 or less
# is typical just above this tau. It is not a bound.
_RANK_RATIO_TAU: Final = 1e-4


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
        Upper-tail survival function ``chi2.sf(W, j)``.  Asymptotic: ``W`` is
        chi-squared only in the limit, for IID, HC0 and HAC covariance alike.
    f_pvalue
        Upper-tail survival function ``f.sf(F, j, dof)``.  Exact only under
        IID covariance and normal errors; an approximation for HC0 or HAC (A.5).
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
        full row rank ``J``.  The rank test runs on the row-normalised matrix
        (amendment to A.5, see Notes).
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
    CoreInputError
        Steps 1-2 and 4-5: ``INPUT_NOT_NDARRAY``, ``INPUT_WRONG_DTYPE``,
        ``INPUT_WRONG_NDIM``, ``INPUT_EMPTY``, ``INPUT_WRONG_SHAPE``,
        ``INPUT_NON_FINITE`` (flat index in ``positions``).
        A ``restriction`` or ``rhs`` that is not exactly ``numpy.ndarray``
        raises ``INPUT_NOT_NDARRAY`` at operand 2 or 3.
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
    8. ``RESTRICTION_RANK_DEFICIENT``: ``J > K``, a zero row, or
       ``s_min <= tau * s_max`` of the row-normalised ``restriction``.
    9. ``RESTRICTION_NON_FINITE``: a non-finite entry in ``gap`` or in ``m``
       (the matrix passed to ``numpy.linalg.cholesky``), checked before the
       Cholesky step; ``numpy.linalg.cholesky`` is not called.  This also
       applies where ``m`` is not positive definite: that input class raised
       ``RESTRICTION_NOT_POSITIVE_DEFINITE`` before this check existed.
       ``RESTRICTION_NOT_POSITIVE_DEFINITE``: Cholesky of the finite ``m``
       fails.
    10. ``RESTRICTION_NON_FINITE``: ``W`` or ``F`` is not finite.

    Amendment to the rank rule of A.5 (owner decision OD-10, option B).  The
    rest of A.5 is unchanged.

    * Each row of ``restriction`` is divided by its 2-norm.  The matching
      entry of ``rhs`` is divided by the same norm.  This happens before the
      rank test and before ``m`` and ``gap`` are formed.  It is an invertible
      row scaling, so ``W`` does not change in exact arithmetic.  The row
      norms use the power-of-two scaling of ``design.py``.  They do not
      overflow or underflow.  A zero row raises ``RESTRICTION_RANK_DEFICIENT``.
    * ``numpy.linalg.svd`` of the normalised ``restriction`` gives ``s_min``
      and ``s_max``.  The module constant ``_RANK_RATIO_TAU = 1e-4`` is the
      threshold ``tau``.  ``RESTRICTION_RANK_DEFICIENT`` is raised when
      ``s_min <= tau * s_max``.  The old rule ``s > 1e-12 * s[0]`` on the raw
      rows depended on the units of the rows.
    * ``J > K`` stays rank deficient.

    The chi-squared p-value is asymptotic.  Only the F test is exact, and only
    under IID covariance and normal errors (A.5).  With HC0 or HAC covariance
    the F p-value is an approximation.

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
    # OD-10, option B: normalise each row, then test s_min against s_max.
    # J > K gives at most K singular values, so the ratio test cannot see it.
    if j > k:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_RANK_DEFICIENT, shape=rest_shape
        )
    peak = np.max(np.abs(restriction), axis=1)
    if bool((peak == 0.0).any()):
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_RANK_DEFICIENT, shape=rest_shape
        )
    # A power-of-two scaling by the binary exponent of the row peak is exact.
    # It keeps the sum of squares in range, as in design.py. A plain sum of
    # squares reads a row near 1e-170 as zero and a row near 1e160 as infinite.
    _, exponent = np.frexp(peak)
    unit = np.ldexp(restriction, -exponent[:, np.newaxis])
    unit_norm = np.sqrt(np.sum(unit * unit, axis=1))  # in [0.5, sqrt(K)]
    rows = unit / unit_norm[:, np.newaxis]
    # rhs / ||row|| is formed as ldexp(rhs / unit_norm, -exponent) so that the
    # row norm is never formed. A result beyond the float64 range is inf, and
    # step 10 reports it.
    with np.errstate(over="ignore"):
        rhs_unit = np.ldexp(rhs / unit_norm, -exponent)
    sv = np.linalg.svd(rows, compute_uv=False)  # descending; the J <= K case
    if sv[-1] <= _RANK_RATIO_TAU * sv[0]:
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_RANK_DEFICIENT, shape=rest_shape
        )

    # ---- 9. RESTRICTION_NON_FINITE (gap, m), NOT_POSITIVE_DEFINITE + solve -----
    # Finite rows, coef, cov and rhs_unit can still give an overflow here: a
    # cov entry near the float64 maximum overflows the matmul, and rhs_unit is
    # inf when the row norm is tiny. numpy.linalg.cholesky returns inf with no
    # LinAlgError on such an m, and the solve then gives a finite wrong W (W = 0
    # for J = 1). The errstate keeps the overflow silent; the check below is the
    # only signal (C.1, C.4).
    with np.errstate(over="ignore", invalid="ignore"):
        gap = rows @ coef - rhs_unit
        m = rows @ cov @ rows.T
        # Symmetrise to guard bit-for-bit asymmetry from floating-point
        # arithmetic. Halve before the add (O-1 = B): 0.5 * (m + m.T) overflows
        # when two finite entries near the float64 maximum are added. Halving is
        # exact outside the subnormal range, so wherever 0.5 * (m + m.T) is
        # finite this form equals it bit for bit (F-7).
        m = 0.5 * m + 0.5 * m.T
    if not (bool(np.isfinite(gap).all()) and bool(np.isfinite(m).all())):
        raise CoreRestrictionError(
            reason=codes.RESTRICTION_NON_FINITE, shape=rest_shape
        )
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
    # numpy 2.4.4 warns on an overflow in dot (sandbox evidence). Step 10 is
    # the only signal.
    with np.errstate(over="ignore", invalid="ignore"):
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
