# tests/factor_regression/test_wald.py
"""
Tests for riskquant.factor_regression.wald  (S5: core joint Wald test).

Coverage
--------
S5.T1  joint alpha=0 AND beta=1, dof=2; Fraction oracle; closed-form p-values.
S5.T2  correlated regressors, J=2; W differs from sum(t^2); oracle check.
S5.T3  J=1 single restriction; F equals t_j^2 to 1e-12 relative.
S5.T4  duplicate row in restriction; wrong shape; NaN in rhs.
S5.T5  HC0 cov; W equals HC0 oracle and differs from IID.
R-012  mutation resistance: F = W / J, the f.sf degrees of freedom, the W or F
       non-finite check, the cov finite check, the coef empty check, the rhs
       ndim check.
R-013  restriction rank rule (OD-10, option B): row normalisation and the
       ratio test s_min <= tau * s_max with tau = 1e-4.

Owner decision 3 (restriction errors):
  WRONG_SHAPE cases, NON_FINITE, RANK_DEFICIENT, NOT_POSITIVE_DEFINITE.

Owner decision 4 (type checks):
  INPUT_NOT_NDARRAY via masked array for coef, cov, restriction, rhs.
  INPUT_WRONG_DTYPE, INPUT_WRONG_NDIM, INPUT_WRONG_SHAPE.
  dof: TypeError for bool/float/str; CoreParameterError for dof < 1.

Owner decision 5 (no mutation): caller arrays unchanged after wald_core.

Tolerances (owner decision 6):
  W vs Fraction oracle:  1e-12 relative.
  Closed-form p-values:  1e-12 relative.
  F = t_j^2:             1e-12 relative.
  F = W / 2 (J = 2):     1e-12 relative.
  f_pvalue (J=1, dof=2): 1e-12 relative.
  W, row-scaled rows:    1e-12 relative.
  W, ratio 2e-4:         1e-8 relative (the error grows as eps / ratio**2).
  All stated in the assertion comments.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from numpy import ma
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.design import assemble_design
from riskquant.factor_regression.errors import (
    CoreInputError,
    CoreParameterError,
    CoreRestrictionError,
)
from riskquant.factor_regression.fit import CoreFit, fit_core
from riskquant.factor_regression.wald import CoreWald, wald_core

# ------------------------------------------------------------------ helpers


def _oracle_w(
    coef_f64: np.ndarray,
    cov_f64: np.ndarray,
    rest_f64: np.ndarray,
    rhs_f64: np.ndarray,
) -> float:
    """
    Exact Wald statistic via ``fractions.Fraction`` arithmetic.

    Reads every float64 element as an exact rational and applies Gaussian
    elimination on the augmented system ``[M | gap]`` where
    ``M = restriction @ cov @ restriction.T`` and
    ``gap = restriction @ coef - rhs``.  Shares no code with ``wald.py``
    (A.7).
    """
    k = coef_f64.shape[0]
    j = rest_f64.shape[0]

    coef_q = [Fraction(c) for c in coef_f64.tolist()]
    rhs_q = [Fraction(r) for r in rhs_f64.tolist()]
    # ki: column index (renamed from l to avoid E741 ambiguous-name lint)
    rest_q = [[Fraction(rest_f64[i, ki]) for ki in range(k)] for i in range(j)]
    cov_q = [[Fraction(cov_f64[i, ki]) for ki in range(k)] for i in range(k)]

    gap = [
        sum(rest_q[i][ki] * coef_q[ki] for ki in range(k)) - rhs_q[i] for i in range(j)
    ]

    # m[i][ip] = sum_{a,b} rest[i,a] * cov[a,b] * rest[ip,b]
    m = [[Fraction(0)] * j for _ in range(j)]
    for i in range(j):
        for ip in range(j):
            for a in range(k):
                for b in range(k):
                    m[i][ip] += rest_q[i][a] * cov_q[a][b] * rest_q[ip][b]

    # Gaussian elimination on augmented [m | gap]
    aug = [[*row[:], g] for row, g in zip(m, gap, strict=True)]
    for col in range(j):
        piv = col
        for row in range(col + 1, j):
            if abs(aug[row][col]) > abs(aug[piv][col]):
                piv = row
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(col + 1, j):
            if aug[col][col] == Fraction(0):
                continue
            fac = aug[row][col] / aug[col][col]
            for ki in range(col, j + 1):
                aug[row][ki] -= fac * aug[col][ki]

    x_sol = [Fraction(0)] * j
    for i in range(j - 1, -1, -1):
        s = aug[i][j]
        for ki in range(i + 1, j):
            s -= aug[i][ki] * x_sol[ki]
        x_sol[i] = s / aug[i][i]

    # W = gap.T @ m^{-1} @ gap = gap.T @ x_sol
    return float(sum(gap[i] * x_sol[i] for i in range(j)))


def _build_fit_iid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    intercept: bool = True,
) -> CoreFit:
    """Build a minimal IID OLS fit from 1-D arrays x and y."""
    regressors = x.reshape(1, -1)
    design = assemble_design(regressors, intercept=intercept)
    return fit_core(
        design,
        y,
        intercept=intercept,
        rank_rtol=1e-10,
        covariance_kind=int(codes.IID),
        covariance_lag=0,
    )


def _build_fit_hc0(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    intercept: bool = True,
) -> CoreFit:
    """Build a minimal HC0 OLS fit from 1-D arrays x and y."""
    regressors = x.reshape(1, -1)
    design = assemble_design(regressors, intercept=intercept)
    return fit_core(
        design,
        y,
        intercept=intercept,
        rank_rtol=1e-10,
        covariance_kind=int(codes.HC0),
        covariance_lag=0,
    )


def _synthetic_moments(k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Coefficient vector and covariance of size ``k`` (``k`` from 1 to 4).

    The covariance is strictly diagonally dominant, so it is positive definite.
    The R-013 tests use it to reach ``k`` above 2 without a regression fit.
    """
    coef = np.array([0.3, -1.2, 0.7, 0.45])[:k].copy()
    cov = np.array([
        [2.0, 0.3, 0.1, -0.2],
        [0.3, 1.5, -0.2, 0.1],
        [0.1, -0.2, 1.0, 0.25],
        [-0.2, 0.1, 0.25, 1.2],
    ])[:k, :k].copy()
    return coef, cov


def _two_row_restriction(ratio: float, k: int) -> NDArray[np.float64]:
    """
    Rows ``e1`` and ``cos(t) * e1 + sin(t) * e_k`` with ``tan(t / 2) = ratio``.

    ``e_k`` is the last axis, so it is ``e2`` for ``k = 2``.  The singular
    values of the two unit rows are ``sqrt(1 + cos t)`` and ``sqrt(1 - cos t)``,
    so ``s_min / s_max = tan(t / 2) = ratio`` (OD-10).
    """
    angle = 2.0 * math.atan(ratio)
    rows = np.zeros((2, k))
    rows[0, 0] = 1.0
    rows[1, 0] = math.cos(angle)
    rows[1, k - 1] = math.sin(angle)
    return rows


# ------------------------------------------------------------------ fixtures


# Tiny dataset: R=4, k=2 (intercept + 1 regressor), dof=2.
# Used for T1 and T3.
_X4 = np.array([1.0, 2.0, 3.0, 4.0])
_Y4 = np.array([0.1, 1.9, 3.2, 3.8])


# Larger dataset: R=8, k=2, dof=6.  Used for T4 (shape/NaN), T5 (HC0).
_X8 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_Y8 = np.array([0.5, 1.8, 3.1, 4.2, 5.3, 5.9, 7.1, 8.5])


# ================================================================== S5 tests


class TestT1JointAlphaBetaDof2:
    """S5.T1: joint alpha=0 AND beta=1, dof=2."""

    def test_w_matches_fraction_oracle(self) -> None:
        """W equals the Fraction oracle to 1e-12 relative (owner decision 6)."""
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        w_exact = _oracle_w(fit.coef, fit.cov, rest, rhs)
        assert result.j == 2
        assert result.dof == 2
        assert abs(result.w - w_exact) / abs(w_exact) < 1e-12  # tol: 1e-12 rel

    def test_chi2_pvalue_closed_form(self) -> None:
        """chi2_pvalue = exp(-W/2) to 1e-12 relative for chi2(2) (owner decision 6)."""
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        # chi2(2) has sf(w) = exp(-w/2).
        expected = math.exp(-result.w / 2)
        assert abs(result.chi2_pvalue - expected) / expected < 1e-12  # tol: 1e-12 rel

    def test_f_pvalue_closed_form(self) -> None:
        """f_pvalue = 1/(1+F) to 1e-12 relative for F(2,2) (owner decision 6)."""
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        # F(2,2) has sf(F) = 1 / (1 + F).
        expected = 1.0 / (1.0 + result.f_stat)
        assert abs(result.f_pvalue - expected) / expected < 1e-12  # tol: 1e-12 rel

    def test_returns_core_wald(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert isinstance(result, CoreWald)
        assert math.isfinite(result.w)
        assert math.isfinite(result.f_stat)
        assert 0.0 < result.chi2_pvalue <= 1.0
        assert 0.0 < result.f_pvalue <= 1.0


class TestT2CorrelatedRegressorsJ2:
    """S5.T2: correlated regressors, J=2; W differs from sum(t^2)."""

    def _make_fit(self) -> CoreFit:
        # Two regressors with high correlation (rho ~ 0.95).
        rng = np.random.default_rng(42)
        n = 30
        x1 = rng.standard_normal(n)
        x2 = x1 * 0.95 + rng.standard_normal(n) * 0.1
        y = 2.0 * x1 + 3.0 * x2 + rng.standard_normal(n) * 0.5
        regressors = np.vstack([x1, x2])
        design = assemble_design(regressors, intercept=False)
        return fit_core(
            design,
            y,
            intercept=False,
            rank_rtol=1e-10,
            covariance_kind=int(codes.IID),
            covariance_lag=0,
        )

    def test_w_differs_from_sum_t_squared(self) -> None:
        """W != sum(t_j^2): the off-diagonal covariance changes the statistic."""
        fit = self._make_fit()
        rest = np.eye(2)
        rhs = np.zeros(2)
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        sum_t2 = float(np.sum(fit.t_stats**2))
        assert abs(result.w - sum_t2) > 1.0, (
            "W should differ from sum(t^2) for correlated regressors"
        )

    def test_w_matches_fraction_oracle(self) -> None:
        """W equals the Fraction oracle to 1e-12 relative (owner decision 6)."""
        fit = self._make_fit()
        rest = np.eye(2)
        rhs = np.zeros(2)
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        w_exact = _oracle_w(fit.coef, fit.cov, rest, rhs)
        assert abs(result.w - w_exact) / abs(w_exact) < 1e-12  # tol: 1e-12 rel

    def test_j_equals_2(self) -> None:
        fit = self._make_fit()
        rest = np.eye(2)
        rhs = np.zeros(2)
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert result.j == 2


class TestT3SingleRestrictionFEqualsTSquared:
    """S5.T3: J=1, test beta=0; F equals t_beta^2 to 1e-12 relative."""

    def test_f_equals_t_squared(self) -> None:
        """F = t_j^2 to 1e-12 relative (owner decision 6)."""
        fit = _build_fit_iid(_X4, _Y4)
        # Test beta_1 = 0 (column index 1 = the regressor).
        rest = np.array([[0.0, 1.0]])
        rhs = np.array([0.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        t_beta = float(fit.t_stats[1])
        expected_f = t_beta**2
        assert abs(result.f_stat - expected_f) / abs(expected_f) < 1e-12  # tol

    def test_j_equals_1(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[0.0, 1.0]])
        rhs = np.array([0.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert result.j == 1

    def test_f_stat_equals_w_for_j_one(self) -> None:
        # F = W / J = W / 1 = W when J=1.
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[0.0, 1.0]])
        rhs = np.array([0.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert result.f_stat == result.w


class TestT4RestrictionErrors:
    """S5.T4: duplicate row, wrong shape, NaN in rhs."""

    def test_duplicate_row_raises_rank_deficient(self) -> None:
        """A rank-deficient restriction raises RESTRICTION_RANK_DEFICIENT."""
        fit = _build_fit_iid(_X8, _Y8)
        # Row 0 and row 1 are identical → rank 1 < J=2.
        rest = np.array([[1.0, 0.0], [1.0, 0.0]])
        rhs = np.zeros(2)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_RANK_DEFICIENT)

    def test_wrong_shape_raises(self) -> None:
        """A 1-D restriction raises RESTRICTION_WRONG_SHAPE."""
        fit = _build_fit_iid(_X8, _Y8)
        rest = np.array([1.0, 0.0])  # 1-D, not 2-D
        rhs = np.zeros(1)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_WRONG_SHAPE)

    def test_nan_in_rhs_raises(self) -> None:
        """NaN in rhs raises RESTRICTION_NON_FINITE."""
        fit = _build_fit_iid(_X8, _Y8)
        rest = np.array([[1.0, 0.0]])
        rhs = np.array([float("nan")])
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_NON_FINITE)

    def test_nan_in_restriction_raises(self) -> None:
        """NaN in restriction raises RESTRICTION_NON_FINITE."""
        fit = _build_fit_iid(_X8, _Y8)
        rest = np.array([[1.0, float("nan")]])
        rhs = np.zeros(1)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_NON_FINITE)

    def test_shape_field_is_restriction_shape(self) -> None:
        """CoreRestrictionError.shape equals the shape of restriction as received."""
        fit = _build_fit_iid(_X8, _Y8)
        rest = np.ones((3, 4))  # wrong K column count
        rhs = np.zeros(3)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.shape == (3, 4)


class TestT5HC0Fit:
    """S5.T5: W uses the HC0 cov and equals the Fraction oracle."""

    def test_hc0_w_matches_oracle(self) -> None:
        """W with HC0 cov equals the Fraction oracle to 1e-12 relative."""
        fit = _build_fit_hc0(_X8, _Y8)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        w_exact = _oracle_w(fit.coef, fit.cov, rest, rhs)
        assert abs(result.w - w_exact) / abs(w_exact) < 1e-12  # tol: 1e-12 rel

    def test_hc0_differs_from_iid(self) -> None:
        """HC0 and IID covariances yield different W values."""
        fit_hc0 = _build_fit_hc0(_X8, _Y8)
        fit_iid = _build_fit_iid(_X8, _Y8)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        res_hc0 = wald_core(fit_hc0.coef, fit_hc0.cov, fit_hc0.dof, rest, rhs)
        res_iid = wald_core(fit_iid.coef, fit_iid.cov, fit_iid.dof, rest, rhs)
        # coef is the same (same data, same design); cov differs.
        assert abs(res_hc0.w - res_iid.w) > 1e-6


# ================================================================== OD3 tests


class TestOD3RestrictionErrors:
    """Owner decision 3: each CoreRestrictionError reason."""

    def test_j_zero_raises_wrong_shape(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.zeros((0, 2))  # J=0
        rhs = np.zeros(0)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_WRONG_SHAPE)

    def test_k_mismatch_raises_wrong_shape(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0, 0.0]])  # 3 columns, K=2
        rhs = np.zeros(1)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_WRONG_SHAPE)

    def test_rhs_length_mismatch_raises_wrong_shape(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.eye(2)
        rhs = np.zeros(1)  # length 1, but J=2
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_WRONG_SHAPE)

    def test_j_above_k_raises_rank_deficient(self) -> None:
        """J > K is always rank-deficient (A.5)."""
        fit = _build_fit_iid(_X4, _Y4)
        # 3 restrictions on a 2-coefficient model.
        rest = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rhs = np.zeros(3)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_RANK_DEFICIENT)

    def test_non_positive_definite_raises(self) -> None:
        """A singular cov block raises RESTRICTION_NOT_POSITIVE_DEFINITE."""
        fit = _build_fit_iid(_X4, _Y4)
        # cov is PD but restriction collapses M to singular:
        # use a restriction that maps to the zero direction of a custom cov.
        # Easier: supply a custom cov with a zero eigenvalue so that M=0.
        coef = np.array([1.0, 2.0])
        # Rank-1 covariance matrix: v v.T where v=(1,1).
        v = np.array([1.0, 1.0])
        cov_sing = np.outer(v, v)  # [[1,1],[1,1]], PSD but not PD
        # Test in the null space of cov_sing (perpendicular to v), but
        # M would still be 0. Instead, use a restriction orthogonal to v:
        rest = np.array([[1.0, -1.0]])  # maps to v.T @ restriction.T = 0
        rhs = np.zeros(1)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(coef, cov_sing, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_NOT_POSITIVE_DEFINITE)


# ================================================================== OD4 tests


class TestOD4TypeChecks:
    """Owner decision 4: type, dtype, ndim, shape and dof checks."""

    # ---- coef (operand 0) -------------------------------------------------------

    def test_coef_masked_array_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        coef_ma = ma.MaskedArray(fit.coef, mask=False)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(coef_ma, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_NOT_NDARRAY)
        assert exc_info.value.operand == 0

    def test_coef_wrong_dtype_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        coef_f32 = fit.coef.astype(np.float32)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(coef_f32, fit.cov, fit.dof, rest, rhs)  # type: ignore[arg-type]
        assert exc_info.value.reason == int(codes.INPUT_WRONG_DTYPE)
        assert exc_info.value.operand == 0

    def test_coef_2d_raises_wrong_ndim(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        coef_2d = fit.coef.reshape(1, -1)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(coef_2d, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_WRONG_NDIM)
        assert exc_info.value.operand == 0

    def test_coef_non_finite_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        coef_bad = fit.coef.copy()
        coef_bad.setflags(write=True)
        coef_bad[0] = float("nan")
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(coef_bad, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_NON_FINITE)
        assert exc_info.value.operand == 0

    # ---- cov (operand 1) --------------------------------------------------------

    def test_cov_masked_array_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        cov_ma = ma.MaskedArray(fit.cov, mask=False)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, cov_ma, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_NOT_NDARRAY)
        assert exc_info.value.operand == 1

    def test_cov_non_square_raises_wrong_shape(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        cov_rect = np.ones((2, 3))
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, cov_rect, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_WRONG_SHAPE)
        assert exc_info.value.operand == 1

    def test_cov_inconsistent_k_raises_wrong_shape(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        cov_3x3 = np.eye(3)  # K=2 coef but 3x3 cov
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, cov_3x3, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_WRONG_SHAPE)
        assert exc_info.value.operand == 1

    def test_cov_wrong_dtype_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        cov_f32 = fit.cov.astype(np.float32)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, cov_f32, fit.dof, rest, rhs)  # type: ignore[arg-type]
        assert exc_info.value.reason == int(codes.INPUT_WRONG_DTYPE)
        assert exc_info.value.operand == 1

    def test_cov_1d_raises_wrong_ndim(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        cov_1d = fit.cov.ravel()
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, cov_1d, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_WRONG_NDIM)
        assert exc_info.value.operand == 1

    # ---- dof --------------------------------------------------------------------

    def test_dof_bool_raises_type_error(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(TypeError):
            wald_core(
                fit.coef, fit.cov, True, rest, rhs
            )  # bool is int subtype; mypy accepts it

    def test_dof_numpy_bool_raises_type_error(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(TypeError):
            wald_core(fit.coef, fit.cov, np.bool_(True), rest, rhs)  # type: ignore[arg-type]

    def test_dof_float_raises_type_error(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(TypeError):
            wald_core(fit.coef, fit.cov, 2.0, rest, rhs)  # type: ignore[arg-type]

    def test_dof_str_raises_type_error(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(TypeError):
            wald_core(fit.coef, fit.cov, "2", rest, rhs)  # type: ignore[arg-type]

    def test_dof_zero_raises_parameter_error(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreParameterError) as exc_info:
            wald_core(fit.coef, fit.cov, 0, rest, rhs)
        assert exc_info.value.reason == int(codes.PARAM_STRIDE_BELOW_ONE)

    def test_dof_negative_raises_parameter_error(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        with pytest.raises(CoreParameterError) as exc_info:
            wald_core(fit.coef, fit.cov, -1, rest, rhs)
        assert exc_info.value.reason == int(codes.PARAM_STRIDE_BELOW_ONE)

    def test_dof_one_is_valid(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[0.0, 1.0]])
        rhs = np.zeros(1)
        result = wald_core(fit.coef, fit.cov, 1, rest, rhs)
        assert result.dof == 1

    def test_dof_numpy_int32_is_valid(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[0.0, 1.0]])
        rhs = np.zeros(1)
        result = wald_core(fit.coef, fit.cov, np.int32(fit.dof), rest, rhs)  # type: ignore[arg-type]
        assert result.dof == fit.dof

    # ---- restriction (operand 2) ------------------------------------------------

    def test_restriction_masked_array_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest_ma = ma.MaskedArray(np.eye(2), mask=False)
        rhs = np.zeros(2)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest_ma, rhs)
        assert exc_info.value.reason == int(codes.INPUT_NOT_NDARRAY)
        assert exc_info.value.operand == 2

    def test_restriction_wrong_dtype_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest_f32 = np.eye(2, dtype=np.float32)
        rhs = np.zeros(2)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest_f32, rhs)  # type: ignore[arg-type]
        assert exc_info.value.reason == int(codes.INPUT_WRONG_DTYPE)
        assert exc_info.value.operand == 2

    # ---- rhs (operand 3) --------------------------------------------------------

    def test_rhs_masked_array_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.eye(2)
        rhs_ma = ma.MaskedArray(np.zeros(2), mask=False)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs_ma)
        assert exc_info.value.reason == int(codes.INPUT_NOT_NDARRAY)
        assert exc_info.value.operand == 3

    def test_rhs_wrong_dtype_raises(self) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.eye(2)
        rhs_f32 = np.zeros(2, dtype=np.float32)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs_f32)  # type: ignore[arg-type]
        assert exc_info.value.reason == int(codes.INPUT_WRONG_DTYPE)
        assert exc_info.value.operand == 3


# ================================================================== OD5: no mutation


class TestOD5NoMutation:
    """Owner decision 5: wald_core does not mutate any caller array (C.19)."""

    def test_caller_arrays_unchanged(self) -> None:
        fit = _build_fit_iid(_X8, _Y8)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])

        # Writable copies with sentinels that would change if mutated.
        coef_copy = fit.coef.copy()
        cov_copy = fit.cov.copy()
        rest_copy = rest.copy()
        rhs_copy = rhs.copy()

        wald_core(coef_copy, cov_copy, fit.dof, rest_copy, rhs_copy)

        np.testing.assert_array_equal(coef_copy, fit.coef)
        np.testing.assert_array_equal(cov_copy, fit.cov)
        np.testing.assert_array_equal(rest_copy, rest)
        np.testing.assert_array_equal(rhs_copy, rhs)

    def test_core_wald_holds_no_arrays(self) -> None:
        """CoreWald fields are Python scalars only (floats and ints)."""
        fit = _build_fit_iid(_X8, _Y8)
        rest = np.array([[1.0, 0.0]])
        rhs = np.zeros(1)
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert isinstance(result.w, float)
        assert isinstance(result.f_stat, float)
        assert isinstance(result.chi2_pvalue, float)
        assert isinstance(result.f_pvalue, float)
        assert isinstance(result.j, int)
        assert isinstance(result.dof, int)

    def test_read_only_coef_still_accepted(self) -> None:
        """read-only arrays from CoreFit pass the ndarray check."""
        fit = _build_fit_iid(_X4, _Y4)
        # fit.coef and fit.cov are already read-only (from fit_core).
        rest = np.array([[0.0, 1.0]])
        rhs = np.zeros(1)
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert math.isfinite(result.w)


# ================================================================== R-012 tests


class TestR012FStatistic:
    """R-012 (a): F = W / J, with an oracle that shares no code with wald.py."""

    def test_f_stat_is_w_over_two_for_j_two(self) -> None:
        """F equals the Fraction W divided by 2 to 1e-12 relative (A.7)."""
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.array([[1.0, 0.0], [0.0, 1.0]])
        rhs = np.array([0.0, 1.0])
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        w_exact = _oracle_w(fit.coef, fit.cov, rest, rhs)
        f_exact = w_exact / 2.0  # J = 2
        assert abs(result.f_stat - f_exact) / f_exact < 1e-12  # tol: 1e-12 rel


class TestR012FPvalueDegreesOfFreedom:
    """R-012 (b): f.sf receives (J, dof) in this order, not (dof, J)."""

    @pytest.mark.parametrize(
        ("restriction", "rhs"),
        [
            ([[0.0, 1.0]], [0.0]),
            ([[1.0, 0.0]], [0.0]),
            ([[0.0, 1.0]], [1.0]),
        ],
    )
    def test_f_pvalue_j_one_dof_two_closed_form(
        self, restriction: list[list[float]], rhs: list[float]
    ) -> None:
        """
        J = 1 on a fit with dof = 2: ``p = 1 - |t| / sqrt(2 + t**2)``.

        ``F(2, 1)`` is not ``F(1, 2)``, so a swap of the two degrees of freedom
        changes the p-value.  (J = 2 with dof = 2 is symmetric and cannot
        detect the swap.)  ``t`` is the square root of the Fraction W because
        ``F = W = t**2`` for J = 1.  Tolerance: 1e-12 relative.
        """
        fit = _build_fit_iid(_X4, _Y4)
        assert fit.dof == 2
        rest = np.array(restriction)
        rhs_arr = np.array(rhs)
        result = wald_core(fit.coef, fit.cov, fit.dof, rest, rhs_arr)
        t_abs = math.sqrt(_oracle_w(fit.coef, fit.cov, rest, rhs_arr))
        expected = 1.0 - t_abs / math.sqrt(2.0 + t_abs**2)
        assert abs(result.f_pvalue - expected) / expected < 1e-12  # tol: 1e-12 rel


class TestR012NonFiniteWald:
    """R-012 (c): finite inputs, W overflows; RESTRICTION_NON_FINITE, no warning."""

    def test_overflowing_w_raises_non_finite_and_no_warning(self) -> None:
        # W = (1e200)**2 / 1e-200 = 1e600 exceeds the float64 range.
        coef = np.array([1e200])
        cov = np.array([[1e-200]])
        rest = np.array([[1.0]])
        rhs = np.array([0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an exception
            with pytest.raises(CoreRestrictionError) as exc_info:
                wald_core(coef, cov, 2, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_NON_FINITE)
        assert exc_info.value.shape == (1, 1)


class TestR012CovFinite:
    """R-012 (d): non-finite cov raises INPUT_NON_FINITE at the first flat index."""

    @pytest.mark.parametrize(
        ("entries", "first_flat_index"),
        [
            (((0, 1, float("nan")),), 1),
            (((1, 0, float("inf")),), 2),
            (((1, 1, float("-inf")),), 3),
            (((1, 1, float("inf")), (1, 0, float("nan"))), 2),
        ],
    )
    def test_cov_non_finite_raises(
        self, entries: tuple[tuple[int, int, float], ...], first_flat_index: int
    ) -> None:
        fit = _build_fit_iid(_X4, _Y4)
        cov_bad = fit.cov.copy()
        for row, col, value in entries:
            cov_bad[row, col] = value
        rest = np.eye(2)
        rhs = np.zeros(2)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(fit.coef, cov_bad, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_NON_FINITE)
        assert exc_info.value.operand == 1
        assert exc_info.value.positions == (first_flat_index,)


class TestR012CoefEmpty:
    """R-012 (e): an empty coef raises INPUT_EMPTY at operand 0."""

    def test_empty_coef_raises_input_empty(self) -> None:
        coef = np.empty(0, dtype=np.float64)
        cov = np.empty((0, 0), dtype=np.float64)
        rest = np.empty((1, 0), dtype=np.float64)
        rhs = np.zeros(1)
        with pytest.raises(CoreInputError) as exc_info:
            wald_core(coef, cov, 2, rest, rhs)
        assert exc_info.value.reason == int(codes.INPUT_EMPTY)
        assert exc_info.value.operand == 0
        assert exc_info.value.positions == ()


class TestR012RhsNdim:
    """R-012 (f): a rhs that is not 1-D raises RESTRICTION_WRONG_SHAPE."""

    @pytest.mark.parametrize("rhs_shape", [(2, 1), (2, 2), ()])
    def test_rhs_not_1d_raises_wrong_shape(self, rhs_shape: tuple[int, ...]) -> None:
        # The first axis of the 2-D cases equals J = 2, so only the ndim check
        # can reject them.  The 0-D case has no first axis.
        fit = _build_fit_iid(_X4, _Y4)
        rest = np.eye(2)
        rhs = np.zeros(rhs_shape)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(fit.coef, fit.cov, fit.dof, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_WRONG_SHAPE)
        assert exc_info.value.shape == (2, 2)


# ================================================================== R-013 tests

# tau = 1e-4 (wald._RANK_RATIO_TAU).  The ratio s_min / s_max of the normalised
# rows must lie above it.  The two ratios below sit 10% from tau on each side.
_RATIO_ABOVE_TAU = 1.1e-4
_RATIO_BELOW_TAU = 0.9e-4


class TestR013RankRuleBoundary:
    """R-013 test a: the accept and reject sides of tau = 1e-4."""

    @pytest.mark.parametrize("k", [2, 3])
    def test_ratio_above_tau_is_accepted(self, k: int) -> None:
        coef, cov = _synthetic_moments(k)
        rest = _two_row_restriction(_RATIO_ABOVE_TAU, k)
        rhs = np.array([0.0, 1.0])
        result = wald_core(coef, cov, 10, rest, rhs)
        assert result.j == 2
        assert math.isfinite(result.w)

    @pytest.mark.parametrize("k", [2, 3])
    def test_ratio_below_tau_raises_rank_deficient(self, k: int) -> None:
        coef, cov = _synthetic_moments(k)
        rest = _two_row_restriction(_RATIO_BELOW_TAU, k)
        rhs = np.array([0.0, 1.0])
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(coef, cov, 10, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_RANK_DEFICIENT)
        assert exc_info.value.shape == (2, k)


class TestR013Accuracy:
    """R-013 test b: W of a near-collinear restriction against the Fraction oracle."""

    def test_w_at_ratio_2e_minus_4_matches_fraction_oracle(self) -> None:
        """
        W equals the Fraction oracle to 1e-8 relative at ratio 2e-4.

        The relative error of W grows as about eps / ratio**2, which is 5.5e-9
        here.  The tolerance is 1e-8, not 1e-12.  The error depends on the data.
        This fixed case gives 5.9e-10 (sandbox evidence).  About 1% of random
        draws at this ratio exceed 1e-8 (sandbox evidence, 2000 draws).
        """
        coef, cov = _synthetic_moments(2)
        rest = _two_row_restriction(2e-4, 2)
        rhs = np.array([0.0, 1.0])
        result = wald_core(coef, cov, 10, rest, rhs)
        w_exact = _oracle_w(coef, cov, rest, rhs)
        assert abs(result.w - w_exact) / abs(w_exact) < 1e-8  # tol: 1e-8 rel


class TestR013RowScale:
    """R-013 test c: a row scale does not change W; row norms do not over/underflow."""

    @pytest.mark.parametrize("scale", [1e-12, 1e-170, 1e160])
    def test_scaled_row_is_accepted_and_w_is_unchanged(self, scale: float) -> None:
        """
        One row and its rhs entry are multiplied by ``scale``.

        W equals the Fraction W of the unscaled rows to 1e-12 relative.  The
        scale 1e-12 gives s_min / s_max of about 1e-12: the old absolute rule
        rejected it.  The scales 1e-170 and 1e160 break a plain sum of squares
        (zero norm, infinite norm).
        """
        coef, cov = _synthetic_moments(2)
        rest = np.array([[1.0, 0.5], [0.0, 1.0]])
        rhs = np.array([0.2, 1.0])
        rest_scaled = rest.copy()
        rest_scaled[1] *= scale
        rhs_scaled = rhs.copy()
        rhs_scaled[1] *= scale
        result = wald_core(coef, cov, 10, rest_scaled, rhs_scaled)
        w_exact = _oracle_w(coef, cov, rest, rhs)
        assert abs(result.w - w_exact) / abs(w_exact) < 1e-12  # tol: 1e-12 rel


class TestR013ZeroRowAndJAboveK:
    """R-013 test d: a zero row and J > K raise RESTRICTION_RANK_DEFICIENT."""

    @pytest.mark.parametrize(
        "restriction",
        [
            [[0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
    )
    def test_zero_row_raises_rank_deficient(
        self, restriction: list[list[float]]
    ) -> None:
        coef, cov = _synthetic_moments(2)
        rest = np.array(restriction)
        rhs = np.zeros(rest.shape[0])
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(coef, cov, 10, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_RANK_DEFICIENT)
        assert exc_info.value.shape == rest.shape

    @pytest.mark.parametrize("k", [2, 3])
    def test_j_above_k_raises_rank_deficient(self, k: int) -> None:
        # k + 1 well-conditioned rows: the ratio test alone would accept them.
        coef, cov = _synthetic_moments(k)
        rest = np.vstack([np.eye(k), np.ones((1, k))])
        rhs = np.zeros(k + 1)
        with pytest.raises(CoreRestrictionError) as exc_info:
            wald_core(coef, cov, 10, rest, rhs)
        assert exc_info.value.reason == int(codes.RESTRICTION_RANK_DEFICIENT)
        assert exc_info.value.shape == (k + 1, k)


class TestR013NoMutation:
    """The row normalisation works on copies (C.19)."""

    def test_caller_restriction_and_rhs_unchanged(self) -> None:
        coef, cov = _synthetic_moments(2)
        rest = np.array([[3e-7, 4e-7], [0.0, 2.0]])
        rhs = np.array([5e-7, 1.0])
        rest_before = rest.copy()
        rhs_before = rhs.copy()
        wald_core(coef, cov, 10, rest, rhs)
        np.testing.assert_array_equal(rest, rest_before)
        np.testing.assert_array_equal(rhs, rhs_before)
