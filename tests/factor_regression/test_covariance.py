# tests/factor_regression/test_covariance.py
"""
S4.T3 to S4.T6: IID, HC0 and HAC_BARTLETT covariance and the lag rules
(A.3, C.13). Owner decisions 3 and 6 of the S4 prompt.

Every oracle is independent of ``covariance.py``, ``fit.py`` and ``design.py``
(A.7): an exact ``fractions.Fraction`` OLS by Gaussian elimination, a Fraction
Gauss-Jordan inverse of ``X.T @ X``, the A.3 sandwich as explicit Fraction
loops over rows and lags on the unscaled design, ``decimal`` square roots at
60 digits, and the closed-form Student t tail for ``dof = 2``:
two-sided ``p = 1 - abs(t) / sqrt(2 + t**2)``. Tests call ``assemble_design``
and ``rank_gate`` only to build inputs.

Tolerances (owner decision 7):

* S4.T3: ``se`` and ``t_stats`` with ``max abs error <= 1e-12 * max abs
  exact``; p-values 1e-12 relative per element against the closed form at
  the core ``t``.
* S4.T4 and S4.T5: ``max abs error <= 1e-12 * max abs exact`` for ``cov``.

Sandbox evidence (Python 3.12.3, numpy 2.4.4, scipy 1.17.1): on the 5-row case
the closed-form p-values match ``scipy.stats.t.sf`` to 3.2e-16 relative for
IID, HC0 and HAC lags 1 and 4.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes, covariance, fit
from riskquant.factor_regression.covariance import (
    check_covariance_parameters,
    coefficient_covariance,
)
from riskquant.factor_regression.design import (
    CoreScaledSvd,
    assemble_design,
    rank_gate,
)
from riskquant.factor_regression.errors import CoreInputError, CoreParameterError
from riskquant.factor_regression.fit import CoreFit, fit_core

IID = int(codes.IID)
HC0 = int(codes.HC0)
HAC = int(codes.HAC_BARTLETT)
TOLERANCE = Fraction(1, 10**12)

# The 5-row case of S4.T1 and S4.T3: intercept and two regressors, dof = 2.
X1 = [0.5, 1.25, 2.0, 3.5, 4.75]
X2 = [1.1, 0.3, 2.7, 1.9, 0.6]
Y5 = [1.0, 2.3, 2.9, 4.8, 5.1]

# A 9-row case with an intercept and one regressor, for lags up to R - 1.
X9 = [0.3, -1.2, 0.8, 2.1, -0.4, 1.7, -2.3, 0.9, 1.1]
Y9 = [1.4, -0.7, 1.9, 2.2, 0.1, 2.9, -1.6, 0.8, 2.5]


# ------------------------------------------------------------------ oracles


Matrix = list[list[Fraction]]


def _rows(columns: list[list[float]], *, intercept: bool) -> Matrix:
    """Exact design rows in the C.8 order, intercept first."""
    n_rows = len(columns[0])
    lead = [Fraction(1)] if intercept else []
    return [lead + [Fraction(c[t]) for c in columns] for t in range(n_rows)]


def _gram(rows: Matrix) -> Matrix:
    k = len(rows[0])
    return [
        [sum((r[i] * r[j] for r in rows), Fraction(0)) for j in range(k)]
        for i in range(k)
    ]


def _inverse(matrix: Matrix) -> Matrix:
    """Gauss-Jordan inverse in Fractions: exact."""
    k = len(matrix)
    work = [
        [*row, *(Fraction(int(i == j)) for j in range(k))]
        for i, row in enumerate(matrix)
    ]
    for col in range(k):
        pivot = next(r for r in range(col, k) if work[r][col] != 0)
        work[col], work[pivot] = work[pivot], work[col]
        lead = work[col][col]
        work[col] = [v / lead for v in work[col]]
        for r in range(k):
            if r != col and work[r][col] != 0:
                factor = work[r][col]
                work[r] = [
                    a - factor * b for a, b in zip(work[r], work[col], strict=True)
                ]
    return [row[k:] for row in work]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [
        [
            sum((a[i][t] * b[t][j] for t in range(len(b))), Fraction(0))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]


def _ols(rows: Matrix, y: list[float]) -> tuple[list[Fraction], list[Fraction]]:
    """Exact OLS: coefficients from the normal equations, then residuals."""
    exact_y = [Fraction(v) for v in y]
    k = len(rows[0])
    moments = [
        sum((r[i] * v for r, v in zip(rows, exact_y, strict=True)), Fraction(0))
        for i in range(k)
    ]
    inverse = _inverse(_gram(rows))
    coef = [
        sum((inverse[i][j] * moments[j] for j in range(k)), Fraction(0))
        for i in range(k)
    ]
    resid = [
        v - sum((r[j] * coef[j] for j in range(k)), Fraction(0))
        for r, v in zip(rows, exact_y, strict=True)
    ]
    return coef, resid


def _sandwich(rows: Matrix, resid: list[Fraction], lag: int) -> Matrix:
    """A.3 HC0 (lag 0) or HAC_BARTLETT: explicit loops over rows and lags."""
    k = len(rows[0])
    n_rows = len(rows)
    meat = [[Fraction(0)] * k for _ in range(k)]
    for t in range(n_rows):
        for a in range(k):
            for b in range(k):
                meat[a][b] += resid[t] * resid[t] * rows[t][a] * rows[t][b]
    for j in range(1, lag + 1):
        weight = 1 - Fraction(j, lag + 1)
        for t in range(j, n_rows):
            product = resid[t] * resid[t - j]
            for a in range(k):
                for b in range(k):
                    # G_j[a][b] + G_j.T[a][b]
                    meat[a][b] += (
                        weight
                        * product
                        * (rows[t][a] * rows[t - j][b] + rows[t - j][a] * rows[t][b])
                    )
    bread = _inverse(_gram(rows))
    return _matmul(_matmul(bread, meat), bread)


def _iid(rows: Matrix, resid: list[Fraction]) -> Matrix:
    dof = len(rows) - len(rows[0])
    scale = sum((e * e for e in resid), Fraction(0)) / dof
    return [[scale * v for v in row] for row in _inverse(_gram(rows))]


def _decimal_sqrt(value: Fraction) -> Decimal:
    with localcontext() as context:
        context.prec = 60
        return (Decimal(value.numerator) / Decimal(value.denominator)).sqrt()


def _max_abs_close(got: NDArray[np.float64], exact: list[Fraction]) -> bool:
    """Owner decision 7: max abs error <= 1e-12 * max abs exact, exactly."""
    errors = [
        abs(Fraction(float(g)) - e)
        for g, e in zip(got.ravel().tolist(), exact, strict=True)
    ]
    return max(errors) <= TOLERANCE * max(abs(e) for e in exact)


def _max_abs_close_decimal(got: NDArray[np.float64], exact: list[Decimal]) -> bool:
    with localcontext() as context:
        context.prec = 60
        errors = [
            abs(Decimal(float(g)) - e) for g, e in zip(got.tolist(), exact, strict=True)
        ]
        return max(errors) <= Decimal("1e-12") * max(abs(e) for e in exact)


def _flat(matrix: Matrix) -> list[Fraction]:
    return [v for row in matrix for v in row]


def _case(name: str) -> tuple[list[list[float]], list[float]]:
    return ([X1, X2], Y5) if name == "five" else ([X9], Y9)


def _fit(name: str, kind: int, lag: int) -> CoreFit:
    columns, y = _case(name)
    design = assemble_design(np.array(columns), intercept=True)
    return fit_core(
        design,
        np.array(y),
        intercept=True,
        rank_rtol=1e-10,
        covariance_kind=kind,
        covariance_lag=lag,
    )


def _no_svd(*args: object, **kwargs: object) -> object:
    raise AssertionError("numpy.linalg.svd must not run")


# ------------------------------------------------------------------ S4.T3


def test_iid_se_t_and_dof_two_p_values_match_the_oracles() -> None:
    result = _fit("five", IID, 0)
    assert result.dof == 2
    rows = _rows([X1, X2], intercept=True)
    coef, resid = _ols(rows, Y5)
    exact_cov = _iid(rows, resid)
    assert _max_abs_close(result.cov, _flat(exact_cov))

    se = [_decimal_sqrt(exact_cov[j][j]) for j in range(3)]
    assert _max_abs_close_decimal(result.se, se)
    with localcontext() as context:
        context.prec = 60
        t_exact = [
            Decimal(c.numerator) / Decimal(c.denominator) / s
            for c, s in zip(coef, se, strict=True)
        ]
    assert _max_abs_close_decimal(result.t_stats, t_exact)
    _assert_dof_two_tail(result)


def _assert_dof_two_tail(result: CoreFit) -> None:
    """p equals 1 - abs(t) / sqrt(2 + t**2) at the core t, 1e-12 relative."""
    assert result.dof == 2
    with localcontext() as context:
        context.prec = 60
        for t, p in zip(result.t_stats.tolist(), result.p_values.tolist(), strict=True):
            exact_t = Decimal(t)
            closed = 1 - abs(exact_t) / (2 + exact_t * exact_t).sqrt()
            assert abs(Decimal(p) - closed) <= Decimal("1e-12") * closed, (t, p)


@pytest.mark.parametrize(
    ("kind", "lag"), [(IID, 0), (HC0, 0), (HAC, 1), (HAC, 4)], ids=str
)
def test_every_kind_uses_the_two_sided_student_t_tail(kind: int, lag: int) -> None:
    # Owner decision 5: Student t with dof = R - k, two-sided, for every kind.
    _assert_dof_two_tail(_fit("five", kind, lag))


# ------------------------------------------------------------------ S4.T4


def test_hc0_equals_the_fraction_sandwich() -> None:
    result = _fit("five", HC0, 0)
    rows = _rows([X1, X2], intercept=True)
    _, resid = _ols(rows, Y5)
    assert _max_abs_close(result.cov, _flat(_sandwich(rows, resid, 0)))
    # se = sqrt(diag(cov)) bit for bit (A.3).
    assert np.array_equal(result.se, np.sqrt(np.diagonal(result.cov)))


# ------------------------------------------------------------------ S4.T5


@pytest.mark.parametrize("name", ["five", "nine"])
def test_hac_lag_zero_equals_hc0(name: str) -> None:
    hac = _fit(name, HAC, 0).cov
    hc0 = _fit(name, HC0, 0).cov
    assert float(np.max(np.abs(hac - hc0))) <= 1e-12 * float(np.max(np.abs(hc0)))


@pytest.mark.parametrize(
    ("name", "lag"),
    [("five", 1), ("five", 2), ("five", 4), ("nine", 1), ("nine", 3), ("nine", 8)],
)
def test_hac_equals_the_fraction_double_sum(name: str, lag: int) -> None:
    columns, y = _case(name)
    result = _fit(name, HAC, lag)
    rows = _rows(columns, intercept=True)
    _, resid = _ols(rows, y)
    assert _max_abs_close(result.cov, _flat(_sandwich(rows, resid, lag)))


def test_the_bartlett_weights_change_the_result_lag_by_lag() -> None:
    # Each added lag must enter: a weight of 0 or a skipped lag would leave
    # two consecutive lags equal.
    covs = [_fit("nine", HAC, lag).cov for lag in range(9)]
    for lower, upper in itertools.pairwise(covs):
        assert not np.array_equal(lower, upper)


# ------------------------------------------------------------------ S4.T6


@pytest.mark.parametrize(
    ("kind", "lag", "reason"),
    [
        pytest.param(HAC, 5, codes.PARAM_LAG_ABOVE_LIMIT, id="hac-lag-5-on-5-rows"),
        pytest.param(HAC, 99, codes.PARAM_LAG_ABOVE_LIMIT, id="hac-lag-99"),
        pytest.param(IID, 1, codes.PARAM_LAG_NOT_ALLOWED, id="iid-lag-1"),
        pytest.param(IID, -1, codes.PARAM_LAG_NOT_ALLOWED, id="iid-lag-minus-1"),
        pytest.param(HC0, 2, codes.PARAM_LAG_NOT_ALLOWED, id="hc0-lag-2"),
        pytest.param(HAC, -1, codes.PARAM_NEGATIVE_LAG, id="hac-negative-lag"),
    ],
)
def test_an_invalid_lag_raises_before_the_svd(
    kind: int, lag: int, reason: np.int8, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(np.linalg, "svd", _no_svd)
    with pytest.raises(CoreParameterError) as info:
        _fit("five", kind, lag)
    assert info.value.reason == reason
    assert info.value.value == float(lag)


def test_the_largest_hac_lag_is_r_minus_one() -> None:
    assert _fit("five", HAC, 4).cov.shape == (3, 3)
    assert check_covariance_parameters(
        covariance_kind=HAC, covariance_lag=4, n_rows=5
    ) == (HAC, 4)


# ------------------------------------------ owner decision 3: parameters


@pytest.mark.parametrize("kind", [3, -1, 127])
def test_an_unknown_covariance_kind_raises_before_the_svd(
    kind: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(np.linalg, "svd", _no_svd)
    with pytest.raises(CoreParameterError) as info:
        _fit("five", kind, 0)
    assert info.value.reason == codes.PARAM_UNKNOWN_COVARIANCE
    assert info.value.value == float(kind)


@pytest.mark.parametrize("name", ["covariance_kind", "covariance_lag"])
@pytest.mark.parametrize(
    "bad", [True, np.True_, False, 1.0, np.float64(0.0), "0", None]
)
def test_a_kind_or_lag_of_the_wrong_type_raises_type_error(
    name: str, bad: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(np.linalg, "svd", _no_svd)
    arguments: dict[str, Any] = {"covariance_kind": HC0, "covariance_lag": 0}
    arguments[name] = bad
    design = assemble_design(np.array([X1, X2]), intercept=True)
    with pytest.raises(TypeError):
        fit_core(design, np.array(Y5), intercept=True, rank_rtol=1e-10, **arguments)
    with pytest.raises(TypeError):
        check_covariance_parameters(n_rows=5, **arguments)


def test_numpy_integer_kinds_and_lags_give_the_int_result() -> None:
    expected = _fit("five", HAC, 2).cov
    design = assemble_design(np.array([X1, X2]), intercept=True)
    kind: Any = codes.HAC_BARTLETT
    lag: Any = np.int64(2)
    got = fit_core(
        design,
        np.array(Y5),
        intercept=True,
        rank_rtol=1e-10,
        covariance_kind=kind,
        covariance_lag=lag,
    ).cov
    assert np.array_equal(got, expected)
    iid: Any = codes.IID
    zero: Any = np.int32(0)
    assert check_covariance_parameters(
        covariance_kind=iid, covariance_lag=zero, n_rows=5
    ) == (IID, 0)


@pytest.mark.parametrize(
    "function", [fit_core, check_covariance_parameters, coefficient_covariance]
)
def test_no_parameter_has_a_default(function: Any) -> None:
    for parameter in inspect.signature(function).parameters.values():
        assert parameter.default is inspect.Parameter.empty, parameter.name


# --------------------------------------------- coefficient_covariance input


def _gate_and_resid() -> tuple[CoreScaledSvd, NDArray[np.float64]]:
    result = _fit("nine", HC0, 0)
    return result.gate, np.array(result.resid)


def _resid_error(resid: object) -> CoreInputError:
    gate, _ = _gate_and_resid()
    with pytest.raises(CoreInputError) as info:
        coefficient_covariance(gate, resid, covariance_kind=HC0, covariance_lag=0)  # type: ignore[arg-type]
    return info.value


class _TrivialSubclass(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """An ndarray subclass with no override: only the exact type differs."""


def test_masked_residuals_with_nan_under_the_mask_are_rejected() -> None:
    _, resid = _gate_and_resid()
    resid[3] = math.nan
    mask = np.zeros(resid.shape[0], dtype=np.bool_)
    mask[3] = True
    masked: NDArray[Any] = np.ma.masked_array(resid, mask=mask)
    assert bool(np.isfinite(masked).all())
    error = _resid_error(masked)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        0,
        (),
    )


def test_the_residual_contract_rejects_without_repair() -> None:
    _, resid = _gate_and_resid()
    cases: list[tuple[object, int]] = [
        (resid.tolist(), int(codes.INPUT_NOT_NDARRAY)),
        (resid.view(_TrivialSubclass), int(codes.INPUT_NOT_NDARRAY)),
        (resid.astype(np.float32), int(codes.INPUT_WRONG_DTYPE)),
        (resid.reshape(-1, 1), int(codes.INPUT_WRONG_NDIM)),
        (resid[:0].copy(), int(codes.INPUT_EMPTY)),
        (resid[:-1].copy(), int(codes.INPUT_WRONG_SHAPE)),
    ]
    for array, reason in cases:
        error = _resid_error(array)
        assert (error.reason, error.operand, error.positions) == (reason, 0, ())
    for bad in (math.nan, math.inf, -math.inf):
        broken = resid.copy()
        broken[6] = bad
        broken[8] = bad
        error = _resid_error(broken)
        assert (error.reason, error.positions) == (int(codes.INPUT_NON_FINITE), (6,))


# ------------------------------------------------- result and scaling


@pytest.mark.parametrize(("kind", "lag"), [(IID, 0), (HC0, 0), (HAC, 3)], ids=str)
def test_the_covariance_is_new_read_only_and_exactly_symmetric(
    kind: int, lag: int
) -> None:
    gate, resid = _gate_and_resid()
    before = resid.copy()
    cov = coefficient_covariance(gate, resid, covariance_kind=kind, covariance_lag=lag)
    assert cov.dtype == np.float64
    assert cov.shape == (2, 2)
    assert cov.flags.c_contiguous
    assert not cov.flags.writeable
    assert np.array_equal(cov, cov.T)
    assert np.array_equal(resid, before)
    assert resid.flags.writeable
    assert not np.shares_memory(cov, resid)


@pytest.mark.parametrize(("kind", "lag"), [(IID, 0), (HC0, 0), (HAC, 3)], ids=str)
@pytest.mark.parametrize("shift", [-560, 520])
def test_extreme_magnitudes_give_the_covariance_of_moderate_ones(
    kind: int, lag: int, shift: int
) -> None:
    # Residuals and columns scaled by the same power of two: cov is unchanged.
    # Unscaled, e**2 underflows to 0 near 1e-169 and overflows near 1e156.
    columns, y = _case("nine")
    base = fit_core(
        assemble_design(np.array(columns), intercept=False),
        np.array(y),
        intercept=False,
        rank_rtol=1e-10,
        covariance_kind=kind,
        covariance_lag=lag,
    )
    resid = np.ldexp(np.array(base.resid), shift)
    with np.errstate(over="ignore", under="ignore"):
        assert not math.isfinite(float(np.dot(resid, resid))) or (
            float(np.dot(resid, resid)) == 0.0
        )
    design = np.ldexp(np.array(base.gate.design), shift)
    cov = coefficient_covariance(
        rank_gate(design, rank_rtol=1e-10),
        resid,
        covariance_kind=kind,
        covariance_lag=lag,
    )
    assert np.array_equal(cov, base.cov)


@pytest.mark.parametrize(("kind", "lag"), [(IID, 0), (HC0, 0), (HAC, 3)], ids=str)
@pytest.mark.parametrize("shift", [-40, 3, 40])
def test_residuals_times_a_power_of_two_scale_the_covariance_exactly(
    kind: int, lag: int, shift: int
) -> None:
    gate, resid = _gate_and_resid()
    base = coefficient_covariance(gate, resid, covariance_kind=kind, covariance_lag=lag)
    scaled = coefficient_covariance(
        gate, np.ldexp(resid, shift), covariance_kind=kind, covariance_lag=lag
    )
    assert np.array_equal(scaled, np.ldexp(base, 2 * shift))


# ------------------------------------- owner decision 6: lags, not rows


def _tree(module: Any) -> ast.Module:
    return ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))


def _comprehension_iterables(tree: ast.Module) -> set[str]:
    return {
        ast.unparse(node.iter)
        for node in ast.walk(tree)
        if isinstance(node, ast.comprehension)
    }


def test_the_only_loop_of_the_covariance_runs_over_the_lags() -> None:
    tree = _tree(covariance)
    loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
    assert len(loops) == 1
    loop = loops[0]
    assert isinstance(loop, ast.For)
    assert ast.unparse(loop.iter) == "range(1, lag + 1)"
    # The one comprehension reads the three covariance codes, not the rows.
    assert _comprehension_iterables(tree) == {"codes.COVARIANCE_CODES"}


def test_the_fit_has_no_loop_over_rows() -> None:
    tree = _tree(fit)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
    # The one comprehension reads the affected column indices (at most k).
    assert _comprehension_iterables(tree) == {"columns"}
