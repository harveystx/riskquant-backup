# tests/factor_regression/test_fit.py
"""
S4.T1, S4.T2, S4.T7, S4.T8 and S4.T13: the core OLS fit on the scaled design
(A.3, C.12, C.14, C.19). Owner decisions 1, 2, 4 and 5 of the S4 prompt.

Every oracle is independent of ``fit.py``, ``covariance.py`` and ``design.py``
(A.7): an exact ``fractions.Fraction`` OLS (the closed form
``b = (n*Sxy - Sx*Sy) / (n*Sxx - Sx**2)``, ``a = (Sy - b*Sx) / n`` for one
regressor, Fraction Gaussian elimination for two), Fraction sums for
``r_squared``, the C.14 bound and the Durbin-Watson statistic. Tests call
``assemble_design`` only to build inputs.

Tolerances (owner decision 7):

* S4.T1: ``coef``, ``resid`` and ``fitted`` with ``max abs error <= 1e-12 *
  max abs exact`` per array; ``rss`` 1e-12 relative; ``dof`` exact.
* S4.T2 (C.12): slope 1e-8 relative; centred intercept ``a + b * mean(x)``
  1e-12 relative, with ``mean(x)`` as a Fraction; fitted values 1e-12
  relative per element. The exact fitted value is rounded once to float64
  (1.1e-16 relative) before the comparison.
* ``r_squared``: 1e-12 relative. Durbin-Watson: 1e-14 relative, as in
  ``test_diagnostics.py``.

Sandbox evidence (Python 3.12.3, numpy 2.4.4), S4.T2 over seeds 1 to 20:
slope 7.0e-10 relative at 250 rows and 5.2e-10 at 2500 rows; centred
intercept 1.9e-15; fitted values 2.1e-15. S4.T7 over seeds 0 to 49: the exact
fits give ``rss / bound`` of 1.1e-2 or less; the ``1e-8`` noise fits give
``rss / bound`` of 2.2e7 or more.
"""

from __future__ import annotations

import dataclasses
import io
import math
import tokenize
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes, covariance, diagnostics, fit
from riskquant.factor_regression.covariance import coefficient_covariance
from riskquant.factor_regression.design import (
    CoreScaledSvd,
    assemble_design,
    rank_gate,
)
from riskquant.factor_regression.errors import (
    CoreDegenerateFitError,
    CoreInputError,
    CoreParameterError,
    CoreSampleSizeError,
)
from riskquant.factor_regression.fit import (
    OPERAND_DESIGN,
    OPERAND_RESPONSE,
    CoreFit,
    fit_core,
)

IID = int(codes.IID)
HC0 = int(codes.HC0)
HAC = int(codes.HAC_BARTLETT)
EPS = float(np.finfo(np.float64).eps)
TOLERANCE = Fraction(1, 10**12)
KINDS = [(IID, 0), (HC0, 0), (HAC, 2)]

# The 5-row case of S4.T1: intercept and two regressors, dof = 2.
X1 = [0.5, 1.25, 2.0, 3.5, 4.75]
X2 = [1.1, 0.3, 2.7, 1.9, 0.6]
Y5 = [1.0, 2.3, 2.9, 4.8, 5.1]


# ------------------------------------------------------------------ helpers


def _run(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    intercept: bool = True,
    kind: int = IID,
    lag: int = 0,
    rank_rtol: float = 1e-10,
) -> CoreFit:
    return fit_core(
        design,
        y,
        intercept=intercept,
        rank_rtol=rank_rtol,
        covariance_kind=kind,
        covariance_lag=lag,
    )


def _five() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return assemble_design(np.array([X1, X2]), intercept=True), np.array(Y5)


def _degenerate(
    design: NDArray[np.float64], y: NDArray[np.float64], **options: Any
) -> CoreDegenerateFitError:
    with pytest.raises(CoreDegenerateFitError) as info:
        _run(design, y, **options)
    return info.value


def _input_error(design: object, y: object) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        _run(design, y)  # type: ignore[arg-type]
    return info.value


def _no_svd(*args: object, **kwargs: object) -> object:
    raise AssertionError("numpy.linalg.svd must not run")


def _exact_ols(
    columns: list[list[float]], y: list[float], *, intercept: bool
) -> tuple[list[Fraction], list[Fraction]]:
    """Exact OLS by Fraction Gaussian elimination of the normal equations."""
    n_rows = len(y)
    lead = [Fraction(1)] if intercept else []
    rows = [lead + [Fraction(c[t]) for c in columns] for t in range(n_rows)]
    exact_y = [Fraction(v) for v in y]
    k = len(rows[0])
    system = [
        [sum((r[i] * r[j] for r in rows), Fraction(0)) for j in range(k)]
        + [sum((r[i] * v for r, v in zip(rows, exact_y, strict=True)), Fraction(0))]
        for i in range(k)
    ]
    for col in range(k):
        pivot = next(r for r in range(col, k) if system[r][col] != 0)
        system[col], system[pivot] = system[pivot], system[col]
        for r in range(col + 1, k):
            factor = system[r][col] / system[col][col]
            system[r] = [
                a - factor * b for a, b in zip(system[r], system[col], strict=True)
            ]
    coef = [Fraction(0)] * k
    for i in reversed(range(k)):
        tail = sum((system[i][j] * coef[j] for j in range(i + 1, k)), Fraction(0))
        coef[i] = (system[i][k] - tail) / system[i][i]
    resid = [
        v - sum((r[j] * coef[j] for j in range(k)), Fraction(0))
        for r, v in zip(rows, exact_y, strict=True)
    ]
    return coef, resid


def _max_abs_close(got: NDArray[np.float64], exact: list[Fraction]) -> bool:
    """Owner decision 7: max abs error <= 1e-12 * max abs exact, exactly."""
    errors = [abs(Fraction(g) - e) for g, e in zip(got.tolist(), exact, strict=True)]
    return max(errors) <= TOLERANCE * max(abs(e) for e in exact)


def _fraction_durbin_watson(resid: list[float]) -> Fraction:
    exact = [Fraction(e) for e in resid]
    numerator = sum(
        ((exact[t] - exact[t - 1]) ** 2 for t in range(1, len(exact))), Fraction(0)
    )
    return numerator / sum((e * e for e in exact), Fraction(0))


def _arrays(result: CoreFit) -> list[NDArray[np.float64]]:
    return [
        result.coef,
        result.se,
        result.t_stats,
        result.p_values,
        result.cov,
        result.resid,
        result.fitted,
    ]


def _assert_finite(result: CoreFit) -> None:
    for array in _arrays(result):
        assert bool(np.isfinite(array).all())
    for value in (result.rss, result.r_squared, result.durbin_watson):
        assert math.isfinite(value)
    assert result.rss > 0.0
    assert bool((np.diagonal(result.cov) > 0.0).all())


# ------------------------------------------------------------------ S4.T1


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
def test_the_five_row_fit_equals_the_fraction_solution(kind: int, lag: int) -> None:
    design, y = _five()
    result = _run(design, y, kind=kind, lag=lag)
    coef, resid = _exact_ols([X1, X2], Y5, intercept=True)
    fitted = [Fraction(v) - e for v, e in zip(Y5, resid, strict=True)]
    rss = sum((e * e for e in resid), Fraction(0))

    assert _max_abs_close(result.coef, coef)
    assert _max_abs_close(result.resid, resid)
    assert _max_abs_close(result.fitted, fitted)
    assert abs(Fraction(result.rss) - rss) <= TOLERANCE * rss
    assert (result.dof, result.n_obs) == (2, 5)
    assert type(result.dof) is int
    assert type(result.n_obs) is int


# ------------------------------------------------------------------ S4.T2


@pytest.mark.parametrize("n_rows", [250, 2500])
@pytest.mark.parametrize("seed", range(1, 21))
def test_a_mean_to_sd_ratio_of_1e7_keeps_slope_centred_intercept_and_fitted(
    seed: int, n_rows: int
) -> None:
    rng = np.random.default_rng(seed)
    x = 1e7 + rng.standard_normal(n_rows)
    y = 3.0 + 0.5 * x + 0.1 * rng.standard_normal(n_rows)
    result = _run(assemble_design(x[np.newaxis, :].copy(), intercept=True), y)
    assert result.gate.rank == 2

    # A.7 closed form, exact.
    xs = [Fraction(v) for v in x.tolist()]
    ys = [Fraction(v) for v in y.tolist()]
    sx = sum(xs, Fraction(0))
    sy = sum(ys, Fraction(0))
    sxx = sum((v * v for v in xs), Fraction(0))
    sxy = sum((a * b for a, b in zip(xs, ys, strict=True)), Fraction(0))
    slope = (n_rows * sxy - sx * sy) / (n_rows * sxx - sx * sx)
    intercept = (sy - slope * sx) / n_rows
    mean_x = sx / n_rows

    core_intercept = Fraction(float(result.coef[0]))
    core_slope = Fraction(float(result.coef[1]))
    assert abs(core_slope - slope) <= Fraction(1, 10**8) * abs(slope)
    centred = intercept + slope * mean_x
    assert abs(core_intercept + core_slope * mean_x - centred) <= TOLERANCE * abs(
        centred
    )
    exact_fitted = np.array([float(intercept + slope * v) for v in xs])
    relative = np.abs(result.fitted - exact_fitted) / np.abs(exact_fitted)
    assert float(np.max(relative)) <= 1e-12


# ------------------------------------------------------------------ S4.T7


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
def test_an_exact_fit_on_small_integers_raises(kind: int, lag: int) -> None:
    x = np.arange(1.0, 9.0)
    y = 1.0 + 2.0 * x
    error = _degenerate(
        assemble_design(x[np.newaxis, :].copy(), intercept=True), y, kind=kind, lag=lag
    )
    assert error.reason == codes.DEGENERATE_EXACT_FIT
    assert error.columns == ()
    assert 0.0 <= error.value <= error.bound
    # The bound is (R * eps)**2 * sum(y**2) (C.14).
    exact_bound = Fraction(8 * EPS) ** 2 * sum(
        (Fraction(v) ** 2 for v in y.tolist()), Fraction(0)
    )
    assert abs(Fraction(error.bound) - exact_bound) <= TOLERANCE * exact_bound


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_rows", [30, 250, 2500])
def test_an_exact_fit_on_non_integer_x_raises_whatever_the_rounding(
    n_rows: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_rows) * 3.7 + 0.3
    y = 1.0 + 2.0 * x
    error = _degenerate(assemble_design(x[np.newaxis, :].copy(), intercept=True), y)
    assert error.reason == codes.DEGENERATE_EXACT_FIT
    # rss is 0.0 when the solve returns (1.0, 2.0) exactly, and a value near
    # 1e-30 otherwise (sandbox: both occur). The rule fires in both cases.
    assert 0.0 <= error.value <= error.bound


def test_an_exact_fit_with_rss_above_zero_raises() -> None:
    # A 16-ulp change of one response puts y off every line, so the residual
    # of that row is not zero; rss stays far below the bound.
    rng = np.random.default_rng(1)
    x = rng.standard_normal(30) * 3.7 + 0.3
    y = 1.0 + 2.0 * x
    y[3] += 16.0 * float(np.spacing(y[3]))
    error = _degenerate(assemble_design(x[np.newaxis, :].copy(), intercept=True), y)
    assert error.reason == codes.DEGENERATE_EXACT_FIT
    assert 0.0 < error.value <= error.bound


def test_an_exact_fit_with_rss_exactly_zero_raises() -> None:
    # One unit column: u, s and vh are exact, so the residual is exactly 0.0.
    design = np.array([[1.0], [0.0], [0.0]])
    error = _degenerate(design, np.array([5.0, 0.0, 0.0]), intercept=False)
    assert error.reason == codes.DEGENERATE_EXACT_FIT
    assert error.value == 0.0


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
@pytest.mark.parametrize("n_rows", [30, 250, 2500])
def test_a_fit_with_1e_8_noise_returns_finite_values(
    n_rows: int, kind: int, lag: int
) -> None:
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n_rows) * 3.7 + 0.3
        y = x + 1e-8 * rng.standard_normal(n_rows)
        design = assemble_design(x[np.newaxis, :].copy(), intercept=True)
        _assert_finite(_run(design, y, kind=kind, lag=lag))


# ------------------------------------------------------------------ S4.T8


def test_an_all_zero_response_with_an_intercept_only_raises() -> None:
    design = assemble_design(np.empty((0, 6), dtype=np.float64), intercept=True)
    error = _degenerate(design, np.zeros(6))
    assert error.reason == codes.DEGENERATE_ZERO_RESPONSE_VARIANCE
    assert (error.value, error.bound, error.columns) == (0.0, 0.0, ())


def test_an_all_zero_response_without_an_intercept_is_an_exact_fit() -> None:
    rng = np.random.default_rng(0)
    design = assemble_design(rng.standard_normal((2, 6)), intercept=False)
    error = _degenerate(design, np.zeros(6), intercept=False)
    assert error.reason == codes.DEGENERATE_EXACT_FIT
    assert (error.value, error.bound) == (0.0, 0.0)


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
def test_the_result_arrays_are_new_read_only_and_typed(kind: int, lag: int) -> None:
    design, y = _five()
    result = _run(design, y, kind=kind, lag=lag)
    shapes = [(3,), (3,), (3,), (3,), (3, 3), (5,), (5,)]
    for array, shape in zip(_arrays(result), shapes, strict=True):
        assert array.dtype == np.float64
        assert array.shape == shape
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        assert not np.shares_memory(array, design)
        assert not np.shares_memory(array, y)
    assert np.array_equal(result.cov, result.cov.T)
    for value in (result.rss, result.r_squared, result.durbin_watson):
        assert type(value) is float
    _assert_finite(result)


def test_the_result_has_no_approval_field_and_is_frozen() -> None:
    names = [field.name for field in dataclasses.fields(CoreFit)]
    for word in ("accepted", "approved", "valid"):
        assert not [name for name in names if word in name]
    design, y = _five()
    result = _run(design, y)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.rss = 0.0  # type: ignore[misc]


def test_caller_mutation_after_the_fit_leaves_the_result_unchanged() -> None:
    design, y = _five()
    writable = np.array(design)
    result = _run(writable, y, kind=HC0)
    snapshot = [array.copy() for array in [*_arrays(result), result.gate.design]]
    writable[:, 1] = 99.0
    y[:] = -7.0
    after = [*_arrays(result), result.gate.design]
    for before, now in zip(snapshot, after, strict=True):
        assert np.array_equal(before, now)


def test_the_fit_changes_no_caller_array() -> None:
    design, y = _five()
    writable = np.asfortranarray(design)
    design_before = writable.copy()
    y_before = y.copy()
    result = _run(writable, y, kind=HAC, lag=2)
    assert np.array_equal(writable, design_before)
    assert np.array_equal(y, y_before)
    assert writable.flags.writeable
    assert y.flags.writeable
    # Any memory layout gives the same fit: the gate copies to C order.
    c_order = _run(np.ascontiguousarray(design), y, kind=HAC, lag=2)
    for a, b in zip(_arrays(result), _arrays(c_order), strict=True):
        assert np.array_equal(a, b)


# ----------------------------------------------------------------- S4.T13


def test_two_rows_and_two_columns_raise_sample_size_before_the_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = assemble_design(np.array([[0.5, 1.5]]), intercept=True)
    assert design.shape == (2, 2)
    monkeypatch.setattr(np.linalg, "svd", _no_svd)
    with pytest.raises(CoreSampleSizeError) as info:
        _run(design, np.array([1.0, 2.0]))
    assert info.value.reason == codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS
    # n_available is R; n_required is k + 1.
    assert (info.value.n_available, info.value.n_required) == (2, 3)


# ----------------------------------- owner decision 1: entry and result


def test_the_result_holds_exactly_the_declared_fields() -> None:
    assert [field.name for field in dataclasses.fields(CoreFit)] == [
        "gate",
        "coef",
        "se",
        "t_stats",
        "p_values",
        "cov",
        "resid",
        "fitted",
        "rss",
        "r_squared",
        "dof",
        "n_obs",
        "durbin_watson",
    ]


@pytest.mark.parametrize("intercept", [1, 0, None, "yes"])
def test_a_non_bool_intercept_raises_type_error(intercept: object) -> None:
    design, y = _five()
    with pytest.raises(TypeError):
        _run(design, y, intercept=intercept)  # type: ignore[arg-type]


def test_a_numpy_bool_intercept_gives_the_bool_result() -> None:
    design, y = _five()
    flag: Any = np.True_
    a = _run(design, y, intercept=True)
    b = _run(design, y, intercept=flag)
    assert a.r_squared == b.r_squared


def test_the_fit_runs_the_gate_once_and_one_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"svd": 0, "gate": 0}
    real_svd = np.linalg.svd
    real_gate = rank_gate

    def counting_svd(*args: Any, **kwargs: Any) -> Any:
        calls["svd"] += 1
        return real_svd(*args, **kwargs)

    def counting_gate(*args: Any, **kwargs: Any) -> CoreScaledSvd:
        calls["gate"] += 1
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counting_svd)
    monkeypatch.setattr(fit, "rank_gate", counting_gate)
    design, y = _five()
    _run(design, y, kind=HAC, lag=1)
    assert calls == {"svd": 1, "gate": 1}


@pytest.mark.parametrize("name", ["lstsq", "inv", "pinv", "solve", "tensorinv"])
def test_the_fit_uses_no_least_squares_routine_and_no_inverse(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"numpy.linalg.{name} must not run")

    monkeypatch.setattr(np.linalg, name, forbidden)
    design, y = _five()
    for kind, lag in KINDS:
        _assert_finite(_run(design, y, kind=kind, lag=lag))


def _executable_names(module: Any) -> set[str]:
    source = Path(module.__file__).read_text(encoding="utf-8")
    return {
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME
    }


@pytest.mark.parametrize("module", [fit, covariance])
def test_the_source_names_no_solver_inverse_or_digest(module: Any) -> None:
    # S4 Prohibited: lstsq, unscaled solves, any digest in the core.
    forbidden = {"svd", "lstsq", "inv", "pinv", "solve", "hashlib", "digest"}
    assert not _executable_names(module) & forbidden


def test_the_solve_is_the_a3_formula_on_the_gate_arrays() -> None:
    design, y = _five()
    result = _run(design, y)
    gate = result.gate
    gamma = gate.vh.T @ ((gate.u.T @ y) / gate.singular_values)
    assert np.array_equal(result.coef, gamma / gate.column_norms)
    # e = y - X @ coef with the design copy of the gate (A.3).
    assert np.array_equal(result.fitted, gate.design @ result.coef)
    assert np.array_equal(result.resid, y - result.fitted)
    assert np.array_equal(gate.design, design)
    assert not np.shares_memory(gate.design, design)
    assert np.array_equal(result.se, np.sqrt(np.diagonal(result.cov)))
    assert np.array_equal(result.t_stats, result.coef / result.se)


# ------------------------------------ owner decision 2: response checks


class _TrivialSubclass(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """An ndarray subclass with no override: only the exact type differs."""


def test_a_masked_response_with_nan_under_the_mask_is_rejected() -> None:
    design, y = _five()
    y[2] = math.nan
    masked: NDArray[Any] = np.ma.masked_array(
        y, mask=[False, False, True, False, False]
    )
    # The mask hides the NaN from a reduction.
    assert bool(np.isfinite(masked).all())
    error = _input_error(design, masked)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_RESPONSE,
        (),
    )


def test_a_masked_design_with_nan_under_the_mask_is_rejected() -> None:
    design, y = _five()
    raw = np.array(design)
    raw[3, 1] = math.nan
    hidden = np.zeros(raw.shape, dtype=np.bool_)
    hidden[3, 1] = True
    masked: NDArray[Any] = np.ma.masked_array(raw, mask=hidden)
    error = _input_error(masked, y)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_DESIGN,
        (),
    )


def test_the_response_contract_rejects_without_repair() -> None:
    design, y = _five()
    cases: list[tuple[object, int]] = [
        (y.tolist(), int(codes.INPUT_NOT_NDARRAY)),
        (y.view(_TrivialSubclass), int(codes.INPUT_NOT_NDARRAY)),
        (y.astype(np.float32), int(codes.INPUT_WRONG_DTYPE)),
        (y.astype(np.int64), int(codes.INPUT_WRONG_DTYPE)),
        (y.reshape(5, 1), int(codes.INPUT_WRONG_NDIM)),
        (y[:0].copy(), int(codes.INPUT_EMPTY)),
        (y[:4].copy(), int(codes.INPUT_WRONG_SHAPE)),
        (np.append(y, 1.0), int(codes.INPUT_WRONG_SHAPE)),
    ]
    for response, reason in cases:
        error = _input_error(design, response)
        assert (error.reason, error.operand, error.positions) == (
            reason,
            OPERAND_RESPONSE,
            (),
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_response_raises_with_its_index(bad: float) -> None:
    design, y = _five()
    y[1] = bad
    y[3] = bad
    error = _input_error(design, y)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        OPERAND_RESPONSE,
        (1,),
    )


def test_the_design_contract_rejects_without_repair() -> None:
    design, y = _five()
    cases: list[tuple[object, int]] = [
        (design.tolist(), int(codes.INPUT_NOT_NDARRAY)),
        (design.view(_TrivialSubclass), int(codes.INPUT_NOT_NDARRAY)),
        (design.astype(np.float32), int(codes.INPUT_WRONG_DTYPE)),
        (design[:, 0].copy(), int(codes.INPUT_WRONG_NDIM)),
        (np.empty((0, 3), dtype=np.float64), int(codes.INPUT_EMPTY)),
        (np.empty((5, 0), dtype=np.float64), int(codes.INPUT_EMPTY)),
    ]
    for array, reason in cases:
        error = _input_error(array, y)
        assert (error.reason, error.operand, error.positions) == (
            reason,
            OPERAND_DESIGN,
            (),
        )
    broken = np.array(design)
    broken[4, 0] = math.nan
    broken[2, 2] = math.inf
    error = _input_error(broken, y)
    # Flat C-order index of the first non-finite element, (2, 2).
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        OPERAND_DESIGN,
        (2 * 3 + 2,),
    )


def test_the_checks_run_in_the_documented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design, y = _five()
    nan_design = np.array(design)
    nan_design[0, 1] = math.nan
    nan_y = y.copy()
    nan_y[0] = math.nan
    # 1 before 2: the design is checked before the response.
    assert _input_error(design.astype(np.float32), y.tolist()).operand == OPERAND_DESIGN
    assert _input_error(nan_design, nan_y).operand == OPERAND_DESIGN
    monkeypatch.setattr(np.linalg, "svd", _no_svd)
    # 2 before 3: the response before the intercept type.
    with pytest.raises(CoreInputError):
        _run(design, nan_y, intercept=1)  # type: ignore[arg-type]
    # 3 before 4: the intercept type before the covariance parameters.
    with pytest.raises(TypeError):
        _run(design, y, intercept=1, kind=99)  # type: ignore[arg-type]
    # 4 before 5: the covariance parameters before rank_rtol and sample size.
    with pytest.raises(CoreParameterError) as info:
        _run(design[:3], y[:3], kind=99, rank_rtol=2.0)
    assert info.value.reason == codes.PARAM_UNKNOWN_COVARIANCE
    # 5: rank_rtol before the sample size, both before the SVD.
    with pytest.raises(CoreParameterError) as info:
        _run(design[:3], y[:3], rank_rtol=2.0)
    assert info.value.reason == codes.PARAM_RANK_RTOL_OUT_OF_RANGE


# --------------------------------------- owner decision 4: degenerate fits


def test_a_constant_response_with_an_intercept_raises_zero_variance() -> None:
    rng = np.random.default_rng(3)
    design = assemble_design(rng.standard_normal((1, 7)), intercept=True)
    y = np.full(7, 0.1)
    # The float64 A.3 expression is not zero here: mean(y) rounds.
    assert float(np.sum((y - np.mean(y)) ** 2)) != 0.0
    # A constant y is also an exact fit with an intercept: zero variance
    # comes first.
    for kind, lag in KINDS:
        error = _degenerate(design, y, kind=kind, lag=lag)
        assert error.reason == codes.DEGENERATE_ZERO_RESPONSE_VARIANCE
        assert (error.value, error.bound, error.columns) == (0.0, 0.0, ())


def test_a_constant_response_without_an_intercept_fits() -> None:
    rng = np.random.default_rng(3)
    design = assemble_design(rng.standard_normal((1, 7)), intercept=False)
    _assert_finite(_run(design, np.full(7, 0.1), intercept=False))


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
def test_an_exact_fit_raises_before_the_covariance_check(kind: int, lag: int) -> None:
    x = np.arange(1.0, 9.0)
    design = assemble_design(x[np.newaxis, :].copy(), intercept=True)
    error = _degenerate(design, 1.0 + 2.0 * x, kind=kind, lag=lag)
    assert error.reason == codes.DEGENERATE_EXACT_FIT


def _scaled_regressor(scale: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(11)
    z = rng.standard_normal(40)
    y = 1.0 + 0.5 * z + 0.1 * rng.standard_normal(40)
    return assemble_design((z * scale)[np.newaxis, :].copy(), intercept=True), y


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
def test_a_covariance_diagonal_that_underflows_to_zero_raises(
    kind: int, lag: int
) -> None:
    # Column 1 near 1e300: coef near 5e-301, its variance near 1e-603.
    design, y = _scaled_regressor(1e300)
    error = _degenerate(design, y, kind=kind, lag=lag)
    assert error.reason == codes.DEGENERATE_NON_POSITIVE_VARIANCE
    assert (error.columns, error.value, error.bound) == ((1,), 0.0, 0.0)


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
def test_a_covariance_diagonal_that_overflows_raises(kind: int, lag: int) -> None:
    # Column 1 near 1e-300: coef near 5e299, its variance near 1e597.
    design, y = _scaled_regressor(1e-300)
    error = _degenerate(design, y, kind=kind, lag=lag)
    assert error.reason == codes.DEGENERATE_NON_POSITIVE_VARIANCE
    assert error.columns == (1,)
    assert error.value == math.inf
    assert error.bound == 0.0


def _patched_covariance(
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[NDArray[np.float64]], None],
) -> list[NDArray[np.float64]]:
    """Patch the covariance of the fit; return the list of unchanged results."""
    real = coefficient_covariance
    seen: list[NDArray[np.float64]] = []

    def patched(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
        cov = real(*args, **kwargs)
        seen.append(cov)
        changed = np.array(cov)
        change(changed)
        return changed

    monkeypatch.setattr(fit, "coefficient_covariance", patched)
    return seen


def test_a_non_finite_covariance_entry_off_the_diagonal_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def off_diagonal(cov: NDArray[np.float64]) -> None:
        cov[0, 2] = cov[2, 0] = math.inf

    seen = _patched_covariance(monkeypatch, off_diagonal)
    design, y = _five()
    error = _degenerate(design, y, kind=HC0)
    assert error.reason == codes.DEGENERATE_NON_POSITIVE_VARIANCE
    # Both indices of the entry; the diagonal itself is positive and finite.
    assert error.columns == (0, 2)
    assert error.bound == 0.0
    (cov,) = seen
    assert error.value == float(min(cov[0, 0], cov[2, 2]))
    assert error.value > 0.0


def test_a_t_statistic_that_overflows_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # A positive subnormal variance passes the diagonal rule; coef / se then
    # overflows for a coefficient near 1e150.
    def tiny_variance(cov: NDArray[np.float64]) -> None:
        cov[1, 1] = 5e-324

    _patched_covariance(monkeypatch, tiny_variance)
    design, y = _five()
    error = _degenerate(design, np.ldexp(y, 500))
    assert error.reason == codes.DEGENERATE_NON_POSITIVE_VARIANCE
    assert (error.columns, error.value, error.bound) == ((1,), 5e-324, 0.0)


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
@pytest.mark.parametrize("shift", [-200, -40, 40, 200])
def test_a_response_times_a_power_of_two_scales_the_result_exactly(
    kind: int, lag: int, shift: int
) -> None:
    design, y = _five()
    base = _run(design, y, kind=kind, lag=lag)
    scaled = _run(design, np.ldexp(y, shift), kind=kind, lag=lag)
    for name in ("coef", "se", "resid", "fitted"):
        assert np.array_equal(
            getattr(scaled, name), np.ldexp(getattr(base, name), shift)
        )
    assert np.array_equal(scaled.cov, np.ldexp(base.cov, 2 * shift))
    assert scaled.rss == math.ldexp(base.rss, 2 * shift)
    assert np.array_equal(scaled.t_stats, base.t_stats)
    assert np.array_equal(scaled.p_values, base.p_values)
    assert (scaled.r_squared, scaled.durbin_watson) == (
        base.r_squared,
        base.durbin_watson,
    )


@pytest.mark.parametrize("shift", [-300, 300])
def test_a_column_times_a_power_of_two_scales_its_coefficient_exactly(
    shift: int,
) -> None:
    design, y = _five()
    base = _run(design, y, kind=HC0)
    stretched = np.array(design)
    stretched[:, 2] = np.ldexp(stretched[:, 2], shift)
    result = _run(stretched, y, kind=HC0)
    factor = np.ldexp(np.ones(3), np.array([0, 0, -shift], dtype=np.int32))
    assert np.array_equal(result.coef, base.coef * factor)
    assert np.array_equal(result.cov, base.cov * np.outer(factor, factor))
    assert np.array_equal(result.fitted, base.fitted)
    assert np.array_equal(result.t_stats, base.t_stats)


def test_a_response_whose_sum_of_squares_overflows_still_fits() -> None:
    # sum(y**2) is above the float64 range, so an unscaled C.14 bound is inf
    # and the rule would fire on any fit. The scaled rule does not.
    rng = np.random.default_rng(5)
    x = rng.standard_normal(50)
    y = 1e155 * (1.0 + 2.0 * x + 1e-10 * rng.standard_normal(50))
    with np.errstate(over="ignore"):
        assert float(np.sum(y**2)) == math.inf
    result = _run(assemble_design(x[np.newaxis, :].copy(), intercept=True), y)
    _assert_finite(result)


def test_an_rss_above_the_float64_range_raises_at_the_largest_response() -> None:
    rng = np.random.default_rng(6)
    z = rng.standard_normal(30)
    y = 1e160 * (0.5 * z + rng.standard_normal(30))
    design = assemble_design((1e160 * z)[np.newaxis, :].copy(), intercept=False)
    error = _input_error(design, y)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        OPERAND_RESPONSE,
        (int(np.argmax(np.abs(y))),),
    )


def test_an_rss_that_underflows_to_zero_counts_as_an_exact_fit() -> None:
    # Owner decision 4: rss <= bound includes rss == 0.0 as a float64.
    rng = np.random.default_rng(7)
    z = rng.standard_normal(30)
    y = 1e-170 * (0.5 * z + rng.standard_normal(30))
    design = assemble_design((1e-170 * z)[np.newaxis, :].copy(), intercept=False)
    error = _degenerate(design, y, intercept=False)
    assert error.reason == codes.DEGENERATE_EXACT_FIT
    assert error.value == 0.0


# ------------------------------------------- owner decision 5: statistics


@pytest.mark.parametrize("intercept", [True, False])
def test_r_squared_is_centred_with_an_intercept_and_uncentred_without(
    intercept: bool,
) -> None:
    design = assemble_design(np.array([X1, X2]), intercept=intercept)
    result = _run(design, np.array(Y5), intercept=intercept)
    _, resid = _exact_ols([X1, X2], Y5, intercept=intercept)
    rss = sum((e * e for e in resid), Fraction(0))
    ys = [Fraction(v) for v in Y5]
    mean = sum(ys, Fraction(0)) / len(ys) if intercept else Fraction(0)
    base = sum(((v - mean) ** 2 for v in ys), Fraction(0))
    exact = 1 - rss / base
    assert abs(Fraction(result.r_squared) - exact) <= TOLERANCE * abs(exact)


@pytest.mark.parametrize(("kind", "lag"), KINDS, ids=str)
def test_durbin_watson_is_the_statistic_of_the_residuals(kind: int, lag: int) -> None:
    design, y = _five()
    result = _run(design, y, kind=kind, lag=lag)
    assert result.durbin_watson == diagnostics.durbin_watson(np.array(result.resid))
    oracle = float(_fraction_durbin_watson(result.resid.tolist()))
    assert abs(result.durbin_watson / oracle - 1.0) <= 1e-14
