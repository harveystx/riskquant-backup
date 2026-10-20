# tests/factor_regression/test_adf.py
"""
S3.T1 to S3.T4: the ADF regression of a combination value
(WS_FACTOR_REGRESSION_COINTEGRATION A.1, C.5, C.6, C.7, C.13). Owner decisions
1 to 7 of the S3 prompt.

Oracles share no code with ``adf.py`` (A.7):

* S3.T1: the 2 x 2 normal equations of the lag 0, CONSTANT regression in
  ``fractions.Fraction``; ``tau**2`` is exact, the square root takes the sign
  of the exact slope.
* S3.T2 and the trend-origin test: a row-by-row construction of the A.1
  design in plain Python (trend origin 1 or 1000, not 0) and the float normal
  equations solved with ``numpy.linalg.solve`` in this file.
* Critical values: the A.2 values at ``n = 100``, typed here to 4 decimals.

Seeds and tolerances:

* S3.T1: random walk with seed 11, ``m = 26``, rounded to multiples of 1/16,
  so every level and every difference is exact in float64. ``tau``,
  ``gamma`` and ``se`` within 1e-12 relative.
* S3.T2: random walks with seeds 1, 2 and 3, ``m = 605``, lag 4,
  CONSTANT_TREND. ``tau`` and ``gamma`` within 1e-10 relative.
* Trend origin: seed 21, ``m = 300``, lag 2. ``tau`` within 1e-9 relative of
  the origin-1000 oracle.

Sandbox evidence (Python 3.12.3, numpy 2.4.4), seeds 1 to 40: S3.T1 worst
relative error 1.4e-15; S3.T2 worst 4.3e-14; origin 1000 worst 1.1e-13. The
exact fit ``z = 0.5**t`` (40 points, dyadic, no rounding) gives
``rss / bound = 3.3e-4`` (DEGENERATE_EXACT_FIT needs at most 1.0); the
alternating series gives ``1.1e-3``. The earlier fixture ``z = 5 + 0.8**t``
gave 0.83 with ``fit_core`` and up to 0.97 with other solvers, so it was
replaced (backlog R-001).

The S3 Prohibited list (no second solver, IID only, no p-value) is checked by
the AST scan of S3.T3 and at run time with a recording ``fit_core``.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import warnings
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import adf, codes, fit, mackinnon
from riskquant.factor_regression.adf import OPERAND_Z, CoreAdf, adf_core
from riskquant.factor_regression.errors import (
    CoreDegenerateFitError,
    CoreInputError,
    CoreParameterError,
    CoreRankError,
    CoreSampleSizeError,
)

CONSTANT = int(codes.DET_CONSTANT)
CONSTANT_TREND = int(codes.DET_CONSTANT_TREND)
RANK_RTOL = 1e-10

_ARRAY_FIELDS = (
    "coef",
    "se",
    "critical_values",
    "response",
    "design",
    "row_start_index",
    "row_end_index",
)


# ------------------------------------------------------------------ helpers


def _walk(seed: int, m: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.asarray(np.cumsum(rng.standard_normal(m)), dtype=np.float64)


def _dyadic_walk(seed: int, m: int) -> NDArray[np.float64]:
    # Multiples of 1/16 below 2**10 in magnitude: every difference is exact.
    return np.asarray(np.round(_walk(seed, m) * 16.0) / 16.0, dtype=np.float64)


def _run(
    z: Any, lag: Any = 0, deterministic: Any = CONSTANT, rank_rtol: Any = RANK_RTOL
) -> CoreAdf:
    return adf_core(z, lag, deterministic, rank_rtol)


def _oracle_rows(
    z: NDArray[np.float64], lag: int, trend: bool, origin: int
) -> tuple[list[list[float]], list[float]]:
    """A.1 row by row: one row per date ``t = lag + 1 .. m - 1``."""
    levels = [float(v) for v in z]
    rows: list[list[float]] = []
    response: list[float] = []
    for t in range(lag + 1, len(levels)):
        row = [1.0]
        if trend:
            row.append(float(origin + t - lag - 1))
        row.append(levels[t - 1])
        row.extend(levels[t - q] - levels[t - q - 1] for q in range(1, lag + 1))
        rows.append(row)
        response.append(levels[t] - levels[t - 1])
    return rows, response


def _oracle_float(
    z: NDArray[np.float64], lag: int, trend: bool, origin: int
) -> tuple[float, float, int, int]:
    """``(tau, gamma, n, k)`` from the float normal equations."""
    rows, response = _oracle_rows(z, lag, trend, origin)
    x = np.array(rows, dtype=np.float64)
    y = np.array(response, dtype=np.float64)
    n, k = x.shape
    xtx = x.T @ x
    beta = np.linalg.solve(xtx, x.T @ y)
    resid = y - x @ beta
    s2 = float(resid @ resid) / (n - k)
    j = 2 if trend else 1
    unit = np.zeros(k, dtype=np.float64)
    unit[j] = 1.0
    variance = s2 * float(np.linalg.solve(xtx, unit)[j])
    return float(beta[j]) / math.sqrt(variance), float(beta[j]), n, k


def _oracle_exact(
    z: NDArray[np.float64],
) -> tuple[Fraction, Fraction, Fraction, list[Fraction]]:
    """Lag 0, CONSTANT: ``(slope, var(slope), tau**2, response)`` exactly."""
    levels = [Fraction(float(v)) for v in z]
    x = levels[:-1]
    y = [levels[t] - levels[t - 1] for t in range(1, len(levels))]
    n = len(y)
    sx = sum(x, Fraction(0))
    sy = sum(y, Fraction(0))
    sxx = sum((v * v for v in x), Fraction(0))
    sxy = sum((a * b for a, b in zip(x, y, strict=True)), Fraction(0))
    det = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / det
    intercept = (sy - slope * sx) / n
    rss = sum(
        ((b - intercept - slope * a) ** 2 for a, b in zip(x, y, strict=True)),
        Fraction(0),
    )
    variance = rss / (n - 2) * n / det
    return slope, variance, slope * slope / variance, y


def _relative(value: float, exact: float) -> float:
    return abs(value - exact) / abs(exact)


def _adf_tree() -> ast.Module:
    return ast.parse(Path(adf.__file__).read_text(encoding="utf-8"))


def _input_error(z: Any, lag: Any = 0, deterministic: Any = CONSTANT) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        _run(z, lag, deterministic)
    return info.value


def _forbid_fit_core(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("fit_core must not run")

    monkeypatch.setattr(fit, "fit_core", forbidden)


def _forbid_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the design must not be built")

    monkeypatch.setattr(adf, "_build_regression", forbidden)


# ===================================================================== S3.T1


def test_t1_lag0_constant_tau_equals_the_fraction_oracle() -> None:
    z = _dyadic_walk(11, 26)
    assert bool(np.all(z * 16.0 == np.round(z * 16.0)))
    assert float(np.max(np.abs(z))) < 2.0**10
    result = _run(z, 0, CONSTANT)
    slope, variance, tau_squared, response = _oracle_exact(z)

    assert (result.n_obs, result.n_columns) == (25, 2)
    # The response is the exact difference on every row: right row offset.
    assert [Fraction(float(v)) for v in result.response] == response
    exact_tau = math.copysign(math.sqrt(float(tau_squared)), float(slope))
    assert math.copysign(1.0, result.gamma) == math.copysign(1.0, float(slope))
    assert _relative(result.tau, exact_tau) <= 1e-12
    assert _relative(result.gamma, float(slope)) <= 1e-12
    # se uses rss / (n - k) with k = 2: a dof of n - 1 is 2 % away.
    assert _relative(float(result.se[1]), math.sqrt(float(variance))) <= 1e-12


# ===================================================================== S3.T2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_t2_lag4_constant_trend_tau_equals_the_normal_equations(seed: int) -> None:
    z = _walk(seed, 605)
    result = _run(z, 4, CONSTANT_TREND)
    tau, gamma, n, k = _oracle_float(z, 4, trend=True, origin=1)
    assert (result.n_obs, result.n_columns) == (n, k) == (600, 7)
    assert _relative(result.tau, tau) <= 1e-10
    assert _relative(result.gamma, gamma) <= 1e-10


# ===================================================================== S3.T3


def test_t3_adf_imports_no_linalg_and_no_scipy() -> None:
    imported: list[str] = []
    for node in ast.walk(_adf_tree()):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported.extend(f"{module}.{alias.name}" for alias in node.names)
    assert imported, "the scan found no import"
    for name in imported:
        assert not name.startswith("scipy"), name
        assert "linalg" not in name.split("."), name


def test_t3_adf_names_no_solver_routine() -> None:
    forbidden = {"lstsq", "solve", "inv", "pinv", "svd", "linalg"}
    names: set[str] = set()
    for node in ast.walk(_adf_tree()):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    assert "fit_core" in names, "the scan found no fit_core reference"
    assert not names & forbidden


def test_t3_adf_calls_fit_core_exactly_once_with_iid() -> None:
    calls = [
        node
        for node in ast.walk(_adf_tree())
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "fit_core")
            or (isinstance(node.func, ast.Name) and node.func.id == "fit_core")
        )
    ]
    assert len(calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert keywords["intercept"] == "True"
    assert keywords["covariance_kind"] == "int(codes.IID)"
    assert keywords["covariance_lag"] == "0"


def test_t3_one_fit_core_call_with_iid_and_the_declared_rank_rtol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    real = fit.fit_core

    def recording(*args: Any, **kwargs: Any) -> fit.CoreFit:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(fit, "fit_core", recording)
    rank_rtol = 3e-11
    _run(_walk(5, 80), 2, CONSTANT_TREND, rank_rtol)
    assert len(calls) == 1
    assert calls[0]["intercept"] is True
    assert calls[0]["covariance_kind"] == int(codes.IID)
    assert type(calls[0]["covariance_kind"]) is int
    assert calls[0]["covariance_lag"] == 0
    assert calls[0]["rank_rtol"] is rank_rtol


@pytest.mark.parametrize("name", ["lstsq", "inv", "pinv", "solve", "tensorinv"])
def test_t3_no_least_squares_routine_or_inverse_runs(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"numpy.linalg.{name} must not run")

    monkeypatch.setattr(np.linalg, name, forbidden)
    result = _run(_walk(6, 120), 3, CONSTANT_TREND)
    assert math.isfinite(result.tau)


# ===================================================================== S3.T4


@pytest.mark.parametrize(
    ("m", "lag", "n_available"),
    [
        pytest.param(20, 0, 19, id="n19-lag0"),
        # k = 1 + 1 + 30 = 32 > n = 9: both rules fail, C.6 order wins.
        pytest.param(40, 30, 9, id="n9-k32-both-rules"),
        pytest.param(3, 5, -3, id="n-minus3"),
    ],
)
def test_t4_rows_below_the_table_minimum_raise_before_any_build(
    m: int, lag: int, n_available: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_fit_core(monkeypatch)
    _forbid_build(monkeypatch)
    with pytest.raises(CoreSampleSizeError) as info:
        _run(_walk(31, m), lag, CONSTANT)
    error = info.value
    assert (error.reason, error.n_available, error.n_required) == (
        int(codes.SAMPLE_ROWS_BELOW_TABLE_MINIMUM),
        n_available,
        20,
    )


def test_t4_twenty_rows_and_twenty_columns_raise_from_the_rank_gate() -> None:
    # 39 aligned dates, lag 18: n = 20, k = 1 + 1 + 18 = 20, so n < k + 1.
    with pytest.raises(CoreSampleSizeError) as info:
        _run(_walk(32, 39), 18, CONSTANT)
    error = info.value
    assert (error.reason, error.n_available, error.n_required) == (
        int(codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS),
        20,
        21,
    )


def test_t4_twenty_rows_pass_the_size_rule() -> None:
    result = _run(_walk(33, 21), 0, CONSTANT_TREND)
    assert (result.n_obs, result.n_columns) == (20, 3)


def test_t4_a_huge_lag_raises_the_size_rule_with_a_negative_count() -> None:
    with pytest.raises(CoreSampleSizeError) as info:
        _run(_walk(34, 30), 2**63, CONSTANT)
    assert info.value.n_available == 30 - 1 - 2**63
    assert info.value.reason == int(codes.SAMPLE_ROWS_BELOW_TABLE_MINIMUM)


# ============================================== design, rows and the table


@pytest.mark.parametrize("deterministic", [CONSTANT, CONSTANT_TREND])
def test_design_and_response_equal_the_a1_rows(deterministic: int) -> None:
    z = _walk(41, 30)
    result = _run(z, 3, deterministic)
    rows, response = _oracle_rows(z, 3, trend=deterministic == CONSTANT_TREND, origin=0)
    assert np.array_equal(result.design, np.array(rows))
    assert np.array_equal(result.response, np.array(response))
    assert result.design.flags.c_contiguous


def test_row_indices_follow_a1() -> None:
    m, lag = 30, 3
    result = _run(_walk(42, m), lag, CONSTANT)
    assert result.n_obs == 26
    # Row r is the date t = lag + 1 + r: start t - 1, end t.
    assert np.array_equal(result.row_start_index, np.arange(3, 29))
    assert np.array_equal(result.row_end_index, np.arange(4, 30))
    assert result.row_start_index.dtype == np.int32
    assert result.row_end_index.dtype == np.int32


@pytest.mark.parametrize(
    ("deterministic", "table", "pinned"),
    [
        (CONSTANT, mackinnon.TABLE_TAU_C, (-3.4975, -2.8909, -2.5824)),
        (CONSTANT_TREND, mackinnon.TABLE_TAU_CT, (-4.0523, -3.4553, -3.1533)),
    ],
)
def test_the_deterministic_case_selects_the_table(
    deterministic: int, table: int, pinned: tuple[float, float, float]
) -> None:
    result = _run(_walk(43, 101), 0, deterministic)
    assert result.n_obs == 100
    assert np.array_equal(
        result.critical_values, mackinnon.critical_values(table, result.n_obs)
    )
    assert np.allclose(result.critical_values, pinned, rtol=0.0, atol=1e-4)


def test_tau_does_not_depend_on_the_trend_origin() -> None:
    z = _walk(21, 300)
    result = _run(z, 2, CONSTANT_TREND)
    tau, _, _, _ = _oracle_float(z, 2, trend=True, origin=1000)
    assert _relative(result.tau, tau) <= 1e-9


def test_tau_is_coef_over_se_of_the_z_lag1_column() -> None:
    for deterministic, column in ((CONSTANT, 1), (CONSTANT_TREND, 2)):
        result = _run(_walk(44, 90), 1, deterministic)
        assert result.gamma == float(result.coef[column])
        assert result.tau == float(result.coef[column]) / float(result.se[column])


# ================================================== errors of fit_core


@pytest.mark.parametrize(
    ("deterministic", "lag", "z"),
    [
        pytest.param(
            CONSTANT_TREND,
            0,
            1.0 + 0.5 * np.arange(40, dtype=np.float64),
            id="linear-z-with-trend",
        ),
        pytest.param(
            CONSTANT,
            2,
            5.0 + 0.8 ** np.arange(60, dtype=np.float64),
            id="geometric-z-lag2",
        ),
    ],
)
def test_a_collinear_design_raises_rank_deficient(
    deterministic: int, lag: int, z: NDArray[np.float64]
) -> None:
    with pytest.raises(CoreRankError) as info:
        _run(z, lag, deterministic)
    assert info.value.reason == int(codes.RANK_DEFICIENT)


def test_a_constant_response_raises_zero_response_variance() -> None:
    z = 1.0 + 0.5 * np.arange(40, dtype=np.float64)
    with pytest.raises(CoreDegenerateFitError) as info:
        _run(z, 0, CONSTANT)
    assert info.value.reason == int(codes.DEGENERATE_ZERO_RESPONSE_VARIANCE)


@pytest.mark.parametrize(
    "z",
    [
        # dz_t = -0.5 z_{t-1}, exact in float64: dyadic values, no rounding.
        # Sandbox rss / bound = 3.3e-4. The earlier fixture 5 + 0.8**t had 0.83
        # (0.97 with another solver): too close to the limit of 1.0 (R-001).
        pytest.param(0.5 ** np.arange(40, dtype=np.float64), id="geometric"),
        # dz_t = 1 - 2 z_{t-1}, exact in float64. Sandbox rss / bound = 1.1e-3.
        pytest.param(np.tile([0.0, 1.0], 30), id="alternating"),
    ],
)
def test_an_exact_linear_relation_raises_exact_fit(z: NDArray[np.float64]) -> None:
    with pytest.raises(CoreDegenerateFitError) as info:
        _run(z, 0, CONSTANT)
    assert info.value.reason == int(codes.DEGENERATE_EXACT_FIT)


@pytest.mark.parametrize(
    ("lag", "operand", "position"),
    [
        # Lag 0: the design [1, z_lag1] is finite; response row 0 is -inf.
        (0, fit.OPERAND_RESPONSE, 0),
        # Lag 2: design row 0, column 2 (dz_lag1) is -inf; flat index 2.
        (2, fit.OPERAND_DESIGN, 2),
    ],
)
def test_an_overflowing_difference_raises_from_fit_core_silently(
    lag: int, operand: int, position: int
) -> None:
    z = np.tile([1e308, -1e308], 15)
    assert bool(np.isfinite(z).all())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        error = _input_error(z, lag, CONSTANT)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        operand,
        (position,),
    )


@pytest.mark.parametrize("rank_rtol", [True, np.bool_(True), "1e-10", None])
def test_a_non_real_rank_rtol_raises_type_error_from_fit_core(rank_rtol: Any) -> None:
    with pytest.raises(TypeError):
        _run(_walk(45, 40), 0, CONSTANT, rank_rtol)


@pytest.mark.parametrize("rank_rtol", [0.0, 1.0, float("nan"), -1e-10])
def test_a_rank_rtol_outside_the_open_unit_interval_raises(rank_rtol: float) -> None:
    with pytest.raises(CoreParameterError) as info:
        _run(_walk(46, 40), 0, CONSTANT, rank_rtol)
    assert info.value.reason == int(codes.PARAM_RANK_RTOL_OUT_OF_RANGE)


# ============================================== checks (owner decision 2)


class _Subclass(np.ndarray[Any, Any]):
    pass


def _masked_nan() -> Any:
    raw = _walk(47, 30)
    raw[4] = np.nan
    hidden = np.zeros(30, dtype=np.bool_)
    hidden[4] = True
    masked = np.ma.masked_array(raw, mask=hidden)
    # The mask hides the NaN from a reduction.
    assert bool(np.isfinite(masked).all())
    return masked


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _walk(48, 30).tolist(), id="list"),
        pytest.param(_masked_nan, id="masked-nan"),
        pytest.param(lambda: _walk(48, 30).view(_Subclass), id="subclass"),
    ],
)
def test_z_that_is_not_exactly_ndarray_is_rejected(make: Callable[[], Any]) -> None:
    error = _input_error(make())
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        OPERAND_Z,
        (),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128])
def test_a_wrong_dtype_is_rejected(dtype: type[np.generic]) -> None:
    error = _input_error(_walk(49, 30).astype(dtype))
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_WRONG_DTYPE),
        OPERAND_Z,
        (),
    )


@pytest.mark.parametrize(
    "z",
    [
        pytest.param(_walk(50, 30).reshape(1, 30), id="2-d"),
        pytest.param(np.array(1.0), id="0-d"),
    ],
)
def test_a_wrong_ndim_is_rejected(z: NDArray[np.float64]) -> None:
    error = _input_error(z)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_WRONG_NDIM),
        OPERAND_Z,
        (),
    )


def test_an_empty_z_is_rejected() -> None:
    error = _input_error(np.empty(0, dtype=np.float64))
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_EMPTY),
        OPERAND_Z,
        (),
    )


@pytest.mark.parametrize(
    ("bad", "first"),
    [({7: np.nan}, 7), ({9: np.nan, 3: np.inf}, 3), ({0: -np.inf}, 0)],
)
def test_a_non_finite_z_is_rejected_at_the_first_position(
    bad: dict[int, float], first: int
) -> None:
    z = _walk(51, 30)
    for index, value in bad.items():
        z[index] = value
    error = _input_error(z)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NON_FINITE),
        OPERAND_Z,
        (first,),
    )


@pytest.mark.parametrize(
    "lag", [True, False, np.bool_(True), 1.0, np.float64(2.0), "2", None]
)
def test_a_lag_without_an_integer_index_raises_type_error(lag: Any) -> None:
    with pytest.raises(TypeError):
        _run(_walk(52, 40), lag, CONSTANT)


@pytest.mark.parametrize("lag", [-1, np.int64(-3)])
def test_a_negative_lag_raises_parameter_error(lag: Any) -> None:
    with pytest.raises(CoreParameterError) as info:
        _run(_walk(53, 40), lag, CONSTANT)
    assert info.value.reason == int(codes.PARAM_NEGATIVE_LAG)
    assert info.value.value == float(lag)


@pytest.mark.parametrize(
    "deterministic", [True, False, np.bool_(False), 0.0, np.float64(1.0), "0", None]
)
def test_a_deterministic_without_an_integer_index_raises_type_error(
    deterministic: Any,
) -> None:
    with pytest.raises(TypeError):
        _run(_walk(54, 40), 0, deterministic)


@pytest.mark.parametrize("deterministic", [2, -1, np.int8(5)])
def test_an_unknown_deterministic_code_raises_value_error(deterministic: Any) -> None:
    with pytest.raises(ValueError, match="unknown deterministic code"):
        _run(_walk(55, 40), 0, deterministic)


def test_numpy_integers_are_accepted_and_stored_as_int() -> None:
    result = _run(_walk(56, 60), np.uint8(2), codes.DET_CONSTANT_TREND)
    assert type(result.lag) is int and result.lag == 2
    assert type(result.deterministic) is int
    assert result.deterministic == CONSTANT_TREND
    assert result.n_columns == 1 + 1 + 1 + 2


def test_the_z_checks_run_before_the_lag_checks() -> None:
    error = _input_error(_walk(57, 30).astype(np.float32), "x", "y")
    assert error.reason == int(codes.INPUT_WRONG_DTYPE)


def test_the_lag_checks_run_before_the_deterministic_checks() -> None:
    with pytest.raises(CoreParameterError):
        _run(_walk(58, 30), -1, 9)


def test_the_deterministic_checks_run_before_the_size_rule() -> None:
    with pytest.raises(ValueError, match="unknown deterministic code"):
        _run(_walk(59, 5), 0, 9)


def test_the_size_rule_runs_before_the_rank_rtol_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_fit_core(monkeypatch)
    with pytest.raises(CoreSampleSizeError):
        _run(_walk(60, 10), 0, CONSTANT, 2.0)


def test_a_non_finite_z_is_rejected_before_the_size_rule() -> None:
    z = _walk(61, 5)
    z[2] = np.nan
    assert _input_error(z).reason == int(codes.INPUT_NON_FINITE)


# ============================================== result contract (C.13)


def test_the_result_holds_exactly_the_declared_fields() -> None:
    assert [field.name for field in dataclasses.fields(CoreAdf)] == [
        "tau",
        "gamma",
        "coef",
        "se",
        "critical_values",
        "condition_number",
        "n_obs",
        "n_columns",
        "lag",
        "deterministic",
        "response",
        "design",
        "row_start_index",
        "row_end_index",
    ]
    for name in (field.name for field in dataclasses.fields(CoreAdf)):
        for word in ("p_value", "accepted", "rejected", "stationary", "valid"):
            assert word not in name


def test_the_result_arrays_are_new_read_only_and_typed() -> None:
    z = _walk(62, 50)
    result = _run(z, 2, CONSTANT_TREND)
    n, k = result.n_obs, result.n_columns
    shapes = {
        "coef": ((k,), np.float64),
        "se": ((k,), np.float64),
        "critical_values": ((3,), np.float64),
        "response": ((n,), np.float64),
        "design": ((n, k), np.float64),
        "row_start_index": ((n,), np.int32),
        "row_end_index": ((n,), np.int32),
    }
    assert set(shapes) == set(_ARRAY_FIELDS)
    for name, (shape, dtype) in shapes.items():
        array = getattr(result, name)
        assert type(array) is np.ndarray, name
        assert array.dtype == dtype, name
        assert array.shape == shape, name
        assert array.flags.c_contiguous, name
        assert not array.flags.writeable, name
        assert not np.shares_memory(array, z), name
        assert bool(np.isfinite(array).all()), name
    for name in ("tau", "gamma", "condition_number"):
        value = getattr(result, name)
        assert type(value) is float and math.isfinite(value), name
    for name in ("n_obs", "n_columns", "lag", "deterministic"):
        assert type(getattr(result, name)) is int, name


def test_the_result_is_frozen_and_compares_by_identity() -> None:
    z = _walk(63, 40)
    first, second = _run(z), _run(z)
    for field in dataclasses.fields(CoreAdf):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(first, field.name, getattr(second, field.name))
    # eq=False (D064): identity equality, no element-wise array comparison.
    assert first != second
    assert first in [first]
    assert first not in [second]


def test_the_caller_array_is_unchanged_and_not_retained() -> None:
    z = _walk(64, 40)
    before = z.copy()
    result = _run(z, 1, CONSTANT)
    snapshot = {name: getattr(result, name).copy() for name in _ARRAY_FIELDS}
    assert np.array_equal(z, before)
    assert z.flags.writeable
    z[:] = -7.0
    for name in _ARRAY_FIELDS:
        assert np.array_equal(getattr(result, name), snapshot[name]), name


def test_a_read_only_z_is_accepted() -> None:
    z = _walk(65, 40)
    z.setflags(write=False)
    assert math.isfinite(_run(z, 1, CONSTANT_TREND).tau)


def test_two_calls_give_equal_results() -> None:
    z = _walk(66, 70)
    first, second = _run(z, 3, CONSTANT_TREND), _run(z, 3, CONSTANT_TREND)
    for name in _ARRAY_FIELDS:
        assert np.array_equal(getattr(first, name), getattr(second, name)), name
    for name in (
        "tau",
        "gamma",
        "condition_number",
        "n_obs",
        "n_columns",
        "lag",
        "deterministic",
    ):
        assert getattr(first, name) == getattr(second, name), name
