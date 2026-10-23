# tests/solutions/factor_regression/test_lags.py
"""
WS_FACTOR_REGRESSION_LAGS S1.T1, S1.T2, S1.T4, S1.T5, S1.T6, S1.T7 and S1.T9.

Facade tests of ``Term.lag``, ``ModelSpec.max_lag``, the lagged column names,
the spec digest fields and the fit on a common sample. Every oracle is
independent of the library: ``hashlib`` for the digests, ``numpy.linalg.lstsq``
on a hand-built design for the coefficients, ``fractions.Fraction`` for the
Wald statistic.

S1.T4 golden literals. ``spec_digest`` holds no data, so the three literals
below were recorded from the code before this change (prod-equal commit) and
hold on every platform.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.errors import CoreParameterError
from riskquant.factor_regression.rows import CoreRows, build_rows
from riskquant.solutions.factor_regression import api, labels
from riskquant.solutions.factor_regression.api import (
    AlignedLevels,
    CovarianceKind,
    CovarianceSpec,
    InsufficientObservationsError,
    ModelSpec,
    ModelSpecError,
    ObservedSeries,
    OlsResult,
    RankDeficiencyError,
    Term,
    Transform,
    align_levels,
    fit_ols,
    wald_test,
)
from riskquant.solutions.factor_regression.contracts import ENGINE_VERSION
from riskquant.solutions.factor_regression.digest import spec_digest

_BASE_ORD = 19_000
_IID = CovarianceSpec(kind=CovarianceKind.IID, lag=None)
_HAC3 = CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=3)

# S1.T4 golden literals, recorded before the change.
_GOLDEN_BASE = "c5d55e66c6b30967ad087a4dc7050571b4fa77ec56df73b81c39972644a7e5ed"
_GOLDEN_USD = "86ba1d977b795509d3999d589915877a1a2f8fbb9efd5a894cd270b01d575524"
_GOLDEN_HAC = "272260a7be05fa992342ab587141df529b0de7e55f049bf52c6371d806f01480"


# ------------------------------------------------------------------ helpers


def _term(
    series: str, lag: int = 0, transform: Transform = Transform.LOG_RETURN
) -> Term:
    return Term(series=series, transform=transform, lag=lag)


def _spec(
    response: str,
    regressors: tuple[Term, ...],
    *,
    intercept: bool = True,
    step: int = 1,
    stride: int = 1,
    covariance: CovarianceSpec = _IID,
    max_lag: int = 0,
    condition_threshold: float | None = None,
) -> ModelSpec:
    return ModelSpec(
        response=_term(response),
        regressors=regressors,
        intercept=intercept,
        step=step,
        stride=stride,
        covariance=covariance,
        rank_rtol=1e-10,
        condition_threshold=condition_threshold,
        max_lag=max_lag,
    )


def _levels(
    seed: int, m: int, names: tuple[str, ...], units: tuple[str, ...]
) -> tuple[AlignedLevels, dict[str, NDArray[np.float64]]]:
    rng = np.random.default_rng(seed)
    dates = (_BASE_ORD + np.arange(m, dtype=np.int64)).astype(np.int32)
    values: dict[str, NDArray[np.float64]] = {}
    series: list[ObservedSeries] = []
    base = np.cumsum(0.01 * rng.standard_normal(m))
    for i, (name, unit) in enumerate(zip(names, units, strict=True)):
        own = np.cumsum(0.01 * rng.standard_normal(m))
        v = (1.0 + 0.5 * i) * np.exp(0.6 * base + own)
        values[name] = v
        series.append(ObservedSeries(name=name, unit=unit, date_ord=dates, values=v))
    levels = align_levels(series, start_ord=_BASE_ORD, end_ord=_BASE_ORD + m)
    return levels, values


def _log_returns(v: NDArray[np.float64]) -> list[float]:
    # r[t] = ln(v[t] / v[t-1]) for t >= 1; r[0] is unused.
    return [math.nan] + [math.log(v[t] / v[t - 1]) for t in range(1, len(v))]


# ------------------------------------------------------------------ digest oracle


def _oracle_digest_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype.str).encode("utf-8"))
    h.update(b"|")
    h.update(str(arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _oracle_digest_named(**arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, arr in arrays.items():
        h.update(name.encode("utf-8"))
        h.update(b"=")
        h.update(_oracle_digest_array(arr).encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def _text(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def _oracle_spec_digest(
    *,
    names: list[str],
    units: list[str],
    transforms: list[str],
    lags: list[int],
    max_lag: int,
    intercept: bool = True,
    step: int = 1,
    stride: int = 1,
    window: tuple[int, int] = (0, 20),
    cov: tuple[int, int] = (0, -1),
    rank_rtol: float = 1e-10,
    threshold: float | None = None,
) -> str:
    lines = ["Y\x1fP\x1fEUR\x1fLOG_RETURN"]
    for n, u, t, lag in zip(names, units, transforms, lags, strict=True):
        line = f"X\x1f{n}\x1f{u}\x1f{t}"
        if lag != 0:
            line += f"\x1f{lag}"
        lines.append(line)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.array([1], dtype=np.int64),
        "engine_version": _text(ENGINE_VERSION),
        "terms": _text("\x1e".join(lines)),
        "intercept": np.array([1 if intercept else 0], dtype=np.int8),
        "step": np.array([step], dtype=np.int64),
        "stride": np.array([stride], dtype=np.int64),
        "window": np.array(list(window), dtype=np.int32),
        "covariance_kind": np.array([cov[0]], dtype=np.int8),
        "covariance_lag": np.array([cov[1]], dtype=np.int64),
        "rank_rtol": np.array([rank_rtol], dtype=np.float64),
        "condition_threshold": (
            np.empty(0, dtype=np.float64)
            if threshold is None
            else np.array([threshold], dtype=np.float64)
        ),
    }
    if max_lag > 0:
        arrays["max_lag"] = np.array([max_lag], dtype=np.int64)
    return _oracle_digest_named(**arrays)


def _facade_spec_digest(
    *,
    names: list[str],
    units: list[str],
    transforms: list[str],
    lags: list[int] | None,
    max_lag: int,
    **overrides: object,
) -> str:
    kwargs: dict[str, object] = dict(
        engine_version=ENGINE_VERSION,
        response_name="P",
        response_unit="EUR",
        response_transform="LOG_RETURN",
        regressor_names=names,
        regressor_units=units,
        regressor_transforms=transforms,
        intercept=True,
        step=1,
        stride=1,
        start_ord=0,
        end_ord=20,
        covariance_kind=0,
        covariance_lag=-1,
        rank_rtol=1e-10,
        condition_threshold=None,
        regressor_lags=lags,
        max_lag=max_lag,
    )
    kwargs.update(overrides)
    return spec_digest(**kwargs)  # type: ignore[arg-type]


# ================================================================ S1.T1


def test_t1_lag_defaults_to_zero_and_is_stored_as_int() -> None:
    term = Term(series="Z", transform=Transform.LOG_RETURN)
    assert term.lag == 0
    assert type(term.lag) is int
    given: list[Any] = [3, -2, np.int32(4), np.int64(-5), np.uint8(1)]
    for value in given:
        stored = Term(series="Z", transform=Transform.LOG_RETURN, lag=value).lag
        assert stored == int(value)
        assert type(stored) is int


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), 1.0, "1", None, 1.5, [1]])
def test_t1_a_bool_or_non_integer_lag_raises_type_error(bad: object) -> None:
    with pytest.raises(TypeError):
        Term(series="Z", transform=Transform.LOG_RETURN, lag=bad)  # type: ignore[arg-type]


def test_t1_the_series_and_transform_checks_still_run_first() -> None:
    with pytest.raises(TypeError, match=r"Term\.series"):
        Term(series=1, transform=Transform.LOG_RETURN, lag=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"Term\.transform"):
        Term(series="Z", transform="LOG_RETURN", lag=True)  # type: ignore[arg-type]


def test_t1_terms_with_different_lags_are_different_values() -> None:
    assert _term("Z", 1) != _term("Z", 0)
    assert _term("Z", 0) == Term(series="Z", transform=Transform.LOG_RETURN)
    assert _term("Z", 1) == _term("Z", 1)


# ================================================================ S1.T2


def test_t2_max_lag_defaults_to_zero_and_is_stored_as_int() -> None:
    spec = ModelSpec(
        response=_term("P"),
        regressors=(_term("Z"),),
        intercept=True,
        step=1,
        stride=1,
        covariance=_IID,
        rank_rtol=1e-10,
        condition_threshold=None,
    )
    assert spec.max_lag == 0
    assert type(spec.max_lag) is int
    spec2 = _spec("P", (_term("Z", 2),), max_lag=np.int64(2))  # type: ignore[arg-type]
    assert spec2.max_lag == 2
    assert type(spec2.max_lag) is int


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.0, "1", None, 2.5])
def test_t2_a_bool_or_non_integer_max_lag_raises_type_error(bad: object) -> None:
    with pytest.raises(TypeError):
        _spec("P", (_term("Z"),), max_lag=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_lag", [-1, -7])
def test_t2_a_negative_max_lag_raises_model_spec_error(max_lag: int) -> None:
    with pytest.raises(ModelSpecError, match="max_lag must be >= 0"):
        _spec("P", (_term("Z"),), max_lag=max_lag)


def test_t2_a_max_lag_above_int32_raises_model_spec_error() -> None:
    with pytest.raises(ModelSpecError, match="int32"):
        _spec("P", (_term("Z"),), max_lag=2**31)
    assert _spec("P", (_term("Z"),), max_lag=2**31 - 1).max_lag == 2**31 - 1


@pytest.mark.parametrize(
    ("lags", "max_lag"),
    [((1,), 0), ((-1,), 0), ((0, 3), 2), ((-3, 0), 2), ((2, -2, 5), 4)],
)
def test_t2_a_regressor_lag_outside_the_window_raises_model_spec_error(
    lags: tuple[int, ...], max_lag: int
) -> None:
    regressors = tuple(_term("Z", lag) for lag in lags)
    with pytest.raises(ModelSpecError, match="outside"):
        _spec("P", regressors, max_lag=max_lag)


@pytest.mark.parametrize("lag", [1, -1, 5])
def test_t2_a_lagged_response_raises_model_spec_error(lag: int) -> None:
    with pytest.raises(ModelSpecError, match="response"):
        ModelSpec(
            response=_term("P", lag),
            regressors=(_term("Z"),),
            intercept=True,
            step=1,
            stride=1,
            covariance=_IID,
            rank_rtol=1e-10,
            condition_threshold=None,
            max_lag=abs(lag),
        )


def test_t2_lags_inside_the_window_are_accepted() -> None:
    spec = _spec("P", (_term("Z", 0), _term("Z", 2), _term("Z", -2)), max_lag=2)
    assert tuple(t.lag for t in spec.regressors) == (0, 2, -2)


def test_t2_the_existing_rules_run_before_the_lag_rules() -> None:
    # step 0 with a return term is reported before the lag outside the window.
    with pytest.raises(ModelSpecError, match="step=0"):
        _spec("P", (_term("Z", 3),), step=0, max_lag=1)


# ================================================================ S1.T5


@pytest.mark.parametrize(
    ("lag", "expected"),
    [(0, "Z"), (1, "Z[-1]"), (3, "Z[-3]"), (-1, "Z[+1]"), (-12, "Z[+12]")],
)
def test_t5_lagged_column_names(lag: int, expected: str) -> None:
    levels, _ = _levels(21, 40, ("P", "Z"), ("EUR", "USD"))
    result = fit_ols(levels, _spec("P", (_term("Z", lag),), max_lag=abs(lag)))
    assert result.column_names == ("const", expected)
    assert result.units == ("", "USD")
    assert result.transforms == (None, Transform.LOG_RETURN)


def test_t5_distributed_lag_names_keep_the_declared_order() -> None:
    levels, _ = _levels(22, 40, ("P", "Z", "S"), ("EUR", "USD", "EUR/USD"))
    spec = _spec(
        "P",
        (_term("Z", 1), _term("Z", 0), _term("S", -1), _term("Z", -1)),
        intercept=False,
        max_lag=1,
    )
    result = fit_ols(levels, spec)
    assert result.column_names == ("Z[-1]", "Z", "S[+1]", "Z[+1]")
    assert result.units == ("USD", "USD", "EUR/USD", "USD")


def test_t5_an_exact_duplicate_term_raises_rank_deficiency_at_the_fit() -> None:
    levels, _ = _levels(23, 40, ("P", "Z"), ("EUR", "USD"))
    with pytest.raises(RankDeficiencyError) as info:
        fit_ols(levels, _spec("P", (_term("Z", 1), _term("Z", 1)), max_lag=1))
    assert set(info.value.column_names) >= {"Z[-1]"}


# ================================================================ S1.T4


def test_t4_golden_lag_zero_spec_digests_are_unchanged() -> None:
    base = _facade_spec_digest(
        names=["Z"], units=["EUR"], transforms=["LOG_RETURN"], lags=None, max_lag=0
    )
    assert base == _GOLDEN_BASE
    assert (
        _facade_spec_digest(
            names=["Z"], units=["EUR"], transforms=["LOG_RETURN"], lags=[0], max_lag=0
        )
        == _GOLDEN_BASE
    )
    assert (
        _facade_spec_digest(
            names=["Z"],
            units=["EUR"],
            transforms=["LOG_RETURN"],
            lags=None,
            max_lag=0,
            response_unit="USD",
        )
        == _GOLDEN_USD
    )
    assert (
        _facade_spec_digest(
            names=["Z", "S"],
            units=["USD", "EUR/USD"],
            transforms=["LOG_RETURN", "DIFFERENCE"],
            lags=[0, 0],
            max_lag=0,
            intercept=False,
            step=5,
            stride=2,
            start_ord=19000,
            end_ord=19400,
            covariance_kind=2,
            covariance_lag=3,
            condition_threshold=30.0,
        )
        == _GOLDEN_HAC
    )


def test_t4_fit_ols_with_no_lag_declared_reproduces_the_golden_digest() -> None:
    levels, _ = _levels(24, 400, ("P", "Z", "S"), ("EUR", "USD", "EUR/USD"))
    spec = ModelSpec(
        response=_term("P"),
        regressors=(
            _term("Z"),
            Term(series="S", transform=Transform.DIFFERENCE),
        ),
        intercept=False,
        step=5,
        stride=2,
        covariance=_HAC3,
        rank_rtol=1e-10,
        condition_threshold=30.0,
    )
    result = fit_ols(levels, spec)
    assert result.spec_digest == _GOLDEN_HAC


def _assert_same_fit(a: OlsResult, b: OlsResult) -> None:
    for name in ("coef", "se", "t_stats", "p_values", "cov", "resid", "fitted"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.column_names == b.column_names
    assert a.diagnostics == b.diagnostics
    assert a.row_start_date_ord.tolist() == b.row_start_date_ord.tolist()
    assert a.row_end_date_ord.tolist() == b.row_end_date_ord.tolist()
    assert (a.spec_digest, a.sample_digest, a.row_digest) == (
        b.spec_digest,
        b.sample_digest,
        b.row_digest,
    )
    assert a.engine_version == b.engine_version == ENGINE_VERSION


def test_t4_explicit_zero_lags_give_the_lag_free_result() -> None:
    levels, _ = _levels(25, 60, ("P", "Z", "S"), ("EUR", "USD", "EUR/USD"))
    legacy = ModelSpec(
        response=Term(series="P", transform=Transform.LOG_RETURN),
        regressors=(
            Term(series="Z", transform=Transform.LOG_RETURN),
            Term(series="S", transform=Transform.SIMPLE_RETURN),
        ),
        intercept=True,
        step=2,
        stride=1,
        covariance=_HAC3,
        rank_rtol=1e-10,
        condition_threshold=None,
    )
    explicit = ModelSpec(
        response=_term("P", 0),
        regressors=(_term("Z", 0), _term("S", 0, Transform.SIMPLE_RETURN)),
        intercept=True,
        step=2,
        stride=1,
        covariance=_HAC3,
        rank_rtol=1e-10,
        condition_threshold=None,
        max_lag=0,
    )
    _assert_same_fit(fit_ols(levels, legacy), fit_ols(levels, explicit))


# ================================================================ S1.T6


@pytest.mark.parametrize(
    ("lags", "max_lag"),
    [([0], 0), ([1], 1), ([-1], 1), ([0, 2, -2], 2), ([0, 0], 3), ([1, 0], 1)],
)
def test_t6_spec_digest_matches_the_hashlib_oracle(
    lags: list[int], max_lag: int
) -> None:
    n = len(lags)
    names = ["Z"] * n
    units = ["USD"] * n
    transforms = ["LOG_RETURN"] * n
    facade = _facade_spec_digest(
        names=names, units=units, transforms=transforms, lags=lags, max_lag=max_lag
    )
    oracle = _oracle_spec_digest(
        names=names, units=units, transforms=transforms, lags=lags, max_lag=max_lag
    )
    assert facade == oracle


def test_t6_one_lag_or_the_window_changes_the_digest() -> None:
    def d(lags: list[int], max_lag: int) -> str:
        return _facade_spec_digest(
            names=["Z", "Z"],
            units=["USD", "USD"],
            transforms=["LOG_RETURN", "LOG_RETURN"],
            lags=lags,
            max_lag=max_lag,
        )

    digests = {
        d([0, 1], 1),
        d([1, 0], 1),
        d([0, 0], 1),
        d([0, 0], 0),
        d([0, 1], 2),
        d([0, -1], 1),
    }
    assert len(digests) == 6


def test_t6_a_lag_of_one_regressor_is_not_a_lag_of_another() -> None:
    a = _facade_spec_digest(
        names=["Z", "S"],
        units=["USD", "USD"],
        transforms=["LOG_RETURN"] * 2,
        lags=[1, 0],
        max_lag=1,
    )
    b = _facade_spec_digest(
        names=["Z", "S"],
        units=["USD", "USD"],
        transforms=["LOG_RETURN"] * 2,
        lags=[0, 1],
        max_lag=1,
    )
    assert a != b


def test_t6_regressor_lags_must_match_the_other_sequences() -> None:
    with pytest.raises(ValueError, match="regressor_lags"):
        _facade_spec_digest(
            names=["Z", "S"],
            units=["USD", "USD"],
            transforms=["LOG_RETURN"] * 2,
            lags=[1],
            max_lag=1,
        )


def test_t6_fit_ols_spec_digest_holds_the_series_name_and_the_lag() -> None:
    m = 40
    levels, _ = _levels(26, m, ("P", "Z"), ("EUR", "USD"))
    result = fit_ols(levels, _spec("P", (_term("Z", 1), _term("Z", 0)), max_lag=1))
    expected = _oracle_spec_digest(
        names=["Z", "Z"],
        units=["USD", "USD"],
        transforms=["LOG_RETURN", "LOG_RETURN"],
        lags=[1, 0],
        max_lag=1,
        window=(_BASE_ORD, _BASE_ORD + m),
    )
    assert result.spec_digest == expected


# ================================================================ S1.T7


def _dimson_fit(
    seed: int, m: int
) -> tuple[OlsResult, NDArray[np.float64], NDArray[np.float64]]:
    levels, values = _levels(seed, m, ("P", "A", "B"), ("EUR", "USD", "USD"))
    spec = _spec(
        "P",
        (_term("A", 0), _term("A", 1), _term("B", -1), _term("B", 0)),
        max_lag=1,
    )
    result = fit_ols(levels, spec)
    r_p = _log_returns(values["P"])
    r_a = _log_returns(values["A"])
    r_b = _log_returns(values["B"])
    # C.3 with h = 1, s = 1, L = 1: row k ends at e = 2 + k, R = m - 3.
    rows = range(2, m - 1)
    y = np.array([r_p[e] for e in rows], dtype=np.float64)
    x = np.array(
        [[1.0, r_a[e], r_a[e - 1], r_b[e + 1], r_b[e]] for e in rows], dtype=np.float64
    )
    return result, y, x


def test_t7_coefficients_and_iid_standard_errors_match_a_hand_built_design() -> None:
    result, y, x = _dimson_fit(31, 120)
    assert result.n_obs == y.shape[0] == 117
    assert result.column_names == ("const", "A", "A[-1]", "B[+1]", "B")
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    dof = y.shape[0] - x.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(result.coef, coef, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(result.se, se, rtol=1e-12, atol=0.0)
    assert result.dof == dof
    assert result.row_end_date_ord.tolist() == [_BASE_ORD + e for e in range(2, 119)]
    assert result.row_start_date_ord.tolist() == [
        _BASE_ORD + e - 1 for e in range(2, 119)
    ]


def test_t7_a_sample_too_short_for_the_window_raises() -> None:
    levels, _ = _levels(32, 4, ("P", "A"), ("EUR", "USD"))
    with pytest.raises(InsufficientObservationsError) as info:
        fit_ols(levels, _spec("P", (_term("A", 1),), max_lag=2))
    assert info.value.reason == "SAMPLE_STEP_NOT_BELOW_COUNT"
    assert info.value.n_available == 4
    assert info.value.n_required == 1 + 2 * 2 + 1


# ================================================================ S1.T9


def test_t9_wald_on_the_sum_of_the_lag_coefficients_matches_a_fraction_oracle() -> None:
    result, _, _ = _dimson_fit(33, 150)
    # Sum of the A coefficients (columns 1 and 2) equals 0.
    restriction = np.array([[0.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    rhs = np.zeros(1, dtype=np.float64)
    wald = wald_test(result, restriction, rhs)
    coef = [Fraction(float(c)) for c in result.coef]
    cov = [[Fraction(float(v)) for v in row] for row in result.cov]
    diff = coef[1] + coef[2]
    variance = cov[1][1] + cov[2][2] + cov[1][2] + cov[2][1]
    w_oracle = diff * diff / variance
    assert wald.n_restrictions == 1
    assert wald.dof == result.dof
    assert abs(Fraction(wald.w) - w_oracle) <= Fraction(1, 10**11) * abs(w_oracle)
    assert wald.spec_digest == result.spec_digest


# ================================================================ labels and core mapping


def test_the_new_reason_has_its_label() -> None:
    code = int(codes.PARAM_LAG_OUTSIDE_WINDOW)
    assert labels.PARAMETER_ERROR_NAMES[code] == "PARAM_LAG_OUTSIDE_WINDOW"
    assert labels.PARAMETER_ERROR_NAME_TO_CODE["PARAM_LAG_OUTSIDE_WINDOW"] is (
        codes.PARAM_LAG_OUTSIDE_WINDOW
    )


def test_a_core_lag_window_error_maps_to_model_spec_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels, _ = _levels(34, 40, ("P", "Z"), ("EUR", "USD"))

    def raise_core(*args: object, **kwargs: object) -> object:
        raise CoreParameterError(reason=codes.PARAM_LAG_OUTSIDE_WINDOW, value=3.0)

    monkeypatch.setattr(api, "build_rows", raise_core)
    with pytest.raises(ModelSpecError) as info:
        fit_ols(levels, _spec("P", (_term("Z", 1),), max_lag=1))
    assert "PARAM_LAG_OUTSIDE_WINDOW" in str(info.value)
    assert isinstance(info.value.__cause__, CoreParameterError)


def test_the_core_receives_the_response_lag_zero_and_the_declared_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels, _ = _levels(35, 40, ("P", "Z", "S"), ("EUR", "USD", "USD"))
    seen: dict[str, Any] = {}

    def spy(*args: Any, **kwargs: Any) -> CoreRows:
        seen.update(kwargs)
        return build_rows(*args, **kwargs)

    monkeypatch.setattr(api, "build_rows", spy)
    fit_ols(levels, _spec("P", (_term("Z", 2), _term("S", -1)), max_lag=3))
    term_lags = seen["term_lags"]
    assert isinstance(term_lags, np.ndarray)
    assert term_lags.dtype == np.int32
    assert term_lags.tolist() == [0, 2, -1]
    assert seen["max_lag"] == 3
