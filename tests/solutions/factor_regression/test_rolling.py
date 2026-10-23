# tests/solutions/factor_regression/test_rolling.py
"""
WS_FACTOR_REGRESSION_ROLLING S1.T1 to S1.T7.

Facade tests of ``rolling_fit``, ``RollingWindow`` and ``RollingFitResult``.

Test IDs
--------
S1.T1 - the contract types: dataclass parameters, fields, the one-of rule (C.1)
S1.T2 - type and value checks of rolling_fit, all before the first window (C.2)
S1.T3 - window count and bounds against the C.3 formula (C.3)
S1.T4 - oracle: every window equals fit_ols on align_levels of the original
        ObservedSeries over the window bounds, bit for bit; the credit fixture
        with NOT_COMMON_DATE exclusions inside the windows (C.4)
S1.T5 - a failing window is recorded; the other windows continue; any other
        error propagates (C.5)
S1.T6 - a lagged spec: n_obs of every window follows the row formula of the
        lags brief for the window length (C.6)
S1.T7 - the public surface gains rolling_fit, RollingFitResult and
        RollingWindow (C.7)

The S1.T4 oracle is the owner loop ("align_levels and fit_ols per window"),
which shares no window logic with ``rolling_fit``: it re-aligns the original
``ObservedSeries`` on the window bounds. The S1.T6 row count is the closed
formula ``R = floor((m - 1 - h - 2L) / s) + 1`` of the lags brief.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.ingest_fixtures.factor_regression.analyses import (
    CANONICAL_OLS_ANALYSES,
)
from riskquant.ingest_fixtures.factor_regression.catalog import (
    CANONICAL_FACTOR_REGRESSION_CASES,
)
from riskquant.ingest_fixtures.factor_regression.observed import (
    aligned_levels_for_case,
    observed_series_for_case,
)
from riskquant.solutions.factor_regression import api
from riskquant.solutions.factor_regression.api import (
    AlignedLevels,
    CovarianceKind,
    CovarianceSpec,
    DegenerateFitError,
    Exclusion,
    InsufficientObservationsError,
    ModelSpec,
    ModelSpecError,
    ObservedSeries,
    OlsResult,
    RankDeficiencyError,
    RollingFitResult,
    RollingWindow,
    Term,
    Transform,
    TransformDomainError,
    align_levels,
    fit_ols,
    rolling_fit,
)

_BASE_ORD = 19_000
_IID = CovarianceSpec(kind=CovarianceKind.IID, lag=None)
_CREDIT_CASE = "credit_factor_loadings"
_CREDIT_ANALYSIS = "credit_weekly_eu_bank"
_NOT_COMMON_DATE = "NOT_COMMON_DATE"
_OUTSIDE_WINDOW = "OUTSIDE_WINDOW"


# ------------------------------------------------------------------ helpers


def _term(series: str, lag: int = 0) -> Term:
    return Term(series=series, transform=Transform.LOG_RETURN, lag=lag)


def _spec(
    response: str,
    regressors: tuple[Term, ...],
    *,
    intercept: bool = True,
    step: int = 1,
    stride: int = 1,
    covariance: CovarianceSpec = _IID,
    max_lag: int = 0,
) -> ModelSpec:
    return ModelSpec(
        response=_term(response),
        regressors=regressors,
        intercept=intercept,
        step=step,
        stride=stride,
        covariance=covariance,
        rank_rtol=1e-10,
        condition_threshold=None,
        max_lag=max_lag,
    )


def _levels(
    seed: int, m: int, names: tuple[str, ...], units: tuple[str, ...]
) -> tuple[AlignedLevels, tuple[ObservedSeries, ...]]:
    """Consecutive dates, no exclusion: every aligned index is one date."""
    rng = np.random.default_rng(seed)
    dates = (_BASE_ORD + np.arange(m, dtype=np.int64)).astype(np.int32)
    series: list[ObservedSeries] = []
    base = np.cumsum(0.01 * rng.standard_normal(m))
    for i, (name, unit) in enumerate(zip(names, units, strict=True)):
        own = np.cumsum(0.01 * rng.standard_normal(m))
        values = (1.0 + 0.5 * i) * np.exp(0.6 * base + own)
        series.append(
            ObservedSeries(name=name, unit=unit, date_ord=dates, values=values)
        )
    levels = align_levels(series, start_ord=_BASE_ORD, end_ord=_BASE_ORD + m)
    return levels, tuple(series)


def _credit() -> tuple[AlignedLevels, tuple[ObservedSeries, ...], ModelSpec]:
    case = CANONICAL_FACTOR_REGRESSION_CASES[_CREDIT_CASE]
    spec = next(a.spec for a in CANONICAL_OLS_ANALYSES if a.name == _CREDIT_ANALYSIS)
    return aligned_levels_for_case(case), observed_series_for_case(case), spec


def _oracle(
    series: Sequence[ObservedSeries], spec: ModelSpec, start_ord: int, end_ord: int
) -> OlsResult:
    """The owner loop: re-align the original series on the window, then fit."""
    return fit_ols(align_levels(series, start_ord=start_ord, end_ord=end_ord), spec)


def _same_bits(a: NDArray[np.generic], b: NDArray[np.generic]) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes(order="C") == b.tobytes(order="C")
    )


def _assert_equal_fits(got: OlsResult, oracle: OlsResult, start: int, end: int) -> None:
    """
    Every field of ``got`` equals the oracle (C.4).

    Arrays compare by bytes. ``exclusions`` compares with the oracle records
    whose ``date_ord`` lies in the window (owner decision 2026-10-23: a
    window holds the exclusions inside its bounds; the oracle re-alignment
    also records OUTSIDE_WINDOW for every observation outside the window).
    """
    for field in dataclasses.fields(OlsResult):
        value = getattr(got, field.name)
        expected = getattr(oracle, field.name)
        if field.name == "exclusions":
            inside = tuple(e for e in expected if start <= e.date_ord < end)
            assert value == inside, field.name
            continue
        if isinstance(value, np.ndarray):
            assert isinstance(expected, np.ndarray)
            assert _same_bits(value, expected), field.name
        else:
            assert value == expected, field.name


def _record_bounds(result: RollingFitResult) -> list[tuple[int, int]]:
    return [(w.start_ord, w.end_ord) for w in result.windows]


# ================================================================ S1.T1


@pytest.mark.parametrize("cls", [RollingWindow, RollingFitResult])
def test_t1_contract_types_are_frozen_slots_kw_only_and_eq_false(cls: type) -> None:
    # Field.kw_only exists from Python 3.10. _DataclassParams.kw_only and .slots
    # exist only from Python 3.12, and the project floor is 3.11.
    params = cls.__dataclass_params__  # type: ignore[attr-defined]
    fields = cls.__dataclass_fields__.values()  # type: ignore[attr-defined]
    assert params.frozen
    assert not params.eq
    assert all(field.kw_only for field in fields)
    assert "__slots__" in cls.__dict__


def test_t1_contract_fields_and_signature() -> None:
    assert [f.name for f in dataclasses.fields(RollingWindow)] == [
        "start_ord",
        "end_ord",
        "result",
        "error_type",
        "error_message",
    ]
    assert [f.name for f in dataclasses.fields(RollingFitResult)] == [
        "window",
        "move",
        "windows",
    ]
    parameters = inspect.signature(rolling_fit).parameters
    assert list(parameters) == ["levels", "spec", "window", "move"]
    assert parameters["window"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["move"].kind is inspect.Parameter.KEYWORD_ONLY


def test_t1_exactly_one_of_result_and_error_type_is_none() -> None:
    levels, _ = _levels(1, 30, ("Y", "X"), ("EUR", "EUR"))
    fit = fit_ols(levels, _spec("Y", (_term("X"),)))
    ok = RollingWindow(
        start_ord=1, end_ord=2, result=fit, error_type=None, error_message=None
    )
    failed = RollingWindow(
        start_ord=1, end_ord=2, result=None, error_type="E", error_message="m"
    )
    assert ok.result is fit and failed.result is None
    assert ok != RollingWindow(
        start_ord=1, end_ord=2, result=fit, error_type=None, error_message=None
    )
    with pytest.raises(ValueError, match="exactly one"):
        RollingWindow(
            start_ord=1, end_ord=2, result=None, error_type=None, error_message=None
        )
    with pytest.raises(ValueError, match="exactly one"):
        RollingWindow(
            start_ord=1, end_ord=2, result=fit, error_type="E", error_message="m"
        )
    with pytest.raises(ValueError, match="error_message"):
        RollingWindow(
            start_ord=1, end_ord=2, result=None, error_type="E", error_message=None
        )
    with pytest.raises(ValueError, match="below"):
        RollingWindow(
            start_ord=2, end_ord=2, result=None, error_type="E", error_message="m"
        )
    with pytest.raises(TypeError, match="bool"):
        RollingWindow(
            start_ord=True, end_ord=2, result=None, error_type="E", error_message="m"
        )
    with pytest.raises(TypeError, match="result"):
        RollingWindow(
            start_ord=1,
            end_ord=2,
            result="fit",  # type: ignore[arg-type]
            error_type=None,
            error_message=None,
        )
    with pytest.raises(TypeError, match="error_type"):
        RollingWindow(
            start_ord=1,
            end_ord=2,
            result=None,
            error_type=1,  # type: ignore[arg-type]
            error_message="m",
        )


def test_t1_rolling_fit_result_checks_its_fields() -> None:
    record = RollingWindow(
        start_ord=1, end_ord=2, result=None, error_type="E", error_message="m"
    )
    stored = RollingFitResult(
        window=np.int64(3),  # type: ignore[arg-type]
        move=np.int32(1),  # type: ignore[arg-type]
        windows=(record,),
    )
    assert type(stored.window) is int and type(stored.move) is int
    with pytest.raises(TypeError, match="exactly tuple"):
        RollingFitResult(window=3, move=1, windows=[record])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"windows\[0\]"):
        RollingFitResult(window=3, move=1, windows=(1,))  # type: ignore[arg-type]
    with pytest.raises(ModelSpecError, match="window must be >= 1"):
        RollingFitResult(window=0, move=1, windows=(record,))
    with pytest.raises(ModelSpecError, match="move must be >= 1"):
        RollingFitResult(window=3, move=0, windows=(record,))
    with pytest.raises(ValueError, match="at least one"):
        RollingFitResult(window=3, move=1, windows=())
    with pytest.raises(TypeError, match="bool"):
        RollingFitResult(window=True, move=1, windows=(record,))


# ================================================================ S1.T2


def test_t2_wrong_types_raise_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    levels, _ = _levels(2, 30, ("Y", "X"), ("EUR", "EUR"))
    spec = _spec("Y", (_term("X"),))

    def _forbidden(*args: object, **kwargs: object) -> OlsResult:
        raise AssertionError("fit_ols ran before the checks")

    monkeypatch.setattr(api, "fit_ols", _forbidden)
    with pytest.raises(TypeError, match="levels must be AlignedLevels"):
        rolling_fit(spec, spec, window=10, move=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spec must be ModelSpec"):
        rolling_fit(levels, levels, window=10, move=1)  # type: ignore[arg-type]
    for bad in (True, np.bool_(True), 10.0, "10", None):
        with pytest.raises(TypeError, match="window"):
            rolling_fit(levels, spec, window=bad, move=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="move"):
            rolling_fit(levels, spec, window=10, move=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("window", "move", "pattern"),
    [
        (0, 1, "window must be >= 1"),
        (-5, 1, "window must be >= 1"),
        (10, 0, "move must be >= 1"),
        (10, -1, "move must be >= 1"),
        (31, 1, "window 31 exceeds the aligned date count 30"),
    ],
)
def test_t2_bad_window_or_move_raises_model_spec_error_before_any_window(
    window: int, move: int, pattern: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    levels, _ = _levels(3, 30, ("Y", "X"), ("EUR", "EUR"))

    def _forbidden(*args: object, **kwargs: object) -> OlsResult:
        raise AssertionError("fit_ols ran before the checks")

    monkeypatch.setattr(api, "fit_ols", _forbidden)
    with pytest.raises(ModelSpecError, match=pattern):
        rolling_fit(levels, _spec("Y", (_term("X"),)), window=window, move=move)


def test_t2_the_level_and_spec_types_are_checked_before_window_and_move() -> None:
    levels, _ = _levels(4, 30, ("Y", "X"), ("EUR", "EUR"))
    with pytest.raises(TypeError, match="levels"):
        rolling_fit(None, None, window=0, move=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spec"):
        rolling_fit(levels, None, window=0, move=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="window"):
        rolling_fit(levels, _spec("Y", (_term("X"),)), window=1.5, move=0)  # type: ignore[arg-type]


def test_t2_numpy_integers_are_accepted_and_stored_as_int() -> None:
    levels, _ = _levels(5, 30, ("Y", "X"), ("EUR", "EUR"))
    result = rolling_fit(
        levels,
        _spec("Y", (_term("X"),)),
        window=np.int32(12),  # type: ignore[arg-type]
        move=np.int64(6),  # type: ignore[arg-type]
    )
    assert type(result.window) is int and result.window == 12
    assert type(result.move) is int and result.move == 6
    assert len(result.windows) == (30 - 12) // 6 + 1


# ================================================================ S1.T3


@pytest.mark.parametrize(
    ("m", "window", "move"),
    [(30, 12, 6), (30, 12, 5), (30, 30, 1), (30, 7, 10), (31, 10, 3), (25, 5, 5)],
)
def test_t3_window_count_and_bounds_follow_the_formula(
    m: int, window: int, move: int
) -> None:
    levels, _ = _levels(6, m, ("Y", "X"), ("EUR", "EUR"))
    result = rolling_fit(levels, _spec("Y", (_term("X"),)), window=window, move=move)
    expected_count = (m - window) // move + 1
    assert len(result.windows) == expected_count
    d = levels.date_ord.tolist()
    expected = [
        (d[i * move], d[i * move + window - 1] + 1) for i in range(expected_count)
    ]
    assert _record_bounds(result) == expected
    for record in result.windows:
        assert record.result is not None
        assert record.result.window == (record.start_ord, record.end_ord)
        assert record.error_type is None and record.error_message is None
        assert record.end_ord - record.start_ord == window


def test_t3_an_incomplete_last_window_is_dropped() -> None:
    # m = 30, window = 12, move = 10: starts 0, 10 complete; 20 would need 32.
    levels, _ = _levels(7, 30, ("Y", "X"), ("EUR", "EUR"))
    result = rolling_fit(levels, _spec("Y", (_term("X"),)), window=12, move=10)
    assert len(result.windows) == 2
    assert result.windows[-1].end_ord == _BASE_ORD + 10 + 12


def test_t3_bounds_use_aligned_dates_not_calendar_days() -> None:
    # The credit fixture has gaps between aligned dates: a window of 120
    # aligned dates spans more than 120 calendar days.
    levels, _, spec = _credit()
    result = rolling_fit(levels, spec, window=120, move=20)
    m = int(levels.date_ord.shape[0])
    assert len(result.windows) == (m - 120) // 20 + 1 == 26
    for i, record in enumerate(result.windows):
        assert record.start_ord == int(levels.date_ord[i * 20])
        assert record.end_ord == int(levels.date_ord[i * 20 + 119]) + 1
        assert record.end_ord - record.start_ord > 120


# ================================================================ S1.T4


def test_t4_every_credit_window_equals_the_realigned_oracle_bit_for_bit() -> None:
    levels, series, spec = _credit()
    result = rolling_fit(levels, spec, window=120, move=20)
    assert len(result.windows) == 26
    windows_with_exclusions = 0
    for record in result.windows:
        assert record.result is not None
        oracle = _oracle(series, spec, record.start_ord, record.end_ord)
        _assert_equal_fits(record.result, oracle, record.start_ord, record.end_ord)
        if record.result.exclusions:
            windows_with_exclusions += 1
            assert all(e.reason == _NOT_COMMON_DATE for e in record.result.exclusions)
            assert all(
                record.start_ord <= e.date_ord < record.end_ord
                for e in record.result.exclusions
            )
        # The re-alignment records OUTSIDE_WINDOW for the observations before
        # and after the window; the slice holds none of them.
        assert any(e.reason == _OUTSIDE_WINDOW for e in oracle.exclusions)
        assert not any(e.reason == _OUTSIDE_WINDOW for e in record.result.exclusions)
    # Non-vacuity: the exclusion slicing is exercised.
    assert windows_with_exclusions >= 20


def test_t4_the_window_exclusions_are_the_slice_of_the_input_exclusions() -> None:
    levels, _, spec = _credit()
    result = rolling_fit(levels, spec, window=90, move=37)
    total = 0
    for record in result.windows:
        assert record.result is not None
        expected = tuple(
            e
            for e in levels.exclusions
            if record.start_ord <= e.date_ord < record.end_ord
        )
        assert record.result.exclusions == expected
        total += len(expected)
    assert total > 0


def test_t4_synthetic_windows_equal_the_oracle_for_hac_and_strides() -> None:
    levels, series = _levels(8, 200, ("Y", "A", "B"), ("EUR", "USD", "USD"))
    hac = CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=2)
    spec = _spec(
        "Y", (_term("A"), _term("B")), intercept=False, step=3, stride=2, covariance=hac
    )
    result = rolling_fit(levels, spec, window=60, move=25)
    assert len(result.windows) == 6
    for record in result.windows:
        assert record.result is not None
        oracle = _oracle(series, spec, record.start_ord, record.end_ord)
        _assert_equal_fits(record.result, oracle, record.start_ord, record.end_ord)


def test_t4_no_caller_array_is_changed_and_no_array_holds_nan() -> None:
    levels, _, spec = _credit()
    dates_before = levels.date_ord.copy()
    levels_before = levels.levels.copy()
    result = rolling_fit(levels, spec, window=120, move=20)
    assert _same_bits(levels.date_ord, dates_before)
    assert _same_bits(levels.levels, levels_before)
    for record in result.windows:
        assert record.result is not None
        for field in dataclasses.fields(OlsResult):
            value = getattr(record.result, field.name)
            if isinstance(value, np.ndarray):
                assert not value.flags.writeable
                if value.dtype == np.float64:
                    assert bool(np.isfinite(value).all())


# ================================================================ S1.T5


def test_t5_too_few_rows_is_recorded_per_window() -> None:
    # window = 3 log-return rows: R = 2 < k + 1 = 3 (intercept and X).
    levels, _ = _levels(9, 30, ("Y", "X"), ("EUR", "EUR"))
    result = rolling_fit(levels, _spec("Y", (_term("X"),)), window=3, move=9)
    assert len(result.windows) == 4
    for record in result.windows:
        assert record.result is None
        assert record.error_type == "InsufficientObservationsError"
        assert record.error_message is not None
        with pytest.raises(InsufficientObservationsError) as info:
            fit_ols(
                AlignedLevels(
                    names=levels.names,
                    units=levels.units,
                    date_ord=levels.date_ord[
                        record.start_ord - _BASE_ORD : record.end_ord - _BASE_ORD
                    ],
                    levels=levels.levels[
                        :, record.start_ord - _BASE_ORD : record.end_ord - _BASE_ORD
                    ],
                    exclusions=(),
                    start_ord=record.start_ord,
                    end_ord=record.end_ord,
                ),
                _spec("Y", (_term("X"),)),
            )
        assert record.error_message == str(info.value)


def _levels_with_a_flat_regressor_span(
    flat_start: int, flat_stop: int, m: int = 60
) -> AlignedLevels:
    rng = np.random.default_rng(10)
    dates = (_BASE_ORD + np.arange(m, dtype=np.int64)).astype(np.int32)
    y = np.exp(np.cumsum(0.01 * rng.standard_normal(m)))
    x = np.exp(np.cumsum(0.01 * rng.standard_normal(m)))
    x[flat_start:flat_stop] = x[flat_start]
    series = (
        ObservedSeries(name="Y", unit="EUR", date_ord=dates, values=y),
        ObservedSeries(name="X", unit="EUR", date_ord=dates, values=x),
    )
    return align_levels(series, start_ord=_BASE_ORD, end_ord=_BASE_ORD + m)


def test_t5_a_constant_regressor_window_is_recorded_and_the_others_continue() -> None:
    # X is flat on aligned indices 20 to 39: the window [20, 40) has zero log
    # returns of X (RANK_ZERO_COLUMN); the windows [0, 20) and [40, 60) fit.
    levels = _levels_with_a_flat_regressor_span(20, 40)
    spec = _spec("Y", (_term("X"),))
    result = rolling_fit(levels, spec, window=20, move=20)
    assert [w.error_type for w in result.windows] == [None, "RankDeficiencyError", None]
    failed = result.windows[1]
    assert failed.result is None
    assert failed.error_message is not None and "X" in failed.error_message
    with pytest.raises(RankDeficiencyError) as info:
        fit_ols(
            AlignedLevels(
                names=levels.names,
                units=levels.units,
                date_ord=levels.date_ord[20:40],
                levels=levels.levels[:, 20:40],
                exclusions=(),
                start_ord=failed.start_ord,
                end_ord=failed.end_ord,
            ),
            spec,
        )
    assert failed.error_message == str(info.value)
    assert failed.error_type == type(info.value).__name__
    for record in (result.windows[0], result.windows[2]):
        assert isinstance(record.result, OlsResult)
        assert record.error_message is None


def test_t5_a_transform_domain_error_is_recorded_for_the_windows_that_hold_it() -> None:
    m = 40
    rng = np.random.default_rng(11)
    dates = (_BASE_ORD + np.arange(m, dtype=np.int64)).astype(np.int32)
    y = np.exp(np.cumsum(0.01 * rng.standard_normal(m)))
    x = np.exp(np.cumsum(0.01 * rng.standard_normal(m)))
    x[25] = -1.0  # a LOG_RETURN level outside its domain, aligned index 25
    series = (
        ObservedSeries(name="Y", unit="EUR", date_ord=dates, values=y),
        ObservedSeries(name="X", unit="EUR", date_ord=dates, values=x),
    )
    levels = align_levels(series, start_ord=_BASE_ORD, end_ord=_BASE_ORD + m)
    result = rolling_fit(levels, _spec("Y", (_term("X"),)), window=10, move=10)
    assert [w.error_type for w in result.windows] == [
        None,
        None,
        "TransformDomainError",
        None,
    ]
    failed = result.windows[2]
    assert failed.result is None
    with pytest.raises(TransformDomainError) as info:
        _oracle(series, _spec("Y", (_term("X"),)), failed.start_ord, failed.end_ord)
    assert failed.error_message == str(info.value)
    assert info.value.date_ord == _BASE_ORD + 25


def test_t5_an_exact_fit_is_recorded_as_degenerate() -> None:
    # Y equals X on aligned indices 20 to 39: the window [20, 40) is an exact
    # fit (DegenerateFitError); the other windows carry noise.
    m = 60
    rng = np.random.default_rng(12)
    dates = (_BASE_ORD + np.arange(m, dtype=np.int64)).astype(np.int32)
    x = np.exp(np.cumsum(0.01 * rng.standard_normal(m)))
    y = np.exp(np.cumsum(0.01 * rng.standard_normal(m)))
    y[20:40] = x[20:40]
    series = (
        ObservedSeries(name="Y", unit="EUR", date_ord=dates, values=y),
        ObservedSeries(name="X", unit="EUR", date_ord=dates, values=x),
    )
    levels = align_levels(series, start_ord=_BASE_ORD, end_ord=_BASE_ORD + m)
    result = rolling_fit(levels, _spec("Y", (_term("X"),)), window=20, move=20)
    assert [w.error_type for w in result.windows] == [None, "DegenerateFitError", None]
    with pytest.raises(DegenerateFitError):
        _oracle(
            series,
            _spec("Y", (_term("X"),)),
            result.windows[1].start_ord,
            result.windows[1].end_ord,
        )


def test_t5_any_other_error_propagates_unchanged() -> None:
    levels, _ = _levels(13, 30, ("Y", "X"), ("EUR", "EUR"))
    # A term that names an absent series: ModelSpecError from fit_ols.
    with pytest.raises(ModelSpecError, match="'Z' is not in the aligned levels"):
        rolling_fit(levels, _spec("Y", (_term("Z"),)), window=10, move=10)
    # A HAC lag above R - 1 of a window: ModelSpecError from the core.
    hac = CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=12)
    with pytest.raises(ModelSpecError):
        rolling_fit(
            levels, _spec("Y", (_term("X"),), covariance=hac), window=10, move=10
        )


def test_t5_an_error_of_a_patched_fit_ols_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels, _ = _levels(14, 30, ("Y", "X"), ("EUR", "EUR"))
    calls: list[tuple[int, int]] = []

    def _raise(window_levels: AlignedLevels, spec: ModelSpec) -> OlsResult:
        calls.append((window_levels.start_ord, window_levels.end_ord))
        raise RuntimeError("not a window error")

    monkeypatch.setattr(api, "fit_ols", _raise)
    with pytest.raises(RuntimeError, match="not a window error"):
        rolling_fit(levels, _spec("Y", (_term("X"),)), window=10, move=10)
    assert calls == [(_BASE_ORD, _BASE_ORD + 10)]


# ================================================================ S1.T6


@pytest.mark.parametrize(
    ("window", "move", "step", "stride", "max_lag"),
    [(40, 15, 1, 1, 1), (40, 15, 1, 1, 3), (45, 20, 2, 3, 2), (50, 11, 5, 5, 1)],
)
def test_t6_lagged_windows_follow_the_row_formula_and_the_oracle(
    window: int, move: int, step: int, stride: int, max_lag: int
) -> None:
    levels, series = _levels(15, 150, ("P", "A", "B"), ("EUR", "USD", "USD"))
    spec = _spec(
        "P",
        (_term("A", 0), _term("A", max_lag), _term("B", -max_lag), _term("B", 0)),
        step=step,
        stride=stride,
        max_lag=max_lag,
    )
    result = rolling_fit(levels, spec, window=window, move=move)
    assert len(result.windows) == (150 - window) // move + 1
    # Lags brief C.3: R = floor((m - 1 - h - 2L) / s) + 1 with m the window length.
    expected_rows = (window - 1 - step - 2 * max_lag) // stride + 1
    for record in result.windows:
        assert record.result is not None, record.error_message
        assert record.result.n_obs == expected_rows
        oracle = _oracle(series, spec, record.start_ord, record.end_ord)
        _assert_equal_fits(record.result, oracle, record.start_ord, record.end_ord)
        # Each window drops max_lag rows at both ends, as a single fit does.
        assert int(record.result.row_end_date_ord[0]) == (
            record.start_ord + step + max_lag
        )
        assert int(record.result.row_end_date_ord[-1]) <= (record.end_ord - 1 - max_lag)


def test_t6_a_lagged_window_too_short_for_the_lag_window_is_recorded() -> None:
    levels, _ = _levels(16, 30, ("P", "A"), ("EUR", "USD"))
    spec = _spec("P", (_term("A", 1),), max_lag=2)
    # window = 5: m - 1 = 4 < h + 2L = 5, so no row exists.
    result = rolling_fit(levels, spec, window=5, move=5)
    assert len(result.windows) == 6
    for record in result.windows:
        assert record.error_type == "InsufficientObservationsError"
        assert record.error_message is not None
        assert "SAMPLE_STEP_NOT_BELOW_COUNT" in record.error_message


# ================================================================ S1.T7


def test_t7_the_public_surface_gains_the_three_names() -> None:
    for name in ("rolling_fit", "RollingFitResult", "RollingWindow"):
        assert name in api.__all__
        assert getattr(api, name) is not None
    assert api.rolling_fit is rolling_fit
    assert api.RollingFitResult is RollingFitResult
    assert api.RollingWindow is RollingWindow
    assert len(api.__all__) == len(set(api.__all__))


def test_t7_a_window_record_holds_the_contract_exclusion_type() -> None:
    levels, _, spec = _credit()
    record = rolling_fit(levels, spec, window=120, move=20).windows[1]
    assert record.result is not None
    assert all(isinstance(e, Exclusion) for e in record.result.exclusions)
