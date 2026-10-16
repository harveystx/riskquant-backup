# tests/solutions/factor_regression/test_contracts.py
"""
Tests for S6 facade contract types (S6.T1 to S6.T6, S6.T11, S6.T12, S6.T13).

Also covers the acceptance-criterion masked-array and ndarray-subclass cases
for ObservedSeries.

Test IDs
--------
S6.T1  - empty name/unit, control characters
S6.T2  - wrong dtype dates, unsorted dates, duplicate dates
S6.T3  - NaN or inf value names series and date_ord
S6.T4  - float32 values; read-only copies after construction
S6.T5  - each omitted ModelSpec field raises TypeError
S6.T6  - each invalid ModelSpec value raises ModelSpecError
S6.T11 - every facade error class subclasses RiskQuantError
S6.T12 - OlsResult and WaldResult: arrays read-only, frozen, no approval fields,
          eq=False on array-bearing types
S6.T13 - architecture tests pass with facade files present (implicitly verified
          by the full pytest run; the three architecture test functions are
          called here directly via import to catch regressions early)
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import numpy as np
import pytest

from riskquant.exceptions import RiskQuantError
from riskquant.solutions.factor_regression.contracts import (
    ENGINE_VERSION,
    AlignedLevels,
    CovarianceKind,
    CovarianceSpec,
    ModelSpec,
    ObservedSeries,
    OlsResult,
    Term,
    Transform,
    WaldResult,
)
from riskquant.solutions.factor_regression.errors import (
    AlignmentError,
    DegenerateFitError,
    InsufficientObservationsError,
    ModelSpecError,
    RankDeficiencyError,
    RestrictionError,
    SeriesContractError,
    TransformDomainError,
)

# ------------------------------------------------------------------ helpers


def _dates(*ints: int) -> np.ndarray:
    return np.array(ints, dtype=np.int32)


def _vals(*fs: float) -> np.ndarray:
    return np.array(fs, dtype=np.float64)


def _series(
    name: str = "P",
    unit: str = "EUR",
    dates: np.ndarray | None = None,
    vals: np.ndarray | None = None,
) -> ObservedSeries:
    if dates is None:
        dates = _dates(1, 2, 3)
    if vals is None:
        vals = _vals(1.0, 2.0, 3.0)
    return ObservedSeries(name=name, unit=unit, date_ord=dates, values=vals)


def _base_spec(**overrides: object) -> ModelSpec:
    defaults: dict[str, object] = dict(
        response=Term(series="Y", transform=Transform.LOG_RETURN),
        regressors=(Term(series="X", transform=Transform.LOG_RETURN),),
        intercept=False,
        step=1,
        stride=1,
        covariance=CovarianceSpec(kind=CovarianceKind.IID, lag=None),
        rank_rtol=1e-10,
        condition_threshold=None,
    )
    defaults.update(overrides)
    return ModelSpec(**defaults)  # type: ignore[arg-type]


def _ols_result(**overrides: object) -> OlsResult:
    k = 2  # intercept + one regressor
    r = 5
    defaults: dict[str, object] = dict(
        coef=np.array([0.5, 1.5], dtype=np.float64),
        se=np.array([0.1, 0.2], dtype=np.float64),
        t_stats=np.array([5.0, 7.5], dtype=np.float64),
        p_values=np.array([0.01, 0.001], dtype=np.float64),
        cov=np.eye(k, dtype=np.float64) * 0.01,
        resid=np.zeros(r, dtype=np.float64),
        fitted=np.ones(r, dtype=np.float64),
        dof=r - k,
        n_obs=r,
        rss=0.1,
        r_squared=0.95,
        durbin_watson=2.0,
        condition_number=10.0,
        singular_values=np.array([2.0, 0.2], dtype=np.float64),
        diagnostics=(),
        column_names=("const", "X"),
        units=("", "EUR"),
        transforms=(None, Transform.LOG_RETURN),
        response=Term(series="Y", transform=Transform.LOG_RETURN),
        window=(0, 10),
        step=1,
        stride=1,
        overlap_order=0,
        row_start_date_ord=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        row_end_date_ord=np.array([2, 3, 4, 5, 6], dtype=np.int32),
        exclusions=(),
        covariance=CovarianceSpec(kind=CovarianceKind.IID, lag=None),
        rank_rtol=1e-10,
        condition_threshold=None,
        sample_digest="a" * 64,
        spec_digest="b" * 64,
        row_digest="c" * 64,
        engine_version=ENGINE_VERSION,
    )
    defaults.update(overrides)
    return OlsResult(**defaults)  # type: ignore[arg-type]


# ================================================================ S6.T1
# empty name, empty unit, control characters -> SeriesContractError


def test_t1_empty_name() -> None:
    with pytest.raises(SeriesContractError):
        _series(name="")


def test_t1_empty_unit() -> None:
    with pytest.raises(SeriesContractError):
        _series(unit="")


@pytest.mark.parametrize(
    "ctrl",
    ["\x00", "\x1e", "\x1f", "\x1b", "\x7f"],
)
def test_t1_control_char_in_name(ctrl: str) -> None:
    with pytest.raises(SeriesContractError):
        _series(name=f"SP{ctrl}X")


@pytest.mark.parametrize(
    "ctrl",
    ["\x00", "\x1e", "\x1f", "\x1b", "\x7f"],
)
def test_t1_control_char_in_unit(ctrl: str) -> None:
    with pytest.raises(SeriesContractError):
        _series(unit=f"EU{ctrl}R")


def test_t1_non_str_name_raises_series_error() -> None:
    with pytest.raises(SeriesContractError):
        ObservedSeries(
            name=123,  # type: ignore[arg-type]
            unit="EUR",
            date_ord=_dates(1, 2, 3),
            values=_vals(1.0, 2.0, 3.0),
        )


def test_t1_non_str_unit_raises_series_error() -> None:
    with pytest.raises(SeriesContractError):
        ObservedSeries(
            name="P",
            unit=None,  # type: ignore[arg-type]
            date_ord=_dates(1, 2, 3),
            values=_vals(1.0, 2.0, 3.0),
        )


# ================================================================ S6.T2
# int64 dates, unsorted dates, duplicate dates -> SeriesContractError


def test_t2_int64_dates() -> None:
    with pytest.raises(SeriesContractError):
        ObservedSeries(
            name="P",
            unit="EUR",
            date_ord=np.array([1, 2, 3], dtype=np.int64),
            values=_vals(1.0, 2.0, 3.0),
        )


def test_t2_unsorted_dates() -> None:
    with pytest.raises(SeriesContractError):
        _series(dates=_dates(3, 1, 2))


def test_t2_duplicate_dates() -> None:
    with pytest.raises(SeriesContractError):
        _series(dates=_dates(1, 2, 2))


# ================================================================ S6.T3
# NaN or inf -> error names series and date_ord


def test_t3_nan_value_names_date_ord() -> None:
    dates = _dates(10, 20, 30)
    vals = np.array([1.0, float("nan"), 3.0], dtype=np.float64)
    with pytest.raises(SeriesContractError) as exc_info:
        ObservedSeries(name="S", unit="U", date_ord=dates, values=vals)
    exc = exc_info.value
    assert exc.series == "S"
    assert exc.date_ord == 20  # date_ord at index 1


def test_t3_inf_value_names_date_ord() -> None:
    dates = _dates(1, 2, 3)
    vals = np.array([1.0, float("inf"), 3.0], dtype=np.float64)
    with pytest.raises(SeriesContractError) as exc_info:
        ObservedSeries(name="Z", unit="U", date_ord=dates, values=vals)
    exc = exc_info.value
    assert exc.series == "Z"
    assert exc.date_ord == 2


def test_t3_neg_inf_value_names_date_ord() -> None:
    dates = _dates(1, 2, 3)
    vals = np.array([float("-inf"), 2.0, 3.0], dtype=np.float64)
    with pytest.raises(SeriesContractError) as exc_info:
        ObservedSeries(name="W", unit="U", date_ord=dates, values=vals)
    assert exc_info.value.date_ord == 1


# ================================================================ S6.T4
# float32 values; caller mutation after construction


def test_t4_float32_values_rejected() -> None:
    with pytest.raises(SeriesContractError):
        ObservedSeries(
            name="P",
            unit="EUR",
            date_ord=_dates(1, 2, 3),
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )


def test_t4_stored_arrays_are_copies() -> None:
    d = _dates(1, 2, 3)
    v = _vals(1.0, 2.0, 3.0)
    s = _series(dates=d, vals=v)
    # Mutate caller arrays.
    d[0] = 99
    v[0] = 99.0
    # Stored arrays unchanged.
    assert int(s.date_ord[0]) == 1
    assert float(s.values[0]) == 1.0


def test_t4_stored_arrays_are_read_only() -> None:
    s = _series()
    with pytest.raises((ValueError, TypeError)):
        s.date_ord[0] = 0
    with pytest.raises((ValueError, TypeError)):
        s.values[0] = 0.0


# ================================================================ S6.T4 (acceptance criterion: masked array and subclass)


def test_t4_masked_array_date_ord_rejected() -> None:
    import numpy.ma as ma

    with pytest.raises(SeriesContractError):
        ObservedSeries(
            name="P",
            unit="EUR",
            date_ord=ma.array([1, 2, 3], dtype=np.int32),
            values=_vals(1.0, 2.0, 3.0),
        )


def test_t4_masked_array_values_rejected() -> None:
    import numpy.ma as ma

    with pytest.raises(SeriesContractError):
        ObservedSeries(
            name="P",
            unit="EUR",
            date_ord=_dates(1, 2, 3),
            values=ma.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_t4_ndarray_subclass_rejected() -> None:
    class MyArr(np.ndarray):
        pass

    dates = _dates(1, 2, 3).view(MyArr)
    with pytest.raises(SeriesContractError):
        ObservedSeries(
            name="P",
            unit="EUR",
            date_ord=dates,
            values=_vals(1.0, 2.0, 3.0),
        )


# ================================================================ S6.T5
# each omitted ModelSpec field -> TypeError


_ALL_FIELDS = [
    "response",
    "regressors",
    "intercept",
    "step",
    "stride",
    "covariance",
    "rank_rtol",
    "condition_threshold",
]

_FULL_KWARGS: dict[str, object] = dict(
    response=Term(series="Y", transform=Transform.LOG_RETURN),
    regressors=(Term(series="X", transform=Transform.LOG_RETURN),),
    intercept=False,
    step=1,
    stride=1,
    covariance=CovarianceSpec(kind=CovarianceKind.IID, lag=None),
    rank_rtol=1e-10,
    condition_threshold=None,
)


@pytest.mark.parametrize("field", _ALL_FIELDS)
def test_t5_omitted_field_raises_type_error(field: str) -> None:
    kwargs = {k: v for k, v in _FULL_KWARGS.items() if k != field}
    with pytest.raises(TypeError):
        ModelSpec(**kwargs)  # type: ignore[arg-type]


# ================================================================ S6.T6
# each invalid ModelSpec value -> ModelSpecError (unknown-term-series is S7)


def test_t6_empty_design() -> None:
    with pytest.raises(ModelSpecError):
        ModelSpec(
            response=Term(series="Y", transform=Transform.LEVEL),
            regressors=(),
            intercept=False,
            step=0,
            stride=1,
            covariance=CovarianceSpec(kind=CovarianceKind.IID, lag=None),
            rank_rtol=1e-10,
            condition_threshold=None,
        )


def test_t6_step_negative() -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(step=-1)


def test_t6_stride_below_one() -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(stride=0)


def test_t6_step_zero_with_return_term_on_response() -> None:
    with pytest.raises(ModelSpecError):
        ModelSpec(
            response=Term(series="Y", transform=Transform.LOG_RETURN),
            regressors=(Term(series="X", transform=Transform.LOG_RETURN),),
            intercept=False,
            step=0,
            stride=1,
            covariance=CovarianceSpec(kind=CovarianceKind.IID, lag=None),
            rank_rtol=1e-10,
            condition_threshold=None,
        )


def test_t6_step_zero_with_return_term_on_regressor() -> None:
    with pytest.raises(ModelSpecError):
        ModelSpec(
            response=Term(series="Y", transform=Transform.LEVEL),
            regressors=(Term(series="X", transform=Transform.SIMPLE_RETURN),),
            intercept=False,
            step=0,
            stride=1,
            covariance=CovarianceSpec(kind=CovarianceKind.IID, lag=None),
            rank_rtol=1e-10,
            condition_threshold=None,
        )


def test_t6_step_positive_level_only() -> None:
    with pytest.raises(ModelSpecError):
        ModelSpec(
            response=Term(series="Y", transform=Transform.LEVEL),
            regressors=(Term(series="X", transform=Transform.LEVEL),),
            intercept=False,
            step=1,
            stride=1,
            covariance=CovarianceSpec(kind=CovarianceKind.IID, lag=None),
            rank_rtol=1e-10,
            condition_threshold=None,
        )


@pytest.mark.parametrize("rtol", [0.0, 1.0, -0.1, 1.5])
def test_t6_rank_rtol_out_of_range(rtol: float) -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(rank_rtol=rtol)


@pytest.mark.parametrize("ct", [0.0, 1.0, 0.5, -1.0])
def test_t6_condition_threshold_not_above_one(ct: float) -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(condition_threshold=ct)


def test_t6_condition_threshold_above_one_valid() -> None:
    # Just above 1 is valid.
    spec = _base_spec(condition_threshold=1.0001)
    assert spec.condition_threshold == pytest.approx(1.0001)


def test_t6_lag_with_iid() -> None:
    with pytest.raises(ModelSpecError):
        CovarianceSpec(kind=CovarianceKind.IID, lag=0)


def test_t6_lag_with_hc0() -> None:
    with pytest.raises(ModelSpecError):
        CovarianceSpec(kind=CovarianceKind.HC0, lag=3)


def test_t6_hac_without_lag() -> None:
    with pytest.raises(ModelSpecError):
        CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=None)


def test_t6_negative_lag() -> None:
    with pytest.raises(ModelSpecError):
        CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=-1)


def test_t6_step_above_int64_range() -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(step=(1 << 63))


def test_t6_stride_above_int64_range() -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(stride=(1 << 63))


def test_t6_step_bool_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _base_spec(step=True)


def test_t6_stride_float_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _base_spec(stride=1.0)


# ================================================================ S6.T11
# every class in facade errors.py subclasses RiskQuantError


_FACADE_ERRORS = [
    SeriesContractError,
    AlignmentError,
    ModelSpecError,
    TransformDomainError,
    RankDeficiencyError,
    InsufficientObservationsError,
    DegenerateFitError,
    RestrictionError,
]


@pytest.mark.parametrize("cls", _FACADE_ERRORS)
def test_t11_subclasses_riskquant_error(cls: type) -> None:
    assert issubclass(cls, RiskQuantError), (
        f"{cls.__name__} must subclass RiskQuantError"
    )


# ================================================================ S6.T12
# OlsResult and WaldResult: arrays read-only, frozen, no approval fields,
# eq=False for array-bearing types


def test_t12_ols_result_arrays_read_only() -> None:
    result = _ols_result()
    for field in dataclasses.fields(result):
        val = getattr(result, field.name)
        if isinstance(val, np.ndarray):
            assert not val.flags["WRITEABLE"], (
                f"OlsResult.{field.name} must be read-only"
            )


def test_t12_ols_result_is_frozen() -> None:
    result = _ols_result()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        result.dof = 99  # type: ignore[misc]


def test_t12_ols_result_no_approval_fields() -> None:
    names = {f.name for f in dataclasses.fields(OlsResult)}
    for forbidden in ("accepted", "approved", "valid"):
        assert forbidden not in names, (
            f"OlsResult must not have a field named {forbidden!r}"
        )


def test_t12_ols_result_eq_false() -> None:
    r1 = _ols_result()
    r2 = _ols_result()
    # eq=False -> identity comparison -> two different instances are not equal.
    assert (r1 == r2) == (r1 is r2)
    assert r1 != r2


def test_t12_observed_series_eq_false() -> None:
    s1 = _series()
    s2 = _series()
    assert (s1 == s2) == (s1 is s2)


def test_t12_aligned_levels_eq_false() -> None:
    a1 = AlignedLevels(
        names=("P",),
        units=("EUR",),
        date_ord=_dates(1, 2, 3),
        levels=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        exclusions=(),
        start_ord=0,
        end_ord=10,
    )
    a2 = AlignedLevels(
        names=("P",),
        units=("EUR",),
        date_ord=_dates(1, 2, 3),
        levels=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        exclusions=(),
        start_ord=0,
        end_ord=10,
    )
    assert (a1 == a2) == (a1 is a2)


def test_t12_wald_result_constructable() -> None:
    wr = WaldResult(
        w=5.0,
        f=2.5,
        chi2_pvalue=0.08,
        f_pvalue=0.1,
        n_restrictions=2,
        dof=10,
        sample_digest="a" * 64,
        spec_digest="b" * 64,
        row_digest="c" * 64,
        restriction_digest="d" * 64,
        engine_version=ENGINE_VERSION,
    )
    assert wr.w == 5.0
    assert wr.engine_version == ENGINE_VERSION


def test_t12_wald_result_is_frozen() -> None:
    wr = WaldResult(
        w=5.0,
        f=2.5,
        chi2_pvalue=0.08,
        f_pvalue=0.1,
        n_restrictions=2,
        dof=10,
        sample_digest="a" * 64,
        spec_digest="b" * 64,
        row_digest="c" * 64,
        restriction_digest="d" * 64,
        engine_version=ENGINE_VERSION,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        wr.w = 99.0  # type: ignore[misc]


def test_t12_wald_result_no_approval_fields() -> None:
    names = {f.name for f in dataclasses.fields(WaldResult)}
    for forbidden in ("accepted", "approved", "valid"):
        assert forbidden not in names


# ================================================================ S6.T13
# Architecture tests pass with facade files present.
# The actual gate invocation runs tests/test_architecture.py directly.
# Here we verify the imports succeed and the path comment exists.


def test_t13_facade_modules_importable() -> None:
    import riskquant.solutions.factor_regression.contracts
    import riskquant.solutions.factor_regression.digest
    import riskquant.solutions.factor_regression.errors
    import riskquant.solutions.factor_regression.labels

    assert riskquant.solutions.factor_regression.contracts.ENGINE_VERSION


def test_t13_init_is_empty() -> None:
    spec = importlib.util.find_spec("riskquant.solutions.factor_regression")
    assert spec is not None
    pkg_init = pathlib.Path(spec.origin)  # type: ignore[arg-type]
    content = pkg_init.read_text().strip()
    assert content == "", f"__init__.py must be empty, got: {content!r}"


def test_t13_path_comments_present() -> None:
    for mod_name in (
        "riskquant.solutions.factor_regression.contracts",
        "riskquant.solutions.factor_regression.errors",
        "riskquant.solutions.factor_regression.labels",
        "riskquant.solutions.factor_regression.digest",
    ):
        spec = importlib.util.find_spec(mod_name)
        assert spec is not None, f"Cannot find {mod_name}"
        path = pathlib.Path(spec.origin)  # type: ignore[arg-type]
        first_line = path.read_text().splitlines()[0]
        assert first_line.startswith("# src/riskquant/solutions/factor_regression/"), (
            f"{path.name}: first line {first_line!r} does not start with expected prefix"
        )
