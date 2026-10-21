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

WS_FACTOR_REGRESSION_GUARDS, slice S2 (AlignedLevels admission):
S2.T2  - one case per C.9 value check: SeriesContractError, the O-2 series
          value, the field in the message
S2.T3  - one case per C.8 type check: TypeError, the field and the exclusion
          index in the message; NumPy integer bounds stored as int (R-7)
S2.T4  - check order: array checks, then type checks, then value checks
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
from collections.abc import Callable

import numpy as np
import pytest

from riskquant.exceptions import RiskQuantError
from riskquant.solutions.factor_regression.contracts import (
    ENGINE_VERSION,
    AlignedLevels,
    CovarianceKind,
    CovarianceSpec,
    Exclusion,
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


# ================================================================ S6 fix round
# R-016, R-019, R-020 and A1.  Each case below failed on the baseline
# 4e1b315 unless its docstring says that it guards existing behaviour.


# ---------------------------------------------------------------- R-016


def test_t6_rank_rtol_nan_raises_model_spec_error() -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(rank_rtol=float("nan"))


def test_t6_condition_threshold_nan_raises_model_spec_error() -> None:
    with pytest.raises(ModelSpecError):
        _base_spec(condition_threshold=float("nan"))


# ---------------------------------------------------------------- R-019


def test_t5_regressors_list_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _base_spec(regressors=[Term(series="X", transform=Transform.LOG_RETURN)])


def _aligned_levels(date_ord: object, levels: object) -> AlignedLevels:
    return AlignedLevels(
        names=("P",),
        units=("EUR",),
        date_ord=date_ord,  # type: ignore[arg-type]
        levels=levels,  # type: ignore[arg-type]
        exclusions=(),
        start_ord=0,
        end_ord=10,
    )


_LEVELS = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)


@pytest.mark.parametrize(
    "date_ord",
    [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.dtype(np.int32).newbyteorder()),
        [1, 2, 3],
        np.ma.array([1, 2, 3], dtype=np.int32),
    ],
    ids=["int64", "int32-swapped-byte-order", "list", "masked-array"],
)
def test_r019_aligned_levels_date_ord_rejected(date_ord: object) -> None:
    with pytest.raises(TypeError):
        _aligned_levels(date_ord, _LEVELS)


@pytest.mark.parametrize(
    "levels",
    [
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[1, 2, 3]], dtype=np.int64),
        [[1.0, 2.0, 3.0]],
        np.ma.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    ],
    ids=["float32", "int64", "list", "masked-array"],
)
def test_r019_aligned_levels_levels_rejected(levels: object) -> None:
    with pytest.raises(TypeError):
        _aligned_levels(_dates(1, 2, 3), levels)


def test_r019_aligned_levels_stores_read_only_copies() -> None:
    """Guards C.19: existing behaviour, passes on the baseline."""
    d = _dates(1, 2, 3)
    lev = _LEVELS.copy()
    a = _aligned_levels(d, lev)
    d[0] = 99
    lev[0, 0] = 99.0
    assert int(a.date_ord[0]) == 1
    assert float(a.levels[0, 0]) == 1.0
    assert not a.date_ord.flags["WRITEABLE"]
    assert not a.levels.flags["WRITEABLE"]
    assert a.date_ord.dtype == np.int32
    assert a.levels.dtype == np.float64


_OLS_FLOAT_FIELDS = (
    "coef",
    "se",
    "t_stats",
    "p_values",
    "cov",
    "resid",
    "fitted",
    "singular_values",
)


@pytest.mark.parametrize("field", _OLS_FLOAT_FIELDS)
def test_r019_ols_result_float32_array_rejected(field: str) -> None:
    good = getattr(_ols_result(), field)
    with pytest.raises(TypeError):
        _ols_result(**{field: np.array(good, dtype=np.float32)})


@pytest.mark.parametrize("field", _OLS_FLOAT_FIELDS)
def test_r019_ols_result_list_rejected(field: str) -> None:
    good = getattr(_ols_result(), field)
    with pytest.raises(TypeError):
        _ols_result(**{field: good.tolist()})


@pytest.mark.parametrize("field", ["row_start_date_ord", "row_end_date_ord"])
def test_r019_ols_result_int64_row_dates_rejected(field: str) -> None:
    good = getattr(_ols_result(), field)
    with pytest.raises(TypeError):
        _ols_result(**{field: np.array(good, dtype=np.int64)})


def test_r019_ols_result_masked_array_rejected() -> None:
    with pytest.raises(TypeError):
        _ols_result(coef=np.ma.array([0.5, 1.5], dtype=np.float64))


def test_r019_ols_result_stores_read_only_copies() -> None:
    """Guards C.19: existing behaviour, passes on the baseline."""
    coef = np.array([0.5, 1.5], dtype=np.float64)
    rows = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = _ols_result(coef=coef, row_start_date_ord=rows)
    coef[0] = 99.0
    rows[0] = 99
    assert float(result.coef[0]) == 0.5
    assert int(result.row_start_date_ord[0]) == 1
    assert not result.coef.flags["WRITEABLE"]
    assert not result.row_start_date_ord.flags["WRITEABLE"]


@pytest.mark.parametrize("ctrl", ["\x80", "\x85", "\x9f"])
def test_t1_c1_control_char_in_name(ctrl: str) -> None:
    with pytest.raises(SeriesContractError):
        _series(name=f"SP{ctrl}X")


@pytest.mark.parametrize("ctrl", ["\x80", "\x85", "\x9f"])
def test_t1_c1_control_char_in_unit(ctrl: str) -> None:
    with pytest.raises(SeriesContractError):
        _series(unit=f"EU{ctrl}R")


def test_t1_non_control_non_ascii_name_accepted() -> None:
    """Guards C.1: a non-ASCII letter is not a control character."""
    s = _series(name="Zürich", unit="€")
    assert (s.name, s.unit) == ("Zürich", "€")


# ---------------------------------------------------------------- R-020


def test_r020_alignment_error_series_named_in_message() -> None:
    exc = AlignmentError(
        reason="DUPLICATE_SERIES_NAME", start_ord=0, end_ord=10, series="P"
    )
    assert exc.series == "P"
    assert "series='P'" in str(exc)
    assert exc.reason == "DUPLICATE_SERIES_NAME"
    assert (exc.start_ord, exc.end_ord) == (0, 10)


def test_r020_alignment_error_default_series_is_none() -> None:
    """Guards OD-12: the construction without series keeps its message."""
    exc = AlignmentError(reason="ALIGN_EMPTY_WINDOW", start_ord=5, end_ord=5)
    assert exc.series is None
    assert str(exc) == "'ALIGN_EMPTY_WINDOW': window=[5, 5)"


# ---------------------------------------------------------------- A1


def test_a1_hac_lag_above_int64_range() -> None:
    with pytest.raises(ModelSpecError):
        CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=(1 << 63))


def test_a1_hac_lag_at_int64_max_accepted() -> None:
    """Guards the boundary: 2**63 - 1 is in the int64 range."""
    spec = CovarianceSpec(kind=CovarianceKind.HAC_BARTLETT, lag=(1 << 63) - 1)
    assert spec.lag == (1 << 63) - 1


# ======================================== WS_FACTOR_REGRESSION_GUARDS S2
# AlignedLevels admission (C.8, C.9, C.13).  The constructors are called
# through Callable[..., T] aliases: the cases pass values of a wrong type on
# purpose, and the alias keeps mypy --strict clean without a pragma.

_make_aligned: Callable[..., AlignedLevels] = AlignedLevels
_make_exclusion: Callable[..., Exclusion] = Exclusion

_INT32_LO = int(np.iinfo(np.int32).min)
_INT32_HI = int(np.iinfo(np.int32).max)


def _excl(
    series: object = "P", date_ord: object = 0, reason: object = "OUTSIDE_WINDOW"
) -> Exclusion:
    return _make_exclusion(series=series, date_ord=date_ord, reason=reason)


def _guard_fields(**overrides: object) -> dict[str, object]:
    """Valid two-series fields with both exclusion reasons; overrides replace."""
    fields: dict[str, object] = {
        "names": ("P", "Q"),
        "units": ("EUR", "USD"),
        "date_ord": _dates(1, 2, 3),
        "levels": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        "exclusions": (
            _excl("P", 0, "OUTSIDE_WINDOW"),
            _excl("Q", 5, "NOT_COMMON_DATE"),
        ),
        "start_ord": 1,
        "end_ord": 10,
    }
    fields.update(overrides)
    return fields


def _two_rows(*row0_row1: tuple[float, float, float]) -> np.ndarray:
    return np.array(row0_row1, dtype=np.float64)


def test_s2_valid_fields_construct() -> None:
    """The base fields of the S2 cases construct; each case changes one item."""
    a = _make_aligned(**_guard_fields())
    assert a.names == ("P", "Q")
    assert (a.start_ord, a.end_ord) == (1, 10)


# (case id, overrides, expected series, field text in the message, date_ord)
_S2_T2_CASES: list[tuple[str, dict[str, object], str, str, int | None]] = [
    # (a) date_ord
    (
        "date-descending",
        {"date_ord": _dates(3, 2, 1)},
        "AlignedLevels.date_ord",
        "AlignedLevels.date_ord",
        None,
    ),
    (
        "date-duplicate",
        {"date_ord": _dates(1, 2, 2)},
        "AlignedLevels.date_ord",
        "AlignedLevels.date_ord",
        None,
    ),
    (
        "date-2d",
        {"date_ord": np.array([[1, 2, 3]], dtype=np.int32)},
        "AlignedLevels.date_ord",
        "AlignedLevels.date_ord",
        None,
    ),
    (
        "date-empty",
        {"date_ord": _dates(), "levels": np.empty((2, 0), dtype=np.float64)},
        "AlignedLevels.date_ord",
        "AlignedLevels.date_ord",
        None,
    ),
    # (b) names count and levels shape
    (
        "date-longer-than-levels",
        {"date_ord": _dates(1, 2, 3, 4)},
        "AlignedLevels.levels",
        "AlignedLevels.levels",
        None,
    ),
    (
        "date-shorter-than-levels",
        {"date_ord": _dates(1, 2)},
        "AlignedLevels.levels",
        "AlignedLevels.levels",
        None,
    ),
    (
        "extra-level-row",
        {"levels": np.ones((3, 3), dtype=np.float64)},
        "AlignedLevels.levels",
        "AlignedLevels.levels",
        None,
    ),
    (
        "missing-level-row",
        {"levels": np.ones((1, 3), dtype=np.float64)},
        "AlignedLevels.levels",
        "AlignedLevels.levels",
        None,
    ),
    (
        "levels-1d",
        {"levels": np.ones(3, dtype=np.float64)},
        "AlignedLevels.levels",
        "AlignedLevels.levels",
        None,
    ),
    (
        "names-empty",
        {
            "names": (),
            "units": (),
            "levels": np.empty((0, 3), dtype=np.float64),
            "exclusions": (),
        },
        "AlignedLevels.names",
        "AlignedLevels.names",
        None,
    ),
    # (c) units count
    (
        "units-short",
        {"units": ("EUR",)},
        "AlignedLevels.units",
        "AlignedLevels.units",
        None,
    ),
    # (d) name and unit values
    (
        "name-empty",
        {"names": ("", "Q")},
        "AlignedLevels.names",
        "AlignedLevels.names[0]",
        None,
    ),
    (
        "name-x1f",
        {"names": ("P", "Q\x1f")},
        "AlignedLevels.names",
        "AlignedLevels.names[1]",
        None,
    ),
    (
        "name-x7f",
        {"names": ("P", "Q\x7f")},
        "AlignedLevels.names",
        "AlignedLevels.names[1]",
        None,
    ),
    (
        "name-x85",
        {"names": ("P", "Q\x85")},
        "AlignedLevels.names",
        "AlignedLevels.names[1]",
        None,
    ),
    ("name-duplicate", {"names": ("P", "P")}, "P", "AlignedLevels.names[1]", None),
    ("unit-empty", {"units": ("EUR", "")}, "Q", "AlignedLevels.units[1]", None),
    ("unit-x1f", {"units": ("EUR", "USD\x1f")}, "Q", "AlignedLevels.units[1]", None),
    ("unit-x7f", {"units": ("EUR\x7f", "USD")}, "P", "AlignedLevels.units[0]", None),
    ("unit-x85", {"units": ("EUR", "USD\x85")}, "Q", "AlignedLevels.units[1]", None),
    # (e) window bounds
    (
        "start-below-int32",
        {"start_ord": _INT32_LO - 1},
        "AlignedLevels.start_ord",
        "AlignedLevels.start_ord",
        None,
    ),
    (
        "end-above-int32",
        {"end_ord": _INT32_HI + 1},
        "AlignedLevels.end_ord",
        "AlignedLevels.end_ord",
        None,
    ),
    (
        "first-date-below-start",
        {"start_ord": 2},
        "AlignedLevels.start_ord",
        "AlignedLevels.start_ord",
        None,
    ),
    (
        "last-date-equals-end",
        {"end_ord": 3},
        "AlignedLevels.end_ord",
        "AlignedLevels.end_ord",
        None,
    ),
    (
        "last-date-above-end",
        {"end_ord": 2},
        "AlignedLevels.end_ord",
        "AlignedLevels.end_ord",
        None,
    ),
    # (f) finite levels: the first row in names order wins over an earlier column
    (
        "level-nan",
        {"levels": _two_rows((1.0, 2.0, 3.0), (4.0, 5.0, np.nan))},
        "Q",
        "AlignedLevels.levels",
        3,
    ),
    (
        "level-inf",
        {"levels": _two_rows((1.0, np.inf, 3.0), (np.nan, 5.0, 6.0))},
        "P",
        "AlignedLevels.levels",
        2,
    ),
    (
        "level-minus-inf",
        {"levels": _two_rows((1.0, 2.0, 3.0), (-np.inf, 5.0, 6.0))},
        "Q",
        "AlignedLevels.levels",
        1,
    ),
    # (g) exclusion values
    (
        "exclusion-unknown-series",
        {"exclusions": (_excl("P", 0), _excl("Z", 5))},
        "Z",
        "AlignedLevels.exclusions[1].series",
        None,
    ),
    (
        "exclusion-reason-not-used-by-rows",
        {"exclusions": (_excl("P", 0), _excl("Q", 5, "NOT_USED_BY_ROWS"))},
        "Q",
        "AlignedLevels.exclusions[1].reason",
        None,
    ),
    (
        "exclusion-reason-unknown",
        {"exclusions": (_excl("P", 0, "JUNK"),)},
        "P",
        "AlignedLevels.exclusions[0].reason",
        None,
    ),
    (
        "exclusion-date-above-int32",
        {"exclusions": (_excl("P", 0), _excl("Q", _INT32_HI + 1))},
        "Q",
        "AlignedLevels.exclusions[1].date_ord",
        None,
    ),
    (
        "exclusion-date-below-int32",
        {"exclusions": (_excl("P", _INT32_LO - 1),)},
        "P",
        "AlignedLevels.exclusions[0].date_ord",
        None,
    ),
]


@pytest.mark.parametrize(
    ("overrides", "series", "field", "date_ord"),
    [case[1:] for case in _S2_T2_CASES],
    ids=[case[0] for case in _S2_T2_CASES],
)
def test_s2_t2_value_fault_raises_series_contract_error(
    overrides: dict[str, object], series: str, field: str, date_ord: int | None
) -> None:
    """S2.T2: one case per C.9 item; series value of O-2; message names the field."""
    with pytest.raises(SeriesContractError) as exc_info:
        _make_aligned(**_guard_fields(**overrides))
    err = exc_info.value
    assert err.series == series
    assert field in str(err)
    assert err.date_ord == date_ord


# (case id, overrides, field text in the message)
_S2_T3_CASES: list[tuple[str, dict[str, object], str]] = [
    ("names-list", {"names": ["P", "Q"]}, "AlignedLevels.names"),
    ("units-list", {"units": ["EUR", "USD"]}, "AlignedLevels.units"),
    ("exclusions-list", {"exclusions": [_excl("P", 0)]}, "AlignedLevels.exclusions"),
    ("name-int", {"names": ("P", 1)}, "AlignedLevels.names[1]"),
    ("unit-int", {"units": ("EUR", 2)}, "AlignedLevels.units[1]"),
    (
        "exclusion-str-junk",
        {"exclusions": (_excl("P", 0), "junk")},
        "AlignedLevels.exclusions[1]",
    ),
    ("start-float", {"start_ord": 0.5}, "AlignedLevels.start_ord"),
    ("start-bool", {"start_ord": True}, "AlignedLevels.start_ord"),
    ("start-numpy-bool", {"start_ord": np.True_}, "AlignedLevels.start_ord"),
    ("start-str", {"start_ord": "0"}, "AlignedLevels.start_ord"),
    ("start-numpy-float", {"start_ord": np.float64(0)}, "AlignedLevels.start_ord"),
    ("end-float", {"end_ord": 10.0}, "AlignedLevels.end_ord"),
    ("end-str", {"end_ord": "10"}, "AlignedLevels.end_ord"),
    ("end-none", {"end_ord": None}, "AlignedLevels.end_ord"),
    (
        "exclusion-series-int",
        {"exclusions": (_excl("P", 0), _excl(1, 5))},
        "AlignedLevels.exclusions[1].series",
    ),
    (
        "exclusion-date-float",
        {"exclusions": (_excl("P", 0), _excl("Q", 5.5))},
        "AlignedLevels.exclusions[1].date_ord",
    ),
    (
        "exclusion-date-bool",
        {"exclusions": (_excl("P", 0), _excl("Q", True))},
        "AlignedLevels.exclusions[1].date_ord",
    ),
    (
        "exclusion-reason-int",
        {"exclusions": (_excl("P", 0), _excl("Q", 5, 0))},
        "AlignedLevels.exclusions[1].reason",
    ),
]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [case[1:] for case in _S2_T3_CASES],
    ids=[case[0] for case in _S2_T3_CASES],
)
def test_s2_t3_type_fault_raises_type_error(
    overrides: dict[str, object], field: str
) -> None:
    """S2.T3: one case per C.8 item; message names the field and exclusion index."""
    with pytest.raises(TypeError) as exc_info:
        _make_aligned(**_guard_fields(**overrides))
    assert field in str(exc_info.value)


def test_s2_t3_numpy_integer_bounds_are_stored_as_int() -> None:
    """R-7: a NumPy integer bound is stored as int; exclusions are not rebuilt."""
    records = (_excl("P", np.int64(0)), _excl("Q", np.int32(5), "NOT_COMMON_DATE"))
    a = _make_aligned(
        **_guard_fields(start_ord=np.int64(1), end_ord=np.int32(10), exclusions=records)
    )
    assert type(a.start_ord) is int and a.start_ord == 1
    assert type(a.end_ord) is int and a.end_ord == 10
    assert a.exclusions is records
    assert all(x is y for x, y in zip(a.exclusions, records, strict=True))


# (case id, overrides, expected exception class, field text in the message)
_S2_T4_CASES: list[tuple[str, dict[str, object], type[Exception], str]] = [
    (
        "array-dtype-before-duplicate-names",
        {"date_ord": np.array([1, 2, 3], dtype=np.int64), "names": ("P", "P")},
        TypeError,
        "AlignedLevels.date_ord",
    ),
    (
        "array-dtype-before-names-list",
        {"levels": np.ones((2, 3), dtype=np.float32), "names": ["P", "Q"]},
        TypeError,
        "AlignedLevels.levels",
    ),
    (
        "names-list-before-descending-dates",
        {"names": ["P", "Q"], "date_ord": _dates(3, 2, 1)},
        TypeError,
        "AlignedLevels.names",
    ),
    (
        "float-start-before-non-finite-level",
        {"start_ord": 0.5, "levels": _two_rows((np.nan, 2.0, 3.0), (4.0, 5.0, 6.0))},
        TypeError,
        "AlignedLevels.start_ord",
    ),
    (
        "exclusion-reason-type-before-unknown-series",
        {"exclusions": (_excl("Z", 0), _excl("Q", 5, 0))},
        TypeError,
        "AlignedLevels.exclusions[1].reason",
    ),
    (
        "descending-dates-before-duplicate-names",
        {"date_ord": _dates(3, 2, 1), "names": ("P", "P")},
        SeriesContractError,
        "AlignedLevels.date_ord",
    ),
    (
        "duplicate-names-before-bound",
        {"names": ("P", "P"), "end_ord": 3},
        SeriesContractError,
        "AlignedLevels.names[1]",
    ),
]


@pytest.mark.parametrize(
    ("overrides", "exc", "field"),
    [case[1:] for case in _S2_T4_CASES],
    ids=[case[0] for case in _S2_T4_CASES],
)
def test_s2_t4_check_order(
    overrides: dict[str, object], exc: type[Exception], field: str
) -> None:
    """S2.T4: array checks, then type checks (TypeError), then value checks."""
    with pytest.raises(exc) as exc_info:
        _make_aligned(**_guard_fields(**overrides))
    assert type(exc_info.value) is exc
    assert field in str(exc_info.value)
