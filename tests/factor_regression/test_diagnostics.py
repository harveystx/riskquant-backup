# tests/factor_regression/test_diagnostics.py
"""
S3.T6, S3.T8, S3.T9 and S3.T10: Durbin-Watson and the integer-coded
diagnostics (A.4, C.10, C.11).

Every oracle is independent of ``diagnostics.py`` (A.7): the hand value
``12 / 4 = 3``, a ``fractions.Fraction`` loop for the statistic, and the A.4
rules written out per case. The overlap order is ``math.ceil(h / s) - 1`` for
``h >= 1`` (A.1, owner decision 4); the test does not import ``rows.py``.

The threshold is a float plus a bool (owner decision 3): the bool False stands
for the facade value None.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.diagnostics import (
    CoreDiagnostics,
    diagnose,
    durbin_watson,
)
from riskquant.factor_regression.errors import (
    CoreDegenerateFitError,
    CoreInputError,
    CoreParameterError,
)

LEVEL = int(codes.LEVEL)
LOG = int(codes.LOG_RETURN)
SIMPLE = int(codes.SIMPLE_RETURN)
DIFF = int(codes.DIFFERENCE)
IID = int(codes.IID)
HC0 = int(codes.HC0)
HAC = int(codes.HAC_BARTLETT)
CONDITION = int(codes.CONDITION_ABOVE_THRESHOLD)
LEVEL_PRESENT = int(codes.LEVEL_TERM_PRESENT)
R2_ABOVE_DW = int(codes.R2_ABOVE_DURBIN_WATSON)
OVERLAP = int(codes.OVERLAP_NOT_CORRECTED)


# ------------------------------------------------------------------ helpers


def _run(**overrides: Any) -> CoreDiagnostics:
    """Diagnose with neutral inputs: no rule fires unless an override fires it."""
    arguments: dict[str, Any] = {
        "term_transforms": np.array([LOG, LOG, LOG], dtype=np.int8),
        "condition_number": 50.0,
        "condition_threshold": 0.0,
        "apply_condition_threshold": False,
        "r_squared": 0.9,
        "durbin_watson_value": 0.5,
        "overlap_order": 0,
        "covariance_kind": IID,
        "covariance_lag": 0,
    }
    arguments.update(overrides)
    return diagnose(**arguments)


def _pairs(result: CoreDiagnostics) -> list[tuple[int, float]]:
    return list(zip(result.codes.tolist(), result.values.tolist(), strict=True))


def _overlap_order(h: int, s: int) -> int:
    return 0 if h == 0 else math.ceil(h / s) - 1


def _fraction_durbin_watson(resid: list[float]) -> Fraction:
    exact = [Fraction(e) for e in resid]
    numerator = sum(
        ((exact[t] - exact[t - 1]) ** 2 for t in range(1, len(exact))), Fraction(0)
    )
    denominator = sum((e * e for e in exact), Fraction(0))
    return numerator / denominator


# ------------------------------------------------------------------ S3.T8


def test_the_durbin_watson_of_an_alternating_sequence_is_three() -> None:
    # (1 - -1)^2 * 3 = 12 over 1 * 4 = 4.
    assert durbin_watson(np.array([1.0, -1.0, 1.0, -1.0])) == 3.0


@pytest.mark.parametrize("seed", range(10))
def test_durbin_watson_matches_an_exact_fraction_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    resid = np.cumsum(rng.standard_normal(60)) * 0.3 + rng.standard_normal(60)
    oracle = float(_fraction_durbin_watson(resid.tolist()))
    assert abs(durbin_watson(resid) / oracle - 1.0) <= 1e-14


def test_durbin_watson_of_a_constant_and_of_one_residual() -> None:
    assert durbin_watson(np.full(5, -2.5)) == 0.0
    assert durbin_watson(np.array([7.0])) == 0.0


@pytest.mark.parametrize("factor", [2.0**-600, 2.0**-40, 2.0**40, 2.0**600])
def test_durbin_watson_is_exact_under_a_power_of_two_scale(factor: float) -> None:
    rng = np.random.default_rng(3)
    resid = rng.standard_normal(40)
    assert durbin_watson(resid * factor) == durbin_watson(resid)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_durbin_watson_survives_extreme_residual_magnitudes(magnitude: float) -> None:
    # An unscaled sum of squares overflows to inf near 1e160 and underflows
    # to 0 near 1e-170.
    rng = np.random.default_rng(4)
    resid = rng.uniform(-1.0, 1.0, 30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = durbin_watson(resid * magnitude)
    assert math.isfinite(got)
    assert abs(got / durbin_watson(resid) - 1.0) <= 1e-14


@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_all_zero_residuals_raise_exact_fit(zero: float) -> None:
    with pytest.raises(CoreDegenerateFitError) as info:
        durbin_watson(np.full(4, zero))
    assert info.value.reason == codes.DEGENERATE_EXACT_FIT
    assert (info.value.value, info.value.bound, info.value.columns) == (0.0, 0.0, ())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_residual_raises_with_its_index(bad: float) -> None:
    resid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    resid[2] = bad
    resid[4] = bad
    with pytest.raises(CoreInputError) as info:
        durbin_watson(resid)
    assert (info.value.reason, info.value.operand, info.value.positions) == (
        int(codes.INPUT_NON_FINITE),
        0,
        (2,),
    )


# ------------------------------------------------------------------ S3.T6


def test_no_threshold_gives_no_flag() -> None:
    assert _pairs(_run(apply_condition_threshold=False)) == []


def test_a_threshold_below_kappa_flags_with_kappa_as_value() -> None:
    result = _run(condition_threshold=10.0, apply_condition_threshold=True)
    assert _pairs(result) == [(CONDITION, 50.0)]


@pytest.mark.parametrize("threshold", [100.0, 50.0])
def test_a_threshold_at_or_above_kappa_gives_no_flag(threshold: float) -> None:
    result = _run(condition_threshold=threshold, apply_condition_threshold=True)
    assert _pairs(result) == []


@pytest.mark.parametrize("threshold", [1.0, 0.5, 0.0, -3.0, math.nan, -math.inf])
def test_an_applied_threshold_not_above_one_raises(threshold: float) -> None:
    with pytest.raises(CoreParameterError) as info:
        _run(condition_threshold=threshold, apply_condition_threshold=True)
    assert info.value.reason == codes.PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE
    assert (math.isnan(threshold) and math.isnan(info.value.value)) or (
        info.value.value == threshold
    )


@pytest.mark.parametrize("threshold", [1.0, 0.5, math.nan, -math.inf])
def test_a_threshold_that_does_not_apply_is_not_read(threshold: float) -> None:
    # Owner decision 3: when the bool is False, the float is ignored.
    result = _run(condition_threshold=threshold, apply_condition_threshold=False)
    assert _pairs(result) == []


def test_a_numpy_bool_flag_and_a_numpy_float_threshold_are_accepted() -> None:
    result = _run(
        condition_threshold=np.float64(10.0), apply_condition_threshold=np.True_
    )
    assert _pairs(result) == [(CONDITION, 50.0)]


@pytest.mark.parametrize(
    ("threshold", "apply"),
    [
        pytest.param(10.0, 1, id="int-flag"),
        pytest.param(10.0, None, id="none-flag"),
        pytest.param(None, False, id="none-threshold"),
        pytest.param("10", True, id="str-threshold"),
        pytest.param(True, True, id="bool-threshold"),
    ],
)
def test_a_threshold_or_flag_of_the_wrong_type_raises_type_error(
    threshold: object, apply: object
) -> None:
    with pytest.raises(TypeError):
        _run(condition_threshold=threshold, apply_condition_threshold=apply)


# ------------------------------------------------------------------ S3.T9


@pytest.mark.parametrize(
    "transforms",
    [[LEVEL, LEVEL], [LEVEL, LOG], [LOG, LEVEL, DIFF]],
    ids=["level-only", "level-response", "level-regressor"],
)
def test_a_level_term_flags_and_r2_flags_only_above_durbin_watson(
    transforms: list[int],
) -> None:
    term_transforms = np.array(transforms, dtype=np.int8)
    above = _run(
        term_transforms=term_transforms, r_squared=0.9, durbin_watson_value=0.5
    )
    assert _pairs(above) == [(LEVEL_PRESENT, 1.0), (R2_ABOVE_DW, 0.9)]
    below = _run(
        term_transforms=term_transforms, r_squared=0.3, durbin_watson_value=0.5
    )
    assert _pairs(below) == [(LEVEL_PRESENT, 1.0)]
    equal = _run(
        term_transforms=term_transforms, r_squared=0.5, durbin_watson_value=0.5
    )
    assert _pairs(equal) == [(LEVEL_PRESENT, 1.0)]


@pytest.mark.parametrize(
    "transforms", [[LOG, LOG, LOG], [SIMPLE, DIFF], [DIFF]], ids=["log", "mixed", "one"]
)
def test_no_level_term_gives_no_code_even_with_r2_above_durbin_watson(
    transforms: list[int],
) -> None:
    result = _run(
        term_transforms=np.array(transforms, dtype=np.int8),
        r_squared=0.99,
        durbin_watson_value=0.01,
    )
    assert _pairs(result) == []


# ----------------------------------------------------------------- S3.T10


@pytest.mark.parametrize(
    ("kind", "lag", "expected"),
    [
        pytest.param(IID, 0, True, id="iid"),
        pytest.param(HC0, 0, True, id="hc0"),
        pytest.param(HAC, 3, True, id="hac-lag-3"),
        pytest.param(HAC, 4, False, id="hac-lag-4"),
        pytest.param(HAC, 0, True, id="hac-lag-0"),
        pytest.param(HAC, 9, False, id="hac-lag-9"),
    ],
)
def test_overlapping_rows_flag_unless_the_hac_lag_covers_the_overlap(
    kind: int, lag: int, expected: bool
) -> None:
    q = _overlap_order(5, 1)
    assert q == 4
    result = _run(overlap_order=q, covariance_kind=kind, covariance_lag=lag)
    assert _pairs(result) == ([(OVERLAP, 4.0)] if expected else [])


@pytest.mark.parametrize(
    ("kind", "lag"), [(IID, 0), (HC0, 0), (HAC, 0), (HAC, 3)], ids=str
)
def test_non_overlapping_rows_never_flag(kind: int, lag: int) -> None:
    q = _overlap_order(5, 5)
    assert q == 0
    result = _run(overlap_order=q, covariance_kind=kind, covariance_lag=lag)
    assert _pairs(result) == []


@pytest.mark.parametrize(("h", "s", "q"), [(0, 1, 0), (1, 1, 0), (3, 1, 2), (7, 3, 2)])
def test_the_overlap_value_is_the_overlap_order(h: int, s: int, q: int) -> None:
    assert _overlap_order(h, s) == q
    result = _run(overlap_order=q, covariance_kind=HC0)
    assert _pairs(result) == ([(OVERLAP, float(q))] if q > 0 else [])


@pytest.mark.parametrize("kind", [IID, HC0])
def test_the_lag_is_not_read_for_iid_and_hc0(kind: int) -> None:
    # The C.13 lag rules belong to the fit (S4); the overlap rule reads the
    # lag only for HAC_BARTLETT.
    result = _run(overlap_order=4, covariance_kind=kind, covariance_lag=-7)
    assert _pairs(result) == [(OVERLAP, 4.0)]


# -------------------------------------------------------- all codes, C.11


def test_all_four_codes_fire_in_ascending_order_with_their_values() -> None:
    result = _run(
        term_transforms=np.array([LEVEL, LOG], dtype=np.int8),
        condition_number=1e4,
        condition_threshold=100.0,
        apply_condition_threshold=True,
        r_squared=0.95,
        durbin_watson_value=0.2,
        overlap_order=2,
        covariance_kind=HAC,
        covariance_lag=1,
    )
    assert _pairs(result) == [
        (CONDITION, 1e4),
        (LEVEL_PRESENT, 1.0),
        (R2_ABOVE_DW, 0.95),
        (OVERLAP, 2.0),
    ]
    assert result.codes.dtype == np.int8
    assert result.values.dtype == np.float64
    assert not result.codes.flags.writeable
    assert not result.values.flags.writeable
    assert result.n_diagnostics == 4
    assert set(result.codes.tolist()) <= {int(code) for code in codes.DIAGNOSTIC_CODES}


def test_an_empty_result_has_typed_empty_arrays() -> None:
    result = _run()
    assert result.codes.shape == (0,)
    assert result.values.shape == (0,)
    assert result.codes.dtype == np.int8
    assert result.values.dtype == np.float64


def test_two_diagnoses_of_the_same_inputs_are_equal_bit_for_bit() -> None:
    first = _run(overlap_order=3, term_transforms=np.array([LEVEL], dtype=np.int8))
    second = _run(overlap_order=3, term_transforms=np.array([LEVEL], dtype=np.int8))
    assert first.codes.tobytes() == second.codes.tobytes()
    assert first.values.tobytes() == second.values.tobytes()


def test_no_python_warning_is_issued() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(
            term_transforms=np.array([LEVEL, LOG], dtype=np.int8),
            condition_threshold=10.0,
            apply_condition_threshold=True,
            overlap_order=4,
        )
        durbin_watson(np.array([1.0, -2.0, 0.5]))


def test_the_term_transforms_are_not_changed() -> None:
    term_transforms = np.array([LEVEL, LOG], dtype=np.int8)
    _run(term_transforms=term_transforms)
    assert term_transforms.tolist() == [LEVEL, LOG]
    assert term_transforms.flags.writeable


def test_the_result_is_frozen() -> None:
    result = _run()
    with pytest.raises(AttributeError):
        result.codes = np.empty(0, dtype=np.int8)  # type: ignore[misc]


# ----------------------------------------------------------- parameters


@pytest.mark.parametrize("kind", [3, -1, 127])
def test_an_unknown_covariance_kind_raises(kind: int) -> None:
    with pytest.raises(CoreParameterError) as info:
        _run(covariance_kind=kind)
    assert info.value.reason == codes.PARAM_UNKNOWN_COVARIANCE
    assert info.value.value == float(kind)


def test_a_negative_hac_lag_raises() -> None:
    with pytest.raises(CoreParameterError) as info:
        _run(covariance_kind=HAC, covariance_lag=-1)
    assert info.value.reason == codes.PARAM_NEGATIVE_LAG
    assert info.value.value == -1.0


def test_a_negative_overlap_order_raises_value_error() -> None:
    with pytest.raises(ValueError, match="overlap_order"):
        _run(overlap_order=-1)


@pytest.mark.parametrize(
    "name", ["condition_number", "r_squared", "durbin_watson_value"]
)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_statistic_raises_value_error(name: str, bad: float) -> None:
    # A NaN makes every comparison False: the rule would miss silently.
    with pytest.raises(ValueError, match="finite"):
        _run(**{name: bad})


@pytest.mark.parametrize("name", ["overlap_order", "covariance_kind", "covariance_lag"])
@pytest.mark.parametrize("bad", [True, np.True_, 1.0, "1"])
def test_an_integer_parameter_of_the_wrong_type_raises_type_error(
    name: str, bad: object
) -> None:
    with pytest.raises(TypeError):
        _run(**{name: bad})


def test_numpy_integer_parameters_give_the_int_result() -> None:
    expected = _pairs(_run(overlap_order=4, covariance_kind=HAC, covariance_lag=2))
    got = _run(
        overlap_order=np.int64(4),
        covariance_kind=codes.HAC_BARTLETT,
        covariance_lag=np.int32(2),
    )
    assert _pairs(got) == expected == [(OVERLAP, 4.0)]


@pytest.mark.parametrize("code", [4, -1, 127])
def test_an_unknown_transform_code_raises(code: int) -> None:
    with pytest.raises(CoreParameterError) as info:
        _run(term_transforms=np.array([LOG, code], dtype=np.int8))
    assert info.value.reason == codes.PARAM_UNKNOWN_TRANSFORM
    assert info.value.value == float(code)


# ------------------------------------------------------- input contract


class _TrivialSubclass(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """An ndarray subclass with no override: only the exact type differs."""


def _residual_error(resid: object) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        durbin_watson(resid)  # type: ignore[arg-type]
    return info.value


def _transform_error(term_transforms: object) -> CoreInputError:
    with pytest.raises(CoreInputError) as info:
        _run(term_transforms=term_transforms)
    return info.value


def test_masked_residuals_with_nan_under_the_mask_are_rejected() -> None:
    raw = np.array([1.0, math.nan, -1.0, 2.0])
    masked: NDArray[Any] = np.ma.masked_array(raw, mask=[False, True, False, False])
    # The mask hides the NaN from a reduction.
    assert bool(np.isfinite(masked).all())
    error = _residual_error(masked)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        0,
        (),
    )


def test_masked_term_transforms_with_an_unknown_code_under_the_mask_are_rejected() -> (
    None
):
    raw = np.array([LEVEL, 99], dtype=np.int8)
    masked: NDArray[Any] = np.ma.masked_array(raw, mask=[False, True])
    error = _transform_error(masked)
    assert (error.reason, error.operand, error.positions) == (
        int(codes.INPUT_NOT_NDARRAY),
        0,
        (),
    )


def test_a_view_of_a_trivial_ndarray_subclass_is_rejected() -> None:
    view = np.array([1.0, -1.0, 2.0]).view(_TrivialSubclass)
    assert isinstance(view, np.ndarray)
    assert _residual_error(view).reason == codes.INPUT_NOT_NDARRAY


def test_the_input_contract_rejects_without_repair() -> None:
    resid = np.array([1.0, -1.0, 2.0])
    residual_cases: list[tuple[object, int]] = [
        (resid.tolist(), int(codes.INPUT_NOT_NDARRAY)),
        (resid.astype(np.float32), int(codes.INPUT_WRONG_DTYPE)),
        (resid.reshape(3, 1), int(codes.INPUT_WRONG_NDIM)),
        (resid[:0].copy(), int(codes.INPUT_EMPTY)),
    ]
    for array, reason in residual_cases:
        error = _residual_error(array)
        assert (error.reason, error.operand, error.positions) == (reason, 0, ())

    transforms = np.array([LEVEL, LOG], dtype=np.int8)
    transform_cases: list[tuple[object, int]] = [
        (transforms.tolist(), int(codes.INPUT_NOT_NDARRAY)),
        (transforms.astype(np.int32), int(codes.INPUT_WRONG_DTYPE)),
        (transforms.reshape(1, 2), int(codes.INPUT_WRONG_NDIM)),
        (transforms[:0].copy(), int(codes.INPUT_EMPTY)),
    ]
    for array, reason in transform_cases:
        error = _transform_error(array)
        assert (error.reason, error.operand, error.positions) == (reason, 0, ())
