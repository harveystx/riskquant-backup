# tests/factor_regression/test_errors.py
"""
S1.T9: the core exceptions subclass ``RiskQuantError`` and carry no text but
the message (C.20, C.22).
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable

import numpy as np
import pytest

from riskquant.exceptions import RiskQuantError
from riskquant.factor_regression import codes, errors

# One factory per class. The coverage test below fails when errors.py gains a
# class without a factory here, so the payload scan cannot go stale.
_FACTORIES: dict[str, Callable[[], BaseException]] = {
    "CoreInputError": lambda: errors.CoreInputError(
        reason=codes.INPUT_NON_FINITE, operand=2, positions=(7,)
    ),
    "CoreAlignmentError": lambda: errors.CoreAlignmentError(
        reason=codes.ALIGN_EMPTY_INTERSECTION, start_ord=10, end_ord=20
    ),
    "CoreParameterError": lambda: errors.CoreParameterError(
        reason=codes.PARAM_NEGATIVE_STEP, value=-1.0
    ),
    "CoreDomainError": lambda: errors.CoreDomainError(
        column=1, transform=codes.LOG_RETURN, aligned_index=4, level=0.0
    ),
    "CoreRankError": lambda: errors.CoreRankError(
        reason=codes.RANK_DEFICIENT,
        n_columns=3,
        rank=2,
        singular_values=np.array([2.0, 1.0, 0.0]),
        participation=np.array([0.5, 0.5, 0.0]),
        involved=np.array([0, 1], dtype=np.int32),
        relations=np.array([[1.0, -1.0, 0.0]]),
    ),
    "CoreSampleSizeError": lambda: errors.CoreSampleSizeError(
        reason=codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS, n_available=2, n_required=3
    ),
    "CoreDegenerateFitError": lambda: errors.CoreDegenerateFitError(
        reason=codes.DEGENERATE_EXACT_FIT, value=0.0, bound=1e-20, columns=()
    ),
    "CoreRestrictionError": lambda: errors.CoreRestrictionError(
        reason=codes.RESTRICTION_WRONG_SHAPE, shape=(2, 3)
    ),
}

_EXPECTED_CLASSES = {
    "CoreInputError",
    "CoreAlignmentError",
    "CoreParameterError",
    "CoreDomainError",
    "CoreRankError",
    "CoreSampleSizeError",
    "CoreDegenerateFitError",
    "CoreRestrictionError",
}


def _defined_classes() -> dict[str, type]:
    return {
        name: obj
        for name, obj in vars(errors).items()
        if inspect.isclass(obj) and obj.__module__ == errors.__name__
    }


def _mentions_str(hint: object) -> bool:
    if hint is str:
        return True
    return any(_mentions_str(arg) for arg in typing.get_args(hint))


def test_the_module_defines_exactly_the_core_exception_set() -> None:
    assert set(_defined_classes()) == _EXPECTED_CLASSES
    assert set(_FACTORIES) == _EXPECTED_CLASSES


@pytest.mark.parametrize("name", sorted(_EXPECTED_CLASSES))
def test_every_core_exception_subclasses_riskquant_error(name: str) -> None:
    cls = _defined_classes()[name]
    assert issubclass(cls, RiskQuantError)
    assert isinstance(_FACTORIES[name](), RiskQuantError)


@pytest.mark.parametrize("name", sorted(_EXPECTED_CLASSES))
def test_no_core_exception_declares_a_str_field_or_parameter(name: str) -> None:
    cls = _defined_classes()[name]
    namespace = {"np": np, "NDArray": np.ndarray}
    hints = typing.get_type_hints(cls, globalns=vars(errors), localns=namespace)
    init_hints = typing.get_type_hints(
        vars(cls)["__init__"], globalns=vars(errors), localns=namespace
    )
    assert hints, f"{name} declares no payload field"
    for field, hint in {**hints, **init_hints}.items():
        assert not _mentions_str(hint), f"{name}.{field} is str-typed"


@pytest.mark.parametrize("name", sorted(_EXPECTED_CLASSES))
def test_the_message_is_the_only_text_on_the_instance(name: str) -> None:
    exc = _FACTORIES[name]()
    assert len(exc.args) == 1
    assert isinstance(exc.args[0], str)
    for field, value in vars(exc).items():
        assert not isinstance(value, str), f"{name}.{field} holds str data"
        if isinstance(value, tuple):
            assert all(isinstance(item, int) for item in value), field
        elif isinstance(value, np.ndarray):
            assert value.dtype.kind in "fi", field
        else:
            assert type(value) in (int, float), field


def test_reason_codes_are_stored_as_python_int() -> None:
    exc = _FACTORIES["CoreInputError"]()
    assert isinstance(exc, errors.CoreInputError)
    assert type(exc.reason) is int
    assert exc.reason == int(codes.INPUT_NON_FINITE)


def test_rank_error_arrays_are_read_only_copies() -> None:
    singular_values = np.array([2.0, 1.0, 0.0])
    involved = np.array([0, 1], dtype=np.int32)
    exc = errors.CoreRankError(
        reason=codes.RANK_DEFICIENT,
        n_columns=3,
        rank=2,
        singular_values=singular_values,
        participation=np.array([0.5, 0.5, 0.0]),
        involved=involved,
        relations=np.array([[1.0, -1.0, 0.0]]),
    )
    singular_values[0] = 99.0
    involved[0] = 5
    assert exc.singular_values.tolist() == [2.0, 1.0, 0.0]
    assert exc.involved.tolist() == [0, 1]
    for array in (
        exc.singular_values,
        exc.participation,
        exc.involved,
        exc.relations,
    ):
        assert not array.flags.writeable
    assert exc.involved.dtype == np.int32


@pytest.mark.parametrize(
    "group",
    [
        codes.TRANSFORM_CODES,
        codes.COVARIANCE_CODES,
        codes.EXCLUSION_CODES,
        codes.DIAGNOSTIC_CODES,
        codes.INPUT_ERROR_CODES,
        codes.ALIGNMENT_ERROR_CODES,
        codes.PARAMETER_ERROR_CODES,
        codes.RANK_ERROR_CODES,
        codes.SAMPLE_SIZE_ERROR_CODES,
        codes.DEGENERATE_FIT_ERROR_CODES,
        codes.RESTRICTION_ERROR_CODES,
    ],
)
def test_every_code_group_is_int8_and_unique(group: tuple[np.int8, ...]) -> None:
    assert group
    assert all(type(code) is np.int8 for code in group)
    assert len({int(code) for code in group}) == len(group)


def test_covariance_codes_match_the_canonical_spec_bytes() -> None:
    # A.6 fixes these values inside spec_digest.
    assert (int(codes.IID), int(codes.HC0), int(codes.HAC_BARTLETT)) == (0, 1, 2)
