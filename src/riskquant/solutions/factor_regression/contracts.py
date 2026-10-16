# src/riskquant/solutions/factor_regression/contracts.py
"""
Facade contract types for the factor-regression library (C.1, C.5, C.15, C.22).

All types are frozen dataclasses with ``kw_only=True`` and no defaults (O-1a).
Array-bearing types declare ``eq=False`` (D064).  The core holds no ``str``
typed data; all names, units and labels live here (C.22).

``ENGINE_VERSION`` is the stable string identifier for this library's canonical
bytes format (A.6).  It changes only when the canonical byte layout changes.
"""

from __future__ import annotations

import enum
import operator
import unicodedata
from dataclasses import dataclass
from typing import Any, Final, SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.solutions.factor_regression.errors import (
    ModelSpecError,
    SeriesContractError,
)

# Stable identifier folded into every spec_digest (A.6).
ENGINE_VERSION: Final = "riskquant.factor_regression/1"

# int64 range: step, stride and the HAC lag are stored as int64 in the spec
# bytes (A.6).
_INT64_MAX: Final = (1 << 63) - 1

# Unicode general category of control characters: C0 (U+0000 to U+001F), DEL
# (U+007F) and C1 (U+0080 to U+009F).  C.1 forbids every one of them.  \x1e and
# \x1f are the field and record separators of the canonical spec bytes (A.6);
# C1 characters such as \x85 (NEL) are line breaks in some readers.
_CONTROL_CATEGORY: Final = "Cc"


def _has_control(s: str) -> bool:
    return any(unicodedata.category(c) == _CONTROL_CATEGORY for c in s)


def _frozen_array_copy(
    owner: str, field: str, value: object, dtype: type[np.generic]
) -> NDArray[Any]:
    """
    Return a read-only copy of ``value`` after an exact type and dtype check.

    The check runs before any reduction or cast.  A subclass of
    ``numpy.ndarray`` (for example ``numpy.ma.MaskedArray``) can redefine
    reductions, and a cast can change values silently, so both raise (C.19).

    Raises
    ------
    TypeError
        ``value`` is not exactly ``numpy.ndarray``, or its dtype is not
        ``dtype`` in native byte order.
    """
    if type(value) is not np.ndarray:
        raise TypeError(
            f"{owner}.{field} must be exactly numpy.ndarray; got {type(value).__name__}"
        )
    expected = np.dtype(dtype)
    if value.dtype != expected or not value.dtype.isnative:
        raise TypeError(
            f"{owner}.{field} must have dtype {expected} in native byte order; "
            f"got {value.dtype.str}"
        )
    out = np.array(value, copy=True)
    out.setflags(write=False)
    return out


# ------------------------------------------------------------------ enums


class Transform(enum.Enum):
    """
    Row transform for one term in the design.

    The value equals the ``int`` form of the matching ``codes.py`` constant so
    that ``int(Transform.LOG_RETURN.value) == int(codes.LOG_RETURN)``.
    """

    LEVEL = int(codes.LEVEL)
    LOG_RETURN = int(codes.LOG_RETURN)
    SIMPLE_RETURN = int(codes.SIMPLE_RETURN)
    DIFFERENCE = int(codes.DIFFERENCE)


class CovarianceKind(enum.Enum):
    """
    Coefficient covariance estimator.

    The value equals the ``int`` form of the matching ``codes.py`` constant;
    it is the ``covariance_kind`` byte in the canonical spec bytes (A.6).
    """

    IID = int(codes.IID)
    HC0 = int(codes.HC0)
    HAC_BARTLETT = int(codes.HAC_BARTLETT)


# ------------------------------------------------------------------ leaf types


@dataclass(frozen=True, slots=True, kw_only=True)
class Exclusion:
    """
    One observed input level that was not kept by the alignment (C.3).

    Parameters
    ----------
    series
        Name of the series whose level was excluded.
    date_ord
        ``int32`` date ordinal of the excluded observation.
    reason
        Label name of the exclusion reason
        (``labels.EXCLUSION_NAMES`` value).
    """

    series: str
    date_ord: int
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Term:
    """
    One regression term: a series name and its transform.

    Parameters
    ----------
    series
        Name of the series to use.
    transform
        ``Transform`` enum value.
    """

    series: str
    transform: Transform

    def __post_init__(self) -> None:
        if not isinstance(self.series, str):
            raise TypeError(
                f"Term.series must be str, got {type(self.series).__name__}"
            )
        if not isinstance(self.transform, Transform):
            raise TypeError(
                f"Term.transform must be Transform, got {type(self.transform).__name__}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Diagnostic:
    """
    One diagnostic record emitted by the fit (A.4, C.11).

    Parameters
    ----------
    code
        Label name of the diagnostic (``labels.DIAGNOSTIC_NAMES`` value).
    value
        The tested quantity (e.g. ``kappa`` for
        ``CONDITION_ABOVE_THRESHOLD``).
    message
        Human-readable message from ``labels.DIAGNOSTIC_MESSAGES``.
    """

    code: str
    value: float
    message: str


# ------------------------------------------------------------------ covariance spec


@dataclass(frozen=True, slots=True, kw_only=True)
class CovarianceSpec:
    """
    Declared covariance estimator and lag (C.13).

    Parameters
    ----------
    kind
        ``CovarianceKind`` enum value.
    lag
        ``None`` for ``IID`` and ``HC0``; a non-negative ``int`` for
        ``HAC_BARTLETT``.

    Raises
    ------
    TypeError
        ``kind`` is not ``CovarianceKind``, or ``lag`` is a ``bool`` or not an
        ``int`` when required.
    ModelSpecError
        ``IID`` or ``HC0`` with a non-None lag; ``HAC_BARTLETT`` without a lag;
        ``HAC_BARTLETT`` with a negative lag or a lag above the ``int64``
        range (the lag is stored as ``int64`` in the spec bytes, A.6).
    """

    kind: CovarianceKind
    lag: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CovarianceKind):
            raise TypeError(
                f"CovarianceSpec.kind must be CovarianceKind, "
                f"got {type(self.kind).__name__}"
            )
        if self.kind in (CovarianceKind.IID, CovarianceKind.HC0):
            if self.lag is not None:
                raise ModelSpecError(
                    reason=(
                        f"covariance {self.kind.name!r} does not accept a lag; "
                        f"got lag={self.lag!r}"
                    )
                )
        else:
            # HAC_BARTLETT
            if self.lag is None:
                raise ModelSpecError(reason="HAC_BARTLETT requires a lag (int >= 0)")
            # bool is a subclass of int; reject it before the int check.
            if isinstance(self.lag, (bool, np.bool_)):
                raise TypeError("lag must be an int, not a bool")
            if not isinstance(self.lag, (int, np.integer)):
                raise TypeError(f"lag must be an int, got {type(self.lag).__name__}")
            lag_int = int(self.lag)
            if lag_int < 0:
                raise ModelSpecError(reason=f"lag must be >= 0, got {lag_int}")
            if lag_int > _INT64_MAX:
                raise ModelSpecError(
                    reason=f"lag {lag_int} exceeds int64 range ({_INT64_MAX})"
                )
            object.__setattr__(self, "lag", lag_int)


# ------------------------------------------------------------------ aligned levels


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AlignedLevels:
    """
    Aligned level series on the common date axis, with all exclusions (C.17).

    ``eq=False`` because an auto-generated ``__eq__`` would compare arrays
    element-wise (D064 precedent).

    Parameters
    ----------
    names
        Series names, in the order they were passed to ``align_levels``.
    units
        Series units, matching ``names``.
    date_ord
        ``int32`` ``(m,)`` kept dates, strictly increasing.
    levels
        ``float64`` ``(n_series, m)`` aligned levels; C-order.
    exclusions
        All exclusion records from the alignment (C.3).
    start_ord, end_ord
        The declared half-open window ``[start_ord, end_ord)``.

    The constructor stores read-only copies of ``date_ord`` and ``levels``.

    Raises
    ------
    TypeError
        ``date_ord`` or ``levels`` is not exactly ``numpy.ndarray`` with its
        declared dtype.  No cast is applied.
    """

    names: tuple[str, ...]
    units: tuple[str, ...]
    date_ord: NDArray[np.int32]
    levels: NDArray[np.float64]
    exclusions: tuple[Exclusion, ...]
    start_ord: int
    end_ord: int

    def __post_init__(self) -> None:
        d = _frozen_array_copy("AlignedLevels", "date_ord", self.date_ord, np.int32)
        lev = _frozen_array_copy("AlignedLevels", "levels", self.levels, np.float64)
        object.__setattr__(self, "date_ord", d)
        object.__setattr__(self, "levels", lev)


# ------------------------------------------------------------------ model spec


def _check_int_param(name: str, value: object) -> int:
    """
    Reject ``bool`` and non-integer; return ``int``.

    ``bool`` is a subclass of ``int`` and ``operator.index(True) == 1``.
    A flag is not a step or stride, so it is rejected before the index call.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an int, not a bool")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None


def _check_float_param(name: str, value: object) -> float:
    """Reject ``bool`` and non-numeric; return ``float``."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a float, not a bool")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"{name} must be a float, got {type(value).__name__}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSpec:
    """
    Declared model specification (C.5).

    Every field is required (``kw_only=True``, no defaults).  Type violations
    raise ``TypeError``; domain violations raise ``ModelSpecError``.

    Parameters
    ----------
    response
        Response ``Term``.
    regressors
        Tuple of regressor ``Term`` values.
    intercept
        ``True`` to include a constant column in the design.
    step
        Look-back length ``h >= 0``.  ``0`` selects LEVEL rows.
    stride
        Sampling stride ``s >= 1``.
    covariance
        ``CovarianceSpec`` for the coefficient covariance estimator.
    rank_rtol
        Relative rank tolerance in ``(0, 1)``.  No default.
    condition_threshold
        When not ``None``: flag the diagnostic
        ``CONDITION_ABOVE_THRESHOLD`` if ``kappa > condition_threshold``.
        Must be strictly above 1.

    Raises
    ------
    TypeError
        A field has the wrong Python type (bool/float/str where int is
        needed, ``regressors`` not a tuple, etc.).
    ModelSpecError
        Empty design, ``step < 0``, ``stride < 1``, step 0 with a return
        term, step > 0 with only LEVEL terms, ``rank_rtol`` outside (0, 1)
        or NaN, ``condition_threshold`` not None and not above 1 (NaN
        included), a lag with IID or HC0, ``HAC_BARTLETT`` without a lag, a
        negative lag, or step/stride above the ``int64`` range.
    """

    response: Term
    regressors: tuple[Term, ...]
    intercept: bool
    step: int
    stride: int
    covariance: CovarianceSpec
    rank_rtol: float
    condition_threshold: float | None

    def __post_init__(self) -> None:
        # ---- response -------------------------------------------------------
        if not isinstance(self.response, Term):
            raise TypeError(
                f"response must be Term, got {type(self.response).__name__}"
            )

        # ---- regressors: a tuple only; no conversion ------------------------
        regs = self.regressors
        if not isinstance(regs, tuple):
            raise TypeError(
                f"regressors must be a tuple of Term, got {type(regs).__name__}"
            )
        for i, reg in enumerate(regs):
            if not isinstance(reg, Term):
                raise TypeError(
                    f"regressors[{i}] must be Term, got {type(reg).__name__}"
                )

        # ---- intercept ------------------------------------------------------
        if not isinstance(self.intercept, (bool, np.bool_)):
            raise TypeError(
                f"intercept must be bool, got {type(self.intercept).__name__}"
            )
        object.__setattr__(self, "intercept", bool(self.intercept))

        # ---- step -----------------------------------------------------------
        step = _check_int_param("step", self.step)
        if step > _INT64_MAX:
            raise ModelSpecError(
                reason=f"step {step} exceeds int64 range ({_INT64_MAX})"
            )
        object.__setattr__(self, "step", step)

        # ---- stride ---------------------------------------------------------
        stride = _check_int_param("stride", self.stride)
        if stride > _INT64_MAX:
            raise ModelSpecError(
                reason=f"stride {stride} exceeds int64 range ({_INT64_MAX})"
            )
        object.__setattr__(self, "stride", stride)

        # ---- covariance (CovarianceSpec validates itself) -------------------
        if not isinstance(self.covariance, CovarianceSpec):
            raise TypeError(
                f"covariance must be CovarianceSpec, "
                f"got {type(self.covariance).__name__}"
            )

        # ---- rank_rtol ------------------------------------------------------
        rank_rtol = _check_float_param("rank_rtol", self.rank_rtol)
        object.__setattr__(self, "rank_rtol", rank_rtol)

        # ---- condition_threshold --------------------------------------------
        ct = self.condition_threshold
        if ct is not None:
            ct_float = _check_float_param("condition_threshold", ct)
            object.__setattr__(self, "condition_threshold", ct_float)

        # ---- domain rules (ModelSpecError) ----------------------------------
        _step = int(self.step)
        _stride = int(self.stride)
        _intercept = bool(self.intercept)
        _regs = tuple(self.regressors)
        _rank_rtol = float(self.rank_rtol)
        _ct = self.condition_threshold

        # Empty design: no regressors and no intercept.
        if len(_regs) == 0 and not _intercept:
            raise ModelSpecError(
                reason="empty design: no regressors and intercept=False"
            )

        # step < 0
        if _step < 0:
            raise ModelSpecError(reason=f"step must be >= 0, got {_step}")

        # stride < 1
        if _stride < 1:
            raise ModelSpecError(reason=f"stride must be >= 1, got {_stride}")

        # step 0 with a return term: any term that is not LEVEL is a return.
        _all_terms = (self.response, *_regs)
        _return_transforms = {
            Transform.LOG_RETURN,
            Transform.SIMPLE_RETURN,
            Transform.DIFFERENCE,
        }
        if _step == 0:
            for term in _all_terms:
                if term.transform in _return_transforms:
                    raise ModelSpecError(
                        reason=(
                            f"step=0 selects LEVEL rows but term "
                            f"{term.series!r} uses {term.transform.name}"
                        )
                    )

        # step > 0 with only LEVEL terms.
        if _step > 0 and all(t.transform == Transform.LEVEL for t in _all_terms):
            raise ModelSpecError(
                reason=(
                    f"step={_step} > 0 expects at least one return transform "
                    "but all terms use LEVEL"
                )
            )

        # rank_rtol outside (0, 1).  The negated interval test also rejects
        # NaN: every comparison with NaN is False.
        if not (0.0 < _rank_rtol < 1.0):
            raise ModelSpecError(
                reason=f"rank_rtol must be in (0, 1), got {_rank_rtol!r}"
            )

        # condition_threshold not None and not above 1.  The negated test also
        # rejects NaN: every comparison with NaN is False.
        if _ct is not None and not (float(_ct) > 1.0):
            raise ModelSpecError(
                reason=(
                    f"condition_threshold must be above 1 when not None, "
                    f"got {float(_ct)!r}"
                )
            )


# ------------------------------------------------------------------ observed series


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ObservedSeries:
    """
    A named, unit-bearing level series aligned to a strictly increasing date axis
    (C.1).

    ``eq=False`` because an auto-generated ``__eq__`` would compare arrays
    element-wise (D064 precedent).

    Parameters
    ----------
    name
        Non-empty str without a character of Unicode category ``Cc``
        (C0 including ``\\x1e`` and ``\\x1f``, DEL, C1 including ``\\x85``).
    unit
        Non-empty str without a character of Unicode category ``Cc``.
    date_ord
        ``int32`` 1-D array, strictly increasing.  Exactly ``numpy.ndarray``;
        subclasses (including ``numpy.ma.MaskedArray``) raise
        ``SeriesContractError``.
    values
        ``float64`` 1-D array, all finite, same length as ``date_ord``.

    The constructor copies both arrays with ``np.array(x, copy=True)`` and
    marks them read-only.

    Raises
    ------
    SeriesContractError
        Any C.1 violation.  For a non-finite value the error carries the
        ``date_ord`` of the first offending element.
    """

    name: str
    unit: str
    date_ord: NDArray[np.int32]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        # ---- name -----------------------------------------------------------
        if not isinstance(self.name, str):
            raise SeriesContractError(
                series=repr(self.name),
                message="name must be str",
            )
        if len(self.name) == 0:
            raise SeriesContractError(
                series=self.name,
                message="name must not be empty",
            )
        if _has_control(self.name):
            raise SeriesContractError(
                series=self.name,
                message="name must not contain control characters",
            )

        # ---- unit -----------------------------------------------------------
        if not isinstance(self.unit, str):
            raise SeriesContractError(
                series=self.name,
                message="unit must be str",
            )
        if len(self.unit) == 0:
            raise SeriesContractError(
                series=self.name,
                message="unit must not be empty",
            )
        if _has_control(self.unit):
            raise SeriesContractError(
                series=self.name,
                message="unit must not contain control characters",
            )

        # ---- exact type check before any reduction (owner decision 2) -------
        # A subclass can redefine comparisons and reductions.
        # numpy.ma.MaskedArray.all() skips masked elements, so a NaN or
        # duplicate date under the mask passes silently.
        for arr, label in ((self.date_ord, "date_ord"), (self.values, "values")):
            if type(arr) is not np.ndarray:
                raise SeriesContractError(
                    series=self.name,
                    message=(
                        f"{label} must be exactly numpy.ndarray; "
                        f"got {type(arr).__name__}"
                    ),
                )

        # ---- dtype ----------------------------------------------------------
        if self.date_ord.dtype != np.int32:
            raise SeriesContractError(
                series=self.name,
                message=(f"date_ord must have dtype int32, got {self.date_ord.dtype}"),
            )
        if self.values.dtype != np.float64:
            raise SeriesContractError(
                series=self.name,
                message=(f"values must have dtype float64, got {self.values.dtype}"),
            )

        # ---- ndim -----------------------------------------------------------
        if self.date_ord.ndim != 1:
            raise SeriesContractError(
                series=self.name,
                message=f"date_ord must be 1-D, got ndim={self.date_ord.ndim}",
            )
        if self.values.ndim != 1:
            raise SeriesContractError(
                series=self.name,
                message=f"values must be 1-D, got ndim={self.values.ndim}",
            )

        # ---- length ---------------------------------------------------------
        n = int(self.date_ord.shape[0])
        if int(self.values.shape[0]) != n:
            raise SeriesContractError(
                series=self.name,
                message=(
                    f"date_ord length {n} != values length {int(self.values.shape[0])}"
                ),
            )
        if n == 0:
            raise SeriesContractError(
                series=self.name,
                message="date_ord and values must have at least one element",
            )

        # ---- strictly increasing --------------------------------------------
        steps_up = self.date_ord[1:] > self.date_ord[:-1]
        if not bool(steps_up.all()):
            first = int(np.flatnonzero(~steps_up)[0]) + 1
            raise SeriesContractError(
                series=self.name,
                message=(
                    f"date_ord not strictly increasing: "
                    f"position {first} is not above position {first - 1}"
                ),
            )

        # ---- finite values --------------------------------------------------
        finite = np.isfinite(self.values)
        if not bool(finite.all()):
            idx = int(np.flatnonzero(~finite)[0])
            bad_date = int(self.date_ord[idx])
            raise SeriesContractError(
                series=self.name,
                message=f"non-finite value {self.values[idx]!r}",
                date_ord=bad_date,
            )

        # ---- copy and freeze (owner decision 2: np.array(x, copy=True)) ----
        d = np.array(self.date_ord, copy=True)
        v = np.array(self.values, copy=True)
        d.setflags(write=False)
        v.setflags(write=False)
        object.__setattr__(self, "date_ord", d)
        object.__setattr__(self, "values", v)


# ------------------------------------------------------------------ OLS result


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class OlsResult:
    """
    Full result of one OLS fit (C.15).

    ``eq=False`` because an auto-generated ``__eq__`` would compare arrays
    element-wise (D064 precedent).

    Array fields are read-only copies; no caller array is retained (C.19).
    No field is named ``accepted``, ``approved`` or ``valid``: the library
    approves no model assumption (C.15).

    Fields follow the order of C.15 as specified by the owner.

    Raises
    ------
    TypeError
        An array field is not exactly ``numpy.ndarray`` with its declared
        dtype (``float64`` values, ``int32`` row dates).  No cast is applied.
    """

    coef: NDArray[np.float64]
    se: NDArray[np.float64]
    t_stats: NDArray[np.float64]
    p_values: NDArray[np.float64]
    cov: NDArray[np.float64]
    resid: NDArray[np.float64]
    fitted: NDArray[np.float64]
    dof: int
    n_obs: int
    rss: float
    r_squared: float
    durbin_watson: float
    condition_number: float
    singular_values: NDArray[np.float64]
    diagnostics: tuple[Diagnostic, ...]
    column_names: tuple[str, ...]
    units: tuple[str, ...]
    transforms: tuple[Transform | None, ...]
    response: Term
    window: tuple[int, int]
    step: int
    stride: int
    overlap_order: int
    row_start_date_ord: NDArray[np.int32]
    row_end_date_ord: NDArray[np.int32]
    exclusions: tuple[Exclusion, ...]
    covariance: CovarianceSpec
    rank_rtol: float
    condition_threshold: float | None
    sample_digest: str
    spec_digest: str
    row_digest: str
    engine_version: str

    def __post_init__(self) -> None:
        # Check, copy and freeze every array field.  A wrong dtype raises; a
        # cast would change the values that the digests describe.
        for _name in (
            "coef",
            "se",
            "t_stats",
            "p_values",
            "cov",
            "resid",
            "fitted",
            "singular_values",
        ):
            _arr = _frozen_array_copy(
                "OlsResult", _name, getattr(self, _name), np.float64
            )
            object.__setattr__(self, _name, _arr)
        for _name in ("row_start_date_ord", "row_end_date_ord"):
            _arr_i = _frozen_array_copy(
                "OlsResult", _name, getattr(self, _name), np.int32
            )
            object.__setattr__(self, _name, _arr_i)


# ------------------------------------------------------------------ Wald result


@dataclass(frozen=True, slots=True, kw_only=True)
class WaldResult:
    """
    Result of one joint Wald test (C.18, O-8).

    No arrays; ``eq`` uses the dataclass default (field-by-field comparison).

    Fields follow the order declared by the owner.
    """

    w: float
    f: float
    chi2_pvalue: float
    f_pvalue: float
    n_restrictions: int
    dof: int
    sample_digest: str
    spec_digest: str
    row_digest: str
    restriction_digest: str
    engine_version: str
