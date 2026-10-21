# src/riskquant/solutions/factor_regression/api.py
"""
Public surface of the factor regression library (C.23, D001).

This module is the only public import surface of
``riskquant.solutions.factor_regression``. It exposes four functions:

* ``align_levels``: integer alignment of named level series (C.2, C.3).
* ``fit_ols``: rows, design, rank gate, OLS fit, diagnostics and digests
  (C.4 to C.17).
* ``wald_test``: joint Wald test on a fit (C.18).
* ``unit_root_test``: ADF unit-root test of a linear combination of aligned
  levels (C.2 to C.10).

``__all__`` also names the contract types of ``contracts.py`` and the facade
exceptions of ``errors.py``.

The facade holds no numerical logic (S7 prohibition). It orders the core
calls, builds integer arrays by indexing, maps names, labels and dates, and
raises. Every number of a result comes from the core.

Error mapping (C.20). One private function, ``_map_core_error``, maps each
core exception to its facade exception. Every facade raise of a mapped error
uses ``raise ... from core_error``, so ``__cause__`` is the core error:

* ``CoreAlignmentError`` to ``AlignmentError``.
* ``CoreScaleDomainError`` to ``ScaleDomainError`` with the series name, the
  date of the offending aligned level and the level.
* ``CoreDomainError`` to ``TransformDomainError`` with the series name, the
  transform name, the date of the offending aligned level and the level.
* ``CoreSampleSizeError`` to ``InsufficientObservationsError`` with the core
  counts. For ``SAMPLE_ROWS_NOT_ABOVE_COLUMNS`` the message names
  ``k = n_required - 1``.
* ``CoreRankError`` to ``RankDeficiencyError`` with the design column names.
  For ``RANK_ZERO_COLUMN`` the message names the zero columns and states no
  rank (owner decision OD-1).
* ``CoreDegenerateFitError`` to ``DegenerateFitError`` with the column names
  that the core gives.
* ``CoreParameterError`` to ``ModelSpecError``.
* ``CoreInputError`` with ``INPUT_NON_FINITE`` during row, design or fit work
  to ``DegenerateFitError``: a transform result overflowed to a non-finite
  number (backlog R-008). ``CoreInputError`` on the restriction or the
  right-hand side of ``wald_test`` to ``RestrictionError``.
  ``CoreInputError`` with ``INPUT_NON_FINITE`` during the ``"adf"`` stage of
  ``unit_root_test`` to ``DegenerateFitError`` with no column names: the
  combination or a difference of it overflowed. Every other
  ``CoreInputError`` is an internal contract violation and propagates
  unchanged.
* ``CoreRestrictionError`` to ``RestrictionError``.

A ``TypeError`` or ``ValueError`` from the core is not mapped: it means a
broken internal contract, not a caller error that the facade can name.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, SupportsIndex, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from riskquant.exceptions import RiskQuantError
from riskquant.factor_regression import adf, codes, combination
from riskquant.factor_regression.align import CoreAlignedLevels, align_core
from riskquant.factor_regression.design import assemble_design
from riskquant.factor_regression.diagnostics import diagnose
from riskquant.factor_regression.errors import (
    CoreAlignmentError,
    CoreDegenerateFitError,
    CoreDomainError,
    CoreInputError,
    CoreParameterError,
    CoreRankError,
    CoreRestrictionError,
    CoreSampleSizeError,
    CoreScaleDomainError,
)
from riskquant.factor_regression.fit import OPERAND_DESIGN, OPERAND_RESPONSE, fit_core
from riskquant.factor_regression.rows import build_rows
from riskquant.factor_regression.wald import wald_core
from riskquant.solutions.factor_regression import digest, labels
from riskquant.solutions.factor_regression.contracts import (
    ENGINE_VERSION,
    AlignedLevels,
    CombinationTerm,
    CovarianceKind,
    CovarianceSpec,
    Deterministic,
    Diagnostic,
    Exclusion,
    LinearCombination,
    ModelSpec,
    ObservedSeries,
    OlsResult,
    Scale,
    Term,
    Transform,
    UnitRootResult,
    UnitRootSpec,
    WaldResult,
)
from riskquant.solutions.factor_regression.errors import (
    AlignmentError,
    DegenerateFitError,
    InsufficientObservationsError,
    ModelSpecError,
    RankDeficiencyError,
    RestrictionError,
    ScaleDomainError,
    SeriesContractError,
    TransformDomainError,
)

__all__ = [
    "AlignedLevels",
    "AlignmentError",
    "CombinationTerm",
    "CovarianceKind",
    "CovarianceSpec",
    "DegenerateFitError",
    "Deterministic",
    "Diagnostic",
    "Exclusion",
    "InsufficientObservationsError",
    "LinearCombination",
    "ModelSpec",
    "ModelSpecError",
    "ObservedSeries",
    "OlsResult",
    "RankDeficiencyError",
    "RestrictionError",
    "Scale",
    "ScaleDomainError",
    "SeriesContractError",
    "Term",
    "Transform",
    "TransformDomainError",
    "UnitRootResult",
    "UnitRootSpec",
    "WaldResult",
    "align_levels",
    "fit_ols",
    "unit_root_test",
    "wald_test",
]

# Name of the intercept column of the design (S6 decision: empty unit, no
# transform).
_INTERCEPT_NAME: Final = "const"
_INTERCEPT_UNIT: Final = ""

# The lag passed to the core when the spec holds no lag (IID, HC0), and the
# lag folded into spec_digest in that case (A.6).
_CORE_NO_LAG: Final = 0
_DIGEST_NO_LAG: Final = -1

# The threshold passed to diagnose when the spec holds None. The core does not
# read it, because the apply flag is False (backlog R-011).
_UNUSED_THRESHOLD: Final = 0.0

_CORE_ERRORS: Final = (
    CoreAlignmentError,
    CoreDegenerateFitError,
    CoreDomainError,
    CoreInputError,
    CoreParameterError,
    CoreRankError,
    CoreRestrictionError,
    CoreSampleSizeError,
    CoreScaleDomainError,
)

# Operand positions of restriction and rhs in wald_core (its check order).
_WALD_OPERAND_RESTRICTION: Final = 2
_WALD_OPERAND_RHS: Final = 3

_Stage = Literal["align", "rows", "design", "fit", "wald", "combination", "adf"]

_ErrorT = TypeVar("_ErrorT", bound=BaseException)
_ScalarT = TypeVar("_ScalarT", bound=np.generic)

# AlignmentError reason of a duplicate series name. No core code exists for
# it: the facade checks the names before the core call (C.2, OD-12).
_DUPLICATE_SERIES_NAME: Final = "DUPLICATE_SERIES_NAME"


def _empty_dates() -> NDArray[np.int32]:
    return np.empty(0, dtype=np.int32)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _ErrorContext:
    """
    What ``_map_core_error`` needs to name a core error of one call.

    Attributes
    ----------
    stage
        The core call: ``align_core``, ``build_rows``, ``assemble_design``,
        ``fit_core`` (with ``diagnose``), ``wald_core``, ``combine_core``
        (stage ``"combination"``) or ``adf_core`` (stage ``"adf"``).
    term_names
        Series name of each term, response first (the ``build_rows`` term
        order).
    date_ord
        The kept aligned dates. ``CoreDomainError.aligned_index`` indexes it.
    column_names
        Design column names, intercept first (C.8 order).
    n_rows
        Row count ``R`` of the fit. It decodes the flat index of a
        non-finite regressor value of ``assemble_design``.
    restriction
        The caller restriction object of ``wald_test``, for the shape of a
        ``RestrictionError``. It is read only for that shape.
    """

    stage: _Stage
    term_names: tuple[str, ...] = ()
    date_ord: NDArray[np.int32] = field(default_factory=_empty_dates)
    column_names: tuple[str, ...] = ()
    n_rows: int = 0
    restriction: object = None


def _restated(error: _ErrorT, message: str) -> _ErrorT:
    # The S6 constructors build a fixed message. Two owner rules need another
    # text (OD-1 zero columns, k of SAMPLE_ROWS_NOT_ABOVE_COLUMNS); the fields
    # stay as constructed. str(error) reads args[0].
    error.args = (message,)
    return error


def _received_shape(value: object) -> tuple[int, ...]:
    # The shape attribute of an ndarray or an ndarray subclass, read without a
    # conversion or a reduction. Any other object has no shape to report.
    if isinstance(value, np.ndarray):
        return tuple(int(n) for n in value.shape)
    return ()


def _non_finite_column(
    core_error: CoreInputError, context: _ErrorContext
) -> tuple[str, ...]:
    """
    Name the series of the first non-finite value of a design or fit operand.

    ``assemble_design`` reports the flat index ``t * R + r`` of its regressor
    rows, so regressor ``t`` is term ``t + 1``. ``fit_core`` reports the flat
    index ``r * k + j`` of the design (operand 0) or the row index of the
    response (operand 1). Integer bookkeeping only.
    """
    if not core_error.positions:
        return ()
    position = core_error.positions[0]
    if context.stage == "design" and context.n_rows > 0:
        return (context.term_names[1 + position // context.n_rows],)
    if context.stage == "fit" and core_error.operand == OPERAND_RESPONSE:
        return (context.term_names[0],)
    if context.stage == "fit" and core_error.operand == OPERAND_DESIGN:
        return (context.column_names[position % len(context.column_names)],)
    return ()


def _map_core_error(
    core_error: RiskQuantError, context: _ErrorContext
) -> RiskQuantError | None:
    """
    Map one core exception to its facade exception (C.20).

    Returns
    -------
    RiskQuantError | None
        The facade exception, not yet raised. ``None`` when the core error is
        an internal contract violation that propagates unchanged.
    """
    if isinstance(core_error, CoreScaleDomainError):
        return ScaleDomainError(
            reason=labels.SCALE_DOMAIN_ERROR_NAMES[core_error.reason],
            series=context.term_names[core_error.term],
            date_ord=int(context.date_ord[core_error.aligned_index]),
            level=core_error.level,
        )
    if isinstance(core_error, CoreAlignmentError):
        return AlignmentError(
            reason=labels.ALIGNMENT_ERROR_NAMES[core_error.reason],
            start_ord=core_error.start_ord,
            end_ord=core_error.end_ord,
        )
    if isinstance(core_error, CoreDomainError):
        return TransformDomainError(
            series=context.term_names[core_error.column],
            transform=labels.TRANSFORM_NAMES[core_error.transform],
            date_ord=int(context.date_ord[core_error.aligned_index]),
            level=core_error.level,
        )
    if isinstance(core_error, CoreSampleSizeError):
        reason = labels.SAMPLE_SIZE_ERROR_NAMES[core_error.reason]
        error = InsufficientObservationsError(
            reason=reason,
            n_available=core_error.n_available,
            n_required=core_error.n_required,
        )
        if core_error.reason == codes.SAMPLE_ROWS_NOT_ABOVE_COLUMNS:
            # The core reports R and k + 1 (C.12 names R and k).
            return _restated(
                error,
                f"{reason!r}: R={core_error.n_available} rows, "
                f"k={core_error.n_required - 1} columns; R must be at least k + 1",
            )
        return error
    if isinstance(core_error, CoreRankError):
        reason = labels.RANK_ERROR_NAMES[core_error.reason]
        names = tuple(context.column_names[int(j)] for j in core_error.involved)
        error_rank = RankDeficiencyError(
            reason=reason,
            rank=core_error.rank,
            n_columns=core_error.n_columns,
            column_names=names,
        )
        if core_error.reason == codes.RANK_ZERO_COLUMN:
            # OD-1: no SVD runs, so no rank is known.
            return _restated(
                error_rank,
                f"{reason!r}: zero columns: {names} of {core_error.n_columns} columns",
            )
        return error_rank
    if isinstance(core_error, CoreDegenerateFitError):
        return DegenerateFitError(
            reason=labels.DEGENERATE_FIT_ERROR_NAMES[core_error.reason],
            column_names=tuple(context.column_names[c] for c in core_error.columns),
        )
    if isinstance(core_error, CoreParameterError):
        reason = labels.PARAMETER_ERROR_NAMES[core_error.reason]
        return ModelSpecError(reason=f"{reason}: value={core_error.value!r}")
    if isinstance(core_error, CoreRestrictionError):
        return RestrictionError(
            reason=labels.RESTRICTION_ERROR_NAMES[core_error.reason],
            shape=core_error.shape,
        )
    if isinstance(core_error, CoreInputError):
        reason = labels.INPUT_ERROR_NAMES[core_error.reason]
        if context.stage == "wald" and core_error.operand in (
            _WALD_OPERAND_RESTRICTION,
            _WALD_OPERAND_RHS,
        ):
            return RestrictionError(
                reason=reason, shape=_received_shape(context.restriction)
            )
        if core_error.reason == codes.INPUT_NON_FINITE and context.stage in (
            "design",
            "fit",
        ):
            names = _non_finite_column(core_error, context)
            return _restated(
                DegenerateFitError(reason=reason, column_names=names),
                f"{reason}: a row value of {names}, or a fit quantity formed "
                "from it, exceeds the float64 range",
            )
        # Overflow of the combination or a difference thereof: z has inf.
        # Positions are not decoded: adf_core and fit_core inside it both
        # report operand 0, so the position index has two meanings (A.1).
        if core_error.reason == codes.INPUT_NON_FINITE and context.stage == "adf":
            return _restated(
                DegenerateFitError(reason=reason, column_names=()),
                f"{reason}: a value of the combination, or a difference of it, "
                "exceeds the float64 range",
            )
    return None


@contextmanager
def _facade_errors(context: _ErrorContext) -> Iterator[None]:
    # Every core call runs inside this block, so every mapped raise chains the
    # core error as __cause__ (C.20, S7.T2). An unmapped core error leaves the
    # block unchanged.
    try:
        yield
    except _CORE_ERRORS as core_error:
        mapped = _map_core_error(core_error, context)
        if mapped is None:
            raise
        raise mapped from core_error


# ------------------------------------------------------------------ align_levels


def _checked_series(series: Sequence[ObservedSeries]) -> tuple[ObservedSeries, ...]:
    # A str is a Sequence of str; it is never a series collection.
    if isinstance(series, (str, bytes)) or not isinstance(series, Sequence):
        raise TypeError(
            f"series must be a sequence of ObservedSeries, got {type(series).__name__}"
        )
    if len(series) == 0:
        raise ValueError("series must hold at least one ObservedSeries")
    items = tuple(series)
    for i, item in enumerate(items):
        if not isinstance(item, ObservedSeries):
            raise TypeError(
                f"series[{i}] must be ObservedSeries, got {type(item).__name__}"
            )
    return items


def _window_bound(name: str, value: int) -> int:
    # Same rule as align_core: operator.index rejects a float or str, so no
    # bound is rounded or parsed. bool is an int subclass and operator.index
    # reads True as 1; a flag is not a date ordinal.
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an int, not a bool")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None


def align_levels(
    series: Sequence[ObservedSeries], *, start_ord: int, end_ord: int
) -> AlignedLevels:
    """
    Align named level series on the dates common to all of them (C.2, C.3).

    Parameters
    ----------
    series
        A non-empty sequence of ``ObservedSeries`` with unique names. The
        order of the sequence is the series order of the result.
    start_ord, end_ord
        Half-open window ``[start_ord, end_ord)`` of ``int32`` date ordinals.
        Each bound is an ``int`` or a NumPy integer.

    Returns
    -------
    AlignedLevels
        Names and units in input order, the kept dates, the ``(n_series, m)``
        levels, and one ``Exclusion`` per input observation that is not kept,
        with the series name, the date and the reason name. The levels are
        the input values bit for bit. No caller array is changed (C.19).

    Raises
    ------
    TypeError
        ``series`` is not a sequence, an item is not ``ObservedSeries``, or a
        bound is not an integer (a ``bool``, float or str included).
    ValueError
        ``series`` is empty.
    AlignmentError
        Two series have one name (``series`` names the duplicate; OD-12);
        ``start_ord >= end_ord``; a bound is outside the ``int32`` range; or
        no date in the window is common to every series.

    Notes
    -----
    The checks run in this order, all before the core call: the series
    sequence, the two bounds, the unique names (S7.T10).
    """
    items = _checked_series(series)
    start = _window_bound("start_ord", start_ord)
    end = _window_bound("end_ord", end_ord)
    names = tuple(item.name for item in items)
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise AlignmentError(
                reason=_DUPLICATE_SERIES_NAME,
                start_ord=start,
                end_ord=end,
                series=name,
            )
        seen.add(name)

    with _facade_errors(_ErrorContext(stage="align")):
        core = align_core(
            [item.date_ord for item in items],
            [item.values for item in items],
            start_ord=start,
            end_ord=end,
        )

    exclusions = tuple(
        Exclusion(
            series=names[j],
            date_ord=d,
            reason=labels.EXCLUSION_NAMES[reason],
        )
        for j, d, reason in zip(
            core.excluded_series.tolist(),
            core.excluded_date_ord.tolist(),
            core.excluded_reason.tolist(),
            strict=True,
        )
    )
    return AlignedLevels(
        names=names,
        units=tuple(item.unit for item in items),
        date_ord=core.date_ord,
        levels=core.levels,
        exclusions=exclusions,
        start_ord=core.start_ord,
        end_ord=core.end_ord,
    )


# ------------------------------------------------------------------ fit_ols


def _read_only(array: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
    array.setflags(write=False)
    return array


def _core_aligned_levels(levels: AlignedLevels) -> CoreAlignedLevels:
    """
    Rebuild the core alignment from a facade ``AlignedLevels``.

    ``AlignedLevels`` holds no input positions, so they are rebuilt by
    integer bookkeeping. Each input date axis is strictly increasing, so the
    input position of kept date ``i`` of series ``j`` is ``i`` plus the count
    of excluded observations of series ``j`` with an earlier date. The
    exclusion arrays keep the record order of ``levels.exclusions``, which
    ``align_levels`` takes from the core unchanged.

    Every array is new and read-only. For an ``AlignedLevels`` from
    ``align_levels`` the result equals the ``align_core`` output field by
    field, bit for bit (S7 test).
    """
    index_of: dict[str, int] = {}
    for j, name in enumerate(levels.names):
        index_of.setdefault(name, j)
    n_series = len(levels.names)
    excluded_series = np.array(
        [index_of[record.series] for record in levels.exclusions], dtype=np.int32
    )
    excluded_date_ord = np.array(
        [record.date_ord for record in levels.exclusions], dtype=np.int32
    )
    excluded_reason = np.array(
        [labels.EXCLUSION_NAME_TO_CODE[record.reason] for record in levels.exclusions],
        dtype=np.int8,
    )
    m = int(levels.date_ord.shape[0])
    positions = np.empty((n_series, m), dtype=np.int32)
    kept_index = np.arange(m, dtype=np.int64)
    for j in range(n_series):
        earlier = np.sort(excluded_date_ord[excluded_series == j])
        positions[j, :] = kept_index + np.searchsorted(
            earlier, levels.date_ord, side="left"
        )
    return CoreAlignedLevels(
        start_ord=levels.start_ord,
        end_ord=levels.end_ord,
        date_ord=_read_only(np.array(levels.date_ord, dtype=np.int32, copy=True)),
        positions=_read_only(positions),
        levels=_read_only(np.array(levels.levels, dtype=np.float64, copy=True)),
        excluded_series=_read_only(excluded_series),
        excluded_date_ord=_read_only(excluded_date_ord),
        excluded_reason=_read_only(excluded_reason),
    )


class _HasSeries(Protocol):
    """Any term type that carries a ``series`` attribute (``Term`` or ``CombinationTerm``).

    ``series`` is a read-only property, so a frozen dataclass field satisfies it.
    """

    @property
    def series(self) -> str: ...


def _missing_term_series(levels: AlignedLevels, terms: Sequence[_HasSeries]) -> None:
    known = set(levels.names)
    for term in terms:
        if term.series not in known:
            raise ModelSpecError(
                reason=(
                    f"term series {term.series!r} is not in the aligned levels; "
                    f"aligned series: {levels.names}"
                )
            )


def fit_ols(levels: AlignedLevels, spec: ModelSpec) -> OlsResult:
    """
    Fit the declared model by OLS on the aligned levels (C.4 to C.17).

    Parameters
    ----------
    levels
        Output of ``align_levels``.
    spec
        The declared model. Every term series must be a name of ``levels``.

    Returns
    -------
    OlsResult
        Coefficients, covariance, statistics, diagnostics, labels, row dates,
        the alignment exclusions, the spec values, three digests and
        ``ENGINE_VERSION``. Every array is read-only. No caller array is
        changed (C.19).

    Raises
    ------
    TypeError
        ``spec`` is not a ``ModelSpec`` or ``levels`` is not an
        ``AlignedLevels``.
    ModelSpecError
        A term names a series that ``levels`` does not hold (S7.T8), or a
        core parameter check fails (for example a HAC lag above ``R - 1``,
        S7.T11).
    TransformDomainError
        An aligned level is outside the domain of its transform (C.7).
    InsufficientObservationsError
        No row exists (``h >= m``), or ``R < k + 1`` (C.12).
    RankDeficiencyError
        A zero design column or ``rank < k`` (C.8, C.9).
    DegenerateFitError
        An exact fit, a constant response with an intercept, a non-positive
        covariance diagonal (C.14), or a transformed value that overflowed to
        a non-finite number (R-008).
    SeriesContractError
        Raised by the ``AlignedLevels`` constructor, before this call, for a
        value fault of a ``levels`` field (O-2). Every ``levels`` that reaches
        this function passed those checks.

    Notes
    -----
    The checks run in this order: ``spec`` type, ``levels`` type, the term
    series names; then the core: rows, design, fit, diagnostics.

    The intercept flag is read once and the same value goes to
    ``assemble_design`` and to ``fit_core`` (OD-9): ``fit_core`` does not
    check that design column 0 holds ones.

    ``OlsResult.exclusions`` holds the alignment exclusions only. The aligned
    dates that no row reads (``NOT_USED_BY_ROWS``) are not records of the
    result: they are the aligned dates that are neither a row start date nor
    a row end date, so they follow from ``levels.date_ord`` and the row dates.
    """
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"spec must be ModelSpec, got {type(spec).__name__}")
    if not isinstance(levels, AlignedLevels):
        raise TypeError(f"levels must be AlignedLevels, got {type(levels).__name__}")
    intercept = spec.intercept
    terms = (spec.response, *spec.regressors)
    _missing_term_series(levels, terms)

    index_of: dict[str, int] = {}
    for j, name in enumerate(levels.names):
        index_of.setdefault(name, j)
    term_names = tuple(term.series for term in terms)
    term_index = tuple(index_of[name] for name in term_names)
    regressor_names = term_names[1:]
    regressor_units = tuple(levels.units[j] for j in term_index[1:])
    regressor_transforms = tuple(term.transform for term in spec.regressors)
    if intercept:
        column_names = (_INTERCEPT_NAME, *regressor_names)
        units = (_INTERCEPT_UNIT, *regressor_units)
        transforms: tuple[Transform | None, ...] = (None, *regressor_transforms)
    else:
        column_names = regressor_names
        units = regressor_units
        transforms = regressor_transforms

    covariance_kind = int(labels.COVARIANCE_NAME_TO_CODE[spec.covariance.kind.name])
    spec_lag = spec.covariance.lag
    threshold = spec.condition_threshold

    core_levels = _core_aligned_levels(levels)
    term_series = np.array(term_index, dtype=np.int32)
    term_transforms = np.array(
        [labels.TRANSFORM_NAME_TO_CODE[term.transform.name] for term in terms],
        dtype=np.int8,
    )
    with _facade_errors(
        _ErrorContext(stage="rows", term_names=term_names, date_ord=levels.date_ord)
    ):
        rows = build_rows(
            core_levels,
            term_series=term_series,
            term_transforms=term_transforms,
            step=spec.step,
            stride=spec.stride,
        )
    y = rows.values[0]
    regressors = rows.values[1:]
    with _facade_errors(
        _ErrorContext(
            stage="design",
            term_names=term_names,
            column_names=column_names,
            n_rows=rows.n_rows,
        )
    ):
        design = assemble_design(regressors, intercept=intercept)

    covariance_lag = _CORE_NO_LAG if spec_lag is None else spec_lag
    with _facade_errors(
        _ErrorContext(
            stage="fit",
            term_names=term_names,
            column_names=column_names,
            n_rows=rows.n_rows,
        )
    ):
        fit = fit_core(
            design,
            y,
            intercept=intercept,
            rank_rtol=spec.rank_rtol,
            covariance_kind=covariance_kind,
            covariance_lag=covariance_lag,
        )
        core_diagnostics = diagnose(
            term_transforms=rows.term_transforms,
            condition_number=fit.gate.condition_number,
            condition_threshold=_UNUSED_THRESHOLD if threshold is None else threshold,
            apply_condition_threshold=threshold is not None,
            r_squared=fit.r_squared,
            durbin_watson_value=fit.durbin_watson,
            overlap_order=rows.overlap_order,
            covariance_kind=covariance_kind,
            covariance_lag=covariance_lag,
        )

    diagnostics = tuple(
        Diagnostic(
            code=labels.DIAGNOSTIC_NAMES[code],
            value=value,
            message=labels.DIAGNOSTIC_MESSAGES[labels.DIAGNOSTIC_NAMES[code]],
        )
        for code, value in zip(
            core_diagnostics.codes.tolist(),
            core_diagnostics.values.tolist(),
            strict=True,
        )
    )

    response_unit = levels.units[term_index[0]]
    return OlsResult(
        coef=fit.coef,
        se=fit.se,
        t_stats=fit.t_stats,
        p_values=fit.p_values,
        cov=fit.cov,
        resid=fit.resid,
        fitted=fit.fitted,
        dof=fit.dof,
        n_obs=fit.n_obs,
        rss=fit.rss,
        r_squared=fit.r_squared,
        durbin_watson=fit.durbin_watson,
        condition_number=fit.gate.condition_number,
        singular_values=fit.gate.singular_values,
        diagnostics=diagnostics,
        column_names=column_names,
        units=units,
        transforms=transforms,
        response=spec.response,
        window=(levels.start_ord, levels.end_ord),
        step=rows.step,
        stride=rows.stride,
        overlap_order=rows.overlap_order,
        row_start_date_ord=rows.row_start_date_ord,
        row_end_date_ord=rows.row_end_date_ord,
        exclusions=levels.exclusions,
        covariance=spec.covariance,
        rank_rtol=spec.rank_rtol,
        condition_threshold=threshold,
        sample_digest=digest.sample_digest(
            row_start_date_ord=rows.row_start_date_ord,
            row_end_date_ord=rows.row_end_date_ord,
            response_name=term_names[0],
            intercept=intercept,
            regressor_names=regressor_names,
            y=y,
            x=design,
        ),
        spec_digest=digest.spec_digest(
            engine_version=ENGINE_VERSION,
            response_name=term_names[0],
            response_unit=response_unit,
            response_transform=spec.response.transform.name,
            regressor_names=regressor_names,
            regressor_units=regressor_units,
            regressor_transforms=tuple(t.name for t in regressor_transforms),
            intercept=intercept,
            step=spec.step,
            stride=spec.stride,
            start_ord=levels.start_ord,
            end_ord=levels.end_ord,
            covariance_kind=covariance_kind,
            covariance_lag=_DIGEST_NO_LAG if spec_lag is None else spec_lag,
            rank_rtol=spec.rank_rtol,
            condition_threshold=threshold,
        ),
        row_digest=digest.row_digest(rows.row_start_date_ord, rows.row_end_date_ord),
        engine_version=ENGINE_VERSION,
    )


# ------------------------------------------------------------------ wald_test


def wald_test(
    result: OlsResult,
    restriction: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> WaldResult:
    """
    Joint Wald test of ``restriction @ coef = rhs`` on one fit (C.18, A.5).

    Parameters
    ----------
    result
        Output of ``fit_ols``. The test uses ``result.coef``, ``result.cov``
        and ``result.dof``.
    restriction
        ``float64`` ``(J, K)``, exactly ``numpy.ndarray``, full row rank,
        ``K`` equal to the coefficient count (design column order).
    rhs
        ``float64`` ``(J,)``, exactly ``numpy.ndarray``.

    Returns
    -------
    WaldResult
        ``W``, ``F = W / J``, both p-values, ``J``, ``dof``, the three digests
        of ``result``, the digest of ``(restriction, rhs)`` and
        ``ENGINE_VERSION`` (O-8). ``restriction`` and ``rhs`` are not changed
        (C.19).

    Raises
    ------
    TypeError
        ``result`` is not an ``OlsResult``.
    RestrictionError
        ``restriction`` or ``rhs`` is not exactly ``numpy.ndarray`` (a
        ``numpy.ma.MaskedArray`` included) or not ``float64``; a wrong shape;
        a non-finite entry; a rank-deficient ``restriction``; or
        ``restriction @ cov @ restriction.T`` is not positive definite.
        ``RESTRICTION_NON_FINITE`` also when ``restriction @ cov @
        restriction.T`` or ``restriction @ coef - rhs`` overflows to a
        non-finite entry (both on the row-normalised restriction); the core
        raises before the Cholesky step and returns no ``W``.

    Notes
    -----
    The core test runs before the restriction digest, so a shape or type
    error raises ``RestrictionError`` first. The facade reads no element of
    ``restriction`` or ``rhs`` before the core has checked the type.
    """
    if not isinstance(result, OlsResult):
        raise TypeError(f"result must be OlsResult, got {type(result).__name__}")
    with _facade_errors(_ErrorContext(stage="wald", restriction=restriction)):
        core = wald_core(result.coef, result.cov, result.dof, restriction, rhs)
    return WaldResult(
        w=core.w,
        f=core.f_stat,
        chi2_pvalue=core.chi2_pvalue,
        f_pvalue=core.f_pvalue,
        n_restrictions=core.j,
        dof=core.dof,
        sample_digest=result.sample_digest,
        spec_digest=result.spec_digest,
        row_digest=result.row_digest,
        restriction_digest=digest.restriction_digest(restriction, rhs),
        engine_version=ENGINE_VERSION,
    )


# ------------------------------------------------------------------ unit_root_test


def unit_root_test(levels: AlignedLevels, spec: UnitRootSpec) -> UnitRootResult:
    """
    ADF unit-root test of a declared linear combination (C.2 to C.10).

    Parameters
    ----------
    levels
        Output of ``align_levels``.  Holds the aligned level series that
        the combination references.
    spec
        The declared test specification.  Every combination series must be
        a name in ``levels``.

    Returns
    -------
    UnitRootResult
        Statistic, MacKinnon critical values, ADF regression output,
        digests, labels and provenance.  Every array is read-only.  No
        caller array is changed (C.13).

    Raises
    ------
    TypeError
        ``spec`` is not a ``UnitRootSpec`` or ``levels`` is not an
        ``AlignedLevels``.
    ModelSpecError
        A combination series is not in ``levels`` (C.2), or a core parameter
        check fails (for example a rank_rtol out of range, which the facade
        checks earlier but the core re-checks).
    ScaleDomainError
        A ``LOG``-scale term has a level at or below zero (C.3).
    InsufficientObservationsError
        ``n = m - 1 - lag < 20`` (C.6) or ``n < k + 1`` from ``fit_core``.
    RankDeficiencyError
        The ADF design is rank-deficient (C.5, C.8).
    DegenerateFitError
        Exact fit, zero response variance, or an overflow to non-finite
        of the combination or a difference of it (C.10).
    SeriesContractError
        Raised by the ``AlignedLevels`` constructor, before this call, for a
        value fault of a ``levels`` field (O-2).  Every ``levels`` that
        reaches this function passed those checks.

    Notes
    -----
    Checks run in this order:

    1. ``spec`` must be ``UnitRootSpec`` and ``levels`` must be
       ``AlignedLevels`` (``TypeError``).
    2. Every combination series must be in ``levels.names``
       (``ModelSpecError``).
    3. Core calls: ``combination.combine_core`` (stage ``"combination"``),
       then ``adf.adf_core`` (stage ``"adf"``).

    Core calls are attributes of their modules, so tests can patch them:
    ``combination.combine_core`` and ``adf.adf_core``.
    """
    # ---- 1. type checks (order fixed: spec first, then levels) ---------------
    if not isinstance(spec, UnitRootSpec):
        raise TypeError(f"spec must be UnitRootSpec, got {type(spec).__name__}")
    if not isinstance(levels, AlignedLevels):
        raise TypeError(f"levels must be AlignedLevels, got {type(levels).__name__}")

    # ---- 2. series check -----------------------------------------------------
    _missing_term_series(levels, spec.combination.terms)

    # ---- index bookkeeping (no numerical logic) ------------------------------
    index_of: dict[str, int] = {}
    for j, name in enumerate(levels.names):
        index_of.setdefault(name, j)

    term_names = tuple(term.series for term in spec.combination.terms)
    term_series = np.array([index_of[name] for name in term_names], dtype=np.int32)
    weights = np.array(
        [term.weight for term in spec.combination.terms], dtype=np.float64
    )
    scale = int(spec.combination.scale.value)
    lag = spec.lag

    # ADF column labels, built before the core calls because a mapping error
    # needs them (C.10, A.3).
    has_trend = spec.deterministic == Deterministic.CONSTANT_TREND
    col_labels: list[str] = [_INTERCEPT_NAME]
    if has_trend:
        col_labels.append("trend")
    col_labels.append("z_lag1")
    # Bound the loop by m. The labels are read only when n = m - 1 - lag is at
    # least 20, so lag is at most m - 21. For lag at least m the core raises
    # the size error before any label is read. An unbounded loop would exhaust
    # memory for a lag near 2**63 - 1, which S4.T2 accepts.
    for q in range(1, min(lag, int(levels.date_ord.shape[0])) + 1):
        col_labels.append(f"dz_lag{q}")
    column_names: tuple[str, ...] = tuple(col_labels)

    # ---- 3a. combination (stage "combination") --------------------------------
    with _facade_errors(
        _ErrorContext(
            stage="combination",
            term_names=term_names,
            date_ord=levels.date_ord,
        )
    ):
        z = combination.combine_core(levels.levels, term_series, weights, scale)

    # ---- 3b. ADF regression (stage "adf") ------------------------------------
    with _facade_errors(
        _ErrorContext(
            stage="adf",
            column_names=column_names,
        )
    ):
        adf_result = adf.adf_core(z, lag, int(spec.deterministic.value), spec.rank_rtol)

    # ---- row date bookkeeping ------------------------------------------------
    # row_start_index[r] and row_end_index[r] are aligned indices; map to
    # date ordinals from levels.date_ord.
    row_start_date_ord = np.array(
        levels.date_ord[adf_result.row_start_index], dtype=np.int32, copy=True
    )
    row_end_date_ord = np.array(
        levels.date_ord[adf_result.row_end_index], dtype=np.int32, copy=True
    )
    row_start_date_ord.setflags(write=False)
    row_end_date_ord.setflags(write=False)

    window = (levels.start_ord, levels.end_ord)
    exclusions = levels.exclusions
    # Unit of each term's series in declared order.
    units: tuple[str, ...] = tuple(
        levels.units[index_of[term.series]] for term in spec.combination.terms
    )
    det_code = int(spec.deterministic.value)

    # ---- digests ------------------------------------------------------------
    # regressor_names: ADF design labels after "const".
    regressor_names = list(column_names[1:])

    _sample_digest = digest.unit_root_sample_digest(
        row_start_date_ord=row_start_date_ord,
        row_end_date_ord=row_end_date_ord,
        y=adf_result.response,
        x=adf_result.design,
        regressor_names=regressor_names,
    )
    _spec_digest = digest.unit_root_spec_digest(
        engine_version=ENGINE_VERSION,
        test_kind=int(codes.TEST_ADF),
        term_names=list(term_names),
        term_units=list(units),
        weights=weights,
        scale=scale,
        deterministic=det_code,
        lag=lag,
        rank_rtol=spec.rank_rtol,
        start_ord=levels.start_ord,
        end_ord=levels.end_ord,
        critical_value_source=labels.CRITICAL_VALUE_SOURCES[det_code],
    )
    _row_digest = digest.row_digest(row_start_date_ord, row_end_date_ord)

    return UnitRootResult(
        statistic=adf_result.tau,
        gamma=adf_result.gamma,
        critical_levels=(0.01, 0.05, 0.10),
        critical_values=adf_result.critical_values,
        deterministic=spec.deterministic,
        lag=spec.lag,
        n_obs=adf_result.n_obs,
        coef=adf_result.coef,
        se=adf_result.se,
        column_names=column_names,
        condition_number=adf_result.condition_number,
        null_hypothesis=labels.NULL_HYPOTHESIS_TEXTS[det_code],
        critical_value_source=labels.CRITICAL_VALUE_SOURCES[det_code],
        combination=spec.combination,
        units=units,
        window=window,
        row_start_date_ord=row_start_date_ord,
        row_end_date_ord=row_end_date_ord,
        exclusions=exclusions,
        rank_rtol=spec.rank_rtol,
        sample_digest=_sample_digest,
        spec_digest=_spec_digest,
        row_digest=_row_digest,
        engine_version=ENGINE_VERSION,
    )
