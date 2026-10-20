# src/riskquant/solutions/factor_regression/errors.py
"""
Facade exceptions for the factor-regression library (C.20).

Every class subclasses ``riskquant.exceptions.RiskQuantError``.  The facade
catches a core exception in S7 and re-raises the matching facade exception
with series names and dates chained as ``__cause__``.  S6 defines the
classes only; S7 does the mapping.

Core-to-facade mapping
----------------------
``CoreInputError``           -> ``SeriesContractError``
``CoreAlignmentError``       -> ``AlignmentError``
``CoreParameterError``       -> ``ModelSpecError``
``CoreDomainError``          -> ``TransformDomainError``
``CoreScaleDomainError``     -> ``ScaleDomainError``
``CoreRankError``            -> ``RankDeficiencyError``
``CoreSampleSizeError``      -> ``InsufficientObservationsError``
``CoreDegenerateFitError``   -> ``DegenerateFitError``
``CoreRestrictionError``     -> ``RestrictionError``
"""

from __future__ import annotations

from riskquant.exceptions import RiskQuantError


class SeriesContractError(RiskQuantError):
    """
    An ``ObservedSeries`` violates the C.1 array contract.

    Parameters
    ----------
    series
        Name of the offending series (or its ``repr`` when ``name`` is not a
        ``str``).
    message
        Human-readable description of the violation.
    date_ord
        For a non-finite value: the ``int32`` date ordinal of the first
        offending element.  ``None`` for violations that do not address a
        specific element (wrong dtype, wrong ndim, unsorted dates, etc.).
    """

    series: str
    date_ord: int | None

    def __init__(
        self,
        *,
        series: str,
        message: str,
        date_ord: int | None = None,
    ) -> None:
        self.series = series
        self.date_ord = date_ord
        if date_ord is not None:
            detail = f"series={series!r} date_ord={date_ord}: {message}"
        else:
            detail = f"series={series!r}: {message}"
        super().__init__(detail)


class AlignmentError(RiskQuantError):
    """
    The alignment window or the common date set is empty or invalid (C.2, C.3).

    Parameters
    ----------
    reason
        Label name of the violated condition (``codes.ALIGNMENT_ERROR_CODES``).
    start_ord, end_ord
        The declared half-open window ``[start_ord, end_ord)``.
    series
        Name of the series that the condition concerns (for example a
        duplicate series name, C.2).  ``None`` when the condition concerns no
        single series; the message is then unchanged (OD-12).
    """

    reason: str
    start_ord: int
    end_ord: int
    series: str | None

    def __init__(
        self,
        *,
        reason: str,
        start_ord: int,
        end_ord: int,
        series: str | None = None,
    ) -> None:
        self.reason = reason
        self.start_ord = int(start_ord)
        self.end_ord = int(end_ord)
        self.series = series
        if series is None:
            detail = f"{reason!r}: window=[{self.start_ord}, {self.end_ord})"
        else:
            detail = (
                f"{reason!r}: series={series!r} "
                f"window=[{self.start_ord}, {self.end_ord})"
            )
        super().__init__(detail)


class ModelSpecError(RiskQuantError):
    """
    A ``ModelSpec`` or ``CovarianceSpec`` field violates the C.5 contract.

    Parameters
    ----------
    reason
        Human-readable description of the violated rule.
    """

    reason: str

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class TransformDomainError(RiskQuantError):
    """
    An aligned level is outside the domain of its declared transform (C.7).

    Parameters
    ----------
    series
        Name of the series with the offending level.
    transform
        Label name of the transform (``codes.TRANSFORM_CODES``).
    date_ord
        Date ordinal of the first offending level.
    level
        The offending level value.
    """

    series: str
    transform: str
    date_ord: int
    level: float

    def __init__(
        self,
        *,
        series: str,
        transform: str,
        date_ord: int,
        level: float,
    ) -> None:
        self.series = series
        self.transform = transform
        self.date_ord = int(date_ord)
        self.level = float(level)
        super().__init__(
            f"series={series!r} transform={transform!r} "
            f"date_ord={self.date_ord} level={self.level!r}"
        )


class ScaleDomainError(RiskQuantError):
    """
    A term's aligned level is outside the domain of the declared scale (C.3).

    Raised when a ``LOG``-scale combination includes a level at or below zero.
    The facade maps ``CoreScaleDomainError`` to this exception, chaining the
    core error as ``__cause__``.

    Parameters
    ----------
    reason
        Label name of the domain violation
        (``codes.SCALE_DOMAIN_ERROR_CODES``).
    series
        Name of the series with the offending level.
    date_ord
        ``int32`` date ordinal of the first offending aligned level.
    level
        The offending level value (at or below zero for ``SCALE_LOG``).
    """

    reason: str
    series: str
    date_ord: int
    level: float

    def __init__(
        self,
        *,
        reason: str,
        series: str,
        date_ord: int,
        level: float,
    ) -> None:
        self.reason = reason
        self.series = series
        self.date_ord = int(date_ord)
        self.level = float(level)
        super().__init__(
            f"series={series!r} reason={reason!r} "
            f"date_ord={self.date_ord} level={self.level!r}"
        )


class RankDeficiencyError(RiskQuantError):
    """
    The design matrix is rank-deficient (C.8, C.9).

    Parameters
    ----------
    reason
        Label name of the rank condition (``codes.RANK_ERROR_CODES``).
    rank
        Detected numerical rank.
    n_columns
        Column count of the design.
    column_names
        Names of the involved columns (those with non-zero null-space
        participation, A.2 step 4).
    """

    reason: str
    rank: int
    n_columns: int
    column_names: tuple[str, ...]

    def __init__(
        self,
        *,
        reason: str,
        rank: int,
        n_columns: int,
        column_names: tuple[str, ...],
    ) -> None:
        self.reason = reason
        self.rank = int(rank)
        self.n_columns = int(n_columns)
        self.column_names = tuple(column_names)
        super().__init__(
            f"{reason!r}: rank={self.rank} of {self.n_columns}; "
            f"involved columns: {self.column_names}"
        )


class InsufficientObservationsError(RiskQuantError):
    """
    The aligned sample is too small for the declared rows or design (C.6, C.12).

    Parameters
    ----------
    reason
        Label name of the constraint (``codes.SAMPLE_SIZE_ERROR_CODES``).
    n_available
        Observations or rows available.
    n_required
        Observations or rows required.
    """

    reason: str
    n_available: int
    n_required: int

    def __init__(
        self,
        *,
        reason: str,
        n_available: int,
        n_required: int,
    ) -> None:
        self.reason = reason
        self.n_available = int(n_available)
        self.n_required = int(n_required)
        super().__init__(
            f"{reason!r}: n_available={self.n_available} n_required={self.n_required}"
        )


class DegenerateFitError(RiskQuantError):
    """
    The OLS fit is degenerate (C.14).

    Parameters
    ----------
    reason
        Label name of the degeneracy kind
        (``codes.DEGENERATE_FIT_ERROR_CODES``).
    column_names
        For ``DEGENERATE_NON_POSITIVE_VARIANCE``: names of the columns with a
        non-positive covariance diagonal.  Empty for the other reasons.
    """

    reason: str
    column_names: tuple[str, ...]

    def __init__(
        self,
        *,
        reason: str,
        column_names: tuple[str, ...] = (),
    ) -> None:
        self.reason = reason
        self.column_names = tuple(column_names)
        detail = reason
        if self.column_names:
            detail = f"{reason}; columns: {self.column_names}"
        super().__init__(detail)


class RestrictionError(RiskQuantError):
    """
    A Wald restriction ``(R, r)`` is invalid (A.5, C.18).

    Parameters
    ----------
    reason
        Label name of the violated condition
        (``codes.RESTRICTION_ERROR_CODES``).
    shape
        Shape of the restriction matrix as received.
    """

    reason: str
    shape: tuple[int, ...]

    def __init__(
        self,
        *,
        reason: str,
        shape: tuple[int, ...] = (),
    ) -> None:
        self.reason = reason
        self.shape = tuple(shape)
        detail = reason
        if self.shape:
            detail = f"{reason} shape={self.shape}"
        super().__init__(detail)
