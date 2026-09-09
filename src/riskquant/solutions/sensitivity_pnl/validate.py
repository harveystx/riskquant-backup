# src/riskquant/solutions/sensitivity_pnl/validate.py
"""
The public-edge gate. Everything the compiler and engine do not catch, or catch
in terms the caller cannot act on, is caught here — and nothing else.

Division of labour
------------------
`compile_sensitivity_book` and `SensitivityPnlEngine` validate their own
contracts thoroughly, and duplicating those checks would guarantee drift the
first time either changes. This module adds exactly what the layer below cannot
say, each item established by probing the engine rather than by reading it:

  * NULL IN AN ID COLUMN. A null `position_id` survives `cast(Int64, strict=True)`
    as a null, and `to_numpy()` renders it `-9223372036854775808`. The compiler
    emits a RuntimeWarning and carries on, so the book gains a phantom position
    and the output gains a phantom row keyed to a sentinel UID no downstream
    join will match.

  * FLOAT IN AN ID COLUMN. `position_id = 2.7` casts to 2 without complaint,
    moving a position's risk onto its neighbour. Excel and pandas round-trips
    produce float id columns routinely.

  * KEY-SPACE MISMATCH. String factor labels against an integer market axis, or
    the reverse, reaches `replace_strict` as an opaque Polars error.

  * COVERAGE, IN THE CALLER'S KEY SPACE. This is the one that matters most under
    string labels. The compiler resolves labels to compacted `int32` slots
    before it validates anything, so a factor missing from `factor_meta` is
    reported as `1 ID(s) absent from the universe. First 1: [1]` — where `1` is
    an internal slot index the caller has never seen and cannot map back. Every
    factor named in a defect below is named in the caller's own keys.

  * ROW-SET DRIFT. A sensitivity for a position outside the declared row set
    would silently widen the output matrix.

Everything else — CSR monotonicity, index bounds, duplicate keys, `i == j` cross
terms, non-positive base levels under a ratio convention — is left to the layer
that owns it, and surfaces as a `ValueError` whose message is already written in
public terms.

Cost model
----------
The happy path is one aggregate `select` per frame returning a handful of
scalars, plus one distinct-value pass per factor column. No masks are
materialised, no frames are filtered, no rows are copied. Row localisation runs
only inside a failure branch, where an extra pass costs nothing that matters
because the run is over.
"""

from __future__ import annotations

__all__ = [
    "validate_book_spec",
    "validate_market_against_book",
]

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.sensitivity_pnl.compile import (
    COL_CROSS_SECOND_ORDER,
    COL_DISPLACEMENT_CONVENTION_ID,
    COL_FACTOR_ID,
    COL_FACTOR_ID_I,
    COL_FACTOR_ID_J,
    COL_FIRST_ORDER,
    COL_POSITION_ID,
    COL_QUOTE_BASE,
    COL_SECOND_ORDER,
)
from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.conventions import (
    POSITIVE_LEVEL_DISPLACEMENT_IDS,
    DisplacementConvention,
)
from riskquant.solutions.sensitivity_pnl.errors import MAX_REPORTED_ROWS, DefectLog
from riskquant.solutions.sensitivity_pnl.market import (
    FactorAxis,
    MarketScenarios,
    ScenarioLevels,
)
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityBookSpec

_DIAGONAL_ID_COLUMNS = (COL_POSITION_ID, COL_FACTOR_ID)
_DIAGONAL_VALUE_COLUMNS = (COL_FIRST_ORDER, COL_SECOND_ORDER)
_CROSS_ID_COLUMNS = (COL_POSITION_ID, COL_FACTOR_ID_I, COL_FACTOR_ID_J)
_CROSS_VALUE_COLUMNS = (COL_CROSS_SECOND_ORDER,)
_META_COLUMNS = (COL_FACTOR_ID, COL_DISPLACEMENT_CONVENTION_ID, COL_QUOTE_BASE)

_VALID_CONVENTION_IDS = sorted(int(c) for c in DisplacementConvention)
_CONVENTION_LABELS = ", ".join(f"{c.name}={int(c)}" for c in DisplacementConvention)

_POSITION_ID_REMEDY = (
    "position ids travel out in PnlMatrix.trade_uids and are joined downstream, "
    "so this layer will not forge them. Cast upstream — a float id truncates "
    "(2.7 becomes position 2) and a null becomes -9223372036854775808, both "
    "silently."
)


def validate_book_spec(spec: SensitivityBookSpec, *, factor_meta: pl.DataFrame) -> None:
    """
    Raise `SensitivityPnlInputError` listing every defect, or return.

    `factor_meta` is the axis-scoped frame from `pipeline.scope_factor_meta`, not
    `spec.factor_meta`: rows for factors outside the axis are dropped before the
    run and must not be able to fail it.

    Runs against the BOOK half of the request only. The scenario matrix is
    validated in `ScenarioShocks.__post_init__` / `ScenarioLevels.__post_init__`,
    which is what lets a cached book price thousands of scenario sets without
    ever revalidating the book and without ever skipping a matrix check.
    """
    log = DefectLog()
    axis = spec.factor_axis
    covered = _validate_factor_meta(frame=factor_meta, axis=axis, log=log)

    _validate_entry_frame(
        frame=spec.sensitivities,
        source="sensitivities",
        id_columns=_DIAGONAL_ID_COLUMNS,
        factor_columns=(COL_FACTOR_ID,),
        value_columns=_DIAGONAL_VALUE_COLUMNS,
        axis=axis,
        covered=covered,
        log=log,
    )
    if spec.cross_sensitivities is not None:
        _validate_entry_frame(
            frame=spec.cross_sensitivities,
            source="cross_sensitivities",
            id_columns=_CROSS_ID_COLUMNS,
            factor_columns=(COL_FACTOR_ID_I, COL_FACTOR_ID_J),
            value_columns=_CROSS_VALUE_COLUMNS,
            axis=axis,
            covered=covered,
            log=log,
        )
    _validate_row_set(spec=spec, log=log)
    log.raise_if_any()


# =========================================================================
# ENTRY FRAMES
# =========================================================================


def _validate_entry_frame(
    *,
    frame: pl.DataFrame,
    source: str,
    id_columns: tuple[str, ...],
    factor_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
    axis: FactorAxis,
    covered: pl.Series | None,
    log: DefectLog,
) -> None:
    if not _require_columns(
        frame=frame, source=source, required=(*id_columns, *value_columns), log=log
    ):
        return

    _require_integer(
        frame=frame,
        source=source,
        column=COL_POSITION_ID,
        remedy=_POSITION_ID_REMEDY,
        log=log,
    )
    aligned_factors = tuple(
        column
        for column in factor_columns
        if _require_key_space(
            frame=frame, source=source, column=column, axis=axis, log=log
        )
    )
    numeric_values = tuple(
        column
        for column in value_columns
        if _require_numeric(frame=frame, source=source, column=column, log=log)
    )

    if frame.height == 0:
        return

    counts = _aggregate(
        frame=frame,
        null_columns=(*id_columns, *numeric_values),
        finite_columns=numeric_values,
    )
    for column in (*id_columns, *numeric_values):
        _report_nulls(
            frame=frame, source=source, column=column, count=counts[column], log=log
        )
    for column in numeric_values:
        _report_non_finite(
            frame=frame,
            source=source,
            column=column,
            count=counts[f"~{column}"],
            log=log,
        )

    for column in aligned_factors:
        distinct = frame.get_column(column).drop_nulls().unique()
        _report_absent(
            source=f"{source}.{column}",
            distinct=distinct,
            present=axis.keys,
            problem_context="no row on the factor axis",
            remedy=(
                "the factor axis defines the universe. A sensitivity to a factor "
                "with no scenario row would contribute zero risk in every "
                "scenario — the quietest way to under-report. Add the factor to "
                "the axis, or drop the sensitivity upstream."
            ),
            log=log,
        )
        if covered is not None:
            # Scoped metadata cannot cover a factor the axis does not carry, so
            # narrowing to the axis first keeps one missing factor from being
            # reported twice under two different headings.
            _report_absent(
                source=f"{source}.{column}",
                distinct=distinct.filter(distinct.is_in(axis.keys.implode())),
                present=covered,
                problem_context="no row in factor_meta",
                remedy=(
                    "every factor needs a displacement convention and a quote "
                    "base; neither has a safe default, because applying a shock "
                    "in the wrong space is wrong by a factor of the level and "
                    "produces no error."
                ),
                log=log,
            )


def _aggregate(
    *,
    frame: pl.DataFrame,
    null_columns: tuple[str, ...],
    finite_columns: tuple[str, ...],
) -> dict[str, int]:
    """
    One pass; a handful of scalars out.

    `is_finite` is asked only of float columns — it rejects integer input, and
    an integer coefficient is finite by construction.
    """
    projections: list[pl.Expr] = [
        pl.col(column).null_count().alias(column) for column in null_columns
    ]
    projections.extend(
        pl.col(column).is_finite().not_().sum().alias(f"~{column}")
        for column in finite_columns
        if frame.schema[column].is_float()
    )
    row = frame.select(projections).row(0, named=True)
    counts = {key: int(value) for key, value in row.items()}
    for column in finite_columns:
        counts.setdefault(f"~{column}", 0)
    return counts


# =========================================================================
# FACTOR METADATA
# =========================================================================


def _validate_factor_meta(
    *, frame: pl.DataFrame, axis: FactorAxis, log: DefectLog
) -> pl.Series | None:
    """
    Validate the market-scoped metadata frame.

    Returns the factor keys it covers, so the entry frames can be checked against
    it in the caller's key space. Returns `None` when the frame is too malformed
    for that to mean anything.
    """
    if not _require_columns(
        frame=frame, source="factor_meta", required=_META_COLUMNS, log=log
    ):
        return None
    if not _require_key_space(
        frame=frame, source="factor_meta", column=COL_FACTOR_ID, axis=axis, log=log
    ):
        return None
    if frame.height == 0:
        log.add(
            source="factor_meta",
            problem="no rows covering any factor in the market.",
            remedy=(
                "every factor a sensitivity references needs a displacement "
                f"convention ({_CONVENTION_LABELS}) and a quote base."
            ),
        )
        return None

    keys = frame.get_column(COL_FACTOR_ID)
    duplicated = keys.is_duplicated()
    if bool(duplicated.any()):
        log.add(
            source=f"factor_meta.{COL_FACTOR_ID}",
            problem=f"{int(duplicated.sum())} duplicate factor row(s).",
            remedy=(
                "two conventions for one factor is an unresolved conflict, not a "
                "choice this layer may make. De-duplicate upstream. Offending "
                f"key(s): {keys.filter(duplicated).unique().sort().head(MAX_REPORTED_ROWS).to_list()}."
            ),
            row_indices=tuple(
                int(i) for i in duplicated.arg_true().head(MAX_REPORTED_ROWS)
            ),
            total_rows=int(duplicated.sum()),
        )

    _validate_convention_column(frame=frame, log=log)
    _validate_quote_base_column(frame=frame, keys=keys, log=log)
    return keys


def _validate_convention_column(*, frame: pl.DataFrame, log: DefectLog) -> None:
    column = COL_DISPLACEMENT_CONVENTION_ID
    dtype = frame.schema[column]
    if not dtype.is_integer():
        log.add(
            source=f"factor_meta.{column}",
            problem=f"dtype {dtype}; the convention is an integer id.",
            remedy=f"use the ids of DisplacementConvention: {_CONVENTION_LABELS}.",
        )
        return
    unknown = frame.select(
        pl.col(column).is_null() | ~pl.col(column).is_in(_VALID_CONVENTION_IDS)
    ).to_series()
    if not bool(unknown.any()):
        return
    log.add(
        source=f"factor_meta.{column}",
        problem=(
            f"{int(unknown.sum())} row(s) carry a null or an id outside "
            f"{_VALID_CONVENTION_IDS}."
        ),
        remedy=(
            f"the displacement convention is the space the sensitivity was "
            f"differentiated in: {_CONVENTION_LABELS}. Applying a shock in the "
            "wrong space is wrong by a factor of the level and produces no error."
        ),
        row_indices=tuple(int(i) for i in unknown.arg_true().head(MAX_REPORTED_ROWS)),
        total_rows=int(unknown.sum()),
    )


def _validate_quote_base_column(
    *, frame: pl.DataFrame, keys: pl.Series, log: DefectLog
) -> None:
    column = COL_QUOTE_BASE
    dtype = frame.schema[column]
    if not dtype.is_numeric():
        log.add(
            source=f"factor_meta.{column}",
            problem=f"dtype {dtype}; quote_base is a positive scalar per factor.",
            remedy=(
                "quote_base is the number of quoted units in one natural unit — "
                "1.0 for natural units, 1e4 for per-basis-point sensitivities."
            ),
        )
        return
    bad = frame.select(
        pl.col(column).is_null()
        | pl.col(column).is_finite().not_()
        | (pl.col(column) <= 0.0)
        if dtype.is_float()
        else pl.col(column).is_null() | (pl.col(column) <= 0)
    ).to_series()
    if not bool(bad.any()):
        return
    log.add(
        source=f"factor_meta.{column}",
        problem=(
            f"{int(bad.sum())} row(s) are null, non-finite, or non-positive. "
            f"Factor(s): {keys.filter(bad).unique().sort().head(MAX_REPORTED_ROWS).to_list()}."
        ),
        remedy=(
            "order-n sensitivities are divided by quote_base ** n, so zero or "
            "negative is not a scaling. Use 1.0 for sensitivities already in "
            "natural units."
        ),
        row_indices=tuple(int(i) for i in bad.arg_true().head(MAX_REPORTED_ROWS)),
        total_rows=int(bad.sum()),
    )


# =========================================================================
# ROW SET
# =========================================================================


def _validate_row_set(*, spec: SensitivityBookSpec, log: DefectLog) -> None:
    declared = spec.position_uids

    if declared is None:
        if _entry_rows(spec) == 0:
            log.add(
                source="sensitivities",
                problem="no rows, and no position_uids were declared.",
                remedy=(
                    "the result would be a (0, S) matrix, which downstream "
                    "aggregation reads as an empty portfolio rather than as an "
                    "error. Declare position_uids to pin the row set."
                ),
            )
        return

    uids = np.asarray(declared)
    if uids.ndim != 1 or not np.issubdtype(uids.dtype, np.integer):
        log.add(
            source="position_uids",
            problem=f"{uids.ndim}-D array of dtype {uids.dtype}.",
            remedy="supply a 1-D sequence of integer position UIDs.",
        )
        return
    unique = np.unique(uids)
    if unique.shape[0] != uids.shape[0]:
        log.add(
            source="position_uids",
            problem=f"{uids.shape[0] - unique.shape[0]} duplicate UID(s).",
            remedy="each declared UID owns exactly one row of the output matrix.",
        )

    referenced = _referenced_positions(spec)
    if referenced is None or referenced.shape[0] == 0:
        return
    undeclared = referenced[~np.isin(referenced, unique)]
    if undeclared.shape[0]:
        log.add(
            source="sensitivities.position_id",
            problem=(
                f"{undeclared.shape[0]} position(s) carry sensitivities but are "
                "absent from position_uids."
            ),
            remedy=(
                "the declared row set is the output contract; an undeclared "
                "position would widen the matrix and break a downstream join. "
                f"First: {undeclared[:MAX_REPORTED_ROWS].tolist()}."
            ),
        )


def _entry_rows(spec: SensitivityBookSpec) -> int:
    cross = spec.cross_sensitivities
    return spec.sensitivities.height + (0 if cross is None else cross.height)


def _referenced_positions(
    spec: SensitivityBookSpec,
) -> npt.NDArray[np.int64] | None:
    """
    Distinct position ids across both entry frames, or `None` if unreadable.

    Computed only when a row set was declared — it is a pass over one column, and
    a run that does not pin its rows has nothing to compare it against.
    """
    columns: list[pl.Series] = []
    for frame in (spec.sensitivities, spec.cross_sensitivities):
        if frame is None or COL_POSITION_ID not in frame.columns:
            continue
        if not frame.schema[COL_POSITION_ID].is_integer():
            return None
        columns.append(frame.get_column(COL_POSITION_ID).drop_nulls().unique())
    if not columns:
        return None
    joined = columns[0] if len(columns) == 1 else pl.concat(columns)
    return np.unique(joined.to_numpy()).astype(np.int64, copy=False)


# =========================================================================
# PRIMITIVES
# =========================================================================


def _require_columns(
    *, frame: pl.DataFrame, source: str, required: tuple[str, ...], log: DefectLog
) -> bool:
    missing = [column for column in required if column not in frame.columns]
    if not missing:
        return True
    log.add(
        source=source,
        problem=f"missing column(s) {missing}.",
        remedy=f"the frame needs {list(required)}; it has {frame.columns[:12]}.",
    )
    return False


def _require_integer(
    *, frame: pl.DataFrame, source: str, column: str, remedy: str, log: DefectLog
) -> bool:
    if frame.schema[column].is_integer():
        return True
    log.add(
        source=f"{source}.{column}",
        problem=f"dtype {frame.schema[column]}; an identifier column must be integer.",
        remedy=remedy,
    )
    return False


def _require_numeric(
    *, frame: pl.DataFrame, source: str, column: str, log: DefectLog
) -> bool:
    if frame.schema[column].is_numeric():
        return True
    log.add(
        source=f"{source}.{column}",
        problem=f"dtype {frame.schema[column]}; a coefficient column must be numeric.",
        remedy=(
            "cast upstream. A string column would be coerced silently and a "
            "malformed entry would land as a null, then as a NaN coefficient."
        ),
    )
    return False


def _require_key_space(
    *, frame: pl.DataFrame, source: str, column: str, axis: FactorAxis, log: DefectLog
) -> bool:
    dtype = frame.schema[column]
    if dtype == axis.keys.dtype:
        return True
    expected = "String labels" if axis.keys.dtype == pl.String else "integer ids"
    log.add(
        source=f"{source}.{column}",
        problem=f"dtype {dtype}, but the factor axis uses {expected} ({axis.keys.dtype}).",
        remedy=(
            "factor keys must be expressed identically on both sides — the axis "
            "defines the universe and sensitivities are resolved into it. Cast "
            "the column, or build the axis from the same key column."
        ),
    )
    return False


def _report_nulls(
    *, frame: pl.DataFrame, source: str, column: str, count: int, log: DefectLog
) -> None:
    if count == 0:
        return
    rows = frame.select(pl.col(column).is_null()).to_series().arg_true()
    log.add(
        source=f"{source}.{column}",
        problem=f"{count} null value(s).",
        remedy=(
            "drop or repair these rows upstream. A null identifier reaches the "
            "compiler as -9223372036854775808 and a null coefficient as NaN."
        ),
        row_indices=tuple(int(i) for i in rows.head(MAX_REPORTED_ROWS)),
        total_rows=count,
    )


def _report_non_finite(
    *, frame: pl.DataFrame, source: str, column: str, count: int, log: DefectLog
) -> None:
    if count == 0:
        return
    rows = frame.select(pl.col(column).is_finite().not_()).to_series().arg_true()
    log.add(
        source=f"{source}.{column}",
        problem=f"{count} non-finite value(s).",
        remedy=(
            "'no exposure' is an absent row, never NaN or inf. A non-finite "
            "coefficient poisons every scenario for that position."
        ),
        row_indices=tuple(int(i) for i in rows.head(MAX_REPORTED_ROWS)),
        total_rows=count,
    )


def _report_absent(
    *,
    source: str,
    distinct: pl.Series,
    present: pl.Series,
    problem_context: str,
    remedy: str,
    log: DefectLog,
) -> None:
    """
    Report keys absent from `present`, named in the caller's own key space.

    `distinct` is a de-duplicated set bound by the CALLER, and the mask is derived
    from that one binding. Polars `.unique()` does not maintain order, so calling
    it twice and filtering one result by a mask built from the other is not merely
    fragile — measured over thirty calls on identical inputs, an earlier version
    of this function returned thirty different answers, each confidently naming
    factors that were not in fact missing.

    `present.implode()`: by the time this line runs, the dtype guard just above
    has already established `distinct.dtype == present.dtype` — exactly the
    "collection of the same datatype" polars now treats as ambiguous between
    element-wise comparison and membership-in-a-collection. `implode()` is the
    library's own documented way to say "the second one, always" without a
    warning; it wraps `present` into a single-row List-typed Series, a metadata
    operation with no data copy.
    """
    if distinct.dtype != present.dtype or distinct.len() == 0:
        return
    absent = distinct.filter(~distinct.is_in(present.implode()))
    if absent.len() == 0:
        return
    shown = absent.sort().head(MAX_REPORTED_ROWS).to_list()
    hidden = absent.len() - len(shown)
    suffix = f" (+{hidden} more)" if hidden > 0 else ""
    log.add(
        source=source,
        problem=f"{absent.len()} factor(s) with {problem_context}: {shown}{suffix}.",
        remedy=remedy,
    )


# =========================================================================
# MARKET AGAINST A COMPILED BOOK
# =========================================================================


def validate_market_against_book(
    *,
    book: CompiledSensitivityBook,
    axis: FactorAxis,
    market: MarketScenarios,
) -> None:
    """
    The checks that need BOTH halves, run before anything is allocated.

    Everything a scenario set can get wrong on its own is a construction
    invariant of `ScenarioShocks` / `ScenarioLevels`. What neither half can see
    alone is whether the LEVELS are admissible under the CONVENTIONS the book
    was compiled with — the market does not know which factors are LOG, and the
    book does not hold the levels.

    `SensitivityPnlEngine` enforces the same rule per batch, and that guard is
    the core-tier invariant rather than a duplicate to be removed: a caller
    reaching the engine directly must not be able to produce a NaN PnL. This
    one exists because the engine names a compiled `factor_id`, which under SLOT
    encoding is a row position and tells a desk user nothing. Here the caller's
    own key is still in scope.
    """
    if not isinstance(market, ScenarioLevels):
        return

    positive_mask = np.isin(
        book.displacement_convention_ids,
        np.asarray(POSITIVE_LEVEL_DISPLACEMENT_IDS, dtype=np.int8),
    )
    if not bool(positive_mask.any()):
        return

    rows = book.active_level_uix[positive_mask]
    log = DefectLog()
    levels = market.shocked_levels[rows]
    row_min = levels.min(axis=1)
    offending = np.flatnonzero(row_min <= 0.0)
    if offending.size:
        keys = axis.keys.to_list()
        named = ", ".join(
            f"{keys[int(rows[i])]!r} (min {float(row_min[i]):.6g})"
            for i in offending[:MAX_REPORTED_ROWS]
        )
        extra = offending.size - min(offending.size, MAX_REPORTED_ROWS)
        log.add(
            source="market.shocked_levels",
            problem=(
                f"{offending.size} factor(s) declared LOG reach a non-positive "
                f"shocked level: {named}" + (f" (+{extra} more)" if extra else "") + "."
            ),
            remedy=(
                "ln of a non-positive level is NaN or -inf. It would reach the "
                "PnL grid silently, poison every scenario of every position "
                "sensitive to that factor, and then sort OUT of the lower tail "
                "— a catastrophic loss becoming an invisible one. Declare the "
                "factor ABSOLUTE if it can cross zero, or repair the scenario."
            ),
            row_indices=tuple(int(rows[i]) for i in offending[:MAX_REPORTED_ROWS]),
            total_rows=int(offending.size),
        )
    log.raise_if_any()
