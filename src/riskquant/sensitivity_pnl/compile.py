# src/riskquant/sensitivity_pnl/compile.py
"""
Sensitivity-book compiler: human-readable Polars at the edge, CSR SoA at the core.

Owns every policy decision. The kernels own none.

  1. Strings die here. `position_id` / `factor_id` resolve to integers via
     caller-supplied lookups; any unmapped label raises.
  2. Factor alignment is a STRICT inner join against the caller's level
     universe. A sensitivity to a factor outside the scenario universe would
     silently zero that risk in every scenario — the single most dangerous
     failure mode in this module — so it raises rather than quarantines.
     Coverage triage belongs upstream, in the assembler.
  3. Factors are compacted to the active set actually referenced by the book,
     never dense-by-universe-row.
  4. Quote conventions are folded into the coefficients (`quote_base ** n`), so
     the compiled arrays are in natural units and the kernels are unit-free.
  5. Rows are lexsorted into CSR order; duplicate `(position, factor)` pairs
     raise rather than being summed — a duplicate is a broken upstream join,
     and leg-level aggregation is an assembler concern.

Edge schema
-----------
`sensitivities`        position_id, factor_id, first_order, second_order
`cross_sensitivities`  position_id, factor_id_i, factor_id_j, cross_second_order
`factor_meta`          factor_id, displacement_convention_id, quote_base

Row order of the compiled book — and therefore of the emitted PnL block — is
ascending `position_uid`.
"""

from __future__ import annotations

__all__ = [
    "COL_CROSS_SECOND_ORDER",
    "COL_DISPLACEMENT_CONVENTION_ID",
    "COL_FACTOR_ID",
    "COL_FACTOR_ID_I",
    "COL_FACTOR_ID_J",
    "COL_FIRST_ORDER",
    "COL_POSITION_ID",
    "COL_QUOTE_BASE",
    "COL_SECOND_ORDER",
    "DuplicatePolicy",
    "compile_sensitivity_book",
]

from collections.abc import Mapping
from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant._arrays import searchsorted_or_raise
from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.conventions import DisplacementConvention

COL_POSITION_ID = "position_id"
COL_FACTOR_ID = "factor_id"
COL_FIRST_ORDER = "first_order"
COL_SECOND_ORDER = "second_order"
COL_FACTOR_ID_I = "factor_id_i"
COL_FACTOR_ID_J = "factor_id_j"
COL_CROSS_SECOND_ORDER = "cross_second_order"
COL_DISPLACEMENT_CONVENTION_ID = "displacement_convention_id"
COL_QUOTE_BASE = "quote_base"

_VALID_CONVENTION_IDS = np.array(
    sorted(int(c) for c in DisplacementConvention), dtype=np.int8
)


class DuplicatePolicy(IntEnum):
    """
    What to do with repeated `(position, factor)` / `(position, i, j)` keys.

    RAISE (default)
        A duplicate is a fanned-out upstream join. Summing it silently would
        mask the fan-out and inflate every risk number for that position.

    SUM
        The feed is leg-level by construction — a vanilla swap carries DV01 to
        the same curve on both legs — and the assembler is not going to
        pre-aggregate. Coefficients add, which is exact: the derivative of a sum
        of legs is the sum of the leg derivatives.

    Set this per feed at the call site. It is not a fallback to retry with when
    RAISE fires; if a feed is not leg-level, a duplicate is still a defect.
    """

    RAISE = 1
    SUM = 2


# =========================================================================
# PUBLIC API
# =========================================================================


def compile_sensitivity_book(
    *,
    sensitivities: pl.DataFrame,
    factor_meta: pl.DataFrame,
    level_universe_factor_ids: npt.NDArray[np.int32],
    cross_sensitivities: pl.DataFrame | None = None,
    position_id_lookup: Mapping[str, int] | None = None,
    factor_id_lookup: Mapping[str, int] | None = None,
    extra_position_uids: npt.NDArray[np.int64] | None = None,
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.RAISE,
) -> CompiledSensitivityBook:
    """
    Compile an edge sensitivity book into kernel-facing CSR SoA arrays.

    Parameters
    ----------
    level_universe_factor_ids
        `(N_req,)` int32, sorted ascending and unique — the row index of the
        base-levels vector and shocked-levels matrix the book will be evaluated
        against. Compiled `active_level_uix` values index into it.
    extra_position_uids
        Positions that must occupy a row of the output even though they carry
        no sensitivities (fully hedged, or awaiting delivery). They emit exact
        zeros rather than being dropped, which keeps the block's row set stable
        across runs.
    duplicate_policy
        `RAISE` treats a repeated key as a broken upstream join; `SUM` treats
        the feed as leg-level and adds the coefficients. See `DuplicatePolicy`.

    Raises
    ------
    ValueError
        On any unmapped label, missing factor, duplicate pair, non-finite
        coefficient, or malformed convention.
    """
    _require_sorted_unique(level_universe_factor_ids, ctx="level_universe_factor_ids")

    diag = _resolve_frame(
        frame=sensitivities,
        id_columns={
            COL_POSITION_ID: (position_id_lookup, pl.Int64),
            COL_FACTOR_ID: (factor_id_lookup, pl.Int32),
        },
        value_columns=(COL_FIRST_ORDER, COL_SECOND_ORDER),
        ctx="sensitivities",
    )
    cross = (
        None
        if cross_sensitivities is None
        else _resolve_frame(
            frame=cross_sensitivities,
            id_columns={
                COL_POSITION_ID: (position_id_lookup, pl.Int64),
                COL_FACTOR_ID_I: (factor_id_lookup, pl.Int32),
                COL_FACTOR_ID_J: (factor_id_lookup, pl.Int32),
            },
            value_columns=(COL_CROSS_SECOND_ORDER,),
            ctx="cross_sensitivities",
        )
    )

    diag_position = _column_i64(diag, COL_POSITION_ID)
    diag_factor = _column_i32(diag, COL_FACTOR_ID)
    first_order = _column_f64(diag, COL_FIRST_ORDER)
    second_order = _column_f64(diag, COL_SECOND_ORDER)

    if cross is None:
        cross_position = np.empty(0, dtype=np.int64)
        cross_factor_i = np.empty(0, dtype=np.int32)
        cross_factor_j = np.empty(0, dtype=np.int32)
        cross_value = np.empty(0, dtype=np.float64)
    else:
        cross_position = _column_i64(cross, COL_POSITION_ID)
        cross_factor_i = _column_i32(cross, COL_FACTOR_ID_I)
        cross_factor_j = _column_i32(cross, COL_FACTOR_ID_J)
        cross_value = _column_f64(cross, COL_CROSS_SECOND_ORDER)

    _require_finite(first_order, ctx="first_order")
    _require_finite(second_order, ctx="second_order")
    _require_finite(cross_value, ctx="cross_second_order")

    # ---- 1. Active-factor compaction ----------------------------------
    active_factor_ids = np.unique(
        np.concatenate((diag_factor, cross_factor_i, cross_factor_j))
    ).astype(np.int32, copy=False)

    active_level_uix = searchsorted_or_raise(
        universe=level_universe_factor_ids,
        raw=active_factor_ids,
        ctx="sensitivity factor_id -> level universe",
    ).astype(np.int32, copy=False)

    convention_ids, quote_base = _gather_factor_meta(
        factor_meta=factor_meta,
        active_factor_ids=active_factor_ids,
        factor_id_lookup=factor_id_lookup,
    )

    # ---- 2. Row (position) axis ---------------------------------------
    position_pool = [diag_position, cross_position]
    if extra_position_uids is not None:
        position_pool.append(np.ascontiguousarray(extra_position_uids, dtype=np.int64))
    position_uids = np.unique(np.concatenate(position_pool)).astype(
        np.int64, copy=False
    )
    num_positions = position_uids.shape[0]

    # ---- 3. Diagonal CSR block ----------------------------------------
    diag_row = searchsorted_or_raise(
        universe=position_uids, raw=diag_position, ctx="diagonal position_id"
    )
    diag_uix = searchsorted_or_raise(
        universe=active_factor_ids, raw=diag_factor, ctx="diagonal factor_id"
    ).astype(np.int32, copy=False)

    order = np.lexsort((diag_uix, diag_row))
    diag_row = diag_row[order]
    diag_uix = np.ascontiguousarray(diag_uix[order])
    first_order = np.ascontiguousarray(first_order[order])
    second_order = np.ascontiguousarray(second_order[order])

    keep, (first_order, second_order) = _collapse_duplicates(
        keys=(diag_row, diag_uix.astype(np.int64, copy=False)),
        values=(first_order, second_order),
        policy=duplicate_policy,
        ctx="(position_id, factor_id)",
    )
    diag_row = diag_row[keep]
    diag_uix = np.ascontiguousarray(diag_uix[keep])

    position_offsets = _csr_offsets(rows=diag_row, num_rows=num_positions)

    # Fold quotation into the coefficients: order-n scales by base**n.
    base_per_entry = quote_base[diag_uix]
    first_order *= base_per_entry
    second_order *= base_per_entry * base_per_entry

    # ---- 4. Cross CSR block -------------------------------------------
    (
        cross_offsets,
        cross_uix_i,
        cross_uix_j,
        cross_value,
    ) = _compile_cross_block(
        cross_position=cross_position,
        cross_factor_i=cross_factor_i,
        cross_factor_j=cross_factor_j,
        cross_value=cross_value,
        position_uids=position_uids,
        active_factor_ids=active_factor_ids,
        quote_base=quote_base,
        num_positions=num_positions,
        duplicate_policy=duplicate_policy,
    )

    return CompiledSensitivityBook(
        position_uids=position_uids,
        active_factor_ids=active_factor_ids,
        active_level_uix=active_level_uix,
        displacement_convention_ids=convention_ids,
        position_offsets=position_offsets,
        factor_uix=diag_uix,
        first_order=first_order,
        second_order=second_order,
        cross_offsets=cross_offsets,
        cross_factor_uix_i=cross_uix_i,
        cross_factor_uix_j=cross_uix_j,
        cross_second_order=cross_value,
    )


# =========================================================================
# CROSS BLOCK
# =========================================================================


def _compile_cross_block(
    *,
    cross_position: npt.NDArray[np.int64],
    cross_factor_i: npt.NDArray[np.int32],
    cross_factor_j: npt.NDArray[np.int32],
    cross_value: npt.NDArray[np.float64],
    position_uids: npt.NDArray[np.int64],
    active_factor_ids: npt.NDArray[np.int32],
    quote_base: npt.NDArray[np.float64],
    num_positions: int,
    duplicate_policy: DuplicatePolicy,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
]:
    """
    Canonicalise cross terms to `i < j`, one entry per unordered pair.

    A pair with `i == j` is diagonal curvature and belongs in `second_order`;
    accepting it here would double-count against the 1/2 factor the Taylor
    kernel applies to the diagonal block.
    """
    if cross_position.size == 0:
        return (
            np.zeros(num_positions + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

    uix_i = searchsorted_or_raise(
        universe=active_factor_ids, raw=cross_factor_i, ctx="cross factor_id_i"
    ).astype(np.int32, copy=False)
    uix_j = searchsorted_or_raise(
        universe=active_factor_ids, raw=cross_factor_j, ctx="cross factor_id_j"
    ).astype(np.int32, copy=False)

    if np.any(uix_i == uix_j):
        raise ValueError(
            "cross_sensitivities: factor_id_i == factor_id_j on "
            f"{int(np.count_nonzero(uix_i == uix_j))} row(s). Diagonal curvature "
            "belongs in `second_order`, not the cross block."
        )

    lo = np.minimum(uix_i, uix_j)
    hi = np.maximum(uix_i, uix_j)

    rows = searchsorted_or_raise(
        universe=position_uids, raw=cross_position, ctx="cross position_id"
    )

    order = np.lexsort((hi, lo, rows))
    rows = rows[order]
    lo = np.ascontiguousarray(lo[order])
    hi = np.ascontiguousarray(hi[order])
    value = np.ascontiguousarray(cross_value[order])

    keep, (value,) = _collapse_duplicates(
        keys=(
            rows,
            lo.astype(np.int64, copy=False),
            hi.astype(np.int64, copy=False),
        ),
        values=(value,),
        policy=duplicate_policy,
        ctx="(position_id, factor_id_i, factor_id_j)",
    )
    rows = rows[keep]
    lo = np.ascontiguousarray(lo[keep])
    hi = np.ascontiguousarray(hi[keep])

    value *= quote_base[lo] * quote_base[hi]

    return _csr_offsets(rows=rows, num_rows=num_positions), lo, hi, value


# =========================================================================
# EDGE RESOLUTION
# =========================================================================


def _resolve_frame(
    *,
    frame: pl.DataFrame,
    id_columns: Mapping[str, tuple[Mapping[str, int] | None, type[pl.DataType]]],
    value_columns: tuple[str, ...],
    ctx: str,
) -> pl.DataFrame:
    """Validate the edge schema and collapse every label column to integers."""
    required = (*id_columns.keys(), *value_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required column(s) {missing}")

    projections: list[pl.Expr] = []
    for column, (lookup, dtype) in id_columns.items():
        projections.append(_label_to_int(column, lookup, dtype, ctx))
    for column in value_columns:
        projections.append(pl.col(column).cast(pl.Float64, strict=True).alias(column))

    return frame.select(projections)


def _label_to_int(
    column: str,
    lookup: Mapping[str, int] | None,
    dtype: type[pl.DataType],
    ctx: str,
) -> pl.Expr:
    """
    Strings die at the perimeter. `replace_strict` raises on any label absent
    from the lookup, which is the fail-fast behaviour we want.
    """
    expr = pl.col(column)
    if lookup is None:
        return expr.cast(dtype, strict=True).alias(column)
    return expr.replace_strict(dict(lookup), return_dtype=dtype).alias(column)


def _gather_factor_meta(
    *,
    factor_meta: pl.DataFrame,
    active_factor_ids: npt.NDArray[np.int32],
    factor_id_lookup: Mapping[str, int] | None,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    """Resolve per-active-factor displacement convention and quote base."""
    required = (COL_FACTOR_ID, COL_DISPLACEMENT_CONVENTION_ID, COL_QUOTE_BASE)
    missing = [column for column in required if column not in factor_meta.columns]
    if missing:
        raise ValueError(f"factor_meta: missing required column(s) {missing}")

    resolved = factor_meta.select(
        _label_to_int(COL_FACTOR_ID, factor_id_lookup, pl.Int32, "factor_meta"),
        pl.col(COL_DISPLACEMENT_CONVENTION_ID).cast(pl.Int8, strict=True),
        pl.col(COL_QUOTE_BASE).cast(pl.Float64, strict=True),
    ).sort(COL_FACTOR_ID)

    meta_ids = _column_i32(resolved, COL_FACTOR_ID)
    if meta_ids.size and int(np.min(np.diff(meta_ids), initial=1)) <= 0:
        raise ValueError("factor_meta: duplicate factor_id rows")

    slots = searchsorted_or_raise(
        universe=meta_ids, raw=active_factor_ids, ctx="factor_meta factor_id"
    )
    convention_ids = np.ascontiguousarray(
        _column_i8(resolved, COL_DISPLACEMENT_CONVENTION_ID)[slots]
    )
    quote_base = np.ascontiguousarray(_column_f64(resolved, COL_QUOTE_BASE)[slots])

    unknown = np.setdiff1d(convention_ids, _VALID_CONVENTION_IDS)
    if unknown.size:
        raise ValueError(
            f"factor_meta: unknown displacement_convention_id {unknown.tolist()}; "
            f"expected one of {_VALID_CONVENTION_IDS.tolist()}"
        )
    if not np.all(np.isfinite(quote_base)) or np.any(quote_base <= 0.0):
        raise ValueError("factor_meta: quote_base must be finite and strictly positive")

    return convention_ids, quote_base


# =========================================================================
# ARRAY PRIMITIVES
# =========================================================================


def _csr_offsets(
    *, rows: npt.NDArray[np.int64], num_rows: int
) -> npt.NDArray[np.int32]:
    """Row pointers from a lexsorted row-index array."""
    offsets = np.empty(num_rows + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(
        np.bincount(rows, minlength=num_rows).astype(np.int32, copy=False),
        out=offsets[1:],
    )
    return offsets


def _collapse_duplicates(
    *,
    keys: tuple[npt.NDArray[np.int64], ...],
    values: tuple[npt.NDArray[np.float64], ...],
    policy: DuplicatePolicy,
    ctx: str,
) -> tuple[npt.NDArray[np.int64], tuple[npt.NDArray[np.float64], ...]]:
    """
    Resolve repeated keys in a lexsorted block.

    Returns the indices of the surviving rows and the resolved value arrays.
    Under `SUM`, group sums come from one `np.add.reduceat` per value column
    over the group-start offsets — no Python loop, no groupby, no dict.
    """
    n = keys[0].shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), values

    is_group_start = np.ones(n, dtype=np.bool_)
    if n > 1:
        same_as_previous = np.ones(n - 1, dtype=np.bool_)
        for key in keys:
            same_as_previous &= key[1:] == key[:-1]
        is_group_start[1:] = ~same_as_previous

    starts = np.flatnonzero(is_group_start)
    if starts.shape[0] == n:
        return starts, values

    if policy is DuplicatePolicy.RAISE:
        raise ValueError(
            f"{n - starts.shape[0]} duplicate {ctx} row(s). Aggregate leg-level "
            "sensitivities upstream, or pass "
            "duplicate_policy=DuplicatePolicy.SUM if the feed is leg-level "
            "by construction."
        )

    collapsed = tuple(
        np.ascontiguousarray(np.add.reduceat(value, starts)) for value in values
    )
    return starts, collapsed


def _require_sorted_unique(arr: npt.NDArray[Any], *, ctx: str) -> None:
    if arr.ndim != 1:
        raise ValueError(f"{ctx} must be 1-D")
    if arr.size > 1 and not bool(np.all(np.diff(arr) > 0)):
        raise ValueError(f"{ctx} must be sorted ascending and unique")


def _require_finite(arr: npt.NDArray[np.float64], *, ctx: str) -> None:
    if arr.size and not bool(np.all(np.isfinite(arr))):
        raise ValueError(
            f"{ctx}: non-finite coefficient(s). Sensitivities use absent CSR "
            "entries for 'no exposure', never NaN."
        )


def _column_i64(frame: pl.DataFrame, column: str) -> npt.NDArray[np.int64]:
    return np.ascontiguousarray(frame.get_column(column).to_numpy(), dtype=np.int64)


def _column_i32(frame: pl.DataFrame, column: str) -> npt.NDArray[np.int32]:
    return np.ascontiguousarray(frame.get_column(column).to_numpy(), dtype=np.int32)


def _column_i8(frame: pl.DataFrame, column: str) -> npt.NDArray[np.int8]:
    return np.ascontiguousarray(frame.get_column(column).to_numpy(), dtype=np.int8)


def _column_f64(frame: pl.DataFrame, column: str) -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(frame.get_column(column).to_numpy(), dtype=np.float64)
