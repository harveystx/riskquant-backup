# src/riskquant/simlab/shocks/axis_operator.py
"""
Execution-axis lowering of the compiled Proxy V1 operator (D088 Slice 3).

The compiled model (``CompiledProxyModel``) is the sole owner of Proxy V1
semantics: its rows are the V1 axis nodes, its terms are canonical
``(d, tau, beta)`` triples. The Slice 2 kernel needs one row per execution-axis
node together with that node's reconstruction chart ``kappa``. This module is
the numerical step between the two; it is not a second compiler.

Row lowering (D088 §12.4, §13.4)
--------------------------------
- V1 row (``kappa`` in LOG, SIMPLE, DIFFERENCE): the model's canonical terms,
  copied unchanged and in canonical order.
- ABSOLUTE_BASE_RETURN row: one unit-loading term on the node's own native
  column ``(A, ABSOLUTE_BASE_RETURN)`` (D025). Numerical representation only;
  it is not part of the proxy model and not part of ``proxy_model_digest``.
- NO_SHIFT row: zero terms; ``T^-1_NO_SHIFT(A_0, X) = A_0`` needs no column.

The model is run-bound (D088 §12.1): its rows must equal, element for element
and in order, the V1 subsequence of the execution axis. A mismatch raises; the
lowering never reorders, sorts or repairs the model or the axis.

Nothing here resolves names, canonicalises, hashes, selects sources, infers a
driver chart from ``kappa`` or knows a horizon. The world request lists the
``(d, tau)`` pairs the lowered operator reads; the horizon is attached by the
world (D088 §12.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from riskquant.config.proxy_model.compiled import (
    V1_TARGET_CHART_IDS,
    CompiledProxyModel,
)
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
)

__all__ = ["AxisOperator", "lower_axis_operator", "world_request"]

_G = TypeVar("_G", bound=np.generic)

_V1 = np.array(sorted(V1_TARGET_CHART_IDS), dtype=np.int8)
_PREVIEW = 10


def _frozen(arr: npt.NDArray[_G]) -> npt.NDArray[_G]:
    out = np.ascontiguousarray(arr)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AxisOperator:
    """
    Slice 2 operator over the execution axis, in axis order (CSR over rows).

    ``row_target_rf_ids``     (N,)    int32   execution-axis node per row
    ``target_shift_type_ids`` (N,)    int8    reconstruction chart kappa per row
    ``row_term_offsets``      (N+1,)  int32   row n owns terms [off[n], off[n+1])
    ``term_driver_rf_ids``    (T,)    int32   driver node d per term
    ``term_chart_ids``        (T,)    int8    driver chart tau per term
    ``term_loadings``         (T,)    float64 loading beta per term

    Every array is read-only and C-contiguous. Built only by
    ``lower_axis_operator``.
    """

    row_target_rf_ids: npt.NDArray[np.int32]
    target_shift_type_ids: npt.NDArray[np.int8]
    row_term_offsets: npt.NDArray[np.int32]
    term_driver_rf_ids: npt.NDArray[np.int32]
    term_chart_ids: npt.NDArray[np.int8]
    term_loadings: npt.NDArray[np.float64]


def lower_axis_operator(
    *,
    model: CompiledProxyModel,
    axis_rf_ids: npt.NDArray[np.int32],
    axis_shift_type_ids: npt.NDArray[np.int8],
) -> AxisOperator:
    """
    Lower the compiled run-bound model onto the execution axis.

    Parameters
    ----------
    model
        The compiled Proxy V1 operator of this run. Not modified.
    axis_rf_ids
        ``(N,)`` int32, unique: the execution-axis nodes, in axis order.
    axis_shift_type_ids
        ``(N,)`` int8: each node's configured ``shift_type`` (the target
        definition's ``kappa``). LOG, SIMPLE, DIFFERENCE, ABSOLUTE_BASE_RETURN
        or NO_SHIFT.

    Returns
    -------
    AxisOperator
        One row per axis node, in axis order.

    Raises
    ------
    TypeError
        Wrong array type, dtype or rank.
    ValueError
        Length mismatch, duplicate axis node, unsupported shift type, or model
        rows that are not exactly the V1 subsequence of the axis.
    """
    for name, arr, dtype in (
        ("axis_rf_ids", axis_rf_ids, np.int32),
        ("axis_shift_type_ids", axis_shift_type_ids, np.int8),
    ):
        if not isinstance(arr, np.ndarray) or arr.dtype != dtype or arr.ndim != 1:
            raise TypeError(f"{name} must be a 1-D {np.dtype(dtype)} ndarray.")
    n = axis_rf_ids.shape[0]
    if axis_shift_type_ids.shape[0] != n:
        raise ValueError(
            f"axis_rf_ids ({n}) and axis_shift_type_ids "
            f"({axis_shift_type_ids.shape[0]}) must have one length."
        )
    if np.unique(axis_rf_ids).shape[0] != n:
        raise ValueError("axis_rf_ids must be unique (one row per node).")

    kappa = axis_shift_type_ids
    v1 = np.isin(kappa, _V1)
    native_abs = kappa == SHIFT_ABSOLUTE_BASE_RETURN_ID
    no_shift = kappa == SHIFT_NO_SHIFT_ID
    unsupported = ~(v1 | native_abs | no_shift)
    if np.any(unsupported):
        raise ValueError(
            "axis_shift_type_ids contains unsupported ids "
            f"{np.unique(kappa[unsupported]).tolist()}."
        )

    # D088 §12.1: the model is run-bound. Its rows are the V1 axis nodes in
    # axis order, exactly; nothing is reordered or repaired.
    v1_axis = axis_rf_ids[v1]
    if not np.array_equal(model.row_target_rf_ids, v1_axis):
        raise ValueError(
            "the compiled proxy model is not bound to this execution axis: its "
            f"{model.row_target_rf_ids.shape[0]} row target(s) "
            f"{model.row_target_rf_ids[:_PREVIEW].tolist()} must equal, in order, "
            f"the axis's {v1_axis.shape[0]} V1 node(s) {v1_axis[:_PREVIEW].tolist()}."
        )

    model_counts = np.diff(model.row_term_offsets).astype(np.int64)
    counts = np.zeros(n, dtype=np.int64)
    counts[v1] = model_counts
    counts[native_abs] = 1
    offsets64 = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets64[1:])
    n_terms = int(offsets64[-1])
    if n_terms > np.iinfo(np.int32).max:
        raise ValueError(f"term count {n_terms} exceeds the int32 offset range.")

    drivers = np.empty(n_terms, dtype=np.int32)
    charts = np.empty(n_terms, dtype=np.int8)
    loadings = np.empty(n_terms, dtype=np.float64)

    # V1 rows: copy each model row's terms into its axis row, order unchanged.
    t_model = model.term_driver_rf_ids.shape[0]
    intra = np.arange(t_model, dtype=np.int64) - np.repeat(
        model.row_term_offsets[:-1].astype(np.int64), model_counts
    )
    dest = np.repeat(offsets64[:-1][v1], model_counts) + intra
    drivers[dest] = model.term_driver_rf_ids
    charts[dest] = model.term_chart_ids
    loadings[dest] = model.term_loadings

    # Native ABSOLUTE_BASE rows: one unit term on the node's own native column.
    abs_dest = offsets64[:-1][native_abs]
    drivers[abs_dest] = axis_rf_ids[native_abs]
    charts[abs_dest] = SHIFT_ABSOLUTE_BASE_RETURN_ID
    loadings[abs_dest] = 1.0

    return AxisOperator(
        row_target_rf_ids=_frozen(np.array(axis_rf_ids, dtype=np.int32, copy=True)),
        target_shift_type_ids=_frozen(np.array(kappa, dtype=np.int8, copy=True)),
        row_term_offsets=_frozen(offsets64.astype(np.int32)),
        term_driver_rf_ids=_frozen(drivers),
        term_chart_ids=_frozen(charts),
        term_loadings=_frozen(loadings),
    )


def world_request(
    operator: AxisOperator,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int8]]:
    """
    The world variables the operator reads: unique ``(d, tau)`` pairs.

    Deterministic first-seen order over the operator's terms (axis order, then
    canonical term order). NO_SHIFT rows contribute nothing. The horizon is not
    part of the request; the world attaches it.

    Returns
    -------
    (driver_rf_ids, chart_ids)
        ``(K,)`` int32 and ``(K,)`` int8, read-only.
    """
    pairs = np.stack(
        [
            operator.term_driver_rf_ids.astype(np.int64),
            operator.term_chart_ids.astype(np.int64),
        ],
        axis=1,
    )
    if pairs.shape[0] == 0:
        return (
            _frozen(np.empty(0, dtype=np.int32)),
            _frozen(np.empty(0, dtype=np.int8)),
        )
    _, first = np.unique(pairs, axis=0, return_index=True)
    first.sort()
    return (
        _frozen(operator.term_driver_rf_ids[first].astype(np.int32)),
        _frozen(operator.term_chart_ids[first].astype(np.int8)),
    )
