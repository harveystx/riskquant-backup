# src/riskquant/simlab/shocks/column_binding.py
"""
World-column binding: ``(d, tau)`` terms plus the world horizon to column indices.

D088 §4 and §12.5: a proxy term declares ``(d, tau)``; a world is one
horizon ``h`` with columns keyed ``(d, tau, h)``; the run binds each term to
the world's declared horizon and the world supplies or rejects the column.
`bind_term_columns` is that binding, on integer keys only.

Rules
-----
- The requested horizon must equal the world's declared horizon. A mismatch
  raises; no column of another horizon is substituted.
- Every requested ``(d, tau, h)`` must exist in the world exactly once. A
  missing key raises; no other chart, driver, node or source is substituted.
- A world with two columns under one key is ambiguous and raises.
- NO_SHIFT is never requested: a NO_SHIFT row needs no column (D088 §13.4).
  Term charts are LOG, SIMPLE, DIFFERENCE (V1 proxy terms) or ABSOLUTE_BASE
  (a native ABS row lowered to one unit-loading term on its own column).

The world's step unit is not represented here; its representation is a
Slice 3 world decision (D088 §13.8(b)). Horizons are compared as the integers
the run and the world declare.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
    VALID_SHIFT_TYPE_IDS,
)

__all__ = ["BINDABLE_CHART_IDS", "bind_term_columns"]

#: Chart ids that may be requested from a world.
BINDABLE_CHART_IDS: frozenset[int] = frozenset(
    int(x)
    for x in (
        SHIFT_LOG_RETURN_ID,
        SHIFT_SIMPLE_RETURN_ID,
        SHIFT_DIFFERENCE_ID,
        SHIFT_ABSOLUTE_BASE_RETURN_ID,
    )
)

_PREVIEW = 10


def bind_term_columns(
    *,
    term_driver_rf_ids: npt.NDArray[np.int32],
    term_chart_ids: npt.NDArray[np.int8],
    requested_horizon: int,
    world_column_driver_rf_ids: npt.NDArray[np.int32],
    world_column_chart_ids: npt.NDArray[np.int8],
    world_horizon: int,
) -> npt.NDArray[np.int32]:
    """
    Resolve each term's ``(d, tau, h)`` to a world column index.

    Parameters
    ----------
    term_driver_rf_ids, term_chart_ids
        ``(T,)`` int32 and int8: the ``(d, tau)`` of each term, for example
        ``CompiledProxyModel.term_driver_rf_ids`` and ``term_chart_ids``.
    requested_horizon
        The horizon the run binds the terms to (a positive ``int``).
    world_column_driver_rf_ids, world_column_chart_ids
        ``(K,)`` int32 and int8: the ``(d, tau)`` of each world column, in
        world column order.
    world_horizon
        The horizon the world declares (a positive ``int``).

    Returns
    -------
    numpy.ndarray
        ``(T,)`` int32, C-contiguous, a new array: the world column index of
        each term, in term order.

    Raises
    ------
    TypeError
        A horizon is not an ``int``, or an array argument is not an ndarray.
    ValueError
        Horizon mismatch, missing or duplicate keys, invalid charts, or any
        dtype or shape violation.
    """
    for name, h in (
        ("requested_horizon", requested_horizon),
        ("world_horizon", world_horizon),
    ):
        if isinstance(h, bool) or not isinstance(h, int):
            raise TypeError(f"{name} must be an int, got {type(h).__name__}.")
        if h <= 0:
            raise ValueError(f"{name} must be positive, got {h}.")
    if requested_horizon != world_horizon:
        raise ValueError(
            f"horizon mismatch: the run requests h={requested_horizon}, the world "
            f"declares h={world_horizon}. No column of another horizon is "
            "substituted (D088 §12.5)."
        )

    _require(term_driver_rf_ids, "term_driver_rf_ids", np.int32)
    _require(term_chart_ids, "term_chart_ids", np.int8)
    _require(world_column_driver_rf_ids, "world_column_driver_rf_ids", np.int32)
    _require(world_column_chart_ids, "world_column_chart_ids", np.int8)
    if term_chart_ids.shape != term_driver_rf_ids.shape:
        raise ValueError("term_driver_rf_ids and term_chart_ids must have one length.")
    if world_column_chart_ids.shape != world_column_driver_rf_ids.shape:
        raise ValueError(
            "world_column_driver_rf_ids and world_column_chart_ids must have one length."
        )

    for name, ids in (
        ("term_driver_rf_ids", term_driver_rf_ids),
        ("world_column_driver_rf_ids", world_column_driver_rf_ids),
    ):
        if ids.size and int(ids.min()) < 0:
            raise ValueError(f"{name} must be non-negative rf_ids.")
    valid = np.array(sorted(VALID_SHIFT_TYPE_IDS), dtype=np.int8)
    bad_world = ~np.isin(world_column_chart_ids, valid)
    if bool(bad_world.any()):
        raise ValueError(
            f"world_column_chart_ids contains ids outside {sorted(VALID_SHIFT_TYPE_IDS)}: "
            f"{np.unique(world_column_chart_ids[bad_world]).tolist()}."
        )

    bindable = np.array(sorted(BINDABLE_CHART_IDS), dtype=np.int8)
    bad = ~np.isin(term_chart_ids, bindable)
    if bool(bad.any()):
        raise ValueError(
            f"term_chart_ids contains ids outside {sorted(BINDABLE_CHART_IDS)}: "
            f"{np.unique(term_chart_ids[bad]).tolist()}. NO_SHIFT rows request "
            "no column (D088 §13.4)."
        )

    world_keys = _keys(world_column_driver_rf_ids, world_column_chart_ids)
    order = np.argsort(world_keys, kind="stable")
    sorted_keys = world_keys[order]
    dup = sorted_keys[1:] == sorted_keys[:-1]
    if bool(dup.any()):
        pairs = [_decode(int(k)) for k in np.unique(sorted_keys[1:][dup])[:_PREVIEW]]
        raise ValueError(
            f"the world has duplicate column keys (d, tau) at h={world_horizon}: "
            f"{pairs}. Binding would be ambiguous."
        )

    term_keys = _keys(term_driver_rf_ids, term_chart_ids)
    n_terms = term_keys.shape[0]
    if n_terms == 0:
        return np.empty(0, dtype=np.int32)
    if sorted_keys.shape[0] == 0:
        found = np.zeros(n_terms, dtype=np.bool_)
        pos = np.zeros(n_terms, dtype=np.int64)
    else:
        pos = np.searchsorted(sorted_keys, term_keys)
        np.clip(pos, 0, sorted_keys.shape[0] - 1, out=pos)
        found = sorted_keys[pos] == term_keys
    if not bool(found.all()):
        missing = [
            (*_decode(int(k)), requested_horizon)
            for k in np.unique(term_keys[~found])[:_PREVIEW]
        ]
        raise ValueError(
            f"{int((~found).sum())} term(s) request world columns (d, tau, h) "
            f"that the world does not supply: {missing}. No other chart, driver, "
            "horizon or source is substituted (D088 §9, §13.1)."
        )
    return np.ascontiguousarray(order[pos], dtype=np.int32)


def _require(arr: object, name: str, dtype: type[np.generic]) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}.")
    if arr.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}, got {arr.dtype}.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")


def _keys(
    driver_rf_ids: npt.NDArray[np.int32], chart_ids: npt.NDArray[np.int8]
) -> npt.NDArray[np.int64]:
    # chart id + 1 lies in [0, 7]; 8 slots per driver keep keys unique.
    return driver_rf_ids.astype(np.int64) * 8 + (chart_ids.astype(np.int64) + 1)


def _decode(key: int) -> tuple[int, int]:
    return key // 8, key % 8 - 1
