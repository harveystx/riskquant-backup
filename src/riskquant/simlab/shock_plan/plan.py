# src/riskquant/simlab/shock_plan/plan.py
"""
Compute plan for one world-column request (D088 §4, §12.5).

A world request is K unique `(driver_rf_id, chart_id)` pairs in a stated
order (`simlab.shocks.axis_operator.world_request`: first-seen over the
compiled operator's terms). Column `k` of every generated batch is the world
variable of request pair `k`. This module validates the request and resolves
it into a kernel-friendly layout:

  - `cols_sorted`           : (K,) int32, the matrix (or parametric data)
                              column of each request driver, sorted ascending
                              so the per-thread gather is monotone and
                              cache-friendly.
  - `shift_types_sorted`    : (K,) int8, the requested chart per sorted
                              column (the chart of the world variable, never
                              a target chart).
  - `inverse_permutation`   : (K,) int32, a GATHER map from request order to
                              sorted compute order: `inverse_permutation[k]`
                              is the position `j` in `cols_sorted` of request
                              pair `k`.

Permutation contract (SimLab P0, D030 amendment). Two orders exist:
  - request order (`k`): the order of every emitted column
    (`ScenarioSetBatch.shocks[:, k]`);
  - sorted compute order (`j`): ascending data column; memory layout for the
    kernels only, never an output order.
Every consumer produces output column `k` by gathering sorted position
`inverse_permutation[k]`:
    out[:, k] = Y(cols_sorted[inverse_permutation[k]],
                  shift_types_sorted[inverse_permutation[k]])
Using `inverse_permutation` as a scatter is wrong for every non-involutive
permutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ShockComputePlan:
    """Frozen kernel-facing column-resolution plan."""

    cols_sorted: npt.NDArray[np.int32]
    shift_types_sorted: npt.NDArray[np.int8]
    inverse_permutation: npt.NDArray[np.int32]


#: Charts a world column can carry (D088 §4): the V1 driver charts and the
#: native ABSOLUTE_BASE_RETURN column. NO_SHIFT requests no column (§13.4).
WORLD_COLUMN_CHART_IDS: frozenset[int] = frozenset(
    int(c)
    for c in (
        SHIFT_LOG_RETURN_ID,
        SHIFT_SIMPLE_RETURN_ID,
        SHIFT_DIFFERENCE_ID,
        SHIFT_ABSOLUTE_BASE_RETURN_ID,
    )
)


def build_world_column_plan(
    *,
    driver_rf_ids: npt.NDArray[np.int32],
    chart_ids: npt.NDArray[np.int8],
    factor_ids: npt.NDArray[np.int32],
) -> ShockComputePlan:
    """
    Validate a world request and resolve it into a sorted compute plan.

    Parameters
    ----------
    driver_rf_ids, chart_ids
        ``(K,)`` int32 and int8: the unique ``(d, tau)`` pairs requested, in
        request order.
    factor_ids
        The world's column ids, sorted ascending and unique (the
        `RiskFactorMatrix` / `ParametricRiskData` invariant).

    Raises
    ------
    TypeError
        Wrong dtype or rank.
    ValueError
        Length mismatch, a chart that no world column carries, or a
        duplicate ``(d, tau)`` pair.
    KeyError
        A requested driver that is not a world column.
    """
    for name, arr, dtype in (
        ("driver_rf_ids", driver_rf_ids, np.int32),
        ("chart_ids", chart_ids, np.int8),
        ("factor_ids", factor_ids, np.int32),
    ):
        if not isinstance(arr, np.ndarray) or arr.ndim != 1 or arr.dtype != dtype:
            raise TypeError(f"{name} must be a 1-D {np.dtype(dtype)} ndarray.")
    if driver_rf_ids.shape != chart_ids.shape:
        raise ValueError(
            "driver_rf_ids and chart_ids must have one length; got "
            f"{driver_rf_ids.shape[0]} and {chart_ids.shape[0]}."
        )
    bad = ~np.isin(chart_ids, np.array(sorted(WORLD_COLUMN_CHART_IDS), dtype=np.int8))
    if np.any(bad):
        raise ValueError(
            f"chart_ids contains ids {np.unique(chart_ids[bad]).tolist()} that no "
            f"world column carries; allowed {sorted(WORLD_COLUMN_CHART_IDS)}."
        )
    if driver_rf_ids.size:
        pairs = np.stack([driver_rf_ids, chart_ids.astype(np.int32)], axis=1)
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("the world request must list each (d, tau) pair once.")

    # One-shot vectorised gather: data column for each requested driver.
    cols = np.searchsorted(factor_ids, driver_rf_ids).astype(np.int32, copy=False)

    if cols.size:
        # Bound-check + identity check; one branch, no Python loop.
        in_bounds = cols < factor_ids.shape[0]
        matches = np.zeros(cols.shape[0], dtype=bool)
        matches[in_bounds] = factor_ids[cols[in_bounds]] == driver_rf_ids[in_bounds]
        if not matches.all():
            missing = np.unique(driver_rf_ids[~matches])
            raise KeyError(
                "Requested world-column drivers missing from the world's "
                f"factor ids: {missing.tolist()}"
            )

    shift_types = chart_ids

    # Sort by column index — monotone gathers are cache-friendly inside
    # the per-thread `for k in prange(K)` generator kernels.
    order = np.argsort(cols, kind="stable")
    cols_sorted = np.ascontiguousarray(cols[order])
    shift_sorted = np.ascontiguousarray(shift_types[order])

    inv = np.empty(order.shape[0], dtype=np.int32)
    inv[order] = np.arange(order.shape[0], dtype=np.int32)

    return ShockComputePlan(
        cols_sorted=cols_sorted,
        shift_types_sorted=shift_sorted,
        inverse_permutation=inv,
    )
