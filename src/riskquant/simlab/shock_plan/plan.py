# src/riskquant/simlab/shock_plan/plan.py
"""
Compiled shock-compute plan.

Resolves the K_req unique `(source_factor_id, shift_type_id)` pairs into
a kernel-friendly layout:

  - `cols_sorted`           : (K_req,) int32 — column indices into the
                              risk-factor matrix, sorted ascending so
                              Numba's per-thread gather is monotone and
                              cache-friendly.
  - `shift_types_sorted`    : (K_req,) int8  — shift-type id per shock,
                              aligned with cols_sorted.
  - `inverse_permutation`   : (K_req,) int32 — for each shock-universe
                              uix `k`, the position of its column in the
                              kernel output (i.e. the inverse of the
                              sort order).

`build_shock_compute_plan` takes `factor_ids: np.ndarray` (the matrix's
sorted-ascending int32 column-id index) instead of a `dict[int, int]`,
which collapses the previous Python-comprehension + dict-lookup gather
into one `np.searchsorted` plus an integrity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class ShockComputePlan:
    """Frozen kernel-facing column-resolution plan."""

    cols_sorted: npt.NDArray[np.int32]
    shift_types_sorted: npt.NDArray[np.int8]
    inverse_permutation: npt.NDArray[np.int32]


def build_shock_compute_plan(
    *,
    shock_source_ids: npt.NDArray[np.int32],
    shock_shift_type_ids: npt.NDArray[np.int8],
    factor_ids: npt.NDArray[np.int32],
) -> ShockComputePlan:
    """
    Resolve `shock_source_ids` into matrix-column indices and produce a
    sorted compute plan.

    `factor_ids` MUST be sorted ascending and unique — that is the
    `RiskFactorMatrix` invariant. Membership is verified via the
    standard `searchsorted` + equality pattern; missing ids raise.
    """
    if factor_ids.dtype != np.int32:
        raise TypeError(f"factor_ids must be int32; got {factor_ids.dtype}")
    if shock_source_ids.dtype != np.int32:
        raise TypeError(f"shock_source_ids must be int32; got {shock_source_ids.dtype}")

    # One-shot vectorised gather: matrix column for each shock source id.
    cols = np.searchsorted(factor_ids, shock_source_ids).astype(np.int32, copy=False)

    if cols.size:
        # Bound-check + identity check; one branch, no Python loop.
        in_bounds = cols < factor_ids.shape[0]
        matches = np.zeros(cols.shape[0], dtype=bool)
        matches[in_bounds] = factor_ids[cols[in_bounds]] == shock_source_ids[in_bounds]
        if not matches.all():
            missing = np.unique(shock_source_ids[~matches])
            raise KeyError(
                f"Shock source factor ids missing from generator's "
                f"available_factor_ids: {missing.tolist()}"
            )

    shift_types = shock_shift_type_ids.astype(np.int8, copy=False)

    # Sort by column index — monotone gathers are cache-friendly inside
    # the per-thread `for k in prange(K)` shock kernel.
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
