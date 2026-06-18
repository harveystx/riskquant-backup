# src/riskquant/_arrays.py
"""
Cross-cutting NumPy array primitives shared across packages.

Top-level private module (mirrors `_njit.py`) so leaf consumers in
`proxy_resolver`, `simlab`, `portfolio`, `market_data`, and `credit` can import
these without a layering inversion.

No domain imports — pure array algebra; domain sentinels (e.g., `INVALID_UIX`
or `NO_UID_MATCH_IDX_I32`) are passed in by callers rather than imported here.

Contents
--------
unique_first_seen                        : dtype-preserving first-seen dedup + inverse.
searchsorted_membership                  : bool found-mask over a sorted-unique universe.
searchsorted_or_raise                    : STRICT INNER JOIN. Raises on any miss.
searchsorted_soft_join                   : SAFE LEFT JOIN. Probes existence, maps misses to a sentinel.
searchsorted_or_propagate_sentinel_int32 : UNSAFE LEFT JOIN. Bypasses probe, blindly maps `< 0` to a sentinel.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

# Centralized preview limit for error-message diagnostics to prevent
# massive stdout dumps on join failures.
_PREVIEW: Final[int] = 10


def _as_c_contig_f64(arr: NDArray) -> NDArray[np.float64]:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_c_contig_f32(arr: NDArray) -> NDArray[np.float32]:
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_c_contig_i64(arr: NDArray) -> NDArray[np.int64]:
    return np.ascontiguousarray(arr, dtype=np.int64)


def _as_c_contig_i32(arr: NDArray) -> NDArray[np.int32]:
    return np.ascontiguousarray(arr, dtype=np.int32)


def _as_c_contig_i16(arr: NDArray) -> NDArray[np.int16]:
    return np.ascontiguousarray(arr, dtype=np.int16)


def _as_c_contig_i8(arr: NDArray) -> NDArray[np.int8]:
    return np.ascontiguousarray(arr, dtype=np.int8)


def _as_c_contig_bool(arr: NDArray) -> NDArray[np.bool_]:
    return np.ascontiguousarray(arr, dtype=np.bool_)


def unique_first_seen(
    arr: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray[np.int32]]:
    """
    First-seen dedup: dtype-preserving in the values, int32 in the inverse.

    `np.unique` sorts; we need first-seen order so shock-compute /
    proxy-resolution plans are deterministic across reruns of identical inputs.

    Returns
    -------
    unique_first_seen
        (K,) — unique values in first-seen order, same dtype as ``arr``.

    inverse_first_seen
        (N,) int32 — for each input position, its index in ``unique_first_seen``.
    """
    if arr.size == 0:
        return arr[:0].copy(), np.empty(0, dtype=np.int32)

    _, first_idx, inverse_sorted = np.unique(
        arr, return_index=True, return_inverse=True
    )
    # Reorder sorted-unique into first-seen order.
    order = np.argsort(first_idx, kind="mergesort")
    out_unique = arr[first_idx[order]]

    # Build sorted_idx -> first_seen_idx remap, then apply to the inverse.
    sorted_to_first_seen = np.empty(order.size, dtype=np.int32)
    sorted_to_first_seen[order] = np.arange(order.size, dtype=np.int32)
    inverse_first_seen = sorted_to_first_seen[inverse_sorted].astype(
        np.int32, copy=False
    )
    return out_unique, inverse_first_seen


# =========================================================================
# VECTORIZED JOIN PRIMITIVES
# =========================================================================
#
# `universe` MUST be sorted ascending and unique (caller's invariant).
# These do NOT validate that invariant; they trust it, exactly as the
# hand-rolled sites they replace did.


def searchsorted_membership(
    *,
    universe: npt.NDArray,
    raw: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    """
    EXISTENCE CHECK: Per-element membership of `raw` in sorted-unique `universe`.

    Returns a `(raw.size,)` bool mask, ``True`` where the value is present.
    No sentinel handling — callers that treat ``raw < 0`` as "not required"
    should mask those out before calling, or ignore the corresponding mask entries.
    """
    if raw.size == 0:
        return np.empty(0, dtype=np.bool_)
    if universe.size == 0:
        return np.zeros(raw.shape[0], dtype=np.bool_)

    idx = np.searchsorted(universe, raw)
    in_bounds = idx < universe.size
    found = np.zeros(raw.shape[0], dtype=np.bool_)
    found[in_bounds] = universe[idx[in_bounds]] == raw[in_bounds]
    return found


def searchsorted_or_raise(
    *,
    universe: npt.NDArray,
    raw: npt.NDArray,
    ctx: str = "ids",
) -> npt.NDArray[np.int64]:
    """
    STRICT INNER JOIN: Locate every `raw` value in sorted-unique `universe`.
    Raises ValueError on any missing target.

    Returns the `(raw.size,)` int64 index array (the natural `np.searchsorted`
    dtype — callers gather with it, no downcast needed).
    """
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)

    n = universe.shape[0]
    if n == 0:
        raise ValueError(
            f"{ctx}: Universe is empty, {raw.shape[0]} target ID(s) cannot be resolved."
        )

    pos = np.searchsorted(universe, raw)
    np.clip(pos, 0, n - 1, out=pos)
    matched = universe[pos] == raw

    if not matched.all():
        bad = np.unique(raw[~matched])[:_PREVIEW].tolist()
        n_bad = int((~matched).sum())
        raise ValueError(
            f"{ctx}: {n_bad} ID(s) absent from the universe. First {len(bad)}: {bad}"
        )

    return pos


def searchsorted_soft_join(
    *,
    universe: npt.NDArray,
    raw: npt.NDArray,
    sentinel: np.int32,
) -> npt.NDArray[np.int32]:
    """
    SAFE LEFT JOIN: Probes existence. Maps ANY missing `raw` value to `sentinel`.

    Use when absent target IDs are a valid, expected runtime case (e.g.,
    a netting set whose obligor is not modelled in the target portfolio).

    Returns `(raw.size,)` int32 indices. Caller filters via ``keep = pos >= 0``
    (assuming sentinel is negative).
    """
    n = universe.shape[0]
    if n == 0:
        return np.full(raw.shape[0], sentinel, dtype=np.int32)

    pos = np.searchsorted(universe, raw)
    np.clip(pos, 0, n - 1, out=pos)
    hit = universe[pos] == raw
    return np.where(hit, pos, sentinel).astype(np.int32, copy=False)


def searchsorted_or_propagate_sentinel_int32(
    *,
    universe: npt.NDArray,
    raw: npt.NDArray,
    sentinel: np.int32,
) -> npt.NDArray[np.int32]:
    """
    UNSAFE LEFT JOIN: High-performance ID translation with sentinel preservation.

    `raw < 0` rows are explicitly treated as sentinels and mapped to `sentinel`.

    CRITICAL: All other IDs (>= 0) are BLINDLY ASSUMED present in `universe`.
    This primitive does NOT probe membership (no equality check, no bounds clip).
    If a missing positive ID is passed, it will silently map to the next-largest
    element's index. Only use this when upstream referential integrity is 100%
    guaranteed by prior validation phases.
    """
    uix = np.searchsorted(universe, raw).astype(np.int32, copy=False)
    uix[raw < 0] = sentinel
    return uix
