# src/riskquant/credit/_uid_join.py
"""
Vectorized uid-join primitives for the credit package.

Every credit module that joins an int64 uid array against an ascending-
sorted master uid array — the compile-edge invariant: every compiled
master in this codebase has `uid` ascending sorted — shares one of
three patterns: strict lookup that raises on absence, soft lookup that
maps absences to a sentinel, and existence-only membership testing.
They are collapsed into three primitives here so the credit code has
exactly one place where the searchsorted / clip / equality dance lives.

`searchsorted_or_raise` returns int64 positions into the master for
every target uid; the caller indexes whichever master column it needs
(collateral, pd, lgd, obligor_uid, ...).

`searchsorted_or_sentinel` returns int32 positions with absences as
`NO_UID_MATCH_IDX_I32`; the caller masks on `>= 0` to filter rows.

`searchsorted_membership` returns a bool mask — the existence-only
variant used by `cross_master/validate.py`.

All three handle the empty-superset edge case explicitly: a missing
master is a join error in the strict variant and a uniform sentinel /
all-False in the soft variants.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

# Centralized preview limit for error-message diagnostics. Anything beyond
# the first ten offending uids is unreadable in a stacktrace.
_PREVIEW: Final[int] = 10

# Sentinel for "no match" in soft-lookup positions. int32 because positions
# index a target array bounded by the obligor / netting-set universe size,
# comfortably under int32 max.
NO_UID_MATCH_IDX_I32: Final[np.int32] = np.int32(-1)


def searchsorted_or_raise(
    target: npt.NDArray[np.int64],
    superset_sorted: npt.NDArray[np.int64],
    *,
    ctx: str,
) -> npt.NDArray[np.int64]:
    """
    Vectorized strict uid lookup: positions of `target` in `superset_sorted`.

    Parameters
    ----------
    target
        int64 uid array to look up. May contain duplicates.
    superset_sorted
        int64 uid array, ascending sorted, unique. Every compiled master
        in this codebase satisfies that invariant.
    ctx
        Caller-side label used in the error message (e.g.
        ``"compile_exposure: netting set uid(s)"``).

    Returns
    -------
    pos
        int64 positions such that ``superset_sorted[pos] == target``
        elementwise. Same shape as ``target``. Caller indexes master
        columns with these positions to materialize joined values.

    Raises
    ------
    ValueError
        If any ``target`` uid is absent from ``superset_sorted``, or if
        ``superset_sorted`` is empty while ``target`` is not.
    """
    n = superset_sorted.shape[0]
    if n == 0:
        if target.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        raise ValueError(
            f"{ctx}: superset is empty, {target.shape[0]} target uid(s) "
            f"cannot be resolved."
        )

    pos = np.searchsorted(superset_sorted, target)
    np.clip(pos, 0, n - 1, out=pos)
    matched = superset_sorted[pos] == target
    if not matched.all():
        bad = target[~matched][:_PREVIEW].tolist()
        n_bad = int((~matched).sum())
        raise ValueError(
            f"{ctx}: {n_bad} uid(s) absent from the superset. First {len(bad)}: {bad}"
        )
    return pos


def searchsorted_or_sentinel(
    target: npt.NDArray[np.int64],
    superset_sorted: npt.NDArray[np.int64],
) -> npt.NDArray[np.int32]:
    """
    Vectorized soft uid lookup: positions or ``NO_UID_MATCH_IDX_I32``.

    Use when absent target uids are a valid runtime case — for example,
    a netting set whose owning obligor has no row in the *target* JTD
    obligor universe and must therefore be dropped, not flagged.

    Returns
    -------
    pos
        int32 positions into ``superset_sorted``. Absent uids map to
        ``NO_UID_MATCH_IDX_I32``. Caller filters via ``keep = pos >= 0``.
    """
    n = superset_sorted.shape[0]
    if n == 0:
        return np.full(target.shape[0], NO_UID_MATCH_IDX_I32, dtype=np.int32)

    pos = np.searchsorted(superset_sorted, target)
    np.clip(pos, 0, n - 1, out=pos)
    hit = superset_sorted[pos] == target
    return np.where(hit, pos, NO_UID_MATCH_IDX_I32).astype(np.int32, copy=False)


def searchsorted_membership(
    target: npt.NDArray[np.int64],
    superset_sorted: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """
    Vectorized existence check: bool mask of which ``target`` entries are
    present in ``superset_sorted``.

    Used by ``cross_master/validate.py`` where presence is the answer
    and no positions or values need to be materialized.
    """
    n = superset_sorted.shape[0]
    if n == 0:
        return np.zeros(target.shape[0], dtype=np.bool_)
    pos = np.searchsorted(superset_sorted, target)
    np.clip(pos, 0, n - 1, out=pos)
    return superset_sorted[pos] == target
