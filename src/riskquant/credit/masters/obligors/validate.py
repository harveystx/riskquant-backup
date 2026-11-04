# src/riskquant/credit/masters/obligors/validate.py
"""
Obligor master validators.

Pure vectorized predicates on the post-ingest SoA arrays. Each function
either passes silently or raises `ValueError` with the (bounded) list of
offending uids. No row loops, no pandas, no string lookups — these run
once at ingest time, on float64/int64/int32/int8 arrays already cast to
their compiled dtypes.

The validators are split (uid uniqueness, range checks, sentinel checks)
rather than fused so any concrete failure points at exactly one invariant.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.constants import (
    NO_COUNTRY_ID,
    NO_PARENT_OBLIGOR,
    NO_RATING_ID,
    NO_SECTOR_ID,
)

_PREVIEW = 10
_PD_LO = 0.0
_PD_HI = 1.0
_LGD_LO = 0.0
_LGD_HI = 1.0


def _bad_uids(uid: npt.NDArray[np.int64], mask: npt.NDArray[np.bool_]) -> list[int]:
    return [int(u) for u in uid[mask][:_PREVIEW]]


def validate_uid_sorted_unique(uid: npt.NDArray[np.int64]) -> None:
    """`uid` must be strictly ascending — uniqueness + sorted in one check.

    Compares neighbours directly, not through ``np.diff``: the int64
    difference of two valid uids can exceed the int64 range and wrap (for
    example ``2**63 - 1 - (-1)``), which made an ascending pair look
    descending.
    """
    if uid.shape[0] == 0:
        raise ValueError("obligor master: empty uid array.")
    if not (uid[1:] > uid[:-1]).all():
        # Surface duplicates first (the common bug); sortedness second.
        unique, counts = np.unique(uid, return_counts=True)
        dups = unique[counts > 1][:_PREVIEW].tolist()
        if dups:
            raise ValueError(f"obligor master: duplicate uid(s) {dups}.")
        raise ValueError("obligor master: uid is not sorted ascending.")


def validate_pd_range(uid: npt.NDArray[np.int64], pd: npt.NDArray[np.float64]) -> None:
    """PD in [0, 1]; NaN is also rejected (np.isnan short-circuits the compare)."""
    bad = ~((pd >= _PD_LO) & (pd <= _PD_HI))
    if bad.any():
        raise ValueError(
            f"obligor master: PD outside [{_PD_LO}, {_PD_HI}] for uid(s) "
            f"{_bad_uids(uid, bad)}."
        )


def validate_lgd_range(
    uid: npt.NDArray[np.int64], lgd: npt.NDArray[np.float64]
) -> None:
    """LGD in [0, 1]; NaN is also rejected."""
    bad = ~((lgd >= _LGD_LO) & (lgd <= _LGD_HI))
    if bad.any():
        raise ValueError(
            f"obligor master: LGD outside [{_LGD_LO}, {_LGD_HI}] for uid(s) "
            f"{_bad_uids(uid, bad)}."
        )


def validate_rating_id(
    uid: npt.NDArray[np.int64], rating_id: npt.NDArray[np.int8]
) -> None:
    """
    `rating_id` must be either the unknown sentinel (`NO_RATING_ID == 0`) or
    a positive int8. The full rating-id taxonomy lives in
    `config/enums/credit_ratings`; this validator only enforces the dtype-
    level invariant. The taxonomy check is deferred until the rating enum
    is promoted to an IntEnum and a canonical {AAA..D} -> int8 mapping
    lands.
    """
    bad = (rating_id < np.int8(0)) & (rating_id != NO_RATING_ID)
    if bad.any():
        raise ValueError(
            f"obligor master: rating_id < 0 (must be NO_RATING_ID={NO_RATING_ID} "
            f"or a positive int8) for uid(s) {_bad_uids(uid, bad)}."
        )


def validate_sector_id(
    uid: npt.NDArray[np.int64], sector_id: npt.NDArray[np.int32]
) -> None:
    bad = (sector_id < np.int32(0)) & (sector_id != NO_SECTOR_ID)
    if bad.any():
        raise ValueError(
            f"obligor master: sector_id < 0 and not NO_SECTOR_ID for uid(s) "
            f"{_bad_uids(uid, bad)}."
        )


def validate_country_id(
    uid: npt.NDArray[np.int64], country_id: npt.NDArray[np.int32]
) -> None:
    bad = (country_id < np.int32(0)) & (country_id != NO_COUNTRY_ID)
    if bad.any():
        raise ValueError(
            f"obligor master: country_id < 0 and not NO_COUNTRY_ID for uid(s) "
            f"{_bad_uids(uid, bad)}."
        )


def validate_parent_uid(
    uid: npt.NDArray[np.int64],
    parent_uid: npt.NDArray[np.int64],
) -> None:
    """
    Validate obligor hierarchy.

    Rules
    -----
    1. Every parent_uid is either NO_PARENT_OBLIGOR or an obligor uid
       present in this master.
    2. An obligor may not list itself as its parent.
    3. The parent hierarchy must be acyclic.
    """

    n = uid.shape[0]

    needs_lookup = parent_uid != NO_PARENT_OBLIGOR

    # Parent row index for each obligor (-1 = no parent).
    parent_idx = np.full(n, -1, dtype=np.int64)

    if needs_lookup.any():
        targets = parent_uid[needs_lookup]

        pos = np.searchsorted(uid, targets)

        matched = (pos < n) & (uid[np.minimum(pos, n - 1)] == targets)

        if not matched.all():
            bad_children = uid[needs_lookup][~matched][:_PREVIEW].tolist()
            bad_parents = targets[~matched][:_PREVIEW].tolist()

            raise ValueError(
                f"obligor master: parent_uid(s) {bad_parents} "
                f"(children {bad_children}) absent from the obligor master."
            )

        parent_idx[needs_lookup] = pos

        self_parent = uid[needs_lookup] == targets

        if self_parent.any():
            bad = uid[needs_lookup][self_parent][:_PREVIEW].tolist()

            raise ValueError(
                f"obligor master: obligor(s) {bad} list themselves as parent_uid."
            )

    # ------------------------------------------------------------------
    # Cycle detection.
    #
    # state:
    #   0 = unseen
    #   1 = processed
    #
    # walk_id:
    #   identifies the current traversal.
    #
    # If we revisit a node with the same walk_id before reaching a
    # processed node or root, we found a cycle.
    # ------------------------------------------------------------------

    state = np.zeros(n, dtype=np.uint8)
    walk_id = np.zeros(n, dtype=np.int64)

    current_walk = 0

    for start_idx in range(n):
        if state[start_idx]:
            continue

        current_walk += 1
        idx = start_idx

        while idx != -1:
            if state[idx]:
                break

            if walk_id[idx] == current_walk:
                raise ValueError(
                    f"obligor master: parent hierarchy cycle detected "
                    f"at obligor {int(uid[idx])}."
                )

            walk_id[idx] = current_walk
            idx = parent_idx[idx]

        idx = start_idx

        while idx != -1 and not state[idx]:
            state[idx] = 1
            idx = parent_idx[idx]


def validate_obligor_master_arrays(
    *,
    uid: npt.NDArray[np.int64],
    pd: npt.NDArray[np.float64],
    lgd: npt.NDArray[np.float64],
    rating_id: npt.NDArray[np.int8],
    sector_id: npt.NDArray[np.int32],
    country_id: npt.NDArray[np.int32],
    parent_uid: npt.NDArray[np.int64],
) -> None:
    """Single-call entry point: every validator, fail-fast on first violation."""
    validate_uid_sorted_unique(uid)
    validate_pd_range(uid, pd)
    validate_lgd_range(uid, lgd)
    validate_rating_id(uid, rating_id)
    validate_sector_id(uid, sector_id)
    validate_country_id(uid, country_id)
    validate_parent_uid(uid, parent_uid)
