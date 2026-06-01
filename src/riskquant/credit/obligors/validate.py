# src/riskquant/credit/obligors/validate.py
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
    return uid[mask][:_PREVIEW].tolist()


def validate_uid_sorted_unique(uid: npt.NDArray[np.int64]) -> None:
    """`uid` must be strictly ascending — uniqueness + sorted in one check."""
    if uid.shape[0] == 0:
        raise ValueError("obligor master: empty uid array.")
    if not (np.diff(uid) > 0).all():
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
    bad = rating_id < np.int8(0)
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
    uid: npt.NDArray[np.int64], parent_uid: npt.NDArray[np.int64]
) -> None:
    """
    Each `parent_uid` is either `NO_PARENT_OBLIGOR` or an obligor uid present
    on this same master. The check is one `searchsorted` on the ascending
    `uid` array — no row loop.
    """
    needs_lookup = parent_uid != NO_PARENT_OBLIGOR
    if not needs_lookup.any():
        return

    targets = parent_uid[needs_lookup]
    pos = np.searchsorted(uid, targets)
    pos = np.clip(pos, 0, uid.shape[0] - 1)
    matched = uid[pos] == targets
    if not matched.all():
        bad_children = uid[needs_lookup][~matched][:_PREVIEW].tolist()
        bad_parents = targets[~matched][:_PREVIEW].tolist()
        raise ValueError(
            f"obligor master: parent_uid(s) {bad_parents} (children "
            f"{bad_children}) absent from the obligor master."
        )

    # Self-parenting check: not allowed.
    self_parent = uid[needs_lookup] == targets
    if self_parent.any():
        bad = uid[needs_lookup][self_parent][:_PREVIEW].tolist()
        raise ValueError(
            f"obligor master: obligor(s) {bad} list themselves as parent_uid."
        )


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
