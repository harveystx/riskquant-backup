# src/riskquant/credit/masters/netting_set/validate.py
"""
Netting-set master validators.

Mirrors `obligors/validate.py`: vectorized predicates on post-ingest SoA,
no row loops, no pandas. Cross-master validation (obligor uids present on
the obligor master) is deliberately out of scope here — it's a layered
check that belongs at the top-level pipeline orchestrator, where both
masters are visible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.exceptions import ConfigurationError

_PREVIEW = 10


def _bad_uids(uid: npt.NDArray[np.int64], mask: npt.NDArray[np.bool_]) -> list[int]:
    return uid[mask][:_PREVIEW].tolist()


def validate_uid_sorted_unique(uid: npt.NDArray[np.int64]) -> None:
    """`uid` (netting-set id) must be strictly ascending."""
    if uid.shape[0] == 0:
        raise ValueError("netting set master: empty uid array.")
    if not (np.diff(uid) > 0).all():
        unique, counts = np.unique(uid, return_counts=True)
        dups = unique[counts > 1][:_PREVIEW].tolist()
        if dups:
            raise ValueError(
                f"netting set master: duplicate netting-set uid(s) {dups}."
            )
        raise ValueError("netting set master: uid is not sorted ascending.")


def validate_obligor_uid_nonnegative(
    uid: npt.NDArray[np.int64],
    obligor_uid: npt.NDArray[np.int64],
) -> None:
    """Every netting set must point at a non-negative obligor uid (no sentinel)."""
    bad = obligor_uid < 0
    if bad.any():
        raise ValueError(
            f"netting set master: negative obligor_uid for netting set(s) "
            f"{_bad_uids(uid, bad)}."
        )


def validate_net_collateral(
    uid: npt.NDArray[np.int64], net_collateral: npt.NDArray[np.float64]
) -> None:
    """
    Validates that net collateral values are finite.
    Negative values are mathematically valid and represent net posted collateral.
    """
    # =====================================================================
    # DOD CONTRACT: Net Collateral = Received - Posted.
    # Negative values (posted) correctly increase exposure via max(V - C, 0)
    # =====================================================================
    bad = ~np.isfinite(net_collateral)

    if bad.any():
        bad_uids = uid[bad][:_PREVIEW].tolist()
        raise ConfigurationError(
            f"NETTING_SET_MASTER: net_collateral must be finite. Invalid at UIDs: {bad_uids}"
        )


def validate_netting_set_master_arrays(
    *,
    uid: npt.NDArray[np.int64],
    obligor_uid: npt.NDArray[np.int64],
    net_collateral: npt.NDArray[np.float64],
) -> None:
    """Single-call entry point: every validator, fail-fast."""
    validate_uid_sorted_unique(uid)
    validate_obligor_uid_nonnegative(uid, obligor_uid)
    validate_net_collateral(uid, net_collateral)
