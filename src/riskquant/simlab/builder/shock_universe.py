# src/riskquant/simlab/builder/shock_universe.py
"""
Canonical shock-universe builder.

Deduplicates `(source_factor_id, shift_type_id)` pairs into K unique
shocks. Each unique shock can be computed once and broadcast to every
required exposure that maps onto it.

Hot-path discipline
-------------------
- No Python loop over per-required ids — dedup is one `np.unique` pass.
- First-seen ordering preserved across reruns of identical inputs so
  downstream hashes are deterministic.
- Bit-packed int64 keys: high 56 bits = `source_factor_id` (int32,
  sign-extended into int64), low 8 bits = `shift_type_id` (uint8 cast
  from int8). The pack/unpack invariant requires
  `0 <= shift_type_id < 128`; enforced at input.
- `available_factor_ids` arrives as a sorted-ascending int32 array
  (the `RiskFactorMatrix` invariant), so the downstream compute-plan
  resolution is one `np.searchsorted`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant._arrays import unique_first_seen
from riskquant.market_data.risk_factors.shift_types import VALID_SHIFT_TYPE_IDS
from riskquant.simlab.shock_plan.plan import (
    ShockComputePlan,
    build_shock_compute_plan,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ShockUniverse:
    """
    Canonical definition of the shock space.

    Deterministic and reusable; zero-allocation at runtime once built.
    """

    # --- UNIQUE SHOCK KEYS (length K) ---
    shock_source_ids: npt.NDArray[np.int32]
    shock_shift_type_ids: npt.NDArray[np.int8]

    # --- REQUIRED -> SHOCK MAPPING (length Nreq) ---
    required_to_shock_uix: npt.NDArray[np.int32]

    # --- COMPUTE PLAN ---
    compute_plan: ShockComputePlan

    # --- RAW PACKED KEYS (debug / hashing; length K) ---
    shock_keys: npt.NDArray[np.int64]


# ============================================================
# Key packing
# ============================================================


_SHIFT_TYPE_BITS: int = 8
_SHIFT_TYPE_MASK: int = (1 << _SHIFT_TYPE_BITS) - 1  # 0xFF


def _pack_keys(
    source_factor_ids: npt.NDArray[np.int32],
    shift_type_ids: npt.NDArray[np.int8],
) -> npt.NDArray[np.int64]:
    """
    Pack `(source_id, shift_type_id)` into int64.

    `source_id` fits in 32 bits but we promote to int64 so the left-shift
    doesn't lose the high bit. `shift_type_id` is masked to its low 8
    bits so a future sign-bit ambiguity can't poison the key.
    """
    src64 = source_factor_ids.astype(np.int64, copy=False)
    st64 = shift_type_ids.astype(np.int64, copy=False) & np.int64(_SHIFT_TYPE_MASK)
    return (src64 << _SHIFT_TYPE_BITS) | st64


# ============================================================
# Public API
# ============================================================


def build_shock_universe(
    *,
    required_shift_type_id: npt.NDArray[np.int8],
    source_factor_ids: npt.NDArray[np.int32],
    available_factor_ids: npt.NDArray[np.int32],
) -> ShockUniverse:
    """
    Build a canonical `ShockUniverse` from proxy-resolution outputs.

    Parameters
    ----------
    required_shift_type_id
        (Nreq,) int8 — shift type id per required exposure.
    source_factor_ids
        (Nreq,) int32 — resolved source rf per required (from
        `ProxyResolution.source_factor_ids`).
    available_factor_ids
        (Ntotal,) int32 — sorted-ascending unique factor ids exposed
        by the generator's risk-factor matrix.
    """
    _smoke_test_inputs(required_shift_type_id, source_factor_ids)

    # --- BIT-PACK KEYS ---
    keys = _pack_keys(source_factor_ids, required_shift_type_id)

    # --- DEDUPLICATE (first-seen) ---
    unique_keys, req_to_uix = unique_first_seen(keys)

    # Unpack unique keys back to int32 / int8 components.
    shock_source_ids = (unique_keys >> _SHIFT_TYPE_BITS).astype(np.int32, copy=False)
    shift_low = (unique_keys & np.int64(_SHIFT_TYPE_MASK)).astype(np.uint8, copy=False)
    shock_shift_type_ids = shift_low.astype(np.int8, copy=False)

    # --- BUILD COMPUTE PLAN ---
    compute_plan = build_shock_compute_plan(
        shock_source_ids=shock_source_ids,
        shock_shift_type_ids=shock_shift_type_ids,
        factor_ids=available_factor_ids,
    )

    return ShockUniverse(
        shock_source_ids=shock_source_ids,
        shock_shift_type_ids=shock_shift_type_ids,
        required_to_shock_uix=req_to_uix,
        compute_plan=compute_plan,
        shock_keys=unique_keys,
    )


# ============================================================
# Input validation
# ============================================================


def _smoke_test_inputs(
    required_shift_type_id: npt.NDArray[np.int8],
    source_factor_ids: npt.NDArray[np.int32],
) -> None:
    if required_shift_type_id.ndim != 1:
        raise TypeError("required_shift_type_id must be 1D.")
    if source_factor_ids.ndim != 1:
        raise TypeError("source_factor_ids must be 1D.")
    if required_shift_type_id.dtype != np.int8:
        raise TypeError(
            f"required_shift_type_id must be int8; got {required_shift_type_id.dtype}"
        )
    if source_factor_ids.dtype != np.int32:
        raise TypeError(
            f"source_factor_ids must be int32; got {source_factor_ids.dtype}"
        )
    if required_shift_type_id.shape != source_factor_ids.shape:
        raise ValueError(
            "required_shift_type_id and source_factor_ids must have the same shape; "
            f"got {required_shift_type_id.shape} and {source_factor_ids.shape}"
        )
    if required_shift_type_id.size == 0:
        return
    # Explicit membership against the canonical valid ids {-1, 1, 2, 3, 4}. A
    # numeric range check is wrong here: NO_SHIFT is -1 (valid, and it
    # round-trips through the low-8-bit dedup key: -1 & 0xFF = 255 -> int8 -1),
    # while 0 is intentionally NOT a valid shift id. Unknown ids fail fast.
    unique_ids = np.unique(required_shift_type_id)
    invalid = [
        int(i) for i in unique_ids.tolist() if int(i) not in VALID_SHIFT_TYPE_IDS
    ]
    if invalid:
        raise ValueError(
            f"required_shift_type_id contains unknown shift ids {invalid}; "
            f"valid ids are {sorted(VALID_SHIFT_TYPE_IDS)}."
        )
