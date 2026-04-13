# src/riskquant/simlab/resolve/shock_universe.py

from dataclasses import dataclass
from typing import Dict
import numpy as np

from riskquant.simlab.shock_plan.plan import (
    ShockComputePlan,
    build_shock_compute_plan,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ShockUniverse:
    """
    Canonical definition of the shock space.

    This object is:
      - deterministic
      - reusable
      - zero-allocation at runtime
    """

    # --- UNIQUE SHOCK KEYS --- #
    shock_source_ids: np.ndarray  # (K,) int32
    shock_shift_type_ids: np.ndarray  # (K,) int8

    # --- REQUIRED → SHOCK MAPPING --- #
    required_to_shock_uix: np.ndarray  # (Nreq,) int32

    # --- COMPUTE PLAN --- #
    compute_plan: ShockComputePlan

    # --- OPTIONAL: RAW KEYS (DEBUG / HASHING) --- #
    shock_keys: np.ndarray  # (K,) int64


def build_shock_universe(
    *,
    required_shift_type_id: np.ndarray,  # (Nreq,) int8
    source_factor_ids: np.ndarray,  # (Nreq,) int32
    id_to_col: Dict[int, int],
) -> ShockUniverse:
    """
    Build a canonical ShockUniverse from proxy resolution outputs.
    """
    # Assertions to enforce preconditions for bit-packing and type assumptions
    assert required_shift_type_id.ndim == 1
    assert (
        required_shift_type_id.max() < 256 and required_shift_type_id.min() >= 0
    )
    assert required_shift_type_id.dtype == np.int8
    assert source_factor_ids.ndim == 1
    assert source_factor_ids.dtype == np.int32
    assert source_factor_ids.shape == required_shift_type_id.shape

    n = int(source_factor_ids.shape[0])

    # --- PACK KEYS --- #
    keys = (source_factor_ids.astype(np.int64) << 8) | (
        required_shift_type_id.astype(np.int64) & 0xFF
    )

    # --- DEDUPLICATION --- #
    key_to_uix: Dict[int, int] = {}
    unique_src: list[np.int32] = []
    unique_st: list[np.int8] = []
    unique_keys: list[np.int64] = []

    req_to_uix = np.empty(n, dtype=np.int32)

    get = key_to_uix.get
    set_ = key_to_uix.__setitem__

    for i in range(n):
        k = int(keys[i])

        uix = get(k)
        if uix is None:
            uix = len(unique_src)
            set_(k, uix)

            unique_src.append(source_factor_ids[i])
            unique_st.append(required_shift_type_id[i])
            unique_keys.append(np.int64(k))

        req_to_uix[i] = np.int32(uix)

    shock_source_ids = np.asarray(unique_src, dtype=np.int32)
    shock_shift_type_ids = np.asarray(unique_st, dtype=np.int8)
    shock_keys = np.asarray(unique_keys, dtype=np.int64)

    # --- BUILD COMPUTE PLAN --- #
    compute_plan = build_shock_compute_plan(
        shock_source_ids=shock_source_ids,
        shock_shift_type_ids=shock_shift_type_ids,
        id_to_col=id_to_col,
    )

    return ShockUniverse(
        shock_source_ids=shock_source_ids,
        shock_shift_type_ids=shock_shift_type_ids,
        required_to_shock_uix=req_to_uix,
        compute_plan=compute_plan,
        shock_keys=shock_keys,
    )
