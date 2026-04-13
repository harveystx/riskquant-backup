# riskquant/simlab/scenarios/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Protocol, Tuple, Dict

import numpy as np


def build_factor_id_to_col_index(factor_ids: np.ndarray) -> Dict[int, int]:
    return {int(fid): i for i, fid in enumerate(factor_ids)}


def build_shock_key_plan(
    *,
    required_shift_type_id: np.ndarray,
    source_factor_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(source_factor_ids.shape[0])
    keys = (source_factor_ids.astype(np.int64) << 8) | (
        required_shift_type_id.astype(np.int64) & 0xFF
    )

    key_to_uix: Dict[int, int] = {}
    unique_src: list[np.int32] = []
    unique_st: list[np.int8] = []
    req_to_uix = np.empty(n, dtype=np.int32)

    get = key_to_uix.get
    set_ = key_to_uix.__setitem__
    us_append = unique_src.append
    ut_append = unique_st.append

    for i in range(n):
        k = int(keys[i])
        uix = get(k)
        if uix is None:
            uix = len(unique_src)
            set_(k, uix)
            us_append(source_factor_ids[i])
            ut_append(required_shift_type_id[i])
        req_to_uix[i] = np.int32(uix)

    return (
        np.asarray(unique_src, dtype=np.int32),
        np.asarray(unique_st, dtype=np.int8),
        req_to_uix,
    )


@dataclass(frozen=True, slots=True)
class ScenarioSetBatch:
    """
    A batch of scenarios represented in shock-space.

    shocks: (B, K) float32/64
        Shock matrix for this batch, in shock-key space
        (unique keys = (source_factor_id, required_shift_type_id)).

    required_to_shock_uix: (Nreq,) int32
        For each required factor (ordered by required_factor_ids contract),
        index into K shock columns.

    shock_scale: (Nreq,) float32
        Scaling to apply to shocks at application time (per required factor).

    required_shift_type_id: (Nreq,) int8
        Shift type id per required factor (determines how to apply shock to snapshot level).

    scenario_offset: int
        Global scenario offset (0-based) of this batch in the full scenario stream.

    scenario_count_total: Optional[int]
        Total scenario count if known (e.g., historical T-horizon), else None.
    """

    shocks: np.ndarray
    required_to_shock_uix: np.ndarray
    shock_scale: np.ndarray
    required_shift_type_id: np.ndarray
    scenario_offset: int
    scenario_count_total: Optional[int] = None


class ScenarioGenerator(Protocol):
    """
    Streaming-first scenario generation.
    """

    def iter_batches(
        self,
        *,
        required_shift_type_id: np.ndarray,  # (Nreq,) int8
        source_factor_ids: np.ndarray,  # (Nreq,) int32 (from ProxyResolution)
        shock_scale: np.ndarray,  # (Nreq,) float32 (from ProxyResolution)
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]: ...
