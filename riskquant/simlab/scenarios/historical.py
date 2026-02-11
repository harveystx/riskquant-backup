# riskquant/simlab/scenarios/historical.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np

from riskquant.exceptions import PrettyKeyError
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN,
    SHIFT_SIMPLE_RETURN,
    SHIFT_DIFFERENCE,
)
from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
)


def _build_factor_id_to_col_index(factor_ids: np.ndarray) -> Dict[int, int]:
    return {int(fid): i for i, fid in enumerate(factor_ids)}


def _build_shock_key_plan(
    *,
    required_shift_type_id: np.ndarray,  # (Nreq,) int8
    source_factor_ids: np.ndarray,  # (Nreq,) int32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build unique keys in first-seen order:
      key = (source_id << 8) | (shift_type_id & 0xFF)

    Returns:
      shock_source_ids: (K,) int32
      shock_shift_type_ids: (K,) int8
      required_to_shock_uix: (Nreq,) int32
    """
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


def _compute_shocks_batch(
    *,
    values: np.ndarray,  # (T,N) contiguous
    id_to_col: Dict[int, int],
    shock_source_ids: np.ndarray,  # (K,) int32
    shock_shift_type_ids: np.ndarray,  # (K,) int8
    t_start: int,  # inclusive end-index in history for this batch (in [horizon, T))
    t_end: int,  # exclusive
    horizon: int,
    out_dtype: np.dtype,
) -> np.ndarray:
    """
    Compute shocks for rows t in [t_start, t_end) where shock compares t vs (t-horizon).
    Output shape: (B, K) where B = t_end - t_start.
    """
    h = int(horizon)
    B = int(t_end - t_start)
    K = int(shock_source_ids.shape[0])

    out = np.empty((B, K), dtype=out_dtype, order="C")

    for k in range(K):
        sid = int(shock_source_ids[k])
        st = np.int8(shock_shift_type_ids[k])

        j = id_to_col.get(sid)
        if j is None:
            raise PrettyKeyError(
                f"shock source factor_id not found in matrix.factor_ids: {sid}"
            )

        x1 = values[t_start:t_end, j]  # (B,)
        x0 = values[t_start - h : t_end - h, j]  # (B,)

        if st == SHIFT_LOG_RETURN:
            out[:, k] = np.log(x1 / x0, dtype=out_dtype)
        elif st == SHIFT_SIMPLE_RETURN:
            out[:, k] = (x1 / x0) - 1.0
        elif st == SHIFT_DIFFERENCE:
            out[:, k] = x1 - x0
        else:
            raise ValueError(f"Unknown shift_type_id={int(st)} for shock key k={k}")

    return out


@dataclass(frozen=True, slots=True)
class HistoricalScenarioGenerator(ScenarioGenerator):
    """
    Historical simulation scenario generator.

    Interpretation:
      - Each scenario corresponds to a historical shock over `horizon` days.
      - Total scenarios = T - horizon (where T = number of history dates).
      - Scenarios are indexed by end-date position t in [horizon, T).
    """

    matrix: RiskFactorMatrix
    horizon: int = 1
    shock_dtype: np.dtype = np.float32

    def iter_batches(
        self,
        *,
        required_factor_ids: np.ndarray,
        required_shift_type_id: np.ndarray,
        source_factor_ids: np.ndarray,
        shock_scale: np.ndarray,
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]:
        # Basic dtype checks (fail fast)
        if required_factor_ids.dtype != np.int32:
            raise TypeError("required_factor_ids must be int32")
        if required_shift_type_id.dtype != np.int8:
            raise TypeError("required_shift_type_id must be int8")
        if source_factor_ids.dtype != np.int32:
            raise TypeError("source_factor_ids must be int32")
        if shock_scale.dtype != np.float32:
            raise TypeError("shock_scale must be float32")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")

        T, N = self.matrix.values.shape
        h = int(self.horizon)
        S_total = T - h
        if S_total <= 0:
            raise ValueError(f"Not enough history: T={T}, horizon={h}")

        # One-time mappings / plans
        id_to_col = _build_factor_id_to_col_index(self.matrix.factor_ids)
        (
            shock_source_ids,
            shock_shift_type_ids,
            required_to_shock_uix,
        ) = _build_shock_key_plan(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
        )

        # Stream scenario batches: end-index t ranges [h, T)
        # Scenario index s = t - h
        scenario_offset = 0
        t = h
        while t < T:
            t_end = min(T, t + int(batch_size))
            shocks = _compute_shocks_batch(
                values=self.matrix.values,
                id_to_col=id_to_col,
                shock_source_ids=shock_source_ids,
                shock_shift_type_ids=shock_shift_type_ids,
                t_start=t,
                t_end=t_end,
                horizon=h,
                out_dtype=self.shock_dtype,
            )
            B = int(shocks.shape[0])

            yield ScenarioSetBatch(
                shocks=shocks,
                required_to_shock_uix=required_to_shock_uix,
                shock_scale=shock_scale,
                required_shift_type_id=required_shift_type_id,
                scenario_offset=scenario_offset,
                scenario_count_total=S_total,
            )

            scenario_offset += B
            t = t_end
