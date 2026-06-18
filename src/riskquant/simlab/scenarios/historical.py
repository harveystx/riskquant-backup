# src/riskquant/simlab/scenarios/historical.py
"""
Standard Historical Simulation (HS) scenario generator.

Computes exact h-day historical changes for each rolling window in the
matrix. No filtering, no volatility scaling — pure empirical.

Implementation
--------------
- One sweep over the (T, K) matrix per batch; per-column branch on
  shift type hoisted around the B-loop.
- Single `shocks` buffer pre-allocated at `batch_size`; each yield
  exposes a view sized to the current batch.
- F-contiguous shocks layout — the apply-shocks kernel reads down
  columns with unit stride.
- Pinned `nb.void(...)` signature, positional kernel call.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numba as nb  # type: ignore[import-untyped]
import numpy as np

from riskquant.numba import parallel_njit
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
)

_F64_2D_F = nb.float64[::1, :]
_F32_2D_F = nb.float32[::1, :]
_I32_1D = nb.int32[::1]
_I8_1D = nb.int8[::1]


# =========================================================================
# ZERO-ALLOCATION KERNEL
# =========================================================================


@parallel_njit(
    nb.void(
        _F64_2D_F,  # values     (T, K_total) F-contig
        _I32_1D,  # cols       (K_req,)
        _I8_1D,  # shift_types(K_req,)
        _I32_1D,  # inv_perm   (K_req,)
        nb.int64,  # t_start
        nb.int64,  # h
        nb.int64,  # B
        _F32_2D_F,  # out        (B, K_req) F-contig
    ),
)
def _kernel_fused_h_day_shock(
    values,
    cols,
    shift_types,
    inv_perm,
    t_start,
    h,
    batch_len,
    out,
) -> None:
    """
    Direct h-day historical changes per column; parallel over factors.
    """
    k_cols = cols.shape[0]
    for k in nb.prange(k_cols):
        col = cols[k]
        st = shift_types[k]
        out_col = inv_perm[k]

        col_vals = values[:, col]
        out_col_vals = out[:, out_col]

        t1 = t_start
        t0 = t_start - h

        if st == SHIFT_LOG_RETURN_ID:
            for b in range(batch_len):
                v1 = col_vals[t1]
                v0 = col_vals[t0]
                out_col_vals[b] = np.float32(math.log(v1 / v0))
                t1 += 1
                t0 += 1
        elif st == SHIFT_SIMPLE_RETURN_ID:
            for b in range(batch_len):
                v1 = col_vals[t1]
                v0 = col_vals[t0]
                out_col_vals[b] = np.float32(v1 / v0 - 1.0)
                t1 += 1
                t0 += 1
        elif st == SHIFT_DIFFERENCE_ID:
            for b in range(batch_len):
                v1 = col_vals[t1]
                v0 = col_vals[t0]
                out_col_vals[b] = np.float32(v1 - v0)
                t1 += 1
                t0 += 1


# =========================================================================
# GENERATOR
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoricalGenerator(ScenarioGenerator):
    """Historical-simulation generator. Shocks are always float32."""

    matrix: RiskFactorMatrix  # MUST be F-contiguous
    horizon: int = field(default=1)

    # Derived in __post_init__.
    total_scenarios: int = field(init=False, repr=False)
    available_factor_ids: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        s_total = self.matrix.values.shape[0] - self.horizon
        if s_total <= 0:
            raise ValueError(
                f"Not enough history: T={self.matrix.values.shape[0]}, "
                f"horizon={self.horizon}"
            )
        object.__setattr__(self, "total_scenarios", s_total)
        object.__setattr__(self, "available_factor_ids", self.matrix.factor_ids)

    def iter_batches(
        self,
        *,
        required_shift_type_id: np.ndarray,
        source_factor_ids: np.ndarray,
        shock_scale: np.ndarray,
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]:
        t_max = self.matrix.values.shape[0]
        h = int(self.horizon)
        s_total = t_max - h

        shock_universe = build_shock_universe(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            available_factor_ids=self.matrix.factor_ids,
        )
        plan = shock_universe.compute_plan
        k_req = plan.cols_sorted.shape[0]

        # Single pre-allocation for the streaming buffer. Tail batch
        # uses a view of the same memory.
        shocks_buf = np.empty((batch_size, k_req), dtype=np.float32, order="F")

        scenario_offset = 0
        t_start = h
        while t_start < t_max:
            t_end = min(t_max, t_start + batch_size)
            batch_len = t_end - t_start
            shocks_view = shocks_buf[:batch_len, :]

            _kernel_fused_h_day_shock(
                self.matrix.values,
                plan.cols_sorted,
                plan.shift_types_sorted,
                plan.inverse_permutation,
                np.int64(t_start),
                np.int64(h),
                np.int64(batch_len),
                shocks_view,
            )

            yield ScenarioSetBatch(
                shocks=shocks_view,
                required_to_shock_uix=shock_universe.required_to_shock_uix,
                shock_scale=shock_scale,
                required_shift_type_id=required_shift_type_id,
                scenario_offset=scenario_offset,
                scenario_count_total=s_total,
            )

            scenario_offset += batch_len
            t_start = t_end
