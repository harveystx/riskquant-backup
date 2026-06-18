# src/riskquant/simlab/scenarios/historical_fhs.py
"""
Filtered Historical Simulation (FHS) scenario generator.

Methodology
-----------
1. AR(1) residualisation of historical returns.
2. EWMA volatility ratios Z_t = sigma_today / sigma_t.
3. Aggregate the scaled residuals over the h-day horizon for each
   rolling window. The same window-step assumption used in standard
   HS applies here: scenario s uses days `[t_start - h + 1 + s, ...,
   t_start + s]`.

Implementation
--------------
- AR(1) + EWMA scratch buffers (`ar1_residuals`, `vol_ratios`) are
  computed lazily on first `iter_batches()` call and cached on the
  generator instance. Subsequent runs against the same matrix reuse
  them. Reset via `reset_caches()`.
- The streaming `shocks` buffer is pre-allocated once at `batch_size`.
- Kernel is pinned, branch-hoisted, and called positionally.
- Shocks are always float32 per the architecture rules.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numba as nb  # type: ignore[import-untyped]
import numpy as np

from riskquant.numba import parallel_njit
from riskquant.constants import EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO
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
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals
from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios_from_residuals,
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
        _F64_2D_F,  # residuals       (T, K_req) F-contig
        _F64_2D_F,  # vol_ratios      (T, K_req) F-contig
        _I32_1D,  # cols            (K_req,) — kept for shape parity / debug
        _I8_1D,  # shift_types     (K_req,)
        _I32_1D,  # inv_perm        (K_req,)
        nb.int64,  # t_start
        nb.int64,  # h
        nb.int64,  # B
        _F32_2D_F,  # out             (B, K_req) F-contig
    ),
)
def _kernel_fused_filtered_h_day_shock(
    residuals,
    vol_ratios,
    cols,
    shift_types,
    inv_perm,
    t_start,
    h,
    batch_len,
    out,
) -> None:
    """
    Aggregate scaled residuals over rolling h-day windows.
    """
    k_cols = cols.shape[0]
    for k in nb.prange(k_cols):
        st = shift_types[k]
        out_col = inv_perm[k]

        col_res = residuals[:, k]
        col_vols = vol_ratios[:, k]
        out_col_vals = out[:, out_col]

        base_t = t_start - h + 1

        if st == SHIFT_LOG_RETURN_ID:
            for b in range(batch_len):
                cum_shock = 0.0
                path_start = base_t + b
                for step in range(h):
                    t1 = path_start + step
                    cum_shock += col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_shock)

        elif st == SHIFT_SIMPLE_RETURN_ID:
            for b in range(batch_len):
                cum_ret = 1.0
                path_start = base_t + b
                for step in range(h):
                    t1 = path_start + step
                    cum_ret *= 1.0 + col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_ret - 1.0)

        elif st == SHIFT_DIFFERENCE_ID:
            for b in range(batch_len):
                cum_diff = 0.0
                path_start = base_t + b
                for step in range(h):
                    t1 = path_start + step
                    cum_diff += col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_diff)


# =========================================================================
# GENERATOR
# =========================================================================


@dataclass(slots=True, kw_only=True)
class FilteredHistoricalGenerator(ScenarioGenerator):
    """
    FHS generator. Shocks are always float32.

    Not `frozen=True` because `ar1_residuals`, `vol_ratios` and the
    cached shock-universe key are populated lazily on first
    `iter_batches()` call.
    """

    matrix: RiskFactorMatrix
    horizon: int = field(default=1)
    decay_factor: float = field(default=EWMA_DECAY_FACTOR)
    max_vol_ratio: float = field(default=MAX_EWMA_VOL_RATIO)

    # Derived in __post_init__.
    total_scenarios: int = field(init=False, repr=False)
    available_factor_ids: np.ndarray = field(init=False, repr=False)

    # Lazy caches (set on first iter_batches).
    _ar1_residuals: np.ndarray | None = field(default=None, init=False, repr=False)
    _vol_ratios: np.ndarray | None = field(default=None, init=False, repr=False)
    _cache_signature: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.matrix.values.shape[0] < 3:
            raise ValueError("History too short for EWMA calculation.")
        s_total = self.matrix.values.shape[0] - self.horizon
        if s_total <= 0:
            raise ValueError(
                f"Not enough history: T={self.matrix.values.shape[0]}, "
                f"horizon={self.horizon}"
            )
        self.total_scenarios = s_total
        self.available_factor_ids = self.matrix.factor_ids

    def reset_caches(self) -> None:
        """Drop the lazy AR(1) + EWMA caches. Next `iter_batches` recomputes."""
        self._ar1_residuals = None
        self._vol_ratios = None
        self._cache_signature = None

    def _ensure_filter_caches(
        self, plan_cols: np.ndarray, plan_shifts: np.ndarray
    ) -> None:
        """Materialise (T, K_req) residuals/vol-ratios if not already present."""
        sig = (
            plan_cols.tobytes(),
            plan_shifts.tobytes(),
            float(self.decay_factor),
            float(self.max_vol_ratio),
        )
        if self._cache_signature == sig:
            return

        t_max = self.matrix.values.shape[0]
        k_req = plan_cols.shape[0]

        ar1 = np.empty((t_max, k_req), dtype=np.float64, order="F")
        kernel_ar1_residuals(
            self.matrix.values,
            plan_cols,
            plan_shifts,
            ar1,
        )

        vols = np.empty((t_max, k_req), dtype=np.float64, order="F")
        kernel_ewma_vol_ratios_from_residuals(
            ar1,
            float(self.decay_factor),
            float(self.max_vol_ratio),
            vols,
        )

        self._ar1_residuals = ar1
        self._vol_ratios = vols
        self._cache_signature = sig

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

        self._ensure_filter_caches(plan.cols_sorted, plan.shift_types_sorted)
        assert self._ar1_residuals is not None
        assert self._vol_ratios is not None

        shocks_buf = np.empty((batch_size, k_req), dtype=np.float32, order="F")

        scenario_offset = 0
        t_start = h
        while t_start < t_max:
            t_end = min(t_max, t_start + batch_size)
            batch_len = t_end - t_start
            shocks_view = shocks_buf[:batch_len, :]

            _kernel_fused_filtered_h_day_shock(
                self._ar1_residuals,
                self._vol_ratios,
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
