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

from riskquant.constants import EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO
from riskquant.jit import (
    _F32_2D_C,
    _F32_2D_F,
    _F64_2D_C,
    _F64_2D_F,
    _I8_1D_C,
    _I32_1D_C,
    _NB_INT64,
    _NB_VOID,
    parallel_njit,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_ABSOLUTE_BASE_RETURN_ID,
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_NO_SHIFT_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.builder.shift_domain import validate_shift_domain
from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
    require_float64_values,
    require_positive_int,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals
from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios_from_residuals,
)

# =========================================================================
# ZERO-ALLOCATION KERNEL
# =========================================================================


@parallel_njit(
    [
        _NB_VOID(
            _F64_2D_F,  # residuals       (T, K_req) F-contig
            _F64_2D_F,  # vol_ratios      (T, K_req) F-contig
            _I32_1D_C,  # cols            (K_req,) — kept for shape parity / debug
            _I8_1D_C,  # shift_types     (K_req,)
            _I32_1D_C,  # inv_perm        (K_req,)
            _NB_INT64,  # t_start
            _NB_INT64,  # h
            _NB_INT64,  # B
            _F32_2D_F,  # out             (B, K_req) F-contig
        ),
        _NB_VOID(
            _F64_2D_F,  # residuals       (T, K_req) F-contig
            _F64_2D_F,  # vol_ratios      (T, K_req) F-contig
            _I32_1D_C,  # cols            (K_req,) — kept for shape parity / debug
            _I8_1D_C,  # shift_types     (K_req,)
            _I32_1D_C,  # inv_perm        (K_req,)
            _NB_INT64,  # t_start
            _NB_INT64,  # h
            _NB_INT64,  # B
            _F32_2D_C,  # out             (B, K_req) F-contig
        ),
        # K_req == 1: residuals/vol_ratios are (T, 1) and out is (B, 1); all three
        # are C=F ambiguous and resolve to 'C'. Same class as D008 (single-column
        # f64 inputs alongside the already-handled single-row f32 output).
        _NB_VOID(
            _F64_2D_C,  # residuals       (T, 1) C-contig
            _F64_2D_C,  # vol_ratios      (T, 1) C-contig
            _I32_1D_C,  # cols            (1,)
            _I8_1D_C,  # shift_types     (1,)
            _I32_1D_C,  # inv_perm        (1,)
            _NB_INT64,  # t_start
            _NB_INT64,  # h
            _NB_INT64,  # B
            _F32_2D_C,  # out             (B, 1) C-contig
        ),
    ]
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
        # Contract (plan.py): inv_perm[k] is the SORTED compute position of
        # shock-universe column k, a gather map. Output column k reads it.
        pos = inv_perm[k]
        st = shift_types[pos]
        out_col = k
        out_col_vals = out[:, out_col]

        if st == SHIFT_NO_SHIFT_ID:
            # Held fixed: short-circuit to zero shock WITHOUT reading the
            # residual / vol-ratio inputs.
            for b in range(batch_len):
                out_col_vals[b] = np.float32(0.0)
            continue

        col_res = residuals[:, pos]
        col_vols = vol_ratios[:, pos]

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

        elif st == SHIFT_ABSOLUTE_BASE_RETURN_ID:
            # Sign-robust SIMPLE_RETURN: aggregate the vol-scaled residuals
            # multiplicatively (identical to the SIMPLE branch, fed
            # absolute-base residuals) so positive levels reduce to SIMPLE
            # exactly, for every horizon.
            for b in range(batch_len):
                cum_ret = 1.0
                path_start = base_t + b
                for step in range(h):
                    t1 = path_start + step
                    cum_ret *= 1.0 + col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_ret - 1.0)


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
        require_float64_values(self.matrix.values)
        object.__setattr__(
            self, "horizon", require_positive_int(self.horizon, "horizon")
        )
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
        validate_shift_domain(
            values=self.matrix.values,
            cols_sorted=plan.cols_sorted,
            shift_type_ids=plan.shift_types_sorted,
            factor_ids=self.matrix.factor_ids,
        )
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
            # See historical.py for the full explanation: a partial-batch
            # slice of an F-contiguous buffer is not itself contiguous, but
            # the kernel's `out` parameter requires genuine F-contiguity.
            # (RQ-AUD-017.)
            kernel_out = (
                shocks_view
                if shocks_view.flags["F_CONTIGUOUS"]
                else np.empty((batch_len, k_req), dtype=np.float32, order="F")
            )

            _kernel_fused_filtered_h_day_shock(
                self._ar1_residuals,
                self._vol_ratios,
                plan.cols_sorted,
                plan.shift_types_sorted,
                plan.inverse_permutation,
                np.int64(t_start),
                np.int64(h),
                np.int64(batch_len),
                kernel_out,
            )

            yield ScenarioSetBatch(
                shocks=kernel_out,
                required_to_shock_uix=shock_universe.required_to_shock_uix,
                shock_scale=shock_scale,
                required_shift_type_id=required_shift_type_id,
                scenario_offset=scenario_offset,
                scenario_count_total=s_total,
            )

            scenario_offset += batch_len
            t_start = t_end