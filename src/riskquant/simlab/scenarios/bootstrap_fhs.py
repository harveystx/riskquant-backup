# src/riskquant/simlab/scenarios/bootstrap_fhs.py
"""
Bootstrapped Filtered Historical Simulation generator.

Methodology
-----------
1. AR(1) residualisation of historical returns.
2. EWMA volatility ratios Z_t = sigma_today / sigma_t.
3. Pure i.i.d. bootstrap of single days drawn uniformly from
   [1, T-1] to construct h-day paths.

Why no block bootstrap: the upstream AR(1) + EWMA pipeline has already
whitened the series, so block-stitching reintroduces artificial seam
artefacts. Per-day i.i.d. is the cleaner aggregator on filtered data.

Implementation
--------------
- AR(1) + EWMA scratch buffers cached lazily on the generator instance
  (same scheme as `FilteredHistoricalGenerator`).
- All MC path indices drawn upfront from one `rng.integers` call.
- Streaming `shocks` buffer pre-allocated once at `batch_size`.
- Kernel pinned, branch-hoisted, called positionally.
- Shocks are always float32.
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
    _I32_2D_C,
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
            _F64_2D_F,  # residuals          (T, K_req) F-contig
            _F64_2D_F,  # vol_ratios         (T, K_req) F-contig
            _I32_1D_C,  # cols               (K_req,)  — debug parity
            _I8_1D_C,  # shift_types        (K_req,)
            _I32_1D_C,  # inv_perm           (K_req,)
            _I32_2D_C,  # random_t_indices   (B, h) C-contig
            _NB_INT64,  # h
            _NB_INT64,  # B
            _F32_2D_F,  # out                (B, K_req) F-contig
        ),
        _NB_VOID(
            _F64_2D_F,  # residuals          (T, K_req) F-contig
            _F64_2D_F,  # vol_ratios         (T, K_req) F-contig
            _I32_1D_C,  # cols               (K_req,)  — debug parity
            _I8_1D_C,  # shift_types        (K_req,)
            _I32_1D_C,  # inv_perm           (K_req,)
            _I32_2D_C,  # random_t_indices   (B, h) C-contig
            _NB_INT64,  # h
            _NB_INT64,  # B
            _F32_2D_C,  # out                (B, K_req) F-contig
        ),
        _NB_VOID(
            _F64_2D_C,  # residuals          (T, 1) C-contig — single-column
            _F64_2D_C,  # vol_ratios         (T, 1) C-contig — single-column
            _I32_1D_C,  # cols               (K_req,)  — debug parity
            _I8_1D_C,  # shift_types        (K_req,)
            _I32_1D_C,  # inv_perm           (K_req,)
            _I32_2D_C,  # random_t_indices   (B, h) C-contig
            _NB_INT64,  # h
            _NB_INT64,  # B
            _F32_2D_C,  # out                (B, 1) C-contig
        ),
    ]
)
def _kernel_fused_filtered_bootstrap_h_day_shock(
    residuals,
    vol_ratios,
    cols,
    shift_types,
    inv_perm,
    random_t_indices,
    h,
    batch_len,
    out,
) -> None:
    """
    Aggregate vol-scaled residuals over a synthetic i.i.d. path.
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
            for b in range(batch_len):
                out_col_vals[b] = np.float32(0.0)
            continue

        col_res = residuals[:, pos]
        col_vols = vol_ratios[:, pos]

        if st == SHIFT_LOG_RETURN_ID:
            for b in range(batch_len):
                cum_shock = 0.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    cum_shock += col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_shock)

        elif st == SHIFT_SIMPLE_RETURN_ID:
            for b in range(batch_len):
                cum_ret = 1.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    cum_ret *= 1.0 + col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_ret - 1.0)

        elif st == SHIFT_DIFFERENCE_ID:
            for b in range(batch_len):
                cum_diff = 0.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    cum_diff += col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_diff)

        elif st == SHIFT_ABSOLUTE_BASE_RETURN_ID:
            # Sign-robust SIMPLE_RETURN: multiplicative aggregation of the
            # vol-scaled residuals (see FilteredHistoricalGenerator).
            for b in range(batch_len):
                cum_ret = 1.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    cum_ret *= 1.0 + col_res[t1] * col_vols[t1]
                out_col_vals[b] = np.float32(cum_ret - 1.0)


# =========================================================================
# GENERATOR
# =========================================================================


@dataclass(slots=True, kw_only=True)
class FilteredBootstrapGenerator(ScenarioGenerator):
    """
    Bootstrapped FHS generator. Shocks are always float32.

    Not `frozen=True` because `ar1_residuals` and `vol_ratios` are
    populated lazily on first `iter_batches()` call.
    """

    matrix: RiskFactorMatrix
    total_scenarios: int
    horizon: int = field(default=1)
    decay_factor: float = field(default=EWMA_DECAY_FACTOR)
    max_vol_ratio: float = field(default=MAX_EWMA_VOL_RATIO)
    seed: int | None = field(default=None)

    # Derived in __post_init__.
    available_factor_ids: np.ndarray = field(init=False, repr=False)

    # Lazy caches.
    _ar1_residuals: np.ndarray | None = field(default=None, init=False, repr=False)
    _vol_ratios: np.ndarray | None = field(default=None, init=False, repr=False)
    _cache_signature: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require_float64_values(self.matrix.values)
        object.__setattr__(
            self,
            "total_scenarios",
            require_positive_int(self.total_scenarios, "total_scenarios"),
        )
        object.__setattr__(
            self, "horizon", require_positive_int(self.horizon, "horizon")
        )
        if self.matrix.values.shape[0] < 3:
            raise ValueError("History too short for EWMA calculation.")
        self.available_factor_ids = self.matrix.factor_ids

    def reset_caches(self) -> None:
        """Drop the AR(1) + EWMA scratch caches."""
        self._ar1_residuals = None
        self._vol_ratios = None
        self._cache_signature = None

    def _ensure_filter_caches(
        self, plan_cols: np.ndarray, plan_shifts: np.ndarray
    ) -> None:
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

        # Single MC draw for all paths upfront. Valid residual indices
        # are [1, T-1] (t=0 has no return).
        rng = np.random.default_rng(self.seed)
        master_t_indices = rng.integers(
            low=1,
            high=t_max,
            size=(self.total_scenarios, h),
            dtype=np.int32,
        )

        shocks_buf = np.empty((batch_size, k_req), dtype=np.float32, order="F")

        scenario_offset = 0
        while scenario_offset < self.total_scenarios:
            batch_len = min(batch_size, self.total_scenarios - scenario_offset)
            shocks_view = shocks_buf[:batch_len, :]
            batch_t_indices = master_t_indices[
                scenario_offset : scenario_offset + batch_len
            ]
            # See historical.py for the full explanation: a partial-batch
            # slice of an F-contiguous buffer is not itself contiguous, but
            # the kernel's `out` parameter requires genuine F-contiguity.
            # (RQ-AUD-017.)
            kernel_out = (
                shocks_view
                if shocks_view.flags["F_CONTIGUOUS"]
                else np.empty((batch_len, k_req), dtype=np.float32, order="F")
            )

            _kernel_fused_filtered_bootstrap_h_day_shock(
                self._ar1_residuals,
                self._vol_ratios,
                plan.cols_sorted,
                plan.shift_types_sorted,
                plan.inverse_permutation,
                np.ascontiguousarray(batch_t_indices),
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
                scenario_count_total=self.total_scenarios,
            )

            scenario_offset += batch_len