# src/riskquant/simlab/scenarios/historical.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import numpy as np
import numba as nb

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
    build_factor_id_to_col_index,
)
from riskquant.simlab.resolve.shock_universe import build_shock_universe

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE,
    SHIFT_LOG_RETURN,
    SHIFT_SIMPLE_RETURN,
)

# =========================================================================
# ZERO-ALLOCATION PATH-AGGREGATION KERNEL - HISTORICAL SCENARIO GENERATOR
# =========================================================================


@nb.njit(parallel=True, fastmath=True, cache=True)
def _kernel_fused_h_day_shock(
    *,
    values: np.ndarray,  # Fortran-contiguous (T x K)
    cols: np.ndarray,  # K, column indices into values
    shift_types: np.ndarray,  # K
    inv_perm: np.ndarray,  # K, output column positions
    t_start: int,
    h: int,  # Effective horizon in days
    B: int,  # Number of scenarios in the batch
    out: np.ndarray,  # Fortran-contiguous (B x K)
):
    """
    Computes h-day shock from 1-day returns by directly indexing into the values array at the start and end of the h-day period.
    Parallelized across factors (columns) for optimal Fortran cache-line usage.
    """
    K = cols.shape[0]

    # Parallel over factors (columns)
    for k in nb.prange(K):
        col = cols[k]
        st = shift_types[k]
        out_col = inv_perm[k]

        # Precompute base indices for the h-day period to avoid redundant arithmetic in the B-loop
        base_t0 = t_start - h

        # Bind column for cache efficiency
        col_vals = values[:, col]
        out_col_vals = out[:, out_col]

        # Bind t1 and t0
        t1 = t_start
        t0 = base_t0

        # Hoist branch out of B-loop
        if st == SHIFT_LOG_RETURN:
            for b in range(B):
                v1 = col_vals[t1]
                v0 = col_vals[t0]
                out_col_vals[b] = np.log(v1 / v0)
                t1 += 1
                t0 += 1

        elif st == SHIFT_SIMPLE_RETURN:
            for b in range(B):
                v1 = col_vals[t1]
                v0 = col_vals[t0]
                out_col_vals[b] = v1 / v0 - 1.0
                t1 += 1
                t0 += 1

        elif st == SHIFT_DIFFERENCE:
            for b in range(B):
                v1 = col_vals[t1]
                v0 = col_vals[t0]
                out_col_vals[b] = v1 - v0
                t1 += 1
                t0 += 1


# =========================================================================
# HISTORICAL SCENARIO GENERATOR
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoricalScenarioGenerator(ScenarioGenerator):
    matrix: RiskFactorMatrix  # MUST be Fortran-contiguous
    horizon: int = 1
    shock_dtype: type[np.float32] | type[np.float64] = np.float32

    def iter_batches(
        self,
        *,
        required_shift_type_id: np.ndarray,
        source_factor_ids: np.ndarray,
        shock_scale: np.ndarray,
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]:

        T = self.matrix.values.shape[0]
        h = int(self.horizon)
        S_total = T - h
        if S_total <= 0:
            raise ValueError(f"Not enough history: T={T}, horizon={h}")

        # Build factor id -> column map
        id_to_col = build_factor_id_to_col_index(self.matrix.factor_ids)

        # Build shock universe and compute plan in one step to avoid redundant key construction
        shock_universe = build_shock_universe(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            id_to_col=id_to_col,
        )

        # Stream scenario batches
        scenario_offset = 0
        t_start = h
        K = shock_universe.compute_plan.cols_sorted.shape[0]

        while t_start < T:
            t_end = min(T, t_start + int(batch_size))
            B = t_end - t_start

            # Allocate shocks in Fortran order for cache-friendly column writes
            shocks = np.empty((B, K), dtype=self.shock_dtype, order="F")

            # Generate shocks for the batch using the fused kernel
            _kernel_fused_h_day_shock(
                values=self.matrix.values,  # Fortran-contiguous
                cols=shock_universe.compute_plan.cols_sorted,
                shift_types=shock_universe.compute_plan.shift_types_sorted,
                inv_perm=shock_universe.compute_plan.inverse_permutation,
                t_start=t_start,
                h=h,
                B=B,
                out=shocks,
            )

            yield ScenarioSetBatch(
                shocks=shocks,
                required_to_shock_uix=shock_universe.required_to_shock_uix,
                shock_scale=shock_scale,
                required_shift_type_id=required_shift_type_id,
                scenario_offset=scenario_offset,
                scenario_count_total=S_total,
            )

            scenario_offset += B
            t_start = t_end
