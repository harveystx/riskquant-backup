# src/riskquant/simlab/scenarios/historical_filtered.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numba as nb
import numpy as np

from riskquant.constants import EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE,
    SHIFT_LOG_RETURN,
    SHIFT_SIMPLE_RETURN,
)
from riskquant.simlab.resolve.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
    build_factor_id_to_col_index,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals
from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios,
    kernel_ewma_vol_ratios_from_residuals,
)


# @nb.njit(parallel=True, fastmath=True, cache=True)
# def _kernel_fused_filtered_h_day_shock(
#     *,
#     values: np.ndarray,  # Fortran-contiguous (T x K_total)
#     vol_ratios: np.ndarray,  # Fortran-contiguous (T x K_req)  <-- Now K_req wide
#     cols: np.ndarray,
#     shift_types: np.ndarray,
#     inv_perm: np.ndarray,
#     t_start: int,
#     h: int,
#     B: int,
#     out: np.ndarray,  # Fortran-contiguous (B x K_req)
# ):
#     K = cols.shape[0]

#     for k in nb.prange(K):
#         col = cols[k]
#         st = shift_types[k]
#         out_col = inv_perm[k]

#         col_vals = values[:, col]
#         col_vols = vol_ratios[:, k]  # <-- Reads from K_req aligned array
#         out_col_vals = out[:, out_col]

#         base_t = t_start - h + 1

#         if st == SHIFT_LOG_RETURN:
#             for b in range(B):
#                 cum_shock = 0.0
#                 path_start = base_t + b
#                 for step in range(h):
#                     t1 = path_start + step
#                     t0 = t1 - 1
#                     v1 = col_vals[t1]
#                     v0 = col_vals[t0]
#                     r_log = np.log(v1 / v0)
#                     cum_shock += r_log * col_vols[t1]
#                 out_col_vals[b] = cum_shock

#         elif st == SHIFT_SIMPLE_RETURN:
#             for b in range(B):
#                 cum_ret = 1.0
#                 path_start = base_t + b
#                 for step in range(h):
#                     t1 = path_start + step
#                     t0 = t1 - 1
#                     v1 = col_vals[t1]
#                     v0 = col_vals[t0]
#                     r_simple = (v1 / v0) - 1.0
#                     cum_ret *= 1.0 + (r_simple * col_vols[t1])
#                 out_col_vals[b] = cum_ret - 1.0

#         elif st == SHIFT_DIFFERENCE:
#             for b in range(B):
#                 cum_diff = 0.0
#                 path_start = base_t + b
#                 for step in range(h):
#                     t1 = path_start + step
#                     t0 = t1 - 1
#                     v1 = col_vals[t1]
#                     v0 = col_vals[t0]
#                     r_diff = v1 - v0
#                     cum_diff += r_diff * col_vols[t1]
#                 out_col_vals[b] = cum_diff


@nb.njit(parallel=True, fastmath=True, cache=True)
def _kernel_fused_filtered_h_day_shock(
    *,
    residuals: np.ndarray,  # Fortran-contiguous (T x K_req)
    vol_ratios: np.ndarray,  # Fortran-contiguous (T x K_req)
    cols: np.ndarray,  # K_req, column indices into original values (for reference only, not used in calculations)
    shift_types: np.ndarray,
    inv_perm: np.ndarray,
    t_start: int,
    h: int,
    B: int,
    out: np.ndarray,  # Fortran-contiguous (B x K_req)
):
    K = cols.shape[0]

    for k in nb.prange(K):
        st = shift_types[k]
        out_col = inv_perm[k]

        # Read directly from the aligned K_req arrays
        col_res = residuals[:, k]
        col_vols = vol_ratios[:, k]
        out_col_vals = out[:, out_col]

        base_t = t_start - h + 1

        if st == SHIFT_LOG_RETURN:
            for b in range(B):
                cum_shock = 0.0
                path_start = base_t + b
                for step in range(h):
                    t1 = path_start + step
                    # Scale the clean residual, not the raw return
                    cum_shock += col_res[t1] * col_vols[t1]
                out_col_vals[b] = cum_shock

        elif st == SHIFT_SIMPLE_RETURN:
            for b in range(B):
                cum_ret = 1.0
                path_start = base_t + b
                for step in range(h):
                    t1 = path_start + step
                    # Scale the clean residual, then compound
                    cum_ret *= 1.0 + (col_res[t1] * col_vols[t1])
                out_col_vals[b] = cum_ret - 1.0

        elif st == SHIFT_DIFFERENCE:
            for b in range(B):
                cum_diff = 0.0
                path_start = base_t + b
                for step in range(h):
                    t1 = path_start + step
                    # Scale the clean residual
                    cum_diff += col_res[t1] * col_vols[t1]
                out_col_vals[b] = cum_diff


# =========================================================================
# FILTERED HISTORICAL SIMULATION (FHS) GENERATOR
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class FilteredHistoricalScenarioGenerator(ScenarioGenerator):
    """
    Generates scenarios via Filtered Historical Simulation (FHS).

    Constructs an h-day synthetic path by applying Exponentially Weighted
    Moving Average (EWMA) volatility scaling to historical returns. This ensures
    that historical shocks are standardized to the current market volatility
    environment before aggregation.

    References
    ----------
    Hull, J., & White, A. (1998). Incorporating volatility updating into the historical simulation method for value-at-risk. Journal of Risk, 1(1), 5-19.
    https://doi.org/10.21314/JOR.1998.001

    Barone-Adesi, G., Bourgoin, F., & Giannopoulos, K. (1998). Don't look back. Risk, 11(8), 100-103.
    (Pioneered the 'Filtered Historical Simulation' terminology and empirical application of scaled residuals).

    J.P. Morgan / Reuters. (1996). RiskMetrics Technical Document. 4th Edition. New York.
    (Industry standard foundation for EWMA volatility tracking and the default lambda=0.94 decay factor).
    """

    matrix: RiskFactorMatrix  # MUST be Fortran-contiguous
    horizon: int = 1
    decay_factor: float = EWMA_DECAY_FACTOR  # Standard RiskMetrics lambda
    max_vol_ratio: float = MAX_EWMA_VOL_RATIO
    shock_dtype: type[np.float32] | type[np.float64] = np.float32

    def __post_init__(self) -> None:
        if self.matrix.values.shape[0] < 3:
            raise ValueError("History too short for EWMA calculation.")

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

        id_to_col = build_factor_id_to_col_index(self.matrix.factor_ids)

        shock_universe = build_shock_universe(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            id_to_col=id_to_col,
        )

        K_req = shock_universe.compute_plan.cols_sorted.shape[0]

        # --- 0. RESIDUALIZE RETURNS VIA JIT-COMPILED AR(1) FILTER ---

        # Get the AR(1) residuals for the entire history and all required columns in a single JIT-compiled pass.
        # Only apply A1(1) to auto-correlated factors - only de-mean non-auto-correlated factors.
        ar1_residuals = np.empty((T, K_req), dtype=np.float64, order="F")

        kernel_ar1_residuals(
            values=self.matrix.values,
            cols=shock_universe.compute_plan.cols_sorted,
            shift_types=shock_universe.compute_plan.shift_types_sorted,
            out_residuals=ar1_residuals,
        )

        # --- 1. JIT COMPILE & EXECUTE EWMA SCALARS ON THE FLY ---
        # We calculate the FHS scalars specifically for the exact shift-type mappings
        # requested by the universe, storing them in a tight (T x K_req) Fortran matrix.
        vol_ratios = np.empty((T, K_req), dtype=np.float64, order="F")

        kernel_ewma_vol_ratios_from_residuals(
            residuals=ar1_residuals,
            decay_factor=self.decay_factor,
            max_vol_ratio=self.max_vol_ratio,
            out_ratios=vol_ratios,
        )

        # --- 2. STREAM SCENARIOS ---
        scenario_offset = 0
        t_start = h

        while t_start < T:
            t_end = min(T, t_start + int(batch_size))
            B = t_end - t_start

            shocks = np.empty((B, K_req), dtype=self.shock_dtype, order="F")

            _kernel_fused_filtered_h_day_shock(
                residuals=ar1_residuals,
                vol_ratios=vol_ratios,
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
