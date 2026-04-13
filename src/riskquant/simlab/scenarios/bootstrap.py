# src/riskquant/simlab/scenarios/bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from operator import ge
from typing import Iterator

import numba as nb
import numpy as np

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
from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)

# =========================================================================
# ZERO-ALLOCATION PATH-AGGREGATION KERNEL - SB BOOTSTRAP SCENARIO GENERATOR
# =========================================================================


@nb.njit(parallel=True, fastmath=True, cache=True)
def _kernel_fused_sbb_h_day_shock(
    *,
    values: np.ndarray,  # Fortran-contiguous (T x K_total)
    cols: np.ndarray,  # K_req, column indices into values
    shift_types: np.ndarray,  # K_req
    inv_perm: np.ndarray,  # K_req, output column positions
    random_t_indices: np.ndarray,  # (B, h) matrix of synthetic daily paths
    h: int,  # Effective horizon in days
    B: int,  # Number of scenarios in the batch
    out: np.ndarray,  # Fortran-contiguous (B x K_req)
) -> None:
    """
    Computes effective h-day shocks by aggregating a synthetic path of 1-day returns.
    Parallelized across requested columns for optimal Fortran cache-line usage.
    """
    K = cols.shape[0]

    for k in nb.prange(K):
        col = cols[k]
        st = shift_types[k]
        out_col = inv_perm[k]

        # Bind column for cache efficiency
        col_vals = values[:, col]
        out_col_vals = out[:, out_col]

        if st == SHIFT_LOG_RETURN:
            for b in range(B):
                # Product of ratios over the h-day path
                cum_ratio = 1.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    t0 = t1 - 1
                    v1 = col_vals[t1]
                    v0 = col_vals[t0]
                    cum_ratio *= v1 / v0
                # Single log call per scenario!
                out_col_vals[b] = np.log(cum_ratio)

        elif st == SHIFT_SIMPLE_RETURN:
            for b in range(B):
                # Compound product of (1 + r) over the h-day path
                cum_ret = 1.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    t0 = t1 - 1
                    v1 = col_vals[t1]
                    v0 = col_vals[t0]
                    cum_ret *= v1 / v0
                out_col_vals[b] = cum_ret - 1.0

        elif st == SHIFT_DIFFERENCE:
            for b in range(B):
                # Sum of absolute differences over the h-day path
                cum_diff = 0.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    t0 = t1 - 1
                    v1 = col_vals[t1]
                    v0 = col_vals[t0]
                    cum_diff += v1 - v0
                out_col_vals[b] = cum_diff


# =========================================================================
# STATIONARY BLOCK BOOTSTRAP GENERATOR
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapMCScenarioGenerator(ScenarioGenerator):
    """
    Generates scenarios via Stationary Block Bootstrap (SBB).

    Constructs an h-day synthetic path for each scenario by stitching together
    blocks of contiguous 1-day historical returns. Block lengths are drawn from
    a Geometric distribution (expected_block_size).

    References
    ----------
    Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. Journal of the American Statistical Association, 89(428), 1303–1313. https://doi.org/10.2307/2290993

    Politis, D. N. (2003). The impact of bootstrap methods on time series analysis. Statistical Science, 18. https://doi.org/10.1214/ss/1063994977

    """

    matrix: RiskFactorMatrix
    total_scenarios: int
    expected_block_size: float = 1.0
    horizon: int = 1
    seed: int | None = None
    shock_dtype: type[np.float32] | type[np.float64] = np.float32

    def __post_init__(self) -> None:
        if self.total_scenarios <= 0:
            raise ValueError("total_scenarios must be positive")
        if self.expected_block_size < 1.0:
            raise ValueError("expected_block_size must be >= 1.0")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")

        T = self.matrix.values.shape[0]
        if T < 2:
            raise ValueError(
                f"Matrix history (T={T}) is too short to compute 1-day returns."
            )

    def iter_batches(
        self,
        *,
        required_shift_type_id: np.ndarray,
        source_factor_ids: np.ndarray,
        shock_scale: np.ndarray,
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]:

        # Build factor id -> column map
        id_to_col = build_factor_id_to_col_index(self.matrix.factor_ids)

        # Build shock universe and compute plan
        shock_universe = build_shock_universe(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            id_to_col=id_to_col,
        )

        T = self.matrix.values.shape[0]
        h = int(self.horizon)
        K_req = shock_universe.compute_plan.cols_sorted.shape[0]

        # --- 1. GENERATE ALL RANDOM PATHS UPFRONT (OUTSIDE THE LOOP) ---
        # This guarantees batch-size invariance and maximizes throughput
        # by only calling the RNG C-API once for the entire simulation.
        rng = np.random.default_rng(self.seed)
        master_t_indices = generate_sbb_path_indices(
            T=T,
            h=h,
            B=self.total_scenarios,  # <-- Generates all paths at once
            expected_block_size=self.expected_block_size,
            rng=rng,
        )

        scenario_offset = 0

        # --- 2. STREAM SCENARIO BATCHES ---
        while scenario_offset < self.total_scenarios:
            B = min(batch_size, self.total_scenarios - scenario_offset)

            # Slice the master index array for this specific batch
            batch_t_indices = master_t_indices[
                scenario_offset : scenario_offset + B
            ]

            shocks = np.empty((B, K_req), dtype=self.shock_dtype, order="F")

            _kernel_fused_sbb_h_day_shock(
                values=self.matrix.values,
                cols=shock_universe.compute_plan.cols_sorted,
                shift_types=shock_universe.compute_plan.shift_types_sorted,
                inv_perm=shock_universe.compute_plan.inverse_permutation,
                random_t_indices=batch_t_indices,
                B=B,
                h=h,
                out=shocks,
            )

            yield ScenarioSetBatch(
                shocks=shocks,
                required_to_shock_uix=shock_universe.required_to_shock_uix,
                shock_scale=shock_scale,
                required_shift_type_id=required_shift_type_id,
                scenario_offset=scenario_offset,
                scenario_count_total=self.total_scenarios,
            )

            scenario_offset += B
