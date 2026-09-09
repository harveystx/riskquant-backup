# src/riskquant/simlab/scenarios/bootstrap_sbb.py
"""
Stationary Block Bootstrap (SBB) scenario generator.

Constructs synthetic h-day paths by stitching contiguous historical
blocks whose lengths are drawn from a Geometric distribution
(parametrised by `expected_block_size`). Per-column aggregation is
log-product, simple-compound, or sum-of-differences according to the
column's shift type.

Implementation
--------------
- All RNG-driven path indices are generated upfront via
  `generate_sbb_path_indices` (RNG draws in NumPy, stitch in a njit
  kernel — see core_math/sbb_block_generator.py).
- Streaming `shocks` buffer is pre-allocated once at `batch_size`.
- Kernel uses a pinned signature with positional args; branch ladder
  hoisted around the B-loop.

References
----------
Politis & Romano (1994); Politis (2003).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numba as nb  # type: ignore[import-untyped]
import numpy as np

from riskquant.jit import (
    _F32_2D_C,
    _F32_2D_F,
    _F64_2D_F,
    _I32_1D_C,
    _I32_2D_C,
    _I8_1D_C,
    parallel_njit,
)
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
from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)



@parallel_njit([
    nb.void(
        _F64_2D_F,  # values            (T, K_total) F-contig
        _I32_1D_C,  # cols              (K_req,)
        _I8_1D_C,  # shift_types       (K_req,)
        _I32_1D_C,  # inv_perm          (K_req,)
        _I32_2D_C,  # random_t_indices  (B, h) C-contig
        nb.int64,  # h
        nb.int64,  # B
        _F32_2D_F,  # out               (B, K_req) F-contig
    ),
    nb.void(
        _F64_2D_F,  # values            (T, K_total) F-contig
        _I32_1D_C,  # cols              (K_req,)
        _I8_1D_C,  # shift_types       (K_req,)
        _I32_1D_C,  # inv_perm          (K_req,)
        _I32_2D_C,  # random_t_indices  (B, h) C-contig
        nb.int64,  # h
        nb.int64,  # B
        _F32_2D_C,  # out               (B, K_req) F-contig
    ),
])
def _kernel_fused_sbb_h_day_shock(
    values,
    cols,
    shift_types,
    inv_perm,
    random_t_indices,
    h,
    batch_len,
    out,
) -> None:
    """
    h-day aggregation of 1-day returns along the SBB-synthetic path.
    Parallelised across columns for F-contig cache-line usage.
    """
    k_cols = cols.shape[0]
    for k in nb.prange(k_cols):
        col = cols[k]
        st = shift_types[k]
        out_col = inv_perm[k]

        col_vals = values[:, col]
        out_col_vals = out[:, out_col]

        if st == SHIFT_LOG_RETURN_ID:
            for b in range(batch_len):
                cum_ratio = 1.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    cum_ratio *= col_vals[t1] / col_vals[t1 - 1]
                out_col_vals[b] = np.float32(math.log(cum_ratio))

        elif st == SHIFT_SIMPLE_RETURN_ID:
            for b in range(batch_len):
                cum_ret = 1.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    cum_ret *= col_vals[t1] / col_vals[t1 - 1]
                out_col_vals[b] = np.float32(cum_ret - 1.0)

        elif st == SHIFT_DIFFERENCE_ID:
            for b in range(batch_len):
                cum_diff = 0.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    cum_diff += col_vals[t1] - col_vals[t1 - 1]
                out_col_vals[b] = np.float32(cum_diff)


# =========================================================================
# GENERATOR
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class StationaryBlockBootstrapGenerator(ScenarioGenerator):
    """SBB generator. Shocks are always float32."""

    matrix: RiskFactorMatrix
    total_scenarios: int
    expected_block_size: float = field(default=1.0)
    horizon: int = field(default=1)
    seed: int | None = field(default=None)

    # Derived in __post_init__.
    available_factor_ids: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total_scenarios <= 0:
            raise ValueError("total_scenarios must be positive")
        if self.expected_block_size < 1.0:
            raise ValueError("expected_block_size must be >= 1.0")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.matrix.values.shape[0] < 2:
            raise ValueError(
                f"Matrix history (T={self.matrix.values.shape[0]}) too short "
                "to compute 1-day returns."
            )
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

        shock_universe = build_shock_universe(
            required_shift_type_id=required_shift_type_id,
            source_factor_ids=source_factor_ids,
            available_factor_ids=self.matrix.factor_ids,
        )
        plan = shock_universe.compute_plan
        k_req = plan.cols_sorted.shape[0]

        # All paths upfront in one njit + RNG burst.
        rng = np.random.default_rng(self.seed)
        master_t_indices = generate_sbb_path_indices(
            t_max=t_max,
            h=h,
            total_scenarios=self.total_scenarios,
            expected_block_size=self.expected_block_size,
            rng=rng,
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

            _kernel_fused_sbb_h_day_shock(
                self.matrix.values,
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
