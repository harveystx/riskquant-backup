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
from riskquant.simlab.shock_plan.plan import build_world_column_plan
from riskquant.simlab.scenarios.base import (
    ScenarioGenerator,
    ScenarioSetBatch,
    require_float64_values,
    require_positive_int,
)
from riskquant.simlab.scenarios.core_math.sbb_block_generator import (
    generate_sbb_path_indices,
)


@parallel_njit([
    _NB_VOID(
        _F64_2D_F,  # values            (T, K_total) F-contig
        _I32_1D_C,  # cols              (K_req,)
        _I8_1D_C,  # shift_types       (K_req,)
        _I32_1D_C,  # inv_perm          (K_req,)
        _I32_2D_C,  # random_t_indices  (B, h) C-contig
        _NB_INT64,  # h
        _NB_INT64,  # B
        _F32_2D_F,  # out               (B, K_req) F-contig
    ),
    _NB_VOID(
        _F64_2D_F,  # values            (T, K_total) F-contig
        _I32_1D_C,  # cols              (K_req,)
        _I8_1D_C,  # shift_types       (K_req,)
        _I32_1D_C,  # inv_perm          (K_req,)
        _I32_2D_C,  # random_t_indices  (B, h) C-contig
        _NB_INT64,  # h
        _NB_INT64,  # B
        _F32_2D_C,  # out               (B, K_req) F-contig
    ),
    _NB_VOID(
        _F64_2D_C,  # values            (T, 1) C-contig — single-column (T,1)
        _I32_1D_C,  # cols              (K_req,)
        _I8_1D_C,  # shift_types       (K_req,)
        _I32_1D_C,  # inv_perm          (K_req,)
        _I32_2D_C,  # random_t_indices  (B, h) C-contig
        _NB_INT64,  # h
        _NB_INT64,  # B
        _F32_2D_C,  # out               (B, 1) C-contig
    ),
    # Single-factor matrix, several charts of that factor (D088 world column
    # (d, tau, h) per chart): values (T, 1) resolves to 'C', out is genuine F.
    _NB_VOID(
        _F64_2D_C,  # values            (T, 1) C-contig — single-column (T,1)
        _I32_1D_C,  # cols              (K_req,)
        _I8_1D_C,  # shift_types       (K_req,)
        _I32_1D_C,  # inv_perm          (K_req,)
        _I32_2D_C,  # random_t_indices  (B, h) C-contig
        _NB_INT64,  # h
        _NB_INT64,  # B
        _F32_2D_F,  # out               (B, K_req) F-contig
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
        # Contract (plan.py): inv_perm[k] is the SORTED compute position of
        # shock-universe column k, a gather map. Output column k reads it.
        pos = inv_perm[k]
        col = cols[pos]
        st = shift_types[pos]
        out_col = k

        col_vals = values[:, col]
        out_col_vals = out[:, out_col]

        if st == SHIFT_NO_SHIFT_ID:
            for b in range(batch_len):
                out_col_vals[b] = np.float32(0.0)

        elif st == SHIFT_LOG_RETURN_ID:
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

        elif st == SHIFT_ABSOLUTE_BASE_RETURN_ID:
            # Sign-robust SIMPLE_RETURN: aggregate the daily gross factors
            # multiplicatively so that on positive levels it equals
            # SIMPLE_RETURN exactly, for every horizon.
            for b in range(batch_len):
                cum_ret = 1.0
                for step in range(h):
                    t1 = random_t_indices[b, step]
                    prev = col_vals[t1 - 1]
                    cum_ret *= 1.0 + (col_vals[t1] - prev) / abs(prev)
                out_col_vals[b] = np.float32(cum_ret - 1.0)


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
        require_float64_values(self.matrix.values)
        object.__setattr__(
            self,
            "total_scenarios",
            require_positive_int(self.total_scenarios, "total_scenarios"),
        )
        object.__setattr__(
            self, "horizon", require_positive_int(self.horizon, "horizon")
        )
        # expected_block_size is the CONTINUOUS mean block length of the
        # stationary bootstrap's geometric block distribution, not a discrete
        # count: accept int|float, reject bool/non-finite, require >= 1.0.
        ebs = self.expected_block_size
        if isinstance(ebs, bool) or not isinstance(
            ebs, (int, float, np.integer, np.floating)
        ):
            raise TypeError("expected_block_size must be a real number.")
        if not math.isfinite(ebs):
            raise ValueError("expected_block_size must be finite.")
        if ebs < 1.0:
            raise ValueError("expected_block_size must be >= 1.0")
        object.__setattr__(self, "expected_block_size", float(ebs))
        if self.matrix.values.shape[0] < 2:
            raise ValueError(
                f"Matrix history (T={self.matrix.values.shape[0]}) too short "
                "to compute 1-day returns."
            )
        object.__setattr__(self, "available_factor_ids", self.matrix.factor_ids)

    def iter_batches(
        self,
        *,
        driver_rf_ids: np.ndarray,
        chart_ids: np.ndarray,
        batch_size: int,
    ) -> Iterator[ScenarioSetBatch]:
        t_max = self.matrix.values.shape[0]
        h = int(self.horizon)

        plan = build_world_column_plan(
            driver_rf_ids=driver_rf_ids,
            chart_ids=chart_ids,
            factor_ids=self.matrix.factor_ids,
        )
        validate_shift_domain(
            values=self.matrix.values,
            cols_sorted=plan.cols_sorted,
            shift_type_ids=plan.shift_types_sorted,
            factor_ids=self.matrix.factor_ids,
        )
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
                scenario_offset=scenario_offset,
                scenario_count_total=self.total_scenarios,
            )

            scenario_offset += batch_len
