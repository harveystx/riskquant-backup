# tests/simlab/scenarios/test_filtered_historical_scenarios.py
"""
Coverage for the Filtered Historical Simulation path.

`FilteredHistoricalGenerator` (`historical_fhs.py`) and its EWMA
volatility-ratio kernel (`core_math/ewma_volatility.py`) shipped with NO
test coverage. `test_partial_batch_contiguity.py` names `historical_fhs.py`
in its docstring but exercises only `HistoricalGenerator`,
`StationaryBlockBootstrapGenerator` and `ParametricGenerator` -- the FHS
class is never actually driven there. This module closes that gap:

1. The EWMA Z-score kernel `kernel_ewma_vol_ratios_from_residuals` matches
   an independent two-pass oracle at machine precision, including the
   `MIN_EWMA_VARIANCE` floor and the `MAX_EWMA_VOL_RATIO` clamp branches.
2. Every FHS batch is `(B, K_req)` float32 F-contiguous, for full,
   partial and single-row trailing batches (the RQ-AUD-017 / D008 case the
   FHS kernel's dual `_F32_2D_F`/`_F32_2D_C` overload exists to survive).
3. The forced-contiguous single-row trailing batch carries the SAME value
   as the equivalent full-batch run -- proving the fresh-F-buffer fallback
   computes correctly, not merely contiguously.
4. `total_scenarios == T - horizon`, and a history shorter than the EWMA
   minimum raises rather than silently degrading.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.constants import (
    EWMA_DECAY_FACTOR,
    MAX_EWMA_VOL_RATIO,
    MIN_EWMA_VARIANCE,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios_from_residuals,
)
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator

_FACTOR_IDS = np.array([101, 202], dtype=np.int32)
_SHIFT_TYPES = np.array([SHIFT_LOG_RETURN_ID, SHIFT_LOG_RETURN_ID], dtype=np.int8)
_SHOCK_SCALES = np.ones(2, dtype=np.float32)


def _build_matrix(t: int) -> RiskFactorMatrix:
    """Two strictly positive columns (LOG_RETURN needs P_t/P_{t-1} > 0)."""
    values = np.empty((t, 2), dtype=np.float64, order="F")
    values[:, 0] = 100.0 + np.arange(t) * 0.10
    values[:, 1] = 80.0 + np.arange(t) * 0.05
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=_FACTOR_IDS,
        values=values,
    )


def _assert_every_batch_f_contiguous(batches) -> None:
    for batch in batches:
        assert batch.shocks.flags["F_CONTIGUOUS"], (
            f"non-F-contiguous shocks batch of shape {batch.shocks.shape}"
        )
        assert batch.shocks.dtype == np.float32


# ------------------------------------------------------------------ EWMA oracle


def _ewma_ratio_oracle(
    col_res: np.ndarray, decay: float, max_ratio: float
) -> np.ndarray:
    """
    Independent two-pass Z_t = sqrt(var_today / var_t), clamped.

    Mirrors the documented recursion exactly: seed var_1 = r_1**2 (index 1),
    var_t = (1-lambda)*r_t**2 + lambda*var_{t-1}; indices t=0 and t=1 share
    the seed variance; a variance below MIN_EWMA_VARIANCE yields ratio 1.0.
    Written from the definition, NOT by calling the kernel under test.
    """
    t_max = col_res.shape[0]
    out = np.empty(t_max, dtype=np.float64)
    if t_max < 2:
        out[:] = 1.0
        return out

    one_minus_lam = 1.0 - decay

    var_t = col_res[1] * col_res[1]
    for t in range(2, t_max):
        var_t = one_minus_lam * col_res[t] * col_res[t] + decay * var_t
    var_today = var_t

    def emit(v: float) -> float:
        if v < MIN_EWMA_VARIANCE:
            return 1.0
        ratio = float(np.sqrt(var_today / v))
        return max_ratio if ratio > max_ratio else ratio

    var_t = col_res[1] * col_res[1]
    out[0] = emit(var_t)
    out[1] = emit(var_t)
    for t in range(2, t_max):
        var_t = one_minus_lam * col_res[t] * col_res[t] + decay * var_t
        out[t] = emit(var_t)
    return out


def test_ewma_vol_ratio_kernel_matches_independent_oracle() -> None:
    t_max = 12
    residuals = np.zeros((t_max, 3), dtype=np.float64, order="F")
    # col0: ordinary alternating residuals.
    residuals[:, 0] = np.array(
        [
            0,
            0.010,
            -0.012,
            0.008,
            -0.006,
            0.011,
            -0.009,
            0.007,
            -0.005,
            0.013,
            -0.004,
            0.006,
        ]
    )
    # col1: an early vol spike that decays -> drives ratios below 1 (no clamp).
    residuals[:, 1] = np.array(
        [
            0,
            0.300,
            -0.001,
            0.0008,
            -0.0006,
            0.0011,
            -0.0009,
            0.0007,
            -0.0005,
            0.0013,
            -0.0004,
            0.0006,
        ]
    )
    # col2: vol RISES toward today from a tiny seed -> exercises the clamp.
    residuals[:, 2] = np.array(
        [0, 1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-2, 5e-2]
    )

    got = np.empty_like(residuals)
    kernel_ewma_vol_ratios_from_residuals(
        residuals, float(EWMA_DECAY_FACTOR), float(MAX_EWMA_VOL_RATIO), got
    )
    expected = np.stack(
        [
            _ewma_ratio_oracle(residuals[:, k], EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO)
            for k in range(residuals.shape[1])
        ],
        axis=1,
    )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
    # "today" (final index) is by definition sigma_today/sigma_today == 1.0.
    np.testing.assert_allclose(got[-1, :], 1.0, rtol=0.0, atol=0.0)
    # The clamp branch was genuinely reached in col2.
    assert np.any(got[:, 2] >= MAX_EWMA_VOL_RATIO - 1e-12)


def test_ewma_variance_floor_yields_unit_ratio() -> None:
    """Dead (all-zero-residual) columns have var_t < floor everywhere -> 1.0.

    Two columns keeps this test's intent (the floor branch) separate from the
    single-column path, which has its own dedicated tests below.
    """
    residuals = np.zeros((8, 2), dtype=np.float64, order="F")
    got = np.empty_like(residuals)
    kernel_ewma_vol_ratios_from_residuals(
        residuals, float(EWMA_DECAY_FACTOR), float(MAX_EWMA_VOL_RATIO), got
    )
    np.testing.assert_allclose(got, 1.0, rtol=0.0, atol=0.0)


# ------------------------------------------------------------ generator contract


@pytest.mark.parametrize(
    "total_scenarios,batch_size",
    [
        (9, 4),  # 4,4,1  -> single-row trailing batch (the ambiguous C=F case)
        (10, 3),  # 3,3,3,1 -> single-row trailing batch
        (10, 4),  # 4,4,2  -> partial (>1) trailing batch
        (10, 5),  # 5,5    -> no partial batch (sanity)
    ],
)
def test_fhs_generator_all_batches_f_contiguous(total_scenarios, batch_size) -> None:
    matrix = _build_matrix(t=total_scenarios + 1)  # horizon=1 -> T-1 scenarios
    gen = FilteredHistoricalGenerator(matrix=matrix, horizon=1)
    assert gen.total_scenarios == total_scenarios
    batches = list(
        gen.iter_batches(
            chart_ids=_SHIFT_TYPES,
            driver_rf_ids=_FACTOR_IDS,
            batch_size=batch_size,
        )
    )
    _assert_every_batch_f_contiguous(batches)
    assert sum(b.shocks.shape[0] for b in batches) == total_scenarios


def test_fhs_single_scenario_final_batch_matches_full_batch_value() -> None:
    """The forced-contiguous single-row trailing batch must hold the SAME
    value as the equivalent full-batch run -- the fresh-F-buffer fallback
    computes correctly, not merely contiguously (RQ-AUD-017 / D008)."""
    matrix = _build_matrix(t=11)  # horizon=1 -> 10 scenarios

    partial = list(
        FilteredHistoricalGenerator(matrix=matrix, horizon=1).iter_batches(
            chart_ids=_SHIFT_TYPES,
            driver_rf_ids=_FACTOR_IDS,
            batch_size=3,  # 3,3,3,1 -> ambiguous single-row final batch
        )
    )
    full = list(
        FilteredHistoricalGenerator(matrix=matrix, horizon=1).iter_batches(
            chart_ids=_SHIFT_TYPES,
            driver_rf_ids=_FACTOR_IDS,
            batch_size=10,  # single full batch, no partial-batch path
        )
    )
    np.testing.assert_array_equal(partial[-1].shocks[0], full[0].shocks[9])


def test_fhs_history_too_short_raises() -> None:
    """Fail fast: fewer than the EWMA minimum observations must raise."""
    with pytest.raises(ValueError):
        FilteredHistoricalGenerator(matrix=_build_matrix(t=2), horizon=1)


# ----------------------------------------------- single-column universe (D024)
#
# A single-column `(T, 1)` f64 array is C=F ambiguous; Numba resolves it to
# 'C', so the `_F64_2D_F`-only kernels once rejected it (TypeError) for any
# single-factor universe. D024 added the `_F64_2D_C` overloads. These cover
# both trigger shapes, for BOTH the plain and filtered historical generators.


def _drive(gen, req_factor_ids: np.ndarray, batch_size: int) -> np.ndarray:
    n = int(req_factor_ids.shape[0])
    batches = list(
        gen.iter_batches(
            chart_ids=np.full(n, SHIFT_LOG_RETURN_ID, dtype=np.int8),
            driver_rf_ids=req_factor_ids,
            batch_size=batch_size,
        )
    )
    for b in batches:
        assert b.shocks.dtype == np.float32
        assert b.shocks.flags["F_CONTIGUOUS"]
    return np.concatenate([b.shocks for b in batches], axis=0)


@pytest.mark.parametrize("gen_cls", [FilteredHistoricalGenerator, HistoricalGenerator])
def test_single_factor_matrix_runs(gen_cls) -> None:
    """Single-factor MATRIX (K_total == 1): values and every scratch buffer are
    `(T, 1)`. The all-C overloads admit them; this once raised for both."""
    values = (100.0 + np.arange(8) * 0.1).reshape(8, 1).copy(order="F")
    matrix = RiskFactorMatrix(
        dates=np.arange(8, dtype=np.int32),
        factor_ids=np.array([101], dtype=np.int32),
        values=values,
    )
    shocks = _drive(
        gen_cls(matrix=matrix, horizon=1), np.array([101], dtype=np.int32), 4
    )
    assert shocks.shape == (7, 1)


@pytest.mark.parametrize("gen_cls", [FilteredHistoricalGenerator, HistoricalGenerator])
def test_single_required_factor_matches_multi_factor_column(gen_cls) -> None:
    """AR(1) + EWMA are per-column independent, so requiring ONE factor from a
    multi-factor matrix (K_total > 1, K_req == 1 -> the mixed F-in / C-out AR1
    overload) reproduces that factor's full-universe column, value for value."""
    matrix = _build_matrix(t=9)  # ids [101, 202] -> K_total == 2
    full = _drive(gen_cls(matrix=matrix, horizon=1), _FACTOR_IDS, 4)
    one = _drive(gen_cls(matrix=matrix, horizon=1), np.array([202], dtype=np.int32), 4)
    np.testing.assert_array_equal(one[:, 0], full[:, 1])
