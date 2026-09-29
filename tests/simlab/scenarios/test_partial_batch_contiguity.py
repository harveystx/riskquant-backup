# tests/simlab/scenarios/test_partial_batch_contiguity.py
"""
Regression test for RQ-AUD-017.

Every streaming scenario generator (`historical.py`, `historical_fhs.py`,
`bootstrap_sbb.py`, `bootstrap_fhs.py`, `parametric.py`) pre-allocates an
F-contiguous `shocks_buf` and yields row-slices of it per batch. Two
contiguity defects existed:

1. A slice `shocks_buf[:batch_len, :]` where `batch_len < batch_size` (the
   final, partial batch) is neither C- nor F-contiguous -- a strided view --
   which fails the strict `_F32_2D_F` Numba kernel signature these
   generators' kernels require for their output parameter.

2. Even a FRESHLY allocated `(1, K)` F-order array is reported by numpy as
   simultaneously C- and F-contiguous (there is only one possible traversal
   order for a single row), and Numba's dispatcher resolves that ambiguity
   to 'C' -- which still fails a strict single-signature `_F32_2D_F` kernel.
   This specifically bites when the trailing batch has exactly one scenario.

Both are exercised here by choosing `total_scenarios` / `batch_size`
combinations that force a partial final batch, including one that forces a
final batch of exactly 1 scenario.
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_DIFFERENCE_ID
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.scenarios.parametric import ParametricGenerator

_FACTOR_IDS = np.array([101, 202], dtype=np.int32)
_SHIFT_TYPES = np.array([SHIFT_DIFFERENCE_ID, SHIFT_DIFFERENCE_ID], dtype=np.int8)
_SHOCK_SCALES = np.ones(2, dtype=np.float32)


def _build_matrix(t: int) -> RiskFactorMatrix:
    values = np.empty((t, 2), dtype=np.float64, order="F")
    values[:, 0] = 100.0 + np.arange(t) * 0.1
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


@pytest.mark.parametrize("total_scenarios,batch_size", [
    (10, 3),   # 3,3,3,1  -> final batch of 1 (the ambiguous C=F case)
    (10, 4),   # 4,4,2    -> final batch >1 but partial
    (10, 5),   # 5,5      -> no partial batch at all (sanity check)
])
def test_historical_generator_all_batches_f_contiguous(total_scenarios, batch_size):
    matrix = _build_matrix(t=total_scenarios + 1)  # horizon=1 -> T-1 scenarios
    gen = HistoricalGenerator(matrix=matrix, horizon=1)
    batches = list(gen.iter_batches(
        chart_ids=_SHIFT_TYPES,
        driver_rf_ids=_FACTOR_IDS,
        batch_size=batch_size,
    ))
    _assert_every_batch_f_contiguous(batches)
    assert sum(b.shocks.shape[0] for b in batches) == total_scenarios


@pytest.mark.parametrize("total_scenarios,batch_size", [
    (10, 3),
    (10, 4),
])
def test_bootstrap_sbb_all_batches_f_contiguous(total_scenarios, batch_size):
    matrix = _build_matrix(t=300)
    gen = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=total_scenarios,
        expected_block_size=5.0,
        horizon=1,
        seed=7,
    )
    batches = list(gen.iter_batches(
        chart_ids=_SHIFT_TYPES,
        driver_rf_ids=_FACTOR_IDS,
        batch_size=batch_size,
    ))
    _assert_every_batch_f_contiguous(batches)
    assert sum(b.shocks.shape[0] for b in batches) == total_scenarios


@pytest.mark.parametrize("total_scenarios,batch_size", [
    (10, 3),
    (10, 4),
])
def test_parametric_generator_all_batches_f_contiguous(total_scenarios, batch_size):
    correlation = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float64)
    gen = ParametricGenerator.standard_normal(
        available_factor_ids=_FACTOR_IDS,
        cholesky_matrix=np.linalg.cholesky(correlation),
        volatilities=np.array([0.2, 0.3], dtype=np.float64),
        total_scenarios=total_scenarios,
        seed=5,
    )
    batches = list(gen.iter_batches(
        chart_ids=_SHIFT_TYPES,
        driver_rf_ids=_FACTOR_IDS,
        batch_size=batch_size,
    ))
    _assert_every_batch_f_contiguous(batches)
    assert sum(b.shocks.shape[0] for b in batches) == total_scenarios


def test_single_scenario_final_batch_matches_full_batch_value():
    """The forced-contiguous single-scenario batch must contain the SAME
    values as an equivalent full-batch (non-partial) run -- proving the
    fresh-buffer fallback path computes correctly, not just contiguously."""
    matrix = _build_matrix(t=11)  # horizon=1 -> 10 scenarios

    gen_partial = HistoricalGenerator(matrix=matrix, horizon=1)
    partial_batches = list(gen_partial.iter_batches(
        chart_ids=_SHIFT_TYPES,
        driver_rf_ids=_FACTOR_IDS,
        batch_size=3,  # 3,3,3,1 -> last batch is the ambiguous 1-row case
    ))
    last_batch_value = partial_batches[-1].shocks[0]

    gen_full = HistoricalGenerator(matrix=matrix, horizon=1)
    full_batches = list(gen_full.iter_batches(
        chart_ids=_SHIFT_TYPES,
        driver_rf_ids=_FACTOR_IDS,
        batch_size=10,  # single full batch, no partial-batch code path at all
    ))
    full_value_at_same_scenario = full_batches[0].shocks[9]

    np.testing.assert_allclose(last_batch_value, full_value_at_same_scenario)
