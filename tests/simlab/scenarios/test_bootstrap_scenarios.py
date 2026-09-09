# tests/simlab/scenarios/test_bootstrap_scenarios.py

from __future__ import annotations

import numpy as np

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_DIFFERENCE_ID
from riskquant.simlab.scenarios.bootstrap_sbb import (
    StationaryBlockBootstrapGenerator,
)

# =========================================================================
# MOCK PORTFOLIO CONSTANTS
# =========================================================================
MOCK_REQ_FACTOR_IDS = np.array([101, 202], dtype=np.int32)
MOCK_REQ_SHIFT_TYPES = np.array(
    [SHIFT_DIFFERENCE_ID, SHIFT_DIFFERENCE_ID], dtype=np.int8
)
MOCK_REQ_SHOCK_SCALES = np.ones(2, dtype=np.float32)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def _build_matrix() -> RiskFactorMatrix:
    t = np.arange(0, 300, dtype=np.float64)
    values = np.empty((t.size, 2), dtype=np.float64, order="F")

    # Create some synthetic cyclical/trending data
    values[:, 0] = 100.0 + 0.05 * t + 0.7 * np.sin(0.07 * t)
    values[:, 1] = 80.0 + 0.03 * t + 0.4 * np.cos(0.11 * t)

    return RiskFactorMatrix(
        dates=np.arange(t.size, dtype=np.int32),
        factor_ids=MOCK_REQ_FACTOR_IDS,
        values=values,
        fortran_order=True,
    )


def _collect_shocks(
    generator: StationaryBlockBootstrapGenerator, batch_size: int
) -> np.ndarray:
    batches = generator.iter_batches(
        required_shift_type_id=MOCK_REQ_SHIFT_TYPES,
        source_factor_ids=MOCK_REQ_FACTOR_IDS,
        shock_scale=MOCK_REQ_SHOCK_SCALES,
        batch_size=batch_size,
    )
    return np.concatenate([batch.shocks.copy() for batch in batches], axis=0)


# =========================================================================
# TESTS
# =========================================================================


def test_seeded_stream_replays_exactly() -> None:
    matrix = _build_matrix()
    generator = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=257,
        expected_block_size=5.0,
        horizon=3,
        seed=11,
    )

    # Because iter_batches re-initializes the RNG with self.seed every time
    # it is called, iterating the same generator twice should produce identical streams.
    first = _collect_shocks(generator, batch_size=19)
    second = _collect_shocks(generator, batch_size=19)

    np.testing.assert_array_equal(first, second)


def test_seeded_stream_is_batch_size_invariant() -> None:
    matrix = _build_matrix()

    generator_a = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=257,
        expected_block_size=7.0,
        horizon=2,
        seed=123,
    )
    generator_b = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=257,
        expected_block_size=7.0,
        horizon=2,
        seed=123,
    )

    shocks_a = _collect_shocks(generator_a, batch_size=13)
    shocks_b = _collect_shocks(generator_b, batch_size=64)

    np.testing.assert_array_equal(shocks_a, shocks_b)


def test_different_seeds_produce_different_paths() -> None:
    matrix = _build_matrix()

    generator_a = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=128,
        expected_block_size=6.0,
        horizon=1,
        seed=7,
    )
    generator_b = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=128,
        expected_block_size=6.0,
        horizon=1,
        seed=8,
    )

    shocks_a = _collect_shocks(generator_a, batch_size=32)
    shocks_b = _collect_shocks(generator_b, batch_size=32)

    assert not np.array_equal(shocks_a, shocks_b)
