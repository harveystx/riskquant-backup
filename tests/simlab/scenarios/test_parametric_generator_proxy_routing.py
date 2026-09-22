# tests/simlab/scenarios/test_parametric_generator_proxy_routing.py
"""
`ParametricGenerator` driving the proxy-routed shock path.

Subject: `riskquant.simlab` -- the generator's reproducibility, dtype/layout
and batching contract, the proxy routing it carries (WTI <- BRENT x 0.95), and
the levels `apply_shocks_to_base_levels` produces from its batches. Expected
levels come from the LOG_RETURN identity `level = base * exp(scale * shock)`.

The P&L of these levels under the Taylor representation is not a simlab
concern: the spot Greeks are reconciled against full revaluation in
`tests/sensitivities/test_spot_full_revaluation_reconciliation.py`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric import ParametricGenerator
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels

_AVAILABLE = np.array([0, 1, 2], dtype=np.int32)
_VOLS = np.array([0.006, 0.015, 0.020], dtype=np.float64)
_CORR = np.array(
    [[1.00, 0.10, 0.05], [0.10, 1.00, 0.35], [0.05, 0.35, 1.00]], dtype=np.float64
)
_SHIFT_LOG_RETURN = np.full(4, SHIFT_LOG_RETURN_ID, dtype=np.int8)

# required = [USDEUR, AAPL, BRENT, WTI]; WTI sources BRENT (col 2) x 0.95.
_SOURCE_FACTOR_IDS = np.array([0, 1, 2, 2], dtype=np.int32)
_SHOCK_SCALE = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32)


def _generator(seed: int, total: int = 256) -> ParametricGenerator:
    return ParametricGenerator.standard_normal(
        available_factor_ids=_AVAILABLE,
        cholesky_matrix=np.linalg.cholesky(_CORR),
        volatilities=_VOLS,
        total_scenarios=total,
        seed=seed,
    )


def _iter_batches(generator: ParametricGenerator, batch_size: int) -> Iterator:
    return generator.iter_batches(
        required_shift_type_id=_SHIFT_LOG_RETURN,
        source_factor_ids=_SOURCE_FACTOR_IDS,
        shock_scale=_SHOCK_SCALE,
        batch_size=batch_size,
    )


def _all_batches(generator: ParametricGenerator, batch_size: int) -> list:
    return list(_iter_batches(generator, batch_size))


def _stacked_shocks(generator: ParametricGenerator, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [b.shocks.copy() for b in _all_batches(generator, batch_size)], axis=0
    )


# =============================================================================
# Scenario generation
# =============================================================================


def test_same_seed_is_bitwise_identical() -> None:
    a = _stacked_shocks(_generator(20_260_913), 64)
    b = _stacked_shocks(_generator(20_260_913), 64)
    assert np.array_equal(a, b)


def test_different_seed_differs() -> None:
    a = _stacked_shocks(_generator(20_260_913), 64)
    b = _stacked_shocks(_generator(7), 64)
    assert not np.array_equal(a, b)


def test_batching_is_reproducible_for_a_fixed_batch_size() -> None:
    """
    Reproducibility is guaranteed per (seed, batch_size), NOT across batch
    sizes.

    `ParametricGenerator` draws a 2-D `(batch_size, k_total)` block per batch
    via `rng.standard_normal(size=work_view.shape, out=work_view)`, so the
    batch geometry determines how the seeded stream is partitioned. Changing
    `batch_size` therefore changes the realised scenarios. That is inherent
    to the generator's streaming design, not a defect -- but it means
    `batch_size` is part of the reproducibility key and must be pinned
    alongside the seed for a reproducible risk run.
    """
    for batch_size in (64, 256):
        first = _stacked_shocks(_generator(20_260_913), batch_size)
        second = _stacked_shocks(_generator(20_260_913), batch_size)
        assert first.shape == (256, 3)
        np.testing.assert_array_equal(first, second)

    # Documented consequence: different batch sizes are NOT interchangeable.
    assert not np.array_equal(
        _stacked_shocks(_generator(20_260_913), 64),
        _stacked_shocks(_generator(20_260_913), 256),
    )


def test_shock_dtype_and_layout_are_canonical() -> None:
    batch = _all_batches(_generator(20_260_913), 64)[0]
    assert batch.shocks.dtype == np.float32
    assert batch.shocks.flags["F_CONTIGUOUS"]
    assert batch.shocks.shape[1] == _AVAILABLE.shape[0]


def test_proxy_routing_survives_the_generator() -> None:
    """
    The generator consumes the resolved proxy outputs and maps WTI onto
    BRENT's shock column with the 0.95 scale -- no demo-side proxy work.
    """
    batch = _all_batches(_generator(20_260_913), 64)[0]
    assert batch.required_to_shock_uix.tolist() == [0, 1, 2, 2]
    np.testing.assert_allclose(batch.shock_scale, [1.0, 1.0, 1.0, 0.95], atol=1e-7)


def test_wti_is_not_simulated_independently() -> None:
    """WTI has no column of its own: only the 3 source factors are drawn."""
    batch = _all_batches(_generator(20_260_913), 64)[0]
    assert batch.shocks.shape[1] == 3
    # WTI's required row points at BRENT's column, not a fourth column.
    assert int(batch.required_to_shock_uix[3]) == int(batch.required_to_shock_uix[2])


# =============================================================================
# Production shock application over generated batches
# =============================================================================

# required = [USDEUR, AAPL, BRENT, WTI] base levels.
_BASE = np.array([1.08, 200.0, 82.30, 70.0], dtype=np.float64)


def _production_levels(
    generator: ParametricGenerator, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    (N_req, S) shocked levels from the production kernel, plus the (S, K)
    shocks that drove them. Each batch is applied while it is live: the
    generator may reuse its shock buffer between batches.
    """
    levels: list[np.ndarray] = []
    shocks: list[np.ndarray] = []
    for batch in _iter_batches(generator, batch_size):
        out = np.empty(
            (_BASE.shape[0], batch.shocks.shape[0]), dtype=np.float64, order="C"
        )
        apply_shocks_to_base_levels(base_levels=_BASE, batch=batch, out=out)
        levels.append(out)
        shocks.append(batch.shocks.copy())
    return np.concatenate(levels, axis=1), np.concatenate(shocks, axis=0)


def test_proxied_wti_level_follows_brents_shock_at_095_scale() -> None:
    """
    WTI keeps its own base level; only the shock provenance is Brent's, and
    the 0.95 scale applies inside the exponent.
    """
    levels, shocks = _production_levels(_generator(20_260_913, total=128), 64)
    expected = _BASE[3] * np.exp(0.95 * shocks[:, 2].astype(np.float64))
    np.testing.assert_allclose(levels[3], expected, rtol=1e-6)


def test_wti_level_is_not_brents_level() -> None:
    """Proxying transfers the SHOCK, never the level."""
    levels, _ = _production_levels(_generator(20_260_913, total=64), 64)
    assert float(np.abs(levels[3] - levels[2]).min()) > 1.0
    assert abs(float(levels[3].mean()) - 70.0) < 2.0


@pytest.mark.parametrize("scenario_count", [1, 7, 256])
def test_generator_emits_the_requested_scenario_count(scenario_count: int) -> None:
    stacked = _stacked_shocks(_generator(20_260_913, total=scenario_count), 64)
    assert stacked.shape[0] == scenario_count
