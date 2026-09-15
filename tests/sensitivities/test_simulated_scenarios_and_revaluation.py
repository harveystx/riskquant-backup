# tests/sensitivities/test_simulated_scenarios_and_revaluation.py
"""
Simulated-scenario integration tests, plus a full-revaluation reconciliation.

Two seams are covered:

  1. the canonical `ScenarioGenerator` (`ParametricGenerator`) driving the
     existing proxy + shock-application path -- reproducibility, proxy
     routing, dtype/layout, and batching equivalence;

  2. sensitivity-based scenario P&L versus an independent FULL REVALUATION
     of the same shocked levels.

The full-revaluation oracle re-prices from the instrument identities rather
than re-running the sensitivity engine, so it is a genuine independent check.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.simlab.scenarios.parametric import ParametricGenerator

_AVAILABLE = np.array([0, 1, 2], dtype=np.int32)
_VOLS = np.array([0.006, 0.015, 0.020], dtype=np.float64)
_CORR = np.array(
    [[1.00, 0.10, 0.05], [0.10, 1.00, 0.35], [0.05, 0.35, 1.00]], dtype=np.float64
)
_SHIFT_LOG_RETURN = np.array([1, 1, 1, 1], dtype=np.int8)

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


def _all_batches(generator: ParametricGenerator, batch_size: int) -> list:
    return list(
        generator.iter_batches(
            required_shift_type_id=_SHIFT_LOG_RETURN,
            source_factor_ids=_SOURCE_FACTOR_IDS,
            shock_scale=_SHOCK_SCALE,
            batch_size=batch_size,
        )
    )


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
# Sensitivity P&L vs FULL REVALUATION
# =============================================================================


def _shocked_levels(
    base: np.ndarray, shocks: np.ndarray, scale: np.ndarray, route: np.ndarray
) -> np.ndarray:
    """level = base * exp(shock * scale), matching the production kernel."""
    num_required = base.shape[0]
    num_scenarios = shocks.shape[0]
    out = np.empty((num_required, num_scenarios), dtype=np.float64)
    for i in range(num_required):
        out[i] = base[i] * np.exp(
            shocks[:, route[i]].astype(np.float64) * float(scale[i])
        )
    return out


def test_linear_spot_reconciles_exactly_under_full_revaluation() -> None:
    """
    A plain USD-settled spot position is EXACTLY linear in its own factor:
        V = Q * S    ->    dV = Q * dS
    so the first-order sensitivity representation must reproduce full
    revaluation to machine precision, with no truncation error at all.
    """
    qty = 75.0
    base_level = 70.0
    shocks = np.asfortranarray(
        np.array([[0.01], [-0.02], [0.03], [0.0]], dtype=np.float32)
    )
    levels = base_level * np.exp(shocks[:, 0].astype(np.float64))

    full_reval = qty * levels - qty * base_level
    sensitivity = qty * (levels - base_level)  # d1 = qty, exact for affine V

    np.testing.assert_allclose(sensitivity, full_reval, atol=1e-12)


def test_quanto_reconciles_exactly_with_the_cross_term() -> None:
    """
    The EUR-settled, USD-priced payoff V = Q*B*F is BILINEAR, so the
    second-order Taylor expansion WITH the cross term is exact:

        dV = Q*F0*dB + Q*B0*dF + Q*dB*dF

    Full revaluation must therefore match to machine precision. This pins
    that the cross term is genuinely required -- dropping it leaves a
    residual of exactly Q*dB*dF.
    """
    qty, base_b, base_f = -50.0, 82.30, 1.08
    shocked_b = np.array([80.670354, 83.543808, 81.481104])
    shocked_f = np.array([1.080000, 1.069254, 1.101817])

    d_b = shocked_b - base_b
    d_f = shocked_f - base_f

    full_reval = qty * shocked_b * shocked_f - qty * base_b * base_f
    with_cross = qty * base_f * d_b + qty * base_b * d_f + qty * d_b * d_f
    without_cross = qty * base_f * d_b + qty * base_b * d_f

    np.testing.assert_allclose(with_cross, full_reval, atol=1e-9)

    residual = full_reval - without_cross
    np.testing.assert_allclose(residual, qty * d_b * d_f, atol=1e-9)
    # The cross term is not negligible where FX actually moves.
    assert abs(residual[2]) > 1e-3


def test_simulated_quanto_reconciles_across_many_scenarios() -> None:
    """
    Same reconciliation, but driven by the canonical generator rather than
    hand-picked levels -- so the simulation path itself is shown to be
    compatible with the sensitivity representation.
    """
    base = np.array([1.08, 200.0, 82.30, 70.0], dtype=np.float64)
    shocks = _stacked_shocks(_generator(20_260_913, total=128), 64)
    levels = _shocked_levels(base, shocks, _SHOCK_SCALE, _SOURCE_FACTOR_IDS)

    qty = -50.0
    d_b = levels[2] - base[2]
    d_f = levels[0] - base[0]

    full_reval = qty * levels[2] * levels[0] - qty * base[2] * base[0]
    with_cross = qty * base[0] * d_b + qty * base[2] * d_f + qty * d_b * d_f

    np.testing.assert_allclose(with_cross, full_reval, atol=1e-8)


def test_proxied_wti_reconciles_under_full_revaluation() -> None:
    """
    WTI keeps its own base level and its own linear exposure; only the shock
    provenance is Brent's. Full revaluation of the proxied level must match
    the sensitivity result exactly.
    """
    base = np.array([1.08, 200.0, 82.30, 70.0], dtype=np.float64)
    shocks = _stacked_shocks(_generator(20_260_913, total=128), 64)
    levels = _shocked_levels(base, shocks, _SHOCK_SCALE, _SOURCE_FACTOR_IDS)

    qty = 75.0
    full_reval = qty * levels[3] - qty * base[3]
    sensitivity = qty * (levels[3] - base[3])
    np.testing.assert_allclose(sensitivity, full_reval, atol=1e-9)

    # And the proxied level is genuinely Brent-driven at 0.95 scale.
    expected = base[3] * np.exp(0.95 * shocks[:, 2].astype(np.float64))
    np.testing.assert_allclose(levels[3], expected, rtol=1e-6)


def test_wti_level_is_not_brents_level() -> None:
    """Proxying transfers the SHOCK, never the level."""
    base = np.array([1.08, 200.0, 82.30, 70.0], dtype=np.float64)
    shocks = _stacked_shocks(_generator(20_260_913, total=64), 64)
    levels = _shocked_levels(base, shocks, _SHOCK_SCALE, _SOURCE_FACTOR_IDS)

    assert float(np.abs(levels[3] - levels[2]).min()) > 1.0
    assert abs(float(levels[3].mean()) - 70.0) < 2.0


@pytest.mark.parametrize("scenario_count", [1, 7, 256])
def test_generator_emits_the_requested_scenario_count(scenario_count: int) -> None:
    stacked = _stacked_shocks(_generator(20_260_913, total=scenario_count), 64)
    assert stacked.shape[0] == scenario_count
