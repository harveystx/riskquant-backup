# tests/solutions/scenario_models/test_sbb_seam.py
"""
SBB (StationaryBlockBootstrap) configuration-seam tests: the second real model
through the solution-tier seam, proving it generalises without a new
abstraction. Model-specific config only; construction routes to the existing
generator; direct-vs-seam equivalence and D029 batch-size invariance hold.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.schemas import (
    HistoricalGeneratorConfig,
    SbbGeneratorConfig,
)


def _matrix(k=2, t=16):
    vals = np.asfortranarray(
        np.array(
            [[100.0 + i + 3 * j for i in range(t)] for j in range(k)], np.float64
        ).T
    )
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.arange(k, dtype=np.int32),
        values=vals,
    )


def _drive(gen, batch_size=4):
    chunks = []
    for b in gen.iter_batches(
        required_shift_type_id=np.array(
            [SHIFT_LOG_RETURN_ID, SHIFT_LOG_RETURN_ID], np.int8
        ),
        source_factor_ids=np.array([0, 1], np.int32),
        shock_scale=np.ones(2, np.float32),
        batch_size=batch_size,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


def _cfg(**over):
    base = {"model": "sbb", "horizon": 2, "total_scenarios": 12, "seed": 7}
    base.update(over)
    return base


# ---- selection & validation ---------------------------------------------------


def test_valid_sbb_config() -> None:
    cfg = parse_scenario_model_config(_cfg(expected_block_size=2.0))
    assert isinstance(cfg, SbbGeneratorConfig)
    assert cfg.horizon == 2 and cfg.total_scenarios == 12
    assert cfg.expected_block_size == 2.0 and cfg.seed == 7
    assert type(cfg.horizon) is int and type(cfg.total_scenarios) is int


def test_expected_block_size_defaults_to_one() -> None:
    cfg = parse_scenario_model_config({
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": 5,
    })
    assert cfg.expected_block_size == 1.0
    assert cfg.seed is None


@pytest.mark.parametrize("field", ["horizon", "total_scenarios"])
@pytest.mark.parametrize("bad", [1.5, 2.0, True, "3"])
def test_strict_integer_rejection(field, bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(**{field: bad}))


@pytest.mark.parametrize("field", ["horizon", "total_scenarios"])
@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_rejection(field, bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(**{field: bad}))


# Negative seeds were accepted here historically, but they can never produce a
# run: numpy's SeedSequence rejects them deep inside the bit generator during
# iteration. The boundary now bounds seed to >= 0; rejection is asserted in
# test_rng_semantics.py.
@pytest.mark.parametrize("seed", [7, 0, None])
def test_seed_accepts_int_or_none(seed) -> None:
    cfg = parse_scenario_model_config(_cfg(seed=seed))
    assert cfg.seed == seed


@pytest.mark.parametrize("bad_seed", [1.5, True, "7"])
def test_seed_strict_rejects_non_int(bad_seed) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(seed=bad_seed))


@pytest.mark.parametrize("good", [1.0, 2.5, 3])  # int and float accepted
def test_expected_block_size_real_accepted(good) -> None:
    cfg = parse_scenario_model_config(_cfg(expected_block_size=good))
    assert cfg.expected_block_size == float(good)
    assert type(cfg.expected_block_size) is float


@pytest.mark.parametrize("bad", [True, 0.5, 0.0, float("inf"), float("nan")])
def test_expected_block_size_invalid_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(expected_block_size=bad))


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(model="block_bootstrap"))


# ---- configuration architecture: no cross-model parameter leakage -------------


@pytest.mark.parametrize(
    "foreign",
    [
        {"decay_factor": 0.9},  # FilteredHistorical / FilteredBootstrap
        {"max_vol_ratio": 8.0},  # FilteredHistorical / FilteredBootstrap
        {"distribution_kind": "student_t"},  # Parametric
        {"student_t_df": 5.0},  # Parametric
        {"vg_nu": 0.5},  # Parametric
        {"cholesky_matrix": [[1.0]]},  # Parametric
    ],
)
def test_sbb_rejects_foreign_model_parameters(foreign) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(**foreign))


def test_historical_rejects_sbb_parameters() -> None:
    """The other direction: Historical config rejects SBB-only parameters."""
    for sbb_only in ("total_scenarios", "expected_block_size", "seed"):
        with pytest.raises(ValidationError):
            parse_scenario_model_config({
                "model": "historical",
                "horizon": 2,
                sbb_only: 3,
            })


def test_sbb_rejects_historical_shaped_but_missing_required() -> None:
    """An SBB selection missing its required total_scenarios fails."""
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": "sbb", "horizon": 2})


# ---- construction & equivalence ----------------------------------------------


def test_build_returns_sbb_generator() -> None:
    m = _matrix()
    cfg = SbbGeneratorConfig(
        model="sbb", horizon=2, total_scenarios=12, expected_block_size=2.0, seed=7
    )
    gen = build_scenario_generator(cfg, data=m)
    assert isinstance(gen, StationaryBlockBootstrapGenerator)
    assert gen.total_scenarios == 12 and gen.horizon == 2
    assert gen.expected_block_size == 2.0 and gen.seed == 7
    assert type(gen.total_scenarios) is int and type(gen.horizon) is int


def test_build_runs_generator_own_validation() -> None:
    """Seam does not bypass the generator's runtime invariants: a float32
    matrix fails at construction (F5/D017), not silently."""
    bad = RiskFactorMatrix(
        dates=np.arange(6, dtype=np.int32),
        factor_ids=np.arange(2, dtype=np.int32),
        values=np.asfortranarray(
            np.array([[100.0 + i for i in range(6)]] * 2, np.float32).T
        ),
    )
    cfg = SbbGeneratorConfig(model="sbb", horizon=2, total_scenarios=5, seed=1)
    with pytest.raises(TypeError, match="must be float64"):
        build_scenario_generator(cfg, data=bad)


@pytest.mark.parametrize("horizon", [1, 3])
def test_seam_equivalent_to_direct_construction(horizon) -> None:
    """count, ordering, values, dtype, layout all identical to direct."""
    m = _matrix()
    direct = StationaryBlockBootstrapGenerator(
        matrix=m, total_scenarios=12, expected_block_size=2.0, horizon=horizon, seed=7
    )
    seamed = build_scenario_generator(
        SbbGeneratorConfig(
            model="sbb",
            horizon=horizon,
            total_scenarios=12,
            expected_block_size=2.0,
            seed=7,
        ),
        data=m,
    )
    rd, rs = _drive(direct), _drive(seamed)
    assert rd.shape == rs.shape
    assert rd.dtype == rs.dtype == np.float32
    assert rs.flags["F_CONTIGUOUS"] == rd.flags["F_CONTIGUOUS"]
    assert np.array_equal(rd, rs)
    assert direct.total_scenarios == seamed.total_scenarios


def test_same_seed_through_seam_preserves_d029_stream() -> None:
    """A seeded SBB built through the seam reproduces the same stream as direct
    construction AND remains batch-size invariant (D029)."""
    m = _matrix()
    seam = build_scenario_generator(
        SbbGeneratorConfig(
            model="sbb", horizon=2, total_scenarios=12, expected_block_size=2.0, seed=42
        ),
        data=m,
    )
    direct = StationaryBlockBootstrapGenerator(
        matrix=m, total_scenarios=12, expected_block_size=2.0, horizon=2, seed=42
    )
    assert np.array_equal(_drive(seam, 1), _drive(direct, 1))  # same stream
    assert np.array_equal(_drive(seam, 1), _drive(seam, 7))  # batch invariant
    assert np.array_equal(_drive(seam, 1), _drive(seam, 12))


def test_different_seed_through_seam_differs() -> None:
    m = _matrix()

    def _seam(seed):
        return build_scenario_generator(
            SbbGeneratorConfig(
                model="sbb",
                horizon=2,
                total_scenarios=12,
                expected_block_size=2.0,
                seed=seed,
            ),
            data=m,
        )

    assert not np.array_equal(_drive(_seam(1)), _drive(_seam(2)))


def test_both_models_coexist_in_union() -> None:
    """The discriminated union routes each tag to its own type."""
    h = parse_scenario_model_config({"model": "historical", "horizon": 2})
    s = parse_scenario_model_config({
        "model": "sbb",
        "horizon": 2,
        "total_scenarios": 5,
    })
    assert isinstance(h, HistoricalGeneratorConfig)
    assert isinstance(s, SbbGeneratorConfig)
