# tests/solutions/scenario_models/test_scenario_model_seam.py
"""
Tests for the scenario-model configuration seam: typed selection + construction
of an existing ScenarioGenerator, with the generator protocol unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig


def _matrix(k=2, t=14):
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


def _drive(gen):
    chunks = []
    for b in gen.iter_batches(
        chart_ids=np.array(
            [SHIFT_LOG_RETURN_ID, SHIFT_LOG_RETURN_ID], np.int8
        ),
        driver_rf_ids=np.array([0, 1], np.int32),
        batch_size=4,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


# ---- selection ----------------------------------------------------------------


def test_valid_model_selection() -> None:
    cfg = parse_scenario_model_config({"model": "historical", "horizon": 3})
    assert isinstance(cfg, HistoricalGeneratorConfig)
    assert cfg.horizon == 3
    assert type(cfg.horizon) is int


@pytest.mark.parametrize("bad_model", ["gbm", "sbb", "parametric", "", "Historical"])
def test_unknown_model_rejected(bad_model) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": bad_model, "horizon": 3})


def test_required_model_specific_parameter_missing() -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": "historical"})  # horizon required


@pytest.mark.parametrize(
    "foreign",
    [
        {"total_scenarios": 10},  # SBB / bootstrap / parametric
        {"expected_block_size": 2.0},  # SBB
        {"decay_factor": 0.9},  # FHS / filtered-bootstrap
        {"distribution_kind": "student_t"},  # parametric
        {"seed": 7},  # stochastic-only; historical is deterministic
    ],
)
def test_foreign_model_parameter_rejected(foreign) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": "historical", "horizon": 3, **foreign})


@pytest.mark.parametrize("bad_horizon", [1.5, 1.0, True, "3"])
def test_horizon_must_be_strict_int(bad_horizon) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": "historical", "horizon": bad_horizon})


@pytest.mark.parametrize("bad_horizon", [0, -1])
def test_horizon_must_be_positive(bad_horizon) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": "historical", "horizon": bad_horizon})


def test_seed_is_not_a_historical_parameter() -> None:
    """Historical is deterministic: the seam must not accept or inject a seed
    for it. (Seed propagation for stochastic models follows the same
    config->constructor passthrough as horizon, verified below.)"""
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": "historical", "horizon": 3, "seed": 1})


# ---- construction + parameter propagation -------------------------------------


def test_build_returns_historical_generator() -> None:
    m = _matrix()
    cfg = HistoricalGeneratorConfig(model="historical", horizon=2)
    gen = build_scenario_generator(cfg, data=m)
    assert isinstance(gen, HistoricalGenerator)
    assert gen.horizon == 2  # config parameter propagated to the constructor
    assert type(gen.horizon) is int


def test_build_runs_generator_own_validation() -> None:
    """The seam does not bypass the generator's constructor validation: a
    horizon that leaves no history still fails at construction."""
    m = _matrix(t=4)
    cfg = HistoricalGeneratorConfig(model="historical", horizon=10)
    with pytest.raises(ValueError):
        build_scenario_generator(cfg, data=m)


# ---- equivalence: direct vs seam ----------------------------------------------


@pytest.mark.parametrize("horizon", [1, 3])
def test_seam_equivalent_to_direct_construction(horizon) -> None:
    """Scenarios built through the seam are identical to direct construction in
    value, ordering, dtype, layout, and count."""
    m = _matrix()
    direct = HistoricalGenerator(matrix=m, horizon=horizon)
    seamed = build_scenario_generator(
        HistoricalGeneratorConfig(model="historical", horizon=horizon), data=m
    )
    rd = _drive(direct)
    rs = _drive(seamed)
    assert rd.shape == rs.shape  # count
    assert rd.dtype == rs.dtype == np.float32  # dtype
    assert rs.flags["F_CONTIGUOUS"] == rd.flags["F_CONTIGUOUS"]  # layout
    assert np.array_equal(rd, rs)  # values + ordering (exact)
    assert direct.total_scenarios == seamed.total_scenarios
