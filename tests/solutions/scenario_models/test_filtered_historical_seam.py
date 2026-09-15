# tests/solutions/scenario_models/test_filtered_historical_seam.py
"""
FilteredHistorical (FHS) configuration-seam tests: the third real model through
the solution-tier seam. Proves the discriminated union handles materially
different model-specific parameters (decay_factor / max_vol_ratio) while keeping
the same RiskFactorMatrix construction-data contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.constants import EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.schemas import (
    FilteredHistoricalGeneratorConfig,
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


def _drive(gen):
    chunks = []
    for b in gen.iter_batches(
        required_shift_type_id=np.array(
            [SHIFT_LOG_RETURN_ID, SHIFT_LOG_RETURN_ID], np.int8
        ),
        source_factor_ids=np.array([0, 1], np.int32),
        shock_scale=np.ones(2, np.float32),
        batch_size=4,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


def _cfg(**over):
    base = {"model": "filtered_historical", "horizon": 2}
    base.update(over)
    return base


# ---- selection & validation ---------------------------------------------------


def test_valid_config_with_explicit_params() -> None:
    cfg = parse_scenario_model_config(_cfg(decay_factor=0.9, max_vol_ratio=8.0))
    assert isinstance(cfg, FilteredHistoricalGeneratorConfig)
    assert cfg.horizon == 2 and cfg.decay_factor == 0.9 and cfg.max_vol_ratio == 8.0
    assert type(cfg.horizon) is int


def test_defaults_match_repository_constants() -> None:
    cfg = parse_scenario_model_config(_cfg())
    assert cfg.decay_factor == EWMA_DECAY_FACTOR
    assert cfg.max_vol_ratio == MAX_EWMA_VOL_RATIO


@pytest.mark.parametrize("bad", [1.5, 2.0, True, "2"])
def test_horizon_strict_int(bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(horizon=bad))


@pytest.mark.parametrize("bad", [0, -1])
def test_horizon_positive(bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(horizon=bad))


@pytest.mark.parametrize("good", [0.5, 0.94, 0.999])
def test_decay_factor_open_unit_interval_accepted(good) -> None:
    cfg = parse_scenario_model_config(_cfg(decay_factor=good))
    assert cfg.decay_factor == good


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1, True, float("inf"), float("nan")])
def test_decay_factor_invalid_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(decay_factor=bad))


@pytest.mark.parametrize("good", [1.0, 5.0, 10, 100.0])  # int and float accepted
def test_max_vol_ratio_ge_one_accepted(good) -> None:
    cfg = parse_scenario_model_config(_cfg(max_vol_ratio=good))
    assert cfg.max_vol_ratio == float(good)
    assert type(cfg.max_vol_ratio) is float


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, True, float("inf"), float("nan")])
def test_max_vol_ratio_invalid_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(max_vol_ratio=bad))


@pytest.mark.parametrize(
    "foreign",
    [
        {"seed": 7},  # stochastic-only
        {"total_scenarios": 10},  # SBB / bootstrap
        {"expected_block_size": 2.0},  # SBB
        {"distribution_kind": "student_t"},  # Parametric
        {"vg_nu": 0.5},  # Parametric
    ],
)
def test_foreign_model_parameter_rejected(foreign) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(**foreign))


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(model="fhs"))  # wrong tag string


# ---- construction, equivalence, determinism -----------------------------------


def test_build_returns_filtered_historical_generator() -> None:
    m = _matrix()
    cfg = FilteredHistoricalGeneratorConfig(
        model="filtered_historical", horizon=2, decay_factor=0.9, max_vol_ratio=8.0
    )
    gen = build_scenario_generator(cfg, data=m)
    assert isinstance(gen, FilteredHistoricalGenerator)
    assert gen.horizon == 2 and gen.decay_factor == 0.9 and gen.max_vol_ratio == 8.0
    assert type(gen.horizon) is int


def test_build_runs_generator_own_validation() -> None:
    """The seam does not bypass runtime invariants: too-short history fails at
    construction (FHS needs T >= 3)."""
    m = _matrix(t=2)
    cfg = FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=1)
    with pytest.raises(ValueError):
        build_scenario_generator(cfg, data=m)


@pytest.mark.parametrize("horizon", [1, 3])
def test_seam_equivalent_to_direct_construction(horizon) -> None:
    """count, ordering, values, dtype, layout identical to direct."""
    m = _matrix()
    direct = FilteredHistoricalGenerator(
        matrix=m, horizon=horizon, decay_factor=0.9, max_vol_ratio=8.0
    )
    seamed = build_scenario_generator(
        FilteredHistoricalGeneratorConfig(
            model="filtered_historical",
            horizon=horizon,
            decay_factor=0.9,
            max_vol_ratio=8.0,
        ),
        data=m,
    )
    rd, rs = _drive(direct), _drive(seamed)
    assert rd.shape == rs.shape
    assert rd.dtype == rs.dtype == np.float32
    assert rs.flags["F_CONTIGUOUS"] == rd.flags["F_CONTIGUOUS"]
    assert np.array_equal(rd, rs)
    assert direct.total_scenarios == seamed.total_scenarios


def test_defaults_equivalent_to_direct_defaults() -> None:
    """Config defaults reproduce the generator's own default parameters."""
    m = _matrix()
    seamed = build_scenario_generator(
        FilteredHistoricalGeneratorConfig(model="filtered_historical", horizon=2),
        data=m,
    )
    direct = FilteredHistoricalGenerator(matrix=m, horizon=2)
    assert np.array_equal(_drive(seamed), _drive(direct))


def test_deterministic_repeatability_through_seam() -> None:
    """FHS is deterministic: the same config builds a generator whose stream is
    identical run to run (no seed involved)."""
    m = _matrix()

    def _seam():
        return build_scenario_generator(
            FilteredHistoricalGeneratorConfig(
                model="filtered_historical", horizon=2, decay_factor=0.9
            ),
            data=m,
        )

    assert np.array_equal(_drive(_seam()), _drive(_seam()))
