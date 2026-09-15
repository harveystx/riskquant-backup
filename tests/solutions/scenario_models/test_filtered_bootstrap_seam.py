# tests/solutions/scenario_models/test_filtered_bootstrap_seam.py
"""
Filtered Bootstrap configuration-seam tests: the fifth and final model. Proves
the existing typed `data` union and single builder absorb another matrix-backed
stochastic model with no new abstraction — schema validation, fail-fast pairing,
direct-vs-seam equivalence, and D029 batch-size invariance through the seam.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.constants import EWMA_DECAY_FACTOR, MAX_EWMA_VOL_RATIO
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.bootstrap_fhs import FilteredBootstrapGenerator
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.schemas import FilteredBootstrapGeneratorConfig

_N = 2


def _matrix(t: int = 40) -> RiskFactorMatrix:
    vals = np.asfortranarray(
        (
            100.0 + np.cumsum(np.sin(np.arange(t * _N).reshape(t, _N) * 0.1), axis=0)
        ).astype(np.float64)
    )
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.arange(_N, dtype=np.int32),
        values=vals,
    )


def _parametric_data() -> ParametricRiskData:
    return ParametricRiskData(
        factor_ids=np.arange(_N, dtype=np.int32),
        cholesky_matrix=np.ascontiguousarray(np.eye(_N, dtype=np.float64)),
        volatilities=np.ascontiguousarray(np.full(_N, 0.1, dtype=np.float64)),
    )


def _cfg(**over):
    base = {
        "model": "filtered_bootstrap",
        "horizon": 2,
        "total_scenarios": 12,
        "decay_factor": 0.9,
        "max_vol_ratio": 8.0,
        "seed": 7,
    }
    base.update(over)
    return base


def _drive(gen, batch_size=4):
    chunks = []
    for b in gen.iter_batches(
        required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * _N, np.int8),
        source_factor_ids=np.arange(_N, dtype=np.int32),
        shock_scale=np.ones(_N, np.float32),
        batch_size=batch_size,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


# ---- schema -------------------------------------------------------------------


def test_valid_config_parses() -> None:
    cfg = parse_scenario_model_config(_cfg())
    assert isinstance(cfg, FilteredBootstrapGeneratorConfig)
    assert cfg.model == "filtered_bootstrap"
    assert cfg.total_scenarios == 12 and cfg.horizon == 2
    assert cfg.decay_factor == 0.9 and cfg.max_vol_ratio == 8.0 and cfg.seed == 7


def test_defaults_match_repository_constants() -> None:
    cfg = FilteredBootstrapGeneratorConfig.model_validate({
        "model": "filtered_bootstrap",
        "horizon": 1,
        "total_scenarios": 5,
    })
    assert cfg.decay_factor == EWMA_DECAY_FACTOR
    assert cfg.max_vol_ratio == MAX_EWMA_VOL_RATIO
    assert cfg.seed is None


@pytest.mark.parametrize("missing", ["horizon", "total_scenarios"])
def test_required_fields(missing) -> None:
    d = _cfg()
    del d[missing]
    with pytest.raises(ValidationError):
        parse_scenario_model_config(d)


@pytest.mark.parametrize("field", ["horizon", "total_scenarios"])
@pytest.mark.parametrize("bad", [True, False, 1.0, 2.0, "10", 0, -1])
def test_discrete_fields_strict_positive_int(field, bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(**{field: bad}))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, True, float("nan"), float("inf")])
def test_decay_factor_invalid(bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(decay_factor=bad))


@pytest.mark.parametrize("bad", [0.5, 0.999999, EWMA_DECAY_FACTOR])
def test_decay_factor_valid_open_interval(bad) -> None:
    cfg = parse_scenario_model_config(_cfg(decay_factor=bad))
    assert cfg.decay_factor == bad


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, True, float("nan"), float("inf")])
def test_max_vol_ratio_invalid(bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(max_vol_ratio=bad))


def test_max_vol_ratio_boundary_one_ok() -> None:
    cfg = parse_scenario_model_config(_cfg(max_vol_ratio=1.0))
    assert cfg.max_vol_ratio == 1.0


def test_seed_strict_int_or_none() -> None:
    assert parse_scenario_model_config(_cfg(seed=None)).seed is None
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(seed=1.5))
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(seed=True))


@pytest.mark.parametrize(
    "foreign",
    [
        {"expected_block_size": 2.0},  # SBB
        {"distribution": {"kind": "laplace"}},  # Parametric
        {"student_t_df": 5.0},  # Parametric
        {"vg_nu": 0.5},  # Parametric
        {"prob_crisis": 0.1},  # Parametric
        {"crisis_vol_multipliers": 2.0},  # Parametric
        {"cholesky_matrix": [[1.0]]},  # Parametric data
        {"volatilities": [0.1]},  # Parametric data
        {"factor_ids": [0, 1]},  # Parametric data
    ],
)
def test_foreign_parameters_rejected(foreign) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(**foreign))


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(model="filtered_boot"))


# ---- builder / data pairing ---------------------------------------------------


def test_filtered_bootstrap_plus_matrix_succeeds() -> None:
    cfg = parse_scenario_model_config(_cfg())
    gen = build_scenario_generator(cfg, data=_matrix())
    assert isinstance(gen, FilteredBootstrapGenerator)


def test_filtered_bootstrap_plus_parametric_data_fails() -> None:
    cfg = parse_scenario_model_config(_cfg())
    with pytest.raises(TypeError):
        build_scenario_generator(cfg, data=_parametric_data())


def test_constructor_parameters_propagate() -> None:
    cfg = parse_scenario_model_config(_cfg(horizon=3, total_scenarios=20, seed=99))
    gen = build_scenario_generator(cfg, data=_matrix())
    assert gen.total_scenarios == 20 and gen.horizon == 3
    assert gen.decay_factor == 0.9 and gen.max_vol_ratio == 8.0 and gen.seed == 99


# ---- equivalence + RNG / D029 -------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 7])
def test_seam_equivalent_to_direct_construction(batch_size) -> None:
    m = _matrix()
    cfg = parse_scenario_model_config(_cfg())
    seamed = build_scenario_generator(cfg, data=m)
    direct = FilteredBootstrapGenerator(
        matrix=m,
        total_scenarios=12,
        horizon=2,
        decay_factor=0.9,
        max_vol_ratio=8.0,
        seed=7,
    )
    rd, rs = _drive(direct, batch_size), _drive(seamed, batch_size)
    assert rd.shape == rs.shape
    assert rd.dtype == rs.dtype == np.float32
    assert rs.flags["F_CONTIGUOUS"] == rd.flags["F_CONTIGUOUS"]
    assert np.array_equal(rd, rs)


def test_seam_preserves_d029_batch_invariance() -> None:
    m = _matrix()
    gen = build_scenario_generator(parse_scenario_model_config(_cfg(seed=42)), data=m)
    full = 12
    assert np.array_equal(_drive(gen, 1), _drive(gen, 7))
    assert np.array_equal(_drive(gen, 1), _drive(gen, full))


def test_same_seed_through_seam_is_deterministic() -> None:
    m = _matrix()

    def seam():
        return build_scenario_generator(
            parse_scenario_model_config(_cfg(seed=123)), data=m
        )

    assert np.array_equal(_drive(seam()), _drive(seam()))


def test_different_seed_through_seam_differs() -> None:
    m = _matrix()

    def seam(s):
        return build_scenario_generator(
            parse_scenario_model_config(_cfg(seed=s)), data=m
        )

    assert not np.array_equal(_drive(seam(1)), _drive(seam(2)))
