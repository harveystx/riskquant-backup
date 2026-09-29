# tests/solutions/scenario_models/test_seam_completeness.py
"""
System-level completeness audit for the scenario-model seam (D036). Proves, as
one matrix over all five models, the properties the per-model seam tests each
assert only for their own model:

  * every ``model`` tag discriminates to its config type;
  * the full config/data pairing matrix fails fast (every matrix-backed config
    rejects ``ParametricRiskData``; the parametric config rejects a
    ``RiskFactorMatrix``);
  * all five build through the single seam and yield the one downstream type
    (``ScenarioSetBatch``) — uniform ``ScenarioGenerator`` protocol, no
    model-specific leakage.

This does not duplicate the per-model equivalence / D029 / numerical tests; it
asserts the cross-model invariants of the seam itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.base import ScenarioGenerator, ScenarioSetBatch
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.schemas import (
    FilteredBootstrapGeneratorConfig,
    FilteredHistoricalGeneratorConfig,
    HistoricalGeneratorConfig,
    ParametricGeneratorConfig,
    SbbGeneratorConfig,
)

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
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(_N, SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )


# (raw config dict, expected config type, is_matrix_backed)
_MODELS = [
    ({"model": "historical", "horizon": 2}, HistoricalGeneratorConfig, True),
    (
        {"model": "sbb", "horizon": 2, "total_scenarios": 6, "seed": 1},
        SbbGeneratorConfig,
        True,
    ),
    (
        {"model": "filtered_historical", "horizon": 2},
        FilteredHistoricalGeneratorConfig,
        True,
    ),
    (
        {"model": "filtered_bootstrap", "horizon": 2, "total_scenarios": 6},
        FilteredBootstrapGeneratorConfig,
        True,
    ),
    (
        {
            "model": "parametric",
            "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
            "total_scenarios": 6,
            "distribution": {"kind": "standard_normal"},
        },
        ParametricGeneratorConfig,
        False,
    ),
]

_IDS = ["historical", "sbb", "filtered_historical", "filtered_bootstrap", "parametric"]


@pytest.mark.parametrize("raw,cfg_type,_matrix_backed", _MODELS, ids=_IDS)
def test_every_tag_discriminates(raw, cfg_type, _matrix_backed) -> None:
    assert isinstance(parse_scenario_model_config(raw), cfg_type)


def test_five_models_and_no_more() -> None:
    """Guards the union arity: exactly the five audited tags parse."""
    tags = {
        "historical",
        "sbb",
        "filtered_historical",
        "filtered_bootstrap",
        "parametric",
    }
    assert {raw["model"] for raw, _, _ in _MODELS} == tags
    with pytest.raises(ValidationError):
        parse_scenario_model_config({"model": "does_not_exist"})


@pytest.mark.parametrize("raw,cfg_type,matrix_backed", _MODELS, ids=_IDS)
def test_correct_pairing_builds(raw, cfg_type, matrix_backed) -> None:
    cfg = parse_scenario_model_config(raw)
    data = _matrix() if matrix_backed else _parametric_data()
    gen = build_scenario_generator(cfg, data=data)
    # ScenarioGenerator is a (non-runtime_checkable) Protocol the generators
    # inherit nominally; assert conformance via the MRO + the protocol method.
    assert ScenarioGenerator in type(gen).__mro__
    assert callable(getattr(gen, "iter_batches", None))


@pytest.mark.parametrize("raw,cfg_type,matrix_backed", _MODELS, ids=_IDS)
def test_wrong_pairing_fails_fast(raw, cfg_type, matrix_backed) -> None:
    """Full pairing matrix: matrix-backed configs reject ParametricRiskData;
    the parametric config rejects a RiskFactorMatrix."""
    cfg = parse_scenario_model_config(raw)
    wrong = _parametric_data() if matrix_backed else _matrix()
    with pytest.raises(TypeError):
        build_scenario_generator(cfg, data=wrong)


@pytest.mark.parametrize("raw,cfg_type,matrix_backed", _MODELS, ids=_IDS)
def test_all_yield_same_downstream_type(raw, cfg_type, matrix_backed) -> None:
    """Every model yields the single ScenarioSetBatch type — no model-specific
    downstream leakage."""
    cfg = parse_scenario_model_config(raw)
    data = _matrix() if matrix_backed else _parametric_data()
    gen = build_scenario_generator(cfg, data=data)
    first = next(
        gen.iter_batches(
            required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * _N, np.int8),
            source_factor_ids=np.arange(_N, dtype=np.int32),
            shock_scale=np.ones(_N, np.float32),
            batch_size=3,
        )
    )
    assert isinstance(first, ScenarioSetBatch)
    assert first.shocks.dtype == np.float32
