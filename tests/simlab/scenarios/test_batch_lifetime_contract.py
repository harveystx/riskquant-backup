# tests/simlab/scenarios/test_batch_lifetime_contract.py
"""
`ScenarioSetBatch` lifetime (simlab/scenarios/base.py): a yielded batch
exposes generator-owned storage that is valid until the next batch is
produced. For every generator family: values copied at yield time reproduce
the single-batch whole-run stream exactly (the guarantee), and the stream
reuses its storage (the reason a consumer must copy).
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)

_IDS = np.array([1, 2], np.int32)
_MATRIX = RiskFactorMatrix(
    dates=np.arange(40, dtype=np.int32), factor_ids=_IDS,
    values=np.asfortranarray(100.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, (40, 2)), axis=0))),
)
_PARAMETRIC = ParametricRiskData(
    factor_ids=_IDS, cholesky_matrix=np.ascontiguousarray(np.eye(2)), volatilities=np.array([0.01, 0.02]),
    effective_period_count=1, effective_period_unit=np.int8(1), chart_ids=np.array([1, 1], np.int8),
)
_FAMILIES = {
    "historical": ({"model": "historical", "horizon": 1}, _MATRIX),
    "sbb": ({"model": "sbb", "horizon": 1, "expected_block_size": 3.0, "total_scenarios": 30, "seed": 1}, _MATRIX),
    "filtered_historical": ({"model": "filtered_historical", "horizon": 1}, _MATRIX),
    "filtered_bootstrap": ({"model": "filtered_bootstrap", "horizon": 1, "total_scenarios": 30, "seed": 1}, _MATRIX),
    "parametric": ({"model": "parametric", "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
                    "total_scenarios": 30, "seed": 1, "distribution": {"kind": "standard_normal"}}, _PARAMETRIC),
}


def _batches(family: str, batch_size: int):
    cfg, data = _FAMILIES[family]
    gen = build_scenario_generator(parse_scenario_model_config(cfg), data=data)
    return gen.iter_batches(driver_rf_ids=_IDS, chart_ids=np.array([1, 1], np.int8), batch_size=batch_size)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_values_copied_at_yield_time_are_the_whole_run_stream(family: str) -> None:
    whole = np.vstack([np.array(b.shocks, copy=True) for b in _batches(family, 10_000)])
    copied = np.vstack([np.array(b.shocks, copy=True) for b in _batches(family, 4)])
    assert copied.tobytes() == whole.tobytes()


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_a_stream_reuses_generator_owned_storage(family: str) -> None:
    kept = list(_batches(family, 4))
    assert len(kept) > 2
    assert any(np.shares_memory(kept[0].shocks, b.shocks) for b in kept[1:])
