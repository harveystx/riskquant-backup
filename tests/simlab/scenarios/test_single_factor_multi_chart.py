# tests/simlab/scenarios/test_single_factor_multi_chart.py
"""
D088 Slice 3, C0: a one-factor matrix world supplies several charts of that
factor, ``(B, LOG, h)``, ``(B, SIMPLE, h)``, ``(B, DIFF, h)``, in every matrix
engine. Before the added layout overloads this raised a Numba dispatch
TypeError (values (T, 1) resolve to 'C', out (B, K>=2) is 'F').

Oracles: the same factor's columns computed in a two-factor matrix (the
already-supported layout), bitwise; and the chart definitions for the
historical engine, where ``Y = T_tau(v[t - h], v[t])`` independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)

_B, _OTHER = 7, 8
_LOG, _SIMPLE, _DIFF = 1, 2, 3
_H = 3
_T = 45

_CONFIGS = {
    "historical": {"model": "historical", "horizon": _H},
    "sbb": {
        "model": "sbb",
        "horizon": _H,
        "expected_block_size": 4.0,
        "total_scenarios": 25,
        "seed": 11,
    },
    "filtered_historical": {"model": "filtered_historical", "horizon": _H},
    "filtered_bootstrap": {
        "model": "filtered_bootstrap",
        "horizon": _H,
        "total_scenarios": 25,
        "seed": 11,
    },
}


def _levels() -> np.ndarray:
    rng = np.random.default_rng(3)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (_T, 2)), axis=0))


def _matrix(two_factor: bool) -> RiskFactorMatrix:
    lv = _levels()
    if two_factor:
        return RiskFactorMatrix(
            dates=np.arange(_T, dtype=np.int32),
            factor_ids=np.array([_B, _OTHER], dtype=np.int32),
            values=np.asfortranarray(lv),
        )
    return RiskFactorMatrix(
        dates=np.arange(_T, dtype=np.int32),
        factor_ids=np.array([_B], dtype=np.int32),
        values=np.asfortranarray(lv[:, :1]),
    )


def _columns(model: str, two_factor: bool, batch_size: int) -> np.ndarray:
    gen = build_scenario_generator(
        parse_scenario_model_config(_CONFIGS[model]), data=_matrix(two_factor)
    )
    charts = np.array([_LOG, _SIMPLE, _DIFF], dtype=np.int8)
    return np.vstack([
        np.array(b.shocks, copy=True)
        for b in gen.iter_batches(
            required_shift_type_id=charts,
            source_factor_ids=np.full(3, _B, dtype=np.int32),
            shock_scale=np.ones(3, dtype=np.float32),
            batch_size=batch_size,
        )
    ])


@pytest.mark.parametrize("model", sorted(_CONFIGS))
def test_one_factor_matrix_supplies_three_charts_of_its_factor(model: str) -> None:
    single = _columns(model, two_factor=False, batch_size=8)
    assert single.shape[1] == 3
    # The three charts are three different random variables.
    assert not np.array_equal(single[:, 0], single[:, 1])
    assert not np.array_equal(single[:, 1], single[:, 2])


@pytest.mark.parametrize("model", sorted(_CONFIGS))
@pytest.mark.parametrize("batch_size", [1, 8, 1000])
def test_one_factor_layout_equals_the_multi_factor_layout_bitwise(
    model: str, batch_size: int
) -> None:
    single = _columns(model, two_factor=False, batch_size=batch_size)
    multi = _columns(model, two_factor=True, batch_size=batch_size)
    assert single.tobytes() == multi.tobytes()


def test_historical_columns_equal_the_chart_definitions() -> None:
    v = _levels()[:, 0]
    start, end = v[:-_H], v[_H:]
    expected = np.stack(
        [np.log(end / start), end / start - 1.0, end - start], axis=1
    ).astype(np.float32)
    got = _columns("historical", two_factor=False, batch_size=8)
    np.testing.assert_allclose(got, expected, rtol=2e-7, atol=1e-9)
