# tests/simlab/scenarios/test_shock_permutation_contract.py
"""
SimLab P0: emitted shock column k belongs to shock-universe factor k, for ANY
source order.

`ShockComputePlan.inverse_permutation` is a GATHER map (shock-universe uix k ->
sorted compute position); every consumer builds output column k from sorted
position `inverse_permutation[k]`. Before the fix, four kernels used it as a
SCATTER map and the parametric generator ignored it, so a 3-cycle source order
assigned each shock to another factor. Identity and involutive orders hid it.

The oracles never call `compute_plan`, `inverse_permutation` or a kernel:
  - historical, sbb: DIFFERENCE shocks of linear prices are exact per-factor
    constants (12 -> 10, 4002 -> 20, 9007 -> 30);
  - filtered historical/bootstrap, parametric: equality, by factor id, with the
    IDENTITY source order (universe order == sorted order, so no permutation is
    applied), plus identifiability of the reference columns.
"""

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_DIFFERENCE_ID, SHIFT_LOG_RETURN_ID
from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.historical import _kernel_fused_h_day_shock
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.simlab.shocks.apply import apply_shocks_to_base_levels
from riskquant.solutions.scenario_models.api import build_scenario_generator, parse_scenario_model_config

IDS = np.array([12, 4002, 9007], np.int32)           # RiskFactorMatrix order: sorted ascending
STEP = {12: 10.0, 4002: 20.0, 9007: 30.0}             # exact DIFFERENCE shock per factor
GROW = {12: 1.1, 4002: 1.2, 9007: 1.3}
VOL = {12: 0.01, 4002: 0.02, 9007: 0.03}
ORDERS = {"identity": (12, 4002, 9007), "involution": (4002, 12, 9007), "three_cycle": (9007, 12, 4002)}
T = 60


def _matrix(kind):
    t = np.arange(T, dtype=np.float64)[:, None]
    cols = [100.0 * (i + 1) + STEP[f] * t if kind == "diff" else 100.0 * np.power(GROW[f], t / 10.0)
            for i, f in enumerate(IDS.tolist())]
    return RiskFactorMatrix(dates=np.arange(T, dtype=np.int32), factor_ids=IDS,
                            values=np.asfortranarray(np.hstack(cols)))


def _plan(order, shift=SHIFT_DIFFERENCE_ID):
    return build_shock_universe(required_shift_type_id=np.full(len(order), shift, np.int8),
                                source_factor_ids=np.array(order, np.int32),
                                available_factor_ids=IDS).compute_plan


def test_the_three_cycle_is_not_an_involution() -> None:
    inv = _plan(ORDERS["three_cycle"]).inverse_permutation
    assert not np.array_equal(inv, np.argsort(inv))                      # perm != perm^-1


@pytest.mark.parametrize("name", list(ORDERS))
def test_inverse_permutation_is_the_universe_to_sorted_gather_map(name) -> None:
    order = ORDERS[name]
    plan = _plan(order)
    matrix_col_of = [IDS.tolist().index(f) for f in order]               # by identity, not by plan
    assert plan.cols_sorted[plan.inverse_permutation].tolist() == matrix_col_of


@pytest.mark.parametrize("name", list(ORDERS))
def test_historical_kernel_gathers_by_factor_identity(name) -> None:
    order, m = ORDERS[name], _matrix("diff")
    plan = _plan(order)
    out = np.empty((T - 1, 3), dtype=np.float32, order="F")
    _kernel_fused_h_day_shock(m.values, plan.cols_sorted, plan.shift_types_sorted,
                              plan.inverse_permutation, np.int64(1), np.int64(1), np.int64(T - 1), out)
    for k, f in enumerate(order):
        assert np.all(out[:, k] == np.float32(STEP[f])), (name, k, f)


_MODELS = {
    "historical": ({"model": "historical", "horizon": 1}, "diff", SHIFT_DIFFERENCE_ID),
    "sbb": ({"model": "sbb", "horizon": 1, "total_scenarios": 40, "expected_block_size": 5.0, "seed": 7},
            "diff", SHIFT_DIFFERENCE_ID),
    "filtered_historical": ({"model": "filtered_historical", "horizon": 1}, "log", SHIFT_LOG_RETURN_ID),
    "filtered_bootstrap": ({"model": "filtered_bootstrap", "horizon": 1, "total_scenarios": 40, "seed": 7},
                           "log", SHIFT_LOG_RETURN_ID),
    "parametric": ({"model": "parametric", "total_scenarios": 4000,
                    "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
                    "distribution": {"kind": "student_t", "df": 5.0}, "seed": 7}, "param", SHIFT_LOG_RETURN_ID),
}


def _data(kind):
    if kind == "param":
        return ParametricRiskData(factor_ids=IDS, cholesky_matrix=np.ascontiguousarray(np.eye(3)),
                                  volatilities=np.ascontiguousarray([VOL[f] for f in IDS.tolist()]),
                                  effective_period_count=1,
                                  effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID, chart_ids=np.full(IDS.shape[0], SHIFT_LOG_RETURN_ID, dtype=np.int8))
    return _matrix(kind)


def _shocks(model, order, batch_size):
    cfg, kind, shift = _MODELS[model]
    gen = build_scenario_generator(parse_scenario_model_config(cfg), data=_data(kind))
    parts = [b.shocks[:, b.required_to_shock_uix].astype(np.float64) for b in gen.iter_batches(
        required_shift_type_id=np.full(3, shift, np.int8), source_factor_ids=np.array(order, np.int32),
        shock_scale=np.ones(3, np.float32), batch_size=batch_size)]
    return np.vstack(parts)                     # column k = required factor order[k]


@pytest.mark.parametrize("batch_size", [1, 7, 64])
@pytest.mark.parametrize("name", list(ORDERS))
@pytest.mark.parametrize("model", list(_MODELS))
def test_generators_emit_each_shock_under_its_own_factor(model, name, batch_size) -> None:
    order = ORDERS[name]
    got = _shocks(model, order, batch_size)
    if _MODELS[model][1] == "diff":
        for k, f in enumerate(order):
            assert np.all(got[:, k] == STEP[f]), (model, name, k, f)
        return
    ref = _shocks(model, ORDERS["identity"], batch_size)                 # no permutation applied
    by_id = {f: ref[:, i] for i, f in enumerate(ORDERS["identity"])}
    assert all(not np.array_equal(by_id[a], by_id[b]) for a in IDS.tolist() for b in IDS.tolist() if a < b)
    for k, f in enumerate(order):
        assert np.array_equal(got[:, k], by_id[f]), (model, name, k, f)
    if model == "parametric":
        stds = [float(np.std(got[:, k])) for k in range(3)]
        assert np.argsort(stds).tolist() == np.argsort([VOL[f] for f in order]).tolist()


def test_shocked_levels_follow_factor_identity_through_proxies() -> None:
    """Required [101, 9007, 12, 4002]; 101 is proxied from 4002 at scale 0.5."""
    required = np.array([101, 9007, 12, 4002], np.int32)
    source = np.array([4002, 9007, 12, 4002], np.int32)
    scale = np.array([0.5, 1.0, 1.0, 1.0], np.float32)
    base = np.array([1000.0, 3000.0, 1000.0, 2000.0])
    gen = build_scenario_generator(parse_scenario_model_config({"model": "historical", "horizon": 1}),
                                   data=_matrix("diff"))
    for batch in gen.iter_batches(required_shift_type_id=np.full(4, SHIFT_DIFFERENCE_ID, np.int8),
                                  source_factor_ids=source, shock_scale=scale, batch_size=16):
        out = np.empty((4, batch.shocks.shape[0]), dtype=np.float64)
        apply_shocks_to_base_levels(base_levels=base, batch=batch, out=out)
        for row, (rid, src, s) in enumerate(zip(required.tolist(), source.tolist(), scale.tolist())):
            assert np.all(out[row] == base[row] + s * STEP[src]), (rid, src)
