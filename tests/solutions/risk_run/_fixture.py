# tests/solutions/risk_run/_fixture.py
"""Shared production-run fixture: a real compiled book with an EUR-settled
position converted through a USDEUR factor, on the official entry point."""

import numpy as np
import polars as pl
from _binding import synthetic_binding, synthetic_declaration

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.solutions.risk_run.schemas import ReportingSpec
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.api import (
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorOrder

K, T, S = 3, 80, 64
FACTOR_IDS = np.array([10, 20, 30], dtype=np.int32)  # 30 = USDEUR fx factor


def matrix():
    rng = np.random.default_rng(0)
    vals = np.asfortranarray(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, K)), axis=0))
    )
    vals[:, 2] = 0.9 * np.exp(np.cumsum(rng.normal(0, 0.005, T)))  # fx level ~0.9
    return RiskFactorMatrix(
        dates=np.arange(T, dtype=np.int32),
        factor_ids=FACTOR_IDS,
        values=np.asfortranarray(vals),
    )


def calculator(base_levels=None):
    abs_id = int(DisplacementConvention.ABSOLUTE)
    sens = pl.DataFrame({
        "position_id": [1, 1, 2],
        "factor_id": [10, 30, 20],
        "first_order": [1000.0, -4115.0, -500.0],
        "second_order": [0.0, 0.0, 0.0],
    })
    meta = pl.DataFrame({
        "factor_id": FACTOR_IDS.tolist(),
        "displacement_convention_id": [abs_id] * K,
        "quote_base": [1.0] * K,
    })
    binding = synthetic_binding(
        factor_ids=FACTOR_IDS,
        base_levels=np.asarray(matrix().values)[-1]
        if base_levels is None
        else base_levels,
        position_uids=[1, 2],
        v0=np.zeros(2),
    )
    return SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=sens,
            factor_meta=meta,
            factor_axis=FactorAxis.from_keys(FACTOR_IDS.tolist()),
            position_uids=[1, 2],
            taylor_declaration=synthetic_declaration(
                sensitivities=sens, order=TaylorOrder.FIRST_ORDER, binding=binding
            ),
        )
    )


def inputs(seed=7, model="sbb"):
    m = matrix()
    cfg = {
        "sbb": {
            "model": "sbb",
            "horizon": 1,
            "total_scenarios": S,
            "expected_block_size": 5.0,
            "seed": seed,
        },
        "historical": {"model": "historical", "horizon": 1},
        "filtered_bootstrap": {
            "model": "filtered_bootstrap",
            "horizon": 1,
            "total_scenarios": S,
            "seed": seed,
        },
    }[model]
    return dict(
        scenario_config=parse_scenario_model_config(cfg),
        data=m,
        required_factor_ids=FACTOR_IDS,
        required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * K, np.int8),
        source_factor_ids=FACTOR_IDS.copy(),
        shock_scale=np.ones(K, np.float32),
        base_levels=np.ascontiguousarray(np.asarray(m.values)[-1].astype(np.float64)),
        calculator=calculator(),
        reporting=ReportingSpec(
            reporting_ccy="USD",
            settlement_ccys=("EUR", "USD"),
            # D081: synthetic Taylor book with an explicitly DECLARED zero base
            # value, i.e. the V0 = 0 special case in which dW = dV*q_s exactly.
            base_settlement_values=np.zeros(2, dtype=np.float64),
            base_binding=calculator().taylor_declaration.base_binding,
            fx_factor_names=("USDEUR",),
            fx_factor_rows=np.array([2], np.int32),
        ),
        confidence_level=0.99,
        batch_size=16,
    )
