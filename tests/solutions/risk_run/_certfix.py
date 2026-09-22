# tests/solutions/risk_run/_certfix.py

"""Certification fixture: production-representative, drives the OFFICIAL API."""

import numpy as np
import polars as pl
from _binding import synthetic_binding, synthetic_declaration

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    TaylorBlock,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.api import (
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorOrder

K, T, S, M = 250, 600, 2000, 200
FIDS = np.arange(1, K + 1, dtype=np.int32)


def _vals():
    r = np.random.default_rng(0)
    v = 100.0 * np.exp(np.cumsum(r.normal(0, 0.01, (T, K)), axis=0))
    v[:, K - 1] = 0.9 * np.exp(np.cumsum(r.normal(0, 0.005, T)))  # FX factor
    return np.asfortranarray(v)


VALS = _vals()


def matrix():
    return RiskFactorMatrix(
        dates=np.arange(T, dtype=np.int32), factor_ids=FIDS, values=VALS
    )


def parametric_data():
    r = np.random.default_rng(1)
    a = r.normal(0, 1, (K, K))
    return ParametricRiskData(
        factor_ids=FIDS,
        cholesky_matrix=np.ascontiguousarray(
            np.linalg.cholesky(a @ a.T / K + np.eye(K) * 0.05)
        ),
        volatilities=np.ascontiguousarray(np.full(K, 0.02)),
    )


def calculator():
    r = np.random.default_rng(2)
    pos = np.repeat(np.arange(1, M + 1), 6)
    fac = r.choice(FIDS, size=M * 6)
    df = (
        pl
        .DataFrame({
            "position_id": pos.tolist(),
            "factor_id": fac.tolist(),
            "first_order": r.normal(0, 100, M * 6).tolist(),
            "second_order": r.normal(0, 5, M * 6).tolist(),
        })
        .unique(subset=["position_id", "factor_id"])
        .sort(["position_id", "factor_id"])
    )
    meta = pl.DataFrame({
        "factor_id": FIDS.tolist(),
        "displacement_convention_id": [int(DisplacementConvention.ABSOLUTE)] * K,
        "quote_base": [1.0] * K,
    })
    binding = synthetic_binding(
        factor_ids=FIDS,
        base_levels=VALS[-1],
        position_uids=list(range(1, M + 1)),
        v0=np.zeros(M),
    )
    return SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=df,
            factor_meta=meta,
            factor_axis=FactorAxis.from_keys(FIDS.tolist()),
            position_uids=list(range(1, M + 1)),
            taylor_declaration=synthetic_declaration(
                sensitivities=df,
                order=TaylorOrder.SECOND_ORDER_TRUNCATED,
                binding=binding,
            ),
        )
    )


CFGS = {
    "historical": {"model": "historical", "horizon": 1},
    "sbb": {
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": S,
        "expected_block_size": 20.0,
        "seed": 7,
    },
    "filtered_bootstrap": {
        "model": "filtered_bootstrap",
        "horizon": 1,
        "total_scenarios": S,
        "seed": 7,
    },
    "filtered_historical": {"model": "filtered_historical", "horizon": 1},
    "parametric": {
        "model": "parametric",
        "total_scenarios": S,
        "distribution": {"kind": "student_t", "df": 5.0},
        "seed": 7,
    },
}


def inputs(model="sbb", batch_size=256, seed=7, calc=None):
    cfg = dict(CFGS[model])
    if "seed" in cfg:
        cfg["seed"] = seed
    m = matrix()
    settle = tuple(("EUR" if i % 2 else "USD") for i in range(M))
    book = calculator() if calc is None else calc
    uids = np.arange(1, M + 1, dtype=np.int64)
    return dict(
        scenario_config=parse_scenario_model_config(cfg),
        data=parametric_data() if model == "parametric" else m,
        required_factor_ids=FIDS,
        required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * K, np.int8),
        source_factor_ids=FIDS.copy(),
        shock_scale=np.ones(K, np.float32),
        base_levels=np.ascontiguousarray(VALS[-1].astype(np.float64)),
        methodology=MethodologyDeclaration(
            production_position_uids=uids,
            full_reval_uids=[],
            taylor_supplied_uids=[],
            taylor_derived_uids=uids,
        ),
        # D081: synthetic Taylor book with an explicitly DECLARED zero base
        # value, i.e. the V0 = 0 special case in which dW = dV*q_s exactly.
        taylor_derived=TaylorBlock(
            calculator=book,
            base_settlement_values=np.zeros(M, dtype=np.float64),
            expected_binding=book.taylor_declaration.base_binding,
        ),
        reporting=ReportingSpec(
            reporting_ccy="USD",
            position_uids=uids,
            settlement_ccys=settle,
            fx_factor_names=("USDEUR",),
            fx_factor_rows=np.array([K - 1], np.int32),
        ),
        confidence_level=0.99,
        batch_size=batch_size,
    )