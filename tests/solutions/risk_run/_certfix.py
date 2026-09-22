# tests/solutions/risk_run/_certfix.py
"""Certification fixture: production-representative, drives the OFFICIAL API.

P1-B: a real valuation world, not a synthetic Taylor book. A static
configuration with 249 equity nodes and one FX node (USDEUR, id 250) is compiled
from the demo sheets; 200 SPOT positions (alternately settled in EUR and USD)
are ingested and compiled; the run axis is the universe map the run builds; the
TAYLOR/DERIVED book is certified at the run base on the producer path, from the
same context the run values. `FIDS` / `K` are the run axis and its length."""

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import _mixedfix as mf

from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.utils.business_days import ord_to_date
from riskquant.valuation.engine import run_valuation

FX = mf.FX
NONE = FX.DiscountingMode.NONE
# Dimensions: the release gate (scripts/release/simlab) may resize the world.
K_TOTAL = int(os.environ.get("RQ_CERT_FACTORS", 250))
T = int(os.environ.get("RQ_CERT_OBSERVATIONS", 600))
S = int(os.environ.get("RQ_CERT_SCENARIOS", 2000))
M = int(os.environ.get("RQ_CERT_POSITIONS", 200))
ALL_IDS = np.arange(1, K_TOTAL + 1, dtype=np.int32)  # 250 = USDEUR
NAMES = [f"EQ{i:04d}" for i in range(1, K_TOTAL)] + ["USDEUR"]
AS_OF = int(FX.date_to_ord32(FX._SNAPSHOT_AS_OF_DATE))


def _static_config():
    sheets = dict(FX._RISK_STATIC_CONFIG_DICT_OF_DFS)
    key = next(k for k, v in sheets.items() if v is FX._NODE_META_DF)
    eq, n = K_TOTAL - 1, K_TOTAL
    sheets[key] = pl.DataFrame({
        "node_id": ALL_IDS.tolist(),
        "node_name": NAMES,
        "quote_ccy": ["USD"] * eq + ["EUR"],
        "asset_class": ["EQUITY"] * eq + ["FX"],
        "asset_subclass": ["EQ_SINGLE_NAME"] * eq + ["FX_G10"],
        "quote_type": ["PRICE"] * eq + ["FX_RATE"],
        "unit_scale": ["DECIMAL"] * n,
        "shift_type": ["LOG_RETURN"] * n,
        "fx_base_ccy": [None] * eq + ["USD"],
    }).cast(FX._NODE_META_DF.schema)
    return FX.compile_risk_static_config_from_dict_of_dfs(sheets)


def _vals():
    r = np.random.default_rng(0)
    v = 100.0 * np.exp(np.cumsum(r.normal(0, 0.01, (T, K_TOTAL)), axis=0))
    v[:, K_TOTAL - 1] = 0.9 * np.exp(np.cumsum(r.normal(0, 0.005, T)))  # FX factor
    return np.asfortranarray(v)


VALS = _vals()
RSC = _static_config()
_r = np.random.default_rng(2)
POSITIONS = pd.DataFrame({
    "uid": list(range(1, M + 1)),
    "instrument": ["SPOT"] * M,
    "quantity": _r.normal(0, 100, M).round(2).tolist(),
    "underlying": [NAMES[i] for i in _r.integers(0, K_TOTAL - 1, M)],
    "settlement_ccy": [("EUR" if i % 2 else "USD") for i in range(M)],
})
UIDS = np.arange(1, M + 1, dtype=np.int64)
_BATCH = FX.compile_raw_df_into_ingest_batch(raw_df=POSITIONS.copy(), static_config=RSC)
MAPPER = FX.build_uix_mapper([_BATCH], as_of_ord32=AS_OF)
PORTFOLIO = FX.compile_portfolio_from_ingest_batch(
    ingest_batch=_BATCH, uix_mapper=MAPPER
)
UNIVERSE_MAP = build_universe_mapping(
    static_config=RSC, uix_mapper=MAPPER, discounting_mode=NONE
)
FIDS = np.ascontiguousarray(
    UNIVERSE_MAP.required_factor_ids, dtype=np.int32
)  # the run axis
K = int(FIDS.shape[0])
BASE = np.ascontiguousarray(VALS[-1][FIDS - 1].astype(np.float64))  # ids are 1-based


def matrix():
    return RiskFactorMatrix(
        dates=np.arange(T, dtype=np.int32), factor_ids=ALL_IDS, values=VALS
    )


def parametric_data():
    r = np.random.default_rng(1)
    a = r.normal(0, 1, (K_TOTAL, K_TOTAL))
    return ParametricRiskData(
        factor_ids=ALL_IDS,
        cholesky_matrix=np.ascontiguousarray(
            np.linalg.cholesky(a @ a.T / K_TOTAL + np.eye(K_TOTAL) * 0.05)
        ),
        volatilities=np.ascontiguousarray(np.full(K_TOTAL, 0.02)),
    )


def v0(base=None):
    """Producer-side base values of U at a base (default: the run base)."""
    return _price(BASE if base is None else np.asarray(base, dtype=np.float64))[1]


def _price(base):
    market = build_market_data_matrix_from_flat_levels(
        as_of_date=ord_to_date(AS_OF),
        uix_mapper=MAPPER,
        universe_map=UNIVERSE_MAP,
        flat_levels=np.ascontiguousarray(base[:, None]),
        num_scenarios=1,
        discounting_mode=NONE,
    )
    rows = np.empty((PORTFOLIO.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=PORTFOLIO,
        out=rows,
        valuation_state=assemble_valuation_state(matrix=market, uix_mapper=MAPPER),
    )
    pv = dict(zip(PORTFOLIO.trade_uids.tolist(), rows[:, 0].tolist(), strict=True))
    return market, np.array([pv[int(u)] for u in UIDS], dtype=np.float64)


def derived(calc):
    return TaylorBlock(calculator=calc)


def calculator(base=None, variant=0):
    """Producer side: certify the DERIVED book at a base (default: the run base).
    `variant=1` certifies the same positions as SECOND_ORDER_TRUNCATED (cross
    terms omitted by declaration): a different compiled book, same base."""
    base = BASE if base is None else np.asarray(base, dtype=np.float64)
    market, values = _price(base)
    sens = FX.run_sensitivities(
        portfolio=PORTFOLIO, market_data=market, uix_mapper=MAPPER
    )
    if variant == 1:
        sens = dataclasses.replace(
            sens,
            **{
                f.name: getattr(sens, f.name)[:0]
                for f in dataclasses.fields(sens)
                if f.name.startswith("cross_")
                and isinstance(getattr(sens, f.name), np.ndarray)
            },
        )
    binding = FX.TaylorBaseBinding(
        as_of_ord32=AS_OF,
        base_levels_digest=FX.base_levels_digest(factor_ids=FIDS, base_levels=base),
        discounting_mode=NONE.value,
        static_config_digest=FX.static_config_content_digest(RSC),
        valuation_contract_fingerprint=FX.valuation_contract_fingerprint(),
        base_settlement_values_digest=FX.base_settlement_values_digest(
            position_uids=UIDS, values=values
        ),
    )
    spec = FX.build_sensitivity_book_spec(sensitivities=sens, required_factor_ids=FIDS)
    declaration = FX.derive_taylor_declaration(
        sensitivities=sens,
        base_binding=binding,
        order=FX.TaylorOrder.SECOND_ORDER_TRUNCATED
        if variant == 1
        else FX.TaylorOrder.SECOND_ORDER_FULL,
    )
    return FX.SensitivityPnlCalculator.compile(
        dataclasses.replace(spec, taylor_declaration=declaration)
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
    settle = dict(zip(POSITIONS["uid"], POSITIONS["settlement_ccy"], strict=True))
    return dict(
        scenario_config=parse_scenario_model_config(cfg),
        data=parametric_data() if model == "parametric" else matrix(),
        required_factor_ids=FIDS,
        required_shift_type_id=np.ascontiguousarray(
            UNIVERSE_MAP.required_shift_type_ids, np.int8
        ),
        source_factor_ids=FIDS.copy(),
        shock_scale=np.ones(K, np.float32),
        base_levels=BASE,
        methodology=MethodologyDeclaration(
            production_position_uids=UIDS,
            full_reval_uids=[],
            taylor_supplied_uids=[],
            taylor_derived_uids=UIDS,
        ),
        valuation_context=ValuationContext(
            static_config=RSC,
            portfolio=PORTFOLIO,
            uix_mapper=MAPPER,
            discounting_mode=NONE,
        ),
        taylor_derived=TaylorBlock(calculator=calculator() if calc is None else calc),
        reporting=ReportingSpec(
            reporting_ccy="USD",
            position_uids=UIDS,
            settlement_ccys=tuple(settle[int(u)] for u in UIDS),
            fx_factor_names=("USDEUR",),
            fx_factor_rows=np.array([list(FIDS).index(K_TOTAL)], np.int32),
        ),
        confidence_level=0.99,
        batch_size=batch_size,
    )
