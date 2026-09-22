# tests/solutions/risk_run/_fixture.py
"""Shared production-run fixture on the REAL production path (P1-B).

Two SPOT positions are ingested, compiled and certified through the demo static
configuration with node ids AAPL_EQ=10, BRENT_CRUDE=20, USDEUR=30: uid 1 holds
AAPL settled in EUR (it depends on the price and on USDEUR), uid 2 holds Brent
settled in USD. The run axis is the universe map the run itself builds (FX
first). The history is a synthetic T=80 random walk keyed by factor id. The
TAYLOR/DERIVED book is certified on the producer side from the same context the
run values: its sensitivities and V_0 come from that base state, so the run's
own base valuation reproduces its binding exactly."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _mixedfix as mf

from riskquant.market_data.assemble import build_market_data_matrix_from_flat_levels
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.utils.business_days import ord_to_date
from riskquant.valuation.engine import run_valuation

RF = (("AAPL_EQ", 10), ("BRENT_CRUDE", 20), ("USDEUR", 30), ("WTI_CRUDE", 40))
POSITIONS = pd.DataFrame({
    "uid": [1, 2],
    "instrument": ["SPOT", "SPOT"],
    "quantity": [1000.0, -500.0],
    "underlying": ["AAPL_EQ", "BRENT_CRUDE"],
    "settlement_ccy": ["EUR", "USD"],
})
POSITION_UIDS = np.array([1, 2], dtype=np.int64)
_W = mf.build_world(POSITIONS, rf=RF)
NONE = mf.FX.DiscountingMode.NONE
FACTOR_IDS = np.ascontiguousarray(_W.universe_map.required_factor_ids, dtype=np.int32)
SHIFT_TYPES = np.ascontiguousarray(
    _W.universe_map.required_shift_type_ids, dtype=np.int8
)
K, T, S = int(FACTOR_IDS.shape[0]), 80, 64
_HIST_IDS = np.array([10, 20, 30], dtype=np.int32)


def matrix():
    rng = np.random.default_rng(0)
    vals = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, 3)), axis=0))
    vals[:, 2] = 0.9 * np.exp(np.cumsum(rng.normal(0, 0.005, T)))  # USDEUR ~0.9
    return RiskFactorMatrix(
        dates=np.arange(T, dtype=np.int32),
        factor_ids=_HIST_IDS,
        values=np.asfortranarray(vals),
    )


def base_levels():
    """The run base on the run axis, looked up by factor id."""
    last = dict(zip(_HIST_IDS.tolist(), np.asarray(matrix().values)[-1].tolist()))
    return np.array([last[int(f)] for f in FACTOR_IDS], dtype=np.float64)


def context():
    return ValuationContext(
        static_config=_W.rsc,
        ingest_batch=_W.batch,
        uix_mapper=_W.mapper,
        discounting_mode=NONE,
    )


def calculator(base_levels_=None):
    """Producer side: certify the DERIVED book at a base (default: the run base)."""
    base = (
        base_levels()
        if base_levels_ is None
        else np.asarray(base_levels_, dtype=np.float64)
    )
    market = build_market_data_matrix_from_flat_levels(
        as_of_date=ord_to_date(_W.as_of),
        uix_mapper=_W.mapper,
        universe_map=_W.universe_map,
        flat_levels=np.ascontiguousarray(base[:, None]),
        num_scenarios=1,
        discounting_mode=NONE,
    )
    sens = mf.FX.run_sensitivities(
        portfolio=_W.portfolio, market_data=market, uix_mapper=_W.mapper
    )
    rows = np.empty((_W.portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=_W.portfolio,
        out=rows,
        valuation_state=assemble_valuation_state(matrix=market, uix_mapper=_W.mapper),
    )
    pv = dict(zip(_W.portfolio.trade_uids.tolist(), rows[:, 0].tolist(), strict=True))
    v0 = np.array([pv[int(u)] for u in POSITION_UIDS], dtype=np.float64)
    binding = mf.FX.TaylorBaseBinding(
        as_of_ord32=_W.as_of,
        base_levels_digest=mf.FX.base_levels_digest(
            factor_ids=FACTOR_IDS, base_levels=base
        ),
        discounting_mode=NONE.value,
        static_config_digest=mf.FX.static_config_content_digest(_W.rsc),
        valuation_contract_fingerprint=mf.FX.valuation_contract_fingerprint(),
        base_settlement_values_digest=mf.FX.base_settlement_values_digest(
            position_uids=POSITION_UIDS, values=v0
        ),
    )
    spec = mf.FX.build_sensitivity_book_spec(
        sensitivities=sens, required_factor_ids=FACTOR_IDS
    )
    declaration = mf.FX.derive_taylor_declaration(
        sensitivities=sens,
        order=mf.FX.TaylorOrder.SECOND_ORDER_FULL,
        base_binding=binding,
    )
    return mf.FX.SensitivityPnlCalculator.compile(
        dataclasses.replace(spec, taylor_declaration=declaration)
    )


def derived(calc=None):
    """The TAYLOR/DERIVED block: the certified calculator only (P1-B)."""
    return TaylorBlock(calculator=calculator() if calc is None else calc)


def methodology():
    """All positions TAYLOR/DERIVED; U stated explicitly."""
    return MethodologyDeclaration(
        production_position_uids=POSITION_UIDS,
        full_reval_uids=[],
        taylor_supplied_uids=[],
        taylor_derived_uids=POSITION_UIDS,
    )


def reporting_spec(reporting_ccy="USD"):
    return ReportingSpec(
        reporting_ccy=reporting_ccy,
        position_uids=POSITION_UIDS,
        settlement_ccys=("EUR", "USD"),
        fx_factor_names=("USDEUR",),
        fx_factor_rows=np.array([list(FACTOR_IDS).index(30)], np.int32),
    )


def inputs(seed=7, model="sbb"):
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
        data=matrix(),
        required_factor_ids=FACTOR_IDS,
        required_shift_type_id=SHIFT_TYPES,
        source_factor_ids=FACTOR_IDS.copy(),
        shock_scale=np.ones(K, np.float32),
        base_levels=base_levels(),
        methodology=methodology(),
        valuation_context=context(),
        taylor_derived=derived(),
        reporting=reporting_spec(),
        confidence_level=0.99,
        batch_size=16,
    )
