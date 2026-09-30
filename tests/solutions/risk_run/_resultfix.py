# tests/solutions/risk_run/_resultfix.py
"""
Test-owned fixture for the persisted-result integrity contract (D083).

A tiny FULL_REVAL SPOT book on the REAL production path: the canonical static
configuration -> `compile_raw_df_into_ingest_batch` -> `build_uix_mapper` ->
`ValuationContext` -> `build_universe_mapping` -> `run_scenario_var_es`. It
depends on nothing under `scripts/`, so the contract is pinned independently of
the demo fixtures (two-layer golden practice). The history is synthetic and
seeded; the numbers are not an economic oracle. The tests using this fixture
assert ARTIFACT behaviour (bytes, digests, acceptance, rejection), never the
level of a VaR.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
)
from riskquant.market_data.assemble import build_universe_mapping
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.portfolio.engine import build_uix_mapper
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.scenario_models.schemas import ScenarioModelConfig
from riskquant.utils.dates import date_to_ord32

AS_OF = dt.date(2024, 6, 28)
UIDS = np.array([101, 102], dtype=np.int64)
S_SBB = 200
_HISTORY_DAYS = 60


def sbb(seed: int | None) -> ScenarioModelConfig:
    """Stochastic model; `seed=None` is the exploratory, non-reproducible case."""
    return parse_scenario_model_config({
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": S_SBB,
        "expected_block_size": 5.0,
        "seed": seed,
    })


def inputs(**over: Any) -> dict[str, Any]:
    """Keyword inputs of `run_scenario_var_es`; `over` replaces any of them."""
    rsc = compile_risk_static_config_from_dict_of_dfs(
        dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    )
    raw = pl.from_dict({
        "uid": UIDS.tolist(),
        "instrument": ["SPOT", "SPOT"],
        "quantity": [100.0, -40.0],
        "underlying": ["AAPL_EQ", "AAPL_EQ"],
        "settlement_ccy": ["USD", "USD"],
    })
    batch = compile_raw_df_into_ingest_batch(
        raw_df=raw, static_config=rsc, as_of_date=AS_OF
    )
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(AS_OF)))
    context = ValuationContext(
        static_config=rsc,
        ingest_batch=batch,
        uix_mapper=mapper,
        discounting_mode=DiscountingMode.NONE,
    )
    layout = build_universe_mapping(
        static_config=rsc, uix_mapper=mapper, discounting_mode=DiscountingMode.NONE
    )
    factor_ids = np.asarray(layout.required_factor_ids, dtype=np.int32)
    k = factor_ids.shape[0]
    rng = np.random.default_rng(7)
    values = np.asfortranarray(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (_HISTORY_DAYS, k)), axis=0))
    )
    end = date_to_ord32(AS_OF)
    dates = np.arange(end - _HISTORY_DAYS + 1, end + 1, dtype=np.int32)
    empty = np.empty(0, dtype=np.int64)
    base: dict[str, Any] = {
        "scenario_config": parse_scenario_model_config({
            "model": "historical",
            "horizon": 1,
        }),
        "data": RiskFactorMatrix(
            dates=dates, factor_ids=factor_ids.copy(), values=values
        ),
        "required_factor_ids": factor_ids,
        "required_shift_type_id": np.asarray(
            layout.required_shift_type_ids, dtype=np.int8
        ),
        "proxy_terms": (),
        "base_levels": np.full(k, 150.0, dtype=np.float64),
        "methodology": MethodologyDeclaration(
            production_position_uids=UIDS,
            full_reval_uids=UIDS,
            taylor_supplied_uids=empty,
            taylor_derived_uids=empty,
        ),
        "valuation_context": context,
        "reporting": ReportingSpec(reporting_ccy="USD"),
        "confidence_level": 0.99,
        "batch_size": 16,
    }
    base.update(over)
    return base
