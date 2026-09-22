# tests/_support/spot_scenario.py
"""
Deterministic proxy-routed scenario over the spot universe.

Three log-return scenarios over the source factors USDEUR, AAPL_EQ and
BRENT_CRUDE. WTI_CRUDE carries no market data of its own, so the proxy path
(WTI <- BRENT x 0.95) is forced rather than incidental.

Two suites consume the same shocked levels with different subjects:
`tests/simlab/shocks` asserts the shock path (routing, LOG_RETURN semantics,
layout); `tests/solutions/sensitivity_pnl` asserts the P&L the calculator
produces from those levels. One definition keeps both on identical inputs.

The fixtures depend on `rsc`, `level_snapshot` and `portfolio_mapper_market`
from `spot_universe`; a consuming conftest imports both modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.market_data.assemble import (
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.proxy_resolver.proxy import (
    resolve_required_factor_ids_to_market_data_factor_ids,
)
from riskquant.simlab.builder.shock_universe import build_shock_universe
from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.shocks.apply import (
    apply_shocks_to_base_levels,
    validate_apply_shocks_buffers,
)

#: Source factors that carry market data, in shock-column order.
SOURCE_NAMES = ("USDEUR", "AAPL_EQ", "BRENT_CRUDE")

#: (S, K_source) log returns.
SHOCKS = (
    (0.000, 0.010, -0.020),
    (-0.010, 0.000, 0.015),
    (0.020, -0.015, -0.010),
)

_PROXY_DF = pl.DataFrame({
    "proxying_level": ["RISK_FACTOR"],
    "required_name": ["WTI_CRUDE"],
    "proxy_name": ["BRENT_CRUDE"],
    "vol_scaling_factor": [0.95],
})


@pytest.fixture(scope="session")
def proxy_topology(rsc):
    return compile_proxy_topology_from_df(df=_PROXY_DF, static_config=rsc)


@pytest.fixture(scope="session")
def raw_shocks() -> np.ndarray:
    return np.asfortranarray(np.array(SHOCKS, dtype=np.float32))


@pytest.fixture(scope="session")
def resolved(
    rsc, proxy_topology, level_snapshot, portfolio_mapper_market, raw_shocks
) -> dict[str, Any]:
    """Run the production scenario path once: proxy -> shock universe -> levels."""
    _portfolio, mapper, _market_data = portfolio_mapper_market
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    required = universe_map.required_factor_ids
    base_levels = extract_levels_from_snapshot(
        required_ids=required, snapshot=level_snapshot
    )

    available = np.ascontiguousarray(
        [rsc.rel_name_to_rf_id[n] for n in SOURCE_NAMES], dtype=np.int32
    )
    proxy_resolution = resolve_required_factor_ids_to_market_data_factor_ids(
        static_config=rsc,
        proxy_topology=proxy_topology,
        required_factor_ids=required,
        market_data_factor_ids=available,
    )
    shock_universe = build_shock_universe(
        required_shift_type_id=universe_map.required_shift_type_ids,
        source_factor_ids=proxy_resolution.source_factor_ids,
        available_factor_ids=available,
    )
    batch = ScenarioSetBatch(
        shocks=raw_shocks,
        required_to_shock_uix=shock_universe.required_to_shock_uix,
        shock_scale=proxy_resolution.shock_scale,
        required_shift_type_id=universe_map.required_shift_type_ids,
        scenario_offset=0,
    )
    shocked = np.empty((required.shape[0], len(SHOCKS)), dtype=np.float64, order="C")
    validate_apply_shocks_buffers(
        base_levels=base_levels, shocks=raw_shocks, out=shocked
    )
    apply_shocks_to_base_levels(base_levels=base_levels, batch=batch, out=shocked)
    return {
        "required": required,
        "base_levels": base_levels,
        "shocked": shocked,
        "proxy_resolution": proxy_resolution,
        "shock_universe": shock_universe,
        "row_of": {int(rf): i for i, rf in enumerate(required)},
    }
