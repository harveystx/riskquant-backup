# tests/_support/spot_scenario.py
"""
Deterministic Proxy V1 scenario over the spot universe.

Three log-return scenarios over the source factors USDEUR, AAPL_EQ and
BRENT_CRUDE. WTI_CRUDE carries no market data of its own, so the proxy path
(an authored term WTI <- 0.95 x BRENT, LOG_RETURN) is forced rather than incidental.

Two suites consume the same shocked levels with different subjects:
`tests/simlab/shocks` asserts the shock path (LOG_RETURN semantics, layout); `tests/solutions/sensitivity_pnl` asserts the P&L the calculator
produces from those levels. One definition keeps both on identical inputs.

The fixtures depend on `rsc`, `level_snapshot` and `portfolio_mapper_market`
from `spot_universe`; a consuming conftest imports both modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.config.proxy_model.schema_objects import NodeProxyTerm
from riskquant.market_data.assemble import (
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID, ShiftType
from riskquant.simlab.shocks.axis_operator import lower_axis_operator
from riskquant.simlab.shocks.column_binding import bind_term_columns
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels

#: Source factors that carry market data, in shock-column order.
SOURCE_NAMES = ("USDEUR", "AAPL_EQ", "BRENT_CRUDE")

#: (S, K_source) log returns.
SHOCKS = (
    (0.000, 0.010, -0.020),
    (-0.010, 0.000, 0.015),
    (0.020, -0.015, -0.010),
)

#: WTI_CRUDE <- 0.95 * Y(BRENT_CRUDE, LOG_RETURN, h): one authored Proxy V1 term.
_PROXY_TERMS: tuple[NodeProxyTerm, ...] = (
    NodeProxyTerm(
        target_name="WTI_CRUDE",
        driver_name="BRENT_CRUDE",
        driver_chart=ShiftType.LOG_RETURN,
        loading=0.95,
    ),
)


@pytest.fixture(scope="session")
def proxy_terms() -> tuple[NodeProxyTerm, ...]:
    return _PROXY_TERMS


@pytest.fixture(scope="session")
def raw_shocks() -> np.ndarray:
    return np.asfortranarray(np.array(SHOCKS, dtype=np.float32))


@pytest.fixture(scope="session")
def resolved(
    rsc, proxy_terms, as_of_date, level_snapshot, portfolio_mapper_market, raw_shocks
) -> dict[str, Any]:
    """Run the production Proxy V1 path once: compile -> lower -> bind -> levels.

    The world is the three hand-authored one-day LOG_RETURN columns of
    `SOURCE_NAMES` (h = 1); every other axis factor keeps its identity row.
    """
    _portfolio, mapper, _market_data = portfolio_mapper_market
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    required = universe_map.required_factor_ids
    base_levels = extract_levels_from_snapshot(
        required_ids=required, snapshot=level_snapshot
    )
    proxy_model = compile_proxy_model(
        authored_terms=proxy_terms,
        static_config=rsc,
        as_of_date=as_of_date,
        run_axis_rf_ids=required,
    )
    operator = lower_axis_operator(
        model=proxy_model,
        axis_rf_ids=required,
        axis_shift_type_ids=universe_map.required_shift_type_ids,
    )
    world = np.ascontiguousarray(
        [rsc.rel_name_to_rf_id[n] for n in SOURCE_NAMES], dtype=np.int32
    )
    term_column_uix = bind_term_columns(
        term_driver_rf_ids=operator.term_driver_rf_ids,
        term_chart_ids=operator.term_chart_ids,
        requested_horizon=1,
        world_column_driver_rf_ids=world,
        world_column_chart_ids=np.full(world.shape[0], SHIFT_LOG_RETURN_ID, np.int8),
        world_horizon=1,
    )
    shocked = np.empty((required.shape[0], len(SHOCKS)), dtype=np.float64, order="C")
    apply_linear_terms_to_base_levels(
        base_levels=base_levels,
        row_term_offsets=operator.row_term_offsets,
        term_column_uix=term_column_uix,
        term_loadings=operator.term_loadings,
        target_shift_types=operator.target_shift_type_ids,
        shocks=raw_shocks,
        out=shocked,
    )
    return {
        "required": required,
        "base_levels": base_levels,
        "shocked": shocked,
        "proxy_model": proxy_model,
        "row_of": {int(rf): i for i, rf in enumerate(required)},
    }
