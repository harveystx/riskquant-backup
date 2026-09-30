# tests/_support/spot_universe.py
"""
Minimal spot universe: one static config, three SPOT positions (AAPL USD,
Brent USD-priced / EUR-settled, WTI USD), one as-of and its golden levels.

Shared by the suites that need it -- `tests/sensitivities` (the subject),
`tests/solutions/sensitivity_pnl` (the solution consuming derived Greeks) and
`tests/simlab/scenarios` (the static config that resolves the factor ids) --
so the universe has ONE definition. Each suite's conftest imports only the
fixtures it uses; the module is not a test module and is never collected.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl
import pytest

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
)
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.sensitivities.result import BlockSensitivities
from riskquant.utils.dates import date_to_ord32

_AS_OF = date(2027, 1, 4)

_LEVELS = {
    "USDEUR": 1.0800,
    "AAPL_EQ": 200.00,
    "BRENT_CRUDE": 82.30,
    "WTI_CRUDE": 70.00,
}
_AAPL_UID, _BRENT_UID, _WTI_UID = 1_010, 1_020, 1_030


@pytest.fixture(scope="session")
def as_of_date() -> date:
    return _AS_OF


@pytest.fixture(scope="session")
def golden_levels() -> dict[str, float]:
    return _LEVELS


@pytest.fixture(scope="session")
def position_uids() -> tuple[int, int, int]:
    return _AAPL_UID, _BRENT_UID, _WTI_UID


@pytest.fixture(scope="session")
def static_config_dfs() -> dict[str, pl.DataFrame]:
    return {
        "ccy": pl.DataFrame({"ccy_name": ["EUR", "USD"], "ccy_id": [1, 2]}),
        "holiday_calendars": pl.DataFrame({
            "holiday_calendar_name": ["NL", "US"],
            "holiday_date": ["2027-01-01", "2027-01-01"],
        }),
        "node_meta": pl.DataFrame({
            "node_id": [0, 1, 2, 3],
            "node_name": ["USDEUR", "AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE"],
            "quote_ccy": ["EUR", "USD", "USD", "USD"],
            "asset_class": ["FX", "EQUITY", "COMMODITY", "COMMODITY"],
            "asset_subclass": ["FX_G10", "EQ_SINGLE_NAME", "CM_OIL", "CM_OIL"],
            "quote_type": ["FX_RATE", "PRICE", "PRICE", "PRICE"],
            "unit_scale": ["DECIMAL"] * 4,
            "shift_type": ["LOG_RETURN"] * 4,
            "fx_base_ccy": ["USD", None, None, None],
        }),
        "curve_meta": pl.DataFrame({
            "curve_id": [250000000, 250001000],
            "curve_name": ["EUR_ESTR_OIS", "USD_SOFR_OIS"],
            "quote_ccy": ["EUR", "USD"],
            "asset_class": ["IR", "IR"],
            "asset_subclass": ["IR_OIS", "IR_OIS"],
            "quote_type": ["RATE", "RATE"],
            "unit_scale": ["PERCENT", "PERCENT"],
            "shift_type": ["DIFFERENCE", "DIFFERENCE"],
            "x_axis_kind": ["LABEL", "LABEL"],
            "x_interp_method": ["LINEAR", "LINEAR"],
            "x_extrap_method": ["FLAT", "FLAT"],
            "roll_method": [None, None],
            "anchor_offset": [None, None],
            "roll_month": [None, None],
            "roll_day": [None, None],
            "fh_bh_mmdd": [None, None],
            "discount_basis": ["OIS", "OIS"],
            "is_default_discount_for_ccy": [True, True],
        }),
        "curve_point_meta": pl.DataFrame({
            "rf_id": [1000, 2000],
            "curve_id": [250000000, 250001000],
            "tenor_token": ["[3M]", "[3M]"],
        }),
        "surface_meta": pl.DataFrame({
            name: pl.Series([], dtype=dtype)
            for name, dtype in {
                "surface_id": pl.Int64,
                "surface_name": pl.String,
                "quote_ccy": pl.String,
                "asset_class": pl.String,
                "asset_subclass": pl.String,
                "quote_type": pl.String,
                "unit_scale": pl.String,
                "shift_type": pl.String,
                "x_axis_kind": pl.String,
                "x_interp_method": pl.String,
                "x_extrap_method": pl.String,
                "roll_method": pl.String,
                "anchor_offset": pl.Int32,
                "roll_month": pl.Int32,
                "roll_day": pl.Int32,
                "fh_bh_mmdd": pl.String,
                "y_axis_kind": pl.String,
                "y_interp_method": pl.String,
                "y_extrap_method": pl.String,
            }.items()
        }),
        "surface_point_meta": pl.DataFrame({
            name: pl.Series([], dtype=dtype)
            for name, dtype in {
                "rf_id": pl.Int64,
                "surface_id": pl.Int64,
                "x_tenor_token": pl.String,
                "y_label": pl.String,
            }.items()
        }),
    }


@pytest.fixture(scope="session")
def raw_positions() -> pd.DataFrame:
    return pd.DataFrame({
        "uid": [_AAPL_UID, _BRENT_UID, _WTI_UID],
        "instrument": ["SPOT", "SPOT", "SPOT"],
        "quantity": [100.0, -50.0, 75.0],
        "underlying": ["AAPL_EQ", "BRENT_CRUDE", "WTI_CRUDE"],
        "settlement_ccy": ["USD", "EUR", "USD"],
    })


@pytest.fixture(scope="session")
def rsc(static_config_dfs):
    return compile_risk_static_config_from_dict_of_dfs(static_config_dfs)


@pytest.fixture(scope="session")
def level_snapshot(rsc, as_of_date, golden_levels):
    return build_risk_factor_level_snapshot_from_dict(
        as_of_date=as_of_date,
        levels_dict=golden_levels,
        static_config=rsc,
    )


@pytest.fixture(scope="session")
def portfolio_mapper_market(rsc, raw_positions, as_of_date, level_snapshot):
    batch = compile_raw_df_into_ingest_batch(raw_df=pl.from_pandas(raw_positions), static_config=rsc)
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(as_of_date)))
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    universe_map = build_universe_mapping(
        discounting_mode=DiscountingMode.NONE, static_config=rsc, uix_mapper=mapper
    )
    market_data = build_market_data_matrix_from_snapshot(
        discounting_mode=DiscountingMode.NONE,
        as_of_date=as_of_date,
        uix_mapper=mapper,
        universe_map=universe_map,
        snapshot=level_snapshot,
        num_scenarios=1,
    )
    return portfolio, mapper, market_data


@pytest.fixture(scope="session")
def sensitivities(portfolio_mapper_market) -> BlockSensitivities:
    portfolio, mapper, market_data = portfolio_mapper_market
    return run_sensitivities(
        portfolio=portfolio, uix_mapper=mapper, market_data=market_data
    )
