# tests/market_data/remediation/hostile_fixture.py
"""
Hostile semantic fixture for the market-data re-certification (D060-D072).

Everything here is PRODUCTION input: static-config tables, a positions frame,
level quotes by NAME. The run helpers call only production entry points:

    compile_risk_static_config_from_dict_of_dfs
    compile_raw_df_into_ingest_batch
    build_uix_mapper -> compile_portfolio_from_ingest_batch
    build_universe_mapping
    build_market_data_matrix_from_snapshot | build_market_data_matrix_from_flat_levels
    assemble_valuation_state -> run_valuation

No MarketDataMatrix, ValuationState or IngestBatch is assembled by hand.

Hostility
---------
* rf ids sparse and interleaved: FX ids (31, 402, 5003) sit between price ids.
* settlement currencies EUR, USD, GBP; asset currencies USD, GBP, CHF, EUR.
* direct quote (USDEUR), inverse-only quotes (EURGBP for GBP->EUR, USDCHF for
  CHF->USD), a missing pair (JPY), identity, a spot ON an FX rate, and a
  negative-level price (a power spread).
* S scenarios with factor-specific, non-proportional scenario paths.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl

AS_OF = date(2027, 1, 4)

CCY = [("EUR", 1), ("USD", 2), ("GBP", 3), ("CHF", 4), ("JPY", 5)]

# (node_id, name, quote_ccy, asset_class, subclass, quote_type, unit, shift, fx_base)
NODES = [
    (
        90001,
        "EQ_EUR",
        "EUR",
        "EQUITY",
        "EQ_SINGLE_NAME",
        "PRICE",
        "DECIMAL",
        "LOG_RETURN",
        None,
    ),
    (5003, "USDCHF", "CHF", "FX", "FX_G10", "FX_RATE", "DECIMAL", "LOG_RETURN", "USD"),
    (
        58,
        "EQ_GBP",
        "GBP",
        "EQUITY",
        "EQ_SINGLE_NAME",
        "PRICE",
        "DECIMAL",
        "LOG_RETURN",
        None,
    ),
    (402, "EURGBP", "GBP", "FX", "FX_G10", "FX_RATE", "DECIMAL", "LOG_RETURN", "EUR"),
    (
        7,
        "EQ_USD",
        "USD",
        "EQUITY",
        "EQ_SINGLE_NAME",
        "PRICE",
        "DECIMAL",
        "LOG_RETURN",
        None,
    ),
    (31, "USDEUR", "EUR", "FX", "FX_G10", "FX_RATE", "DECIMAL", "LOG_RETURN", "USD"),
    (
        1200,
        "EQ_CHF",
        "CHF",
        "EQUITY",
        "EQ_SINGLE_NAME",
        "PRICE",
        "DECIMAL",
        "LOG_RETURN",
        None,
    ),
    (
        333,
        "EQ_JPY",
        "JPY",
        "EQUITY",
        "EQ_SINGLE_NAME",
        "PRICE",
        "DECIMAL",
        "LOG_RETURN",
        None,
    ),
    (
        64,
        "PWR_SPREAD",
        "EUR",
        "COMMODITY",
        "CM_POWER",
        "PRICE",
        "DECIMAL",
        "DIFFERENCE",
        None,
    ),
    (
        17,
        "VOL_EQ_USD",
        "USD",
        "EQUITY",
        "EQ_SINGLE_NAME",
        "VOL",
        "PERCENT",
        "LOG_RETURN",
        None,
    ),
]

_IR = [
    ("EUR", 250_004_000),
    ("USD", 250_001_000),
    ("GBP", 250_009_000),
    ("CHF", 250_002_000),
    ("JPY", 250_017_000),
]
BRENT_CURVE_ID = 250_030_000


def static_config_dfs() -> dict[str, pl.DataFrame]:
    n = len(_IR) + 1
    curve_meta = pl.DataFrame({
        "curve_id": [c for _, c in _IR] + [BRENT_CURVE_ID],
        "curve_name": [f"{k}_OIS" for k, _ in _IR] + ["BRENT_FWD"],
        "quote_ccy": [k for k, _ in _IR] + ["USD"],
        "asset_class": ["IR"] * len(_IR) + ["COMMODITY"],
        "asset_subclass": ["IR_OIS"] * len(_IR) + ["CM_OIL"],
        "quote_type": ["RATE"] * len(_IR) + ["PRICE"],
        "unit_scale": ["PERCENT"] * len(_IR) + ["DECIMAL"],
        "shift_type": ["DIFFERENCE"] * len(_IR) + ["LOG_RETURN"],
        "x_axis_kind": ["LABEL"] * len(_IR) + ["OFFSET_M"],
        "x_interp_method": ["LINEAR"] * n,
        "x_extrap_method": ["FLAT"] * n,
        "roll_method": [None] * len(_IR) + ["CALENDAR_PERIOD"],
        "anchor_offset": pl.Series([None] * len(_IR) + [1], dtype=pl.Int32),
        "roll_month": pl.Series([None] * n, dtype=pl.Int32),
        "roll_day": pl.Series([None] * n, dtype=pl.Int32),
        "fh_bh_mmdd": pl.Series([None] * n, dtype=pl.String),
        "discount_basis": ["OIS"] * len(_IR) + ["NOT_A_DISCOUNT_CURVE"],
        "is_default_discount_for_ccy": [True] * len(_IR) + [False],
    })
    curve_points = pl.DataFrame({
        "rf_id": [1001, 1002, 1003, 1004, 1005, 2001, 2002],
        "curve_id": [c for _, c in _IR] + [BRENT_CURVE_ID, BRENT_CURVE_ID],
        "tenor_token": ["[3M]"] * len(_IR) + ["[M+0]", "[M+1]"],
    })
    cols = [
        "node_id",
        "node_name",
        "quote_ccy",
        "asset_class",
        "asset_subclass",
        "quote_type",
        "unit_scale",
        "shift_type",
        "fx_base_ccy",
    ]
    node_meta = pl.DataFrame({c: [r[i] for r in NODES] for i, c in enumerate(cols)})
    surface_meta = pl.DataFrame({
        "surface_id": pl.Series([], dtype=pl.Int64),
        "surface_name": pl.Series([], dtype=pl.String),
        "quote_ccy": pl.Series([], dtype=pl.String),
        "asset_class": pl.Series([], dtype=pl.String),
        "asset_subclass": pl.Series([], dtype=pl.String),
        "quote_type": pl.Series([], dtype=pl.String),
        "unit_scale": pl.Series([], dtype=pl.String),
        "shift_type": pl.Series([], dtype=pl.String),
        "x_axis_kind": pl.Series([], dtype=pl.String),
        "x_interp_method": pl.Series([], dtype=pl.String),
        "x_extrap_method": pl.Series([], dtype=pl.String),
        "roll_method": pl.Series([], dtype=pl.String),
        "anchor_offset": pl.Series([], dtype=pl.Int32),
        "roll_month": pl.Series([], dtype=pl.Int32),
        "roll_day": pl.Series([], dtype=pl.Int32),
        "fh_bh_mmdd": pl.Series([], dtype=pl.String),
        "y_axis_kind": pl.Series([], dtype=pl.String),
        "y_interp_method": pl.Series([], dtype=pl.String),
        "y_extrap_method": pl.Series([], dtype=pl.String),
    })
    surface_points = pl.DataFrame({
        "rf_id": pl.Series([], dtype=pl.Int64),
        "surface_id": pl.Series([], dtype=pl.Int64),
        "x_tenor_token": pl.Series([], dtype=pl.String),
        "y_label": pl.Series([], dtype=pl.String),
    })
    return {
        "ccy": pl.DataFrame({
            "ccy_name": [c for c, _ in CCY],
            "ccy_id": [i for _, i in CCY],
        }),
        "holiday_calendars": pl.DataFrame({
            "holiday_calendar_name": ["NL", "US"],
            "holiday_date": ["2027-01-01", "2027-01-01"],
        }),
        "node_meta": node_meta,
        "curve_meta": curve_meta,
        "curve_point_meta": curve_points,
        "surface_meta": surface_meta,
        "surface_point_meta": surface_points,
    }


# Economic market, in canonical DECIMAL units, by NAME. Scenario paths differ by
# factor so a row or scenario substitution cannot cancel.
S = 5
_PATH = {
    "EQ_USD": [1.0, 1.021, 0.973, 1.044, 0.912],
    "EQ_GBP": [1.0, 0.987, 1.032, 0.955, 1.061],
    "EQ_CHF": [1.0, 1.009, 0.991, 1.017, 0.978],
    "EQ_EUR": [1.0, 0.962, 1.013, 1.071, 0.941],
    "PWR_SPREAD": [1.0, 1.8, -0.4, 2.3, 0.6],
    "USDEUR": [1.0, 0.994, 1.006, 0.989, 1.012],
    "EURGBP": [1.0, 1.004, 0.997, 1.011, 0.992],
    "USDCHF": [1.0, 0.998, 1.013, 0.987, 1.003],
}
_BASE = {
    "EQ_USD": 212.37,
    "EQ_GBP": 148.91,
    "EQ_CHF": 91.44,
    "EQ_EUR": 53.12,
    "PWR_SPREAD": -7.25,
    "USDEUR": 0.9259,
    "EURGBP": 0.8561,
    "USDCHF": 0.8834,
}
LEVELS = {k: _BASE[k] * np.asarray(_PATH[k]) for k in _BASE}
FACTOR_CCY = {
    "EQ_USD": "USD",
    "EQ_GBP": "GBP",
    "EQ_CHF": "CHF",
    "EQ_EUR": "EUR",
    "PWR_SPREAD": "EUR",
    "USDEUR": "EUR",
    "EQ_JPY": "JPY",
}
FX_QUOTES = {
    ("USD", "EUR"): "USDEUR",
    ("EUR", "GBP"): "EURGBP",
    ("USD", "CHF"): "USDCHF",
}

SPOT_BOOK = pd.DataFrame({
    "uid": [101, 102, 103, 104, 105, 106],
    "instrument": ["SPOT"] * 6,
    "quantity": [100.0, -250.0, 40.0, 1_000_000.0, 3.0, 12.5],
    "underlying": ["EQ_USD", "EQ_GBP", "EQ_CHF", "USDEUR", "EQ_EUR", "PWR_SPREAD"],
    "settlement_ccy": ["EUR", "EUR", "USD", "EUR", "GBP", "EUR"],
})


def flat_levels_for(
    universe_map, static_config, levels: dict[str, np.ndarray]
) -> np.ndarray:
    """Lay levels out in the production `required_factor_ids` order, by NAME."""
    id_to_name = {v: k for k, v in static_config.rel_name_to_rf_id.items()}
    rows = [
        levels[id_to_name[int(rf)]] for rf in universe_map.required_factor_ids.tolist()
    ]
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def run_spot_book(
    *,
    static_config,
    book: pd.DataFrame,
    levels: dict[str, np.ndarray],
    num_scenarios: int,
):
    """Real production path, flat-levels route (S scenarios). Returns (portfolio, mapper, matrix, state, pv)."""
    from riskquant.market_data.assemble import (
        build_market_data_matrix_from_flat_levels,
        build_universe_mapping,
    )
    from riskquant.market_data.compile import assemble_valuation_state
    from riskquant.portfolio.engine import (
        build_uix_mapper,
        compile_portfolio_from_ingest_batch,
    )
    from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
    from riskquant.utils.dates import date_to_ord32
    from riskquant.valuation.engine import run_valuation

    batch = compile_raw_df_into_ingest_batch(
        raw_df=book, static_config=static_config, copy=True
    )
    mapper = build_uix_mapper([batch], as_of_ord32=int(date_to_ord32(AS_OF)))
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    um = build_universe_mapping(static_config=static_config, uix_mapper=mapper)
    matrix = build_market_data_matrix_from_flat_levels(
        as_of_date=AS_OF,
        uix_mapper=mapper,
        universe_map=um,
        flat_levels=flat_levels_for(um, static_config, levels),
        num_scenarios=num_scenarios,
    )
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((portfolio.total_trades, num_scenarios), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return portfolio, mapper, matrix, state, out
