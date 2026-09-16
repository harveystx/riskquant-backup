# tests/valuation/oracle/forward_fixture.py
"""
Production-path forward fixture: a real `IngestBatch` compiled by the real
compiler, priced by the real engine.

This is the Phase-4 difference from Phase 3. Nothing here hand-builds a
`VanillaForwardBlock`; the block is produced by
`compile_portfolio_from_ingest_batch`, and its UIXs by `build_uix_mapper` plus
`compile_vanilla_forward_block`. The only hand-built object is the raw
`MarketDataMatrix`, which is the market-data PRODUCER boundary and out of scope
for this phase.

    IngestBatch (global ids)
        -> build_uix_mapper                  effective-factor universes
        -> compile_portfolio_from_ingest_batch   VanillaForwardBlock + SpotBlock
        -> MarketDataMatrix                  raw tier (hand-built here)
        -> assemble_valuation_state          the real fold
        -> run_valuation                     the real engine
        -> PV

Hostile by construction
-----------------------
    trade_uids      [9007, 41, 120003, 5]   non-ascending, sparse
    rf ids          sparse, non-ascending; id != row position everywhere
    currencies      USD asset / GBP strike / EUR settlement
    FX              non-unit; the GBP leg is quoted EUR->GBP and carries
                    `inv = 1`, so the upstream resolution must invert it
    curves          DF_asset != DF_strike, and different dates
    quantities      mixed long and short
    strike          K != F on every scenario

`instrument_ids` must be sorted for the portfolio compiler's contiguous-run
partitioning, so the batch is laid out SPOT, FORWARD, FORWARD -- which still
yields the required non-ascending uid ordering.
"""

from __future__ import annotations

import numpy as np
from oracle.economic_oracle import (
    EconomicMarket,
    ForwardTrade,
    SpotTrade,
)

from riskquant.constants import NO_FX_RF_REQUIRED
from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.instruments import SPOT_ID, VANILLA_FORWARD_NODE_ID

SETTLE_CCY = "EUR"
N_SCENARIOS = 3

# --- global risk-factor ids (sparse, non-ascending presentation) -------------
RF_BRENT = 7001  # spot underlying, quoted USD
RF_WTI = 88011  # forward node, quoted USD
RF_ALU = 12  # forward node, quoted GBP
RF_FX_USD_EUR = 501  # USD -> EUR, quoted direct
RF_FX_EUR_GBP = 40003  # EUR -> GBP, quoted REVERSE (inv = 1 gives GBP -> EUR)

CURVE_USD = 900_001
CURVE_GBP = 31
CURVE_EUR = 700_019

ORD_DELIVERY = 20_450
ORD_SETTLE = 20_455
ORD_OTHER = 20_100

FX_USD_EUR = np.array([0.91370, 0.90118, 0.92744])
FX_EUR_GBP = np.array([0.84655, 0.85901, 0.83402])  # GBP->EUR = 1 / this

LVL_BRENT = np.array([82.30, 91.77, 61.05])
LVL_WTI = np.array([78.44, 88.12, 59.37])
LVL_ALU = np.array([2451.0, 2199.5, 2610.25])

DF_USD_DELIVERY = np.array([0.96150, 0.95022, 0.97311])
DF_USD_SETTLE = np.array([0.96010, 0.94880, 0.97180])
DF_GBP_SETTLE = np.array([0.94270, 0.93115, 0.95402])
DF_EUR_SETTLE = np.array([0.88120, 0.86033, 0.90277])


# =========================================================================
# ECONOMIC SIDE -- consumed by the independent oracle
# =========================================================================


def economic_market() -> EconomicMarket:
    return EconomicMarket(
        n_scenarios=N_SCENARIOS,
        spot_native={RF_BRENT: LVL_BRENT},
        forward_native={RF_WTI: LVL_WTI, RF_ALU: LVL_ALU},
        discount={  # type: ignore[arg-type]
            (CURVE_USD, ORD_DELIVERY): DF_USD_DELIVERY,
            (CURVE_USD, ORD_SETTLE): DF_USD_SETTLE,
            (CURVE_GBP, ORD_SETTLE): DF_GBP_SETTLE,
            (CURVE_EUR, ORD_SETTLE): DF_EUR_SETTLE,
        },
        fx={("USD", "EUR"): FX_USD_EUR, ("EUR", "GBP"): FX_EUR_GBP},
        vol={},
        forward_rate={},
    )


def economic_book() -> list:
    """Four economic trades, in the order their block rows will appear."""
    return [
        # SPOT block -- short Brent, USD quoted, EUR settled
        SpotTrade(
            trade_uid=9007, qty=-1500.0, spot_factor_id=RF_BRENT, quote_ccy="USD"
        ),
        # FORWARD block, row 0 -- CROSS-CURRENCY: USD asset leg, GBP strike leg
        ForwardTrade(
            trade_uid=41,
            qty=250.0,
            forward_factor_id=RF_WTI,
            asset_ccy="USD",
            asset_discount_factor_id=(CURVE_USD, ORD_DELIVERY),  # type: ignore[arg-type]
            strike=61.25,
            strike_ccy="GBP",
            strike_discount_factor_id=(CURVE_GBP, ORD_SETTLE),  # type: ignore[arg-type]
        ),
        # FORWARD block, row 1 -- SINGLE-CURRENCY degeneration: EUR throughout,
        # one curve and one date on both legs, so PV must reduce to Q(F-K)DF.
        ForwardTrade(
            trade_uid=120003,
            qty=-90.0,
            forward_factor_id=RF_ALU,
            asset_ccy="GBP",
            asset_discount_factor_id=(CURVE_GBP, ORD_SETTLE),  # type: ignore[arg-type]
            strike=2380.0,
            strike_ccy="GBP",
            strike_discount_factor_id=(CURVE_GBP, ORD_SETTLE),  # type: ignore[arg-type]
        ),
        # FORWARD block, row 2 -- EUR asset, EUR strike, EUR settle: identity FX
        ForwardTrade(
            trade_uid=5,
            qty=400.0,
            forward_factor_id=RF_ALU,
            asset_ccy="GBP",
            asset_discount_factor_id=(CURVE_GBP, ORD_SETTLE),  # type: ignore[arg-type]
            strike=2600.0,
            strike_ccy="EUR",
            strike_discount_factor_id=(CURVE_EUR, ORD_SETTLE),  # type: ignore[arg-type]
        ),
    ]


# =========================================================================
# PRODUCTION SIDE
# =========================================================================

# Universe row layouts. Deliberately NOT the sorted id order, so a factor id is
# never equal to its row position.
MARKET_ROWS = sorted([RF_BRENT, RF_WTI, RF_ALU, RF_FX_USD_EUR, RF_FX_EUR_GBP])
CURVE_ROWS = sorted([CURVE_USD, CURVE_GBP, CURVE_EUR])
DATE_ROWS = sorted([ORD_DELIVERY, ORD_SETTLE, ORD_OTHER])


def _ascontiguousarray(x, dtype):
    return np.ascontiguousarray(x, dtype=dtype)


def _ascontiguous_i32(x):
    return _ascontiguousarray(x, dtype=np.int32)


def _ascontiguous_i8(x):
    return _ascontiguousarray(x, dtype=np.int8)


def _ascontiguous_f64(x):
    return _ascontiguousarray(x, dtype=np.float64)


def ingest_batch():
    """A real `IngestBatch`: one spot, three forwards, hostile uids."""
    from riskquant.portfolio.ingest.batch import IngestBatch

    n = 4
    i32 = _ascontiguous_i32
    i8 = _ascontiguous_i8
    f64 = _ascontiguous_f64

    return IngestBatch(
        size=n,
        instrument_ids=i32([
            SPOT_ID,
            VANILLA_FORWARD_NODE_ID,
            VANILLA_FORWARD_NODE_ID,
            VANILLA_FORWARD_NODE_ID,
        ]),
        uids=np.ascontiguousarray([9007, 41, 120003, 5], dtype=np.int64),
        qty=f64([-1500.0, 250.0, -90.0, 400.0]),
        strike_price=f64([0.0, 61.25, 2380.0, 2600.0]),
        cp_mult=i8([0, 0, 0, 0]),
        settle_ord=i32([ORD_OTHER, ORD_SETTLE, ORD_SETTLE, ORD_SETTLE]),
        delivery_ord=i32([ORD_OTHER, ORD_DELIVERY, ORD_SETTLE, ORD_SETTLE]),
        expiry_ord=i32([ORD_OTHER, ORD_DELIVERY, ORD_SETTLE, ORD_SETTLE]),
        underlying_ids=i32([RF_BRENT, RF_WTI, RF_ALU, RF_ALU]),
        volatility_ids=i32([-1, -1, -1, -1]),
        # asset -> settle FX
        fx_u2s_ids=i32([RF_FX_USD_EUR, RF_FX_USD_EUR, RF_FX_EUR_GBP, RF_FX_EUR_GBP]),
        fx_u2s_inv=i8([0, 0, 1, 1]),  # GBP legs are the REVERSE quote
        # strike -> settle FX
        fx_k2s_ids=i32([
            NO_FX_RF_REQUIRED,
            RF_FX_EUR_GBP,
            RF_FX_EUR_GBP,
            NO_FX_RF_REQUIRED,
        ]),
        fx_k2s_inv=i8([0, 1, 1, 0]),
        under_discount_curve_ids=i32([-1, CURVE_USD, CURVE_GBP, CURVE_GBP]),
        strike_discount_curve_ids=i32([-1, CURVE_GBP, CURVE_GBP, CURVE_EUR]),
        vol_surface_y_axis_type_ids=i8([-99, -99, -99, -99]),
        netting_set_id=np.ascontiguousarray([1, 1, 1, 1], dtype=np.int64),
    )


def market_data_matrix() -> MarketDataMatrix:
    """
    Raw market tier, hand-built. Rows are keyed by the universes above.

    `fx_spots_effective` rows are keyed by `UixMapper.fx_effective_pairs`, so
    they are built in `build_market_matrix` where the mapper is available.
    """
    raise NotImplementedError("use build_market_matrix(uix_mapper)")


def build_market_matrix(uix_mapper) -> MarketDataMatrix:
    """Materialise the raw tier against the run's actual effective-FX table."""
    s = N_SCENARIOS
    market = np.array(MARKET_ROWS, dtype=np.int32)
    curves = np.array(CURVE_ROWS, dtype=np.int32)
    dates = np.array(DATE_ROWS, dtype=np.int32)

    # --- node levels, market_id_universe aligned ---
    levels = {RF_BRENT: LVL_BRENT, RF_WTI: LVL_WTI, RF_ALU: LVL_ALU}
    prices = np.zeros((market.shape[0], s), dtype=np.float64)
    for rf, lvl in levels.items():
        prices[int(np.searchsorted(market, rf))] = lvl
    prices = np.ascontiguousarray(prices)

    # --- discount factor grid, (curve, date, scenario) ---
    df = np.ones((curves.shape[0], dates.shape[0], s), dtype=np.float64)
    grid = {
        (CURVE_USD, ORD_DELIVERY): DF_USD_DELIVERY,
        (CURVE_USD, ORD_SETTLE): DF_USD_SETTLE,
        (CURVE_GBP, ORD_SETTLE): DF_GBP_SETTLE,
        (CURVE_EUR, ORD_SETTLE): DF_EUR_SETTLE,
    }
    for (c, d), v in grid.items():
        df[int(np.searchsorted(curves, c)), int(np.searchsorted(dates, d))] = v
    df = np.ascontiguousarray(df)

    # --- effective FX, keyed by fx_effective_pairs rows ---
    quotes = {RF_FX_USD_EUR: FX_USD_EUR, RF_FX_EUR_GBP: FX_EUR_GBP}
    pairs = uix_mapper.fx_effective_pairs
    fx_eff = np.empty((pairs.shape[0], s), dtype=np.float64)
    for row in range(pairs.shape[0]):
        rf = int(uix_mapper.market_id_universe[pairs[row, 0]])
        raw = quotes[rf]
        fx_eff[row] = (1.0 / raw) if int(pairs[row, 1]) == 1 else raw
    fx_eff = np.ascontiguousarray(fx_eff)

    return MarketDataMatrix(
        num_scenarios=s,
        fx_spots_effective=fx_eff,
        prices=prices,
        forward_nodes=prices,
        vol_nodes=prices,
        discount_factor_grid=df,
        price_curve_grid=np.zeros((market.shape[0], dates.shape[0], s)),
        # No options in this fixture: zero surfaces, but the structural guard
        # still anchors every date axis to len(tte_grid).
        tte_grid=np.ascontiguousarray(np.full(dates.shape[0], 1.0)),
        vol_surface_strikes=np.zeros((0, 0)),
        vol_surfaces=np.ascontiguousarray(np.zeros((0, dates.shape[0], s, 0))),
        surface_y_axis_type_ids=np.zeros(0, dtype=np.int8),
    )
