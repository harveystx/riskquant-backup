# tests/valuation/oracle/whole_surface_fixture.py
"""
The whole supported valuation surface in ONE production-path fixture.

    SpotBlock + VanillaForwardBlock (NODE) + VanillaEuropeanOptionBlock (NODE)

Every block is produced by the real compilers from one `IngestBatch`; every
factor row by the real `assemble_valuation_state`. Nothing is hand-built except
the raw `MarketDataMatrix`, which is the market-data PRODUCER boundary.

Hostile by construction
-----------------------
    trade_uids      [9007, 41, 120003, 9007, 5]   non-ascending, 9007 REPEATED
                    across the spot block and the option block
    rf ids          sparse, non-ascending; id != row position everywhere
    currencies      USD asset / GBP strike / EUR settlement
    FX              non-unit both ways; the GBP leg is quoted EUR->GBP with
                    `inv = 1`, so the resolution must invert it
    curves          asset curve != strike curve
    dates           delivery, two expiries, a deferred settlement
    options         call + put, long + short, two strikes, two maturities
    spot            short position, USD quoted, EUR settled

`instrument_ids` must be sorted for the compiler's contiguous-run partitioning,
so the layout is SPOT, FORWARD, OPTION, OPTION, OPTION. uid 9007 therefore owns
the spot row AND an option row -- rows separated by two block boundaries.
"""

from __future__ import annotations

import numpy as np

from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.instruments import (
    SPOT_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_NODE_ID,
)

from .economic_oracle import (
    CrossCurrencyOptionTrade,
    EconomicMarket,
    ForwardTrade,
    SpotTrade,
)

SETTLE_CCY = "EUR"
N_SCENARIOS = 3
AS_OF_ORD = 20_000

# --- global risk-factor ids (sparse, non-ascending presentation) -------------
RF_BRENT = 7_001  # spot underlying, quoted USD
RF_WTI = 88_011  # forward node, quoted USD
RF_GOLD = 12  # option underlying forward, quoted USD
RF_SILVER = 400_003  # option underlying forward, quoted USD
RF_VOL_GOLD = 91_004
RF_VOL_SILVER = 6
RF_FX_USD_EUR = 501  # USD -> EUR, quoted direct
RF_FX_EUR_GBP = 40_009  # EUR -> GBP, quoted REVERSE (inv = 1 gives GBP -> EUR)

CURVE_USD = 900_001
CURVE_GBP = 31

ORD_SPOT = 20_001  # spot carries no time value; a date is still required
ORD_DELIVERY = 20_450
ORD_EXP_A = 20_365
ORD_EXP_B = 20_200
ORD_SETTLE_LATE = 20_370

FX_USD_EUR = np.array([0.91370, 0.90118, 0.92744])
FX_EUR_GBP = np.array([0.84655, 0.85901, 0.83402])  # GBP->EUR = 1 / this

LVL_BRENT = np.array([82.30, 91.77, 61.05])
LVL_WTI = np.array([78.44, 88.12, 59.37])
LVL_GOLD = np.array([2410.55, 2688.20, 2101.40])
LVL_SILVER = np.array([28.115, 31.44, 24.07])

VOL_GOLD = np.array([0.2150, 0.2795, 0.1830])
VOL_SILVER = np.array([0.3410, 0.4102, 0.2988])

DF_USD_DELIVERY = np.array([0.96150, 0.95022, 0.97311])
DF_USD_EXP_A = np.array([0.96410, 0.95380, 0.97540])
DF_USD_EXP_B = np.array([0.98020, 0.97455, 0.98630])
DF_GBP_DELIVERY = np.array([0.93110, 0.91905, 0.94290])
DF_GBP_EXP_A = np.array([0.94270, 0.93115, 0.95402])
DF_GBP_EXP_B = np.array([0.96880, 0.96215, 0.97540])
DF_GBP_SETTLE_LATE = np.array([0.94180, 0.93020, 0.95315])

_DAYS = 365.25

# Row order of the compiled book: SPOT, FORWARD, then three OPTION rows.
UIDS = [9007, 41, 120003, 9007, 5]
MULTI_ROW_UID = 9007


def _tte(ordinal: int) -> float:
    return (ordinal - AS_OF_ORD) / _DAYS


# =========================================================================
# ECONOMIC SIDE -- consumed by the independent oracle
# =========================================================================


def economic_market() -> EconomicMarket:
    return EconomicMarket(
        n_scenarios=N_SCENARIOS,
        spot_native={RF_BRENT: LVL_BRENT},
        forward_native={RF_WTI: LVL_WTI, RF_GOLD: LVL_GOLD, RF_SILVER: LVL_SILVER},
        discount={
            (CURVE_USD, ORD_DELIVERY): DF_USD_DELIVERY,
            (CURVE_USD, ORD_EXP_A): DF_USD_EXP_A,
            (CURVE_USD, ORD_EXP_B): DF_USD_EXP_B,
            (CURVE_GBP, ORD_DELIVERY): DF_GBP_DELIVERY,
            (CURVE_GBP, ORD_EXP_A): DF_GBP_EXP_A,
            (CURVE_GBP, ORD_EXP_B): DF_GBP_EXP_B,
            (CURVE_GBP, ORD_SETTLE_LATE): DF_GBP_SETTLE_LATE,
        },  # type: ignore[arg-type]
        fx={("USD", "EUR"): FX_USD_EUR, ("EUR", "GBP"): FX_EUR_GBP},
        vol={RF_VOL_GOLD: VOL_GOLD, RF_VOL_SILVER: VOL_SILVER},
        forward_rate={},
    )


def economic_book() -> list:
    """Five economic rows, in the order their block rows appear."""
    return [
        # SPOT -- short Brent, USD quoted, EUR settled. uid shared with row 3.
        SpotTrade(
            trade_uid=9007, qty=-1500.0, spot_factor_id=RF_BRENT, quote_ccy="USD"
        ),
        # FORWARD -- long WTI, USD asset leg, GBP strike leg
        ForwardTrade(
            trade_uid=41,
            qty=250.0,
            forward_factor_id=RF_WTI,
            asset_ccy="USD",
            asset_discount_factor_id=(CURVE_USD, ORD_DELIVERY),  # type: ignore[arg-type]
            strike=61.25,
            strike_ccy="GBP",
            strike_discount_factor_id=(CURVE_GBP, ORD_DELIVERY),  # type: ignore[arg-type]
        ),
        # OPTION -- SHORT gold CALL, settlement on the expiry date
        CrossCurrencyOptionTrade(
            trade_uid=120003,
            qty=-80.0,
            is_call=True,
            strike=2500.0,
            asset_forward_factor_id=RF_GOLD,
            asset_ccy="USD",
            asset_expiry_discount_id=(CURVE_USD, ORD_EXP_A),  # type: ignore[arg-type]
            strike_ccy="GBP",
            strike_expiry_discount_id=(CURVE_GBP, ORD_EXP_A),  # type: ignore[arg-type]
            strike_settle_discount_id=(CURVE_GBP, ORD_EXP_A),  # type: ignore[arg-type]
            vol_factor_id=RF_VOL_GOLD,
            tte_years=_tte(ORD_EXP_A),
        ),
        # OPTION -- LONG silver PUT, DEFERRED settlement. uid shared with row 0.
        CrossCurrencyOptionTrade(
            trade_uid=9007,
            qty=600.0,
            is_call=False,
            strike=30.00,
            asset_forward_factor_id=RF_SILVER,
            asset_ccy="USD",
            asset_expiry_discount_id=(CURVE_USD, ORD_EXP_B),  # type: ignore[arg-type]
            strike_ccy="GBP",
            strike_expiry_discount_id=(CURVE_GBP, ORD_EXP_B),  # type: ignore[arg-type]
            strike_settle_discount_id=(CURVE_GBP, ORD_SETTLE_LATE),  # type: ignore[arg-type]
            vol_factor_id=RF_VOL_SILVER,
            tte_years=_tte(ORD_EXP_B),
        ),
        # OPTION -- LONG gold PUT, same underlying as row 2, different strike
        CrossCurrencyOptionTrade(
            trade_uid=5,
            qty=140.0,
            is_call=False,
            strike=2300.0,
            asset_forward_factor_id=RF_GOLD,
            asset_ccy="USD",
            asset_expiry_discount_id=(CURVE_USD, ORD_EXP_A),  # type: ignore[arg-type]
            strike_ccy="GBP",
            strike_expiry_discount_id=(CURVE_GBP, ORD_EXP_A),  # type: ignore[arg-type]
            strike_settle_discount_id=(CURVE_GBP, ORD_EXP_A),  # type: ignore[arg-type]
            vol_factor_id=RF_VOL_GOLD,
            tte_years=_tte(ORD_EXP_A),
        ),
    ]


# =========================================================================
# PRODUCTION SIDE
# =========================================================================

MARKET_ROWS = sorted([
    RF_BRENT,
    RF_WTI,
    RF_GOLD,
    RF_SILVER,
    RF_VOL_GOLD,
    RF_VOL_SILVER,
    RF_FX_USD_EUR,
    RF_FX_EUR_GBP,
])
CURVE_ROWS = sorted([CURVE_USD, CURVE_GBP])
DATE_ROWS = sorted([ORD_SPOT, ORD_DELIVERY, ORD_EXP_A, ORD_EXP_B, ORD_SETTLE_LATE])


def ingest_batch():
    """One spot, one forward, three options. Hostile, repeated uids."""
    from riskquant.portfolio.ingest.batch import IngestBatch

    def i32(x):
        return np.ascontiguousarray(x, dtype=np.int32)

    def i8(x):
        return np.ascontiguousarray(x, dtype=np.int8)

    def f64(x):
        return np.ascontiguousarray(x, dtype=np.float64)

    OPT = VANILLA_EUROPEAN_OPTION_NODE_ID
    FWD = VANILLA_FORWARD_NODE_ID

    return IngestBatch(
        size=5,
        instrument_ids=i32([SPOT_ID, FWD, OPT, OPT, OPT]),
        uids=np.ascontiguousarray(UIDS, dtype=np.int64),
        qty=f64([-1500.0, 250.0, -80.0, 600.0, 140.0]),
        strike_price=f64([0.0, 61.25, 2500.0, 30.00, 2300.0]),
        cp_mult=i8([0, 0, 1, -1, -1]),
        settle_ord=i32([ORD_SPOT, ORD_DELIVERY, ORD_EXP_A, ORD_SETTLE_LATE, ORD_EXP_A]),
        delivery_ord=i32([ORD_SPOT, ORD_DELIVERY, ORD_EXP_A, ORD_EXP_B, ORD_EXP_A]),
        expiry_ord=i32([ORD_SPOT, ORD_DELIVERY, ORD_EXP_A, ORD_EXP_B, ORD_EXP_A]),
        underlying_ids=i32([RF_BRENT, RF_WTI, RF_GOLD, RF_SILVER, RF_GOLD]),
        volatility_ids=i32([-1, -1, RF_VOL_GOLD, RF_VOL_SILVER, RF_VOL_GOLD]),
        # asset -> settle FX: every underlying is USD quoted
        fx_u2s_ids=i32([RF_FX_USD_EUR] * 5),
        fx_u2s_inv=i8([0, 0, 0, 0, 0]),
        # strike -> settle FX: GBP strikes, quoted EUR->GBP so inverted
        fx_k2s_ids=i32([RF_FX_EUR_GBP] * 5),
        fx_k2s_inv=i8([1, 1, 1, 1, 1]),
        under_discount_curve_ids=i32([-1, CURVE_USD, CURVE_USD, CURVE_USD, CURVE_USD]),
        strike_discount_curve_ids=i32([-1, CURVE_GBP, CURVE_GBP, CURVE_GBP, CURVE_GBP]),
        vol_surface_y_axis_type_ids=i8([-99] * 5),
        netting_set_id=np.ascontiguousarray([1] * 5, dtype=np.int64),
    )


def build_market_matrix(uix_mapper) -> MarketDataMatrix:
    """Materialise the raw tier against the run's actual effective-FX table."""
    s = N_SCENARIOS
    market = np.array(MARKET_ROWS, dtype=np.int32)
    curves = np.array(CURVE_ROWS, dtype=np.int32)
    dates = np.array(DATE_ROWS, dtype=np.int32)

    levels = {
        RF_BRENT: LVL_BRENT,
        RF_WTI: LVL_WTI,
        RF_GOLD: LVL_GOLD,
        RF_SILVER: LVL_SILVER,
        RF_VOL_GOLD: VOL_GOLD,
        RF_VOL_SILVER: VOL_SILVER,
    }
    prices = np.zeros((market.shape[0], s), dtype=np.float64)
    for rf, lvl in levels.items():
        prices[int(np.searchsorted(market, rf))] = lvl
    prices = np.ascontiguousarray(prices)

    df = np.ones((curves.shape[0], dates.shape[0], s), dtype=np.float64)
    grid = {
        (CURVE_USD, ORD_DELIVERY): DF_USD_DELIVERY,
        (CURVE_USD, ORD_EXP_A): DF_USD_EXP_A,
        (CURVE_USD, ORD_EXP_B): DF_USD_EXP_B,
        (CURVE_GBP, ORD_DELIVERY): DF_GBP_DELIVERY,
        (CURVE_GBP, ORD_EXP_A): DF_GBP_EXP_A,
        (CURVE_GBP, ORD_EXP_B): DF_GBP_EXP_B,
        (CURVE_GBP, ORD_SETTLE_LATE): DF_GBP_SETTLE_LATE,
    }
    for (c, d), v in grid.items():
        df[int(np.searchsorted(curves, c)), int(np.searchsorted(dates, d))] = v
    df = np.ascontiguousarray(df)

    quotes = {RF_FX_USD_EUR: FX_USD_EUR, RF_FX_EUR_GBP: FX_EUR_GBP}
    pairs = uix_mapper.fx_effective_pairs
    fx_eff = np.empty((pairs.shape[0], s), dtype=np.float64)
    for row in range(pairs.shape[0]):
        rf = int(uix_mapper.market_id_universe[pairs[row, 0]])
        raw = quotes[rf]
        fx_eff[row] = (1.0 / raw) if int(pairs[row, 1]) == 1 else raw
    fx_eff = np.ascontiguousarray(fx_eff)

    tte = np.ascontiguousarray(
        np.array([_tte(int(d)) for d in dates], dtype=np.float64)
    )

    return MarketDataMatrix(
        num_scenarios=s,
        uix_fingerprint=uix_mapper.fingerprint,
        fx_spots_effective=fx_eff,
        prices=prices,
        forward_nodes=prices,
        vol_nodes=prices,
        discount_factor_grid=df,
        price_curve_grid=np.zeros((market.shape[0], dates.shape[0], s)),
        tte_grid=tte,
        vol_surface_strikes=np.zeros((0, 0)),
        vol_surfaces=np.ascontiguousarray(np.zeros((0, dates.shape[0], s, 0))),
        surface_y_axis_type_ids=np.zeros(0, dtype=np.int8),
    )
