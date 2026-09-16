# tests/valuation/oracle/option_fixture.py
"""
Production-path option fixture: a real `IngestBatch` compiled by the real
option compiler, priced by the real Black-76 kernel.

Nothing here hand-builds a `VanillaEuropeanOptionBlock`. The block, every UIX it
carries, the strike-currency forward, its log cache, the volatility row and the
settlement discount row are all produced by `build_uix_mapper`,
`compile_portfolio_from_ingest_batch` and `assemble_valuation_state`.

Hostile by construction
-----------------------
    trade_uids      [9007, 41, 120003, 5]   non-ascending, sparse
    rf ids          sparse, non-ascending; id != row position everywhere
    currencies      USD asset / GBP strike / EUR settlement  (genuinely cross-ccy)
    FX              non-unit; the GBP leg is quoted EUR->GBP with `inv = 1`, so
                    the resolution must invert it
    curves          asset curve != strike curve, and expiry != settlement
    options         call + put, long + short, two strikes, two maturities
    forwards        two distinct underlyings, so a forward-row permutation moves
                    the price

Row 0 is a NODE forward, kept so the Phase-4 certification stays exercised
alongside the options and so the fixture spans two block types.

Identifiability
---------------
The two options reference DIFFERENT underlyings, different vols, different
strikes and different maturities, so permuting any of the forward, log-forward,
vol or discount rows changes the expected PV. `trade_uid 5` deliberately has a
settlement date later than its expiry, which makes the expiry/settlement
discount rows distinct and therefore swappable.
"""

from __future__ import annotations

import numpy as np
from oracle.economic_oracle import (
    CrossCurrencyOptionTrade,
    EconomicMarket,
    ForwardTrade,
)

from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.instruments import (
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_NODE_ID,
)

SETTLE_CCY = "EUR"
N_SCENARIOS = 3

AS_OF_ORD = 20_000

# --- global risk-factor ids (sparse, non-ascending presentation) -------------
RF_WTI = 88_011  # forward node, quoted USD
RF_GOLD = 12  # option underlying forward, quoted USD
RF_SILVER = 400_003  # option underlying forward, quoted USD
RF_VOL_GOLD = 91_004  # vol node
RF_VOL_SILVER = 6  # vol node
RF_FX_USD_EUR = 501  # USD -> EUR, quoted direct
RF_FX_EUR_GBP = 40_009  # EUR -> GBP, quoted REVERSE (inv = 1 gives GBP -> EUR)

CURVE_USD = 900_001
CURVE_GBP = 31

ORD_DELIVERY = 20_450  # forward delivery
ORD_EXP_A = 20_365  # gold option expiry   (~1.0y)
ORD_EXP_B = 20_200  # silver option expiry (~0.55y)
ORD_SETTLE_LATE = 20_370  # silver option settles AFTER its expiry

FX_USD_EUR = np.array([0.91370, 0.90118, 0.92744])
FX_EUR_GBP = np.array([0.84655, 0.85901, 0.83402])  # GBP->EUR = 1 / this

LVL_WTI = np.array([78.44, 88.12, 59.37])
LVL_GOLD = np.array([2410.55, 2688.20, 2101.40])
LVL_SILVER = np.array([28.115, 31.44, 24.07])

VOL_GOLD = np.array([0.2150, 0.2795, 0.1830])
VOL_SILVER = np.array([0.3410, 0.4102, 0.2988])

DF_USD_DELIVERY = np.array([0.96150, 0.95022, 0.97311])
DF_USD_EXP_A = np.array([0.96410, 0.95380, 0.97540])
DF_USD_EXP_B = np.array([0.98020, 0.97455, 0.98630])
DF_GBP_EXP_A = np.array([0.94270, 0.93115, 0.95402])
DF_GBP_EXP_B = np.array([0.96880, 0.96215, 0.97540])
DF_GBP_SETTLE_LATE = np.array([0.94180, 0.93020, 0.95315])
DF_GBP_DELIVERY = np.array([0.93110, 0.91905, 0.94290])

_DAYS = 365.25


def _tte(ordinal: int) -> float:
    return (ordinal - AS_OF_ORD) / _DAYS


# =========================================================================
# ECONOMIC SIDE -- consumed by the independent oracle
# =========================================================================


def economic_market() -> EconomicMarket:
    return EconomicMarket(
        n_scenarios=N_SCENARIOS,
        spot_native={},
        forward_native={
            RF_WTI: LVL_WTI,
            RF_GOLD: LVL_GOLD,
            RF_SILVER: LVL_SILVER,
        },
        discount={  # type: ignore[arg-type]
            (CURVE_USD, ORD_DELIVERY): DF_USD_DELIVERY,
            (CURVE_USD, ORD_EXP_A): DF_USD_EXP_A,
            (CURVE_USD, ORD_EXP_B): DF_USD_EXP_B,
            (CURVE_GBP, ORD_EXP_A): DF_GBP_EXP_A,
            (CURVE_GBP, ORD_EXP_B): DF_GBP_EXP_B,
            (CURVE_GBP, ORD_SETTLE_LATE): DF_GBP_SETTLE_LATE,
            (CURVE_GBP, ORD_DELIVERY): DF_GBP_DELIVERY,
        },
        fx={("USD", "EUR"): FX_USD_EUR, ("EUR", "GBP"): FX_EUR_GBP},
        vol={RF_VOL_GOLD: VOL_GOLD, RF_VOL_SILVER: VOL_SILVER},
        forward_rate={},
    )


def economic_book() -> list:
    """Four economic trades, in the order their block rows will appear."""
    return [
        # FORWARD block -- long WTI, USD asset leg, GBP strike leg
        ForwardTrade(
            trade_uid=9007,
            qty=250.0,
            forward_factor_id=RF_WTI,
            asset_ccy="USD",
            asset_discount_factor_id=(CURVE_USD, ORD_DELIVERY),  # type: ignore[arg-type]
            strike=61.25,
            strike_ccy="GBP",
            strike_discount_factor_id=(CURVE_GBP, ORD_DELIVERY),  # type: ignore[arg-type]
        ),
        # OPTION block, row 0 -- SHORT gold CALL, USD asset / GBP strike / EUR settle,
        # settlement on the expiry date.
        CrossCurrencyOptionTrade(
            trade_uid=41,
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
        # OPTION block, row 1 -- LONG silver PUT, DEFERRED settlement: the
        # discount row is the settlement date, not the expiry date.
        CrossCurrencyOptionTrade(
            trade_uid=120003,
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
        # OPTION block, row 2 -- LONG gold PUT, same underlying as row 0 but a
        # different strike, so a block-row permutation is visible.
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
    RF_WTI,
    RF_GOLD,
    RF_SILVER,
    RF_VOL_GOLD,
    RF_VOL_SILVER,
    RF_FX_USD_EUR,
    RF_FX_EUR_GBP,
])
CURVE_ROWS = sorted([CURVE_USD, CURVE_GBP])
DATE_ROWS = sorted([ORD_DELIVERY, ORD_EXP_A, ORD_EXP_B, ORD_SETTLE_LATE])


def ingest_batch():
    """A real `IngestBatch`: one forward, three options, hostile uids."""
    from riskquant.portfolio.ingest.batch import IngestBatch

    def i32(x):
        return np.ascontiguousarray(x, dtype=np.int32)

    def i8(x):
        return np.ascontiguousarray(x, dtype=np.int8)

    def f64(x):
        return np.ascontiguousarray(x, dtype=np.float64)

    _opt = VANILLA_EUROPEAN_OPTION_NODE_ID
    _fwd = VANILLA_FORWARD_NODE_ID

    return IngestBatch(
        size=4,
        instrument_ids=i32([_fwd, _opt, _opt, _opt]),
        uids=np.ascontiguousarray([9007, 41, 120003, 5], dtype=np.int64),
        qty=f64([250.0, -80.0, 600.0, 140.0]),
        strike_price=f64([61.25, 2500.0, 30.00, 2300.0]),
        cp_mult=i8([0, 1, -1, -1]),
        settle_ord=i32([ORD_DELIVERY, ORD_EXP_A, ORD_SETTLE_LATE, ORD_EXP_A]),
        delivery_ord=i32([ORD_DELIVERY, ORD_EXP_A, ORD_EXP_B, ORD_EXP_A]),
        expiry_ord=i32([ORD_DELIVERY, ORD_EXP_A, ORD_EXP_B, ORD_EXP_A]),
        underlying_ids=i32([RF_WTI, RF_GOLD, RF_SILVER, RF_GOLD]),
        volatility_ids=i32([-1, RF_VOL_GOLD, RF_VOL_SILVER, RF_VOL_GOLD]),
        # asset -> settle FX: every underlying is USD quoted
        fx_u2s_ids=i32([RF_FX_USD_EUR] * 4),
        fx_u2s_inv=i8([0, 0, 0, 0]),
        # strike -> settle FX: every strike is GBP, quoted EUR->GBP so inverted
        fx_k2s_ids=i32([RF_FX_EUR_GBP] * 4),
        fx_k2s_inv=i8([1, 1, 1, 1]),
        under_discount_curve_ids=i32([CURVE_USD] * 4),
        strike_discount_curve_ids=i32([CURVE_GBP] * 4),
        vol_surface_y_axis_type_ids=i8([-99, -99, -99, -99]),
        netting_set_id=np.ascontiguousarray([1, 1, 1, 1], dtype=np.int64),
    )


def build_market_matrix(uix_mapper) -> MarketDataMatrix:
    """Materialise the raw tier against the run's actual effective-FX table."""
    s = N_SCENARIOS
    market = np.array(MARKET_ROWS, dtype=np.int32)
    curves = np.array(CURVE_ROWS, dtype=np.int32)
    dates = np.array(DATE_ROWS, dtype=np.int32)

    # --- node levels (forwards AND vols share the market_id_universe) ---
    levels = {
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

    # --- discount factor grid, (curve, date, scenario) ---
    df = np.ones((curves.shape[0], dates.shape[0], s), dtype=np.float64)
    grid = {
        (CURVE_USD, ORD_DELIVERY): DF_USD_DELIVERY,
        (CURVE_USD, ORD_EXP_A): DF_USD_EXP_A,
        (CURVE_USD, ORD_EXP_B): DF_USD_EXP_B,
        (CURVE_GBP, ORD_EXP_A): DF_GBP_EXP_A,
        (CURVE_GBP, ORD_EXP_B): DF_GBP_EXP_B,
        (CURVE_GBP, ORD_SETTLE_LATE): DF_GBP_SETTLE_LATE,
        (CURVE_GBP, ORD_DELIVERY): DF_GBP_DELIVERY,
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

    tte = np.ascontiguousarray(
        np.array([_tte(int(d)) for d in dates], dtype=np.float64)
    )

    return MarketDataMatrix(
        num_scenarios=s,
        fx_spots_effective=fx_eff,
        prices=prices,
        forward_nodes=prices,
        vol_nodes=prices,
        discount_factor_grid=df,
        price_curve_grid=np.zeros((market.shape[0], dates.shape[0], s)),
        tte_grid=tte,
        # NODE options only: no surfaces, but the structural guard still anchors
        # every date axis to len(tte_grid).
        vol_surface_strikes=np.zeros((0, 0)),
        vol_surfaces=np.ascontiguousarray(np.zeros((0, dates.shape[0], s, 0))),
        surface_y_axis_type_ids=np.zeros(0, dtype=np.int8),
    )
