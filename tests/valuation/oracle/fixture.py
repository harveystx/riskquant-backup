# tests/valuation/oracle/fixture.py
"""
The hostile, identity-sensitive valuation fixture.

One economic book, described twice:

    ECONOMIC SIDE   -> `economic_book()`   consumed by the independent oracle
    PRODUCTION SIDE -> `compile_book()`    consumed by `run_valuation`

The production side is a TEST-OWNED stand-in for the block compilers that
`portfolio/compilers/registry.py` has not yet authored (only SPOT is
registered). It is the only place in this suite that maps a factor id onto an
array row; the oracle never sees a row index.

Hostility requirements (D045 Phase 3 / section 12), all satisfied here:

    trade_uids            [9007, 41, 120003, 5]     non-ascending, sparse
    factor ids            sparse, non-ascending, id != row position everywhere
    currencies            USD, GBP, EUR (settlement)
    FX                    non-unit in both directions; EUR/USD quoted the
                          REVERSE way round so the oracle must invert it
    discount factors      non-unit, all distinct
    quantities            mixed long and short
    block types           spot, vanilla forward, vanilla european option
    maturities            distinct per option row
    market values         deliberately asymmetric across every universe

The asymmetry is the point: every factor universe below has rows of clearly
different magnitude, so permuting any two rows moves the PV.
"""

from __future__ import annotations

import numpy as np
from oracle.economic_oracle import (
    EconomicMarket,
    ForwardTrade,
    OptionTrade,
    SpotTrade,
)

from riskquant.market_data.state import ValuationState
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.valuation.blocks.spot_block import SpotBlock
from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionBlock,
)
from riskquant.valuation.blocks.vanilla_forward_block import VanillaForwardBlock

SETTLE_CCY = "EUR"
N_SCENARIOS = 3

# --- factor identities -------------------------------------------------------
# Sparse and non-ascending. The ROW ORDER below is deliberately NOT the sorted
# order of these ids, so `factor_id == row_position` is false for every factor.
SPOT_BRENT = 7001  # quoted USD
SPOT_ALU = 400003  # quoted GBP   (present but unused by any trade)
SPOT_TTF = 12  # quoted EUR   (present but unused by any trade)

FWD_WTI = 88011  # quoted USD
FWD_COPPER = 3  # quoted GBP   (present but unused by any trade)

FWD_K_GOLD = 50021  # option underlying, already in STRIKE ccy (USD)
FWD_K_SILVER = 17  # option underlying, already in STRIKE ccy (GBP)

VOL_GOLD = 91004
VOL_SILVER = 6

DF_USD_1Y = 220007  # 0.9615
DF_GBP_1Y = 31  # 0.9427
DF_EUR_2Y = 750019  # 0.8812
DF_USD_18M = 404  # 0.9401

# --- FX ---------------------------------------------------------------------
# USD->EUR quoted directly; GBP quoted the REVERSE way (EUR->GBP), so the
# oracle must invert it. Valuation itself never inverts: the fold pre-resolves.
FX_USD_EUR = 0.91370
FX_EUR_GBP = 0.84655  # => GBP->EUR = 1 / 0.84655 = 1.18127...


def economic_market() -> EconomicMarket:
    s = N_SCENARIOS

    def v(*xs: float) -> np.ndarray:
        assert len(xs) == s
        return np.array(xs, dtype=np.float64)

    return EconomicMarket(
        n_scenarios=s,
        spot_native={
            SPOT_BRENT: v(82.30, 91.77, 61.05),  # USD
            SPOT_ALU: v(2451.0, 2199.5, 2610.25),  # GBP
            SPOT_TTF: v(34.115, 41.90, 28.605),  # EUR
        },
        forward_native={
            FWD_WTI: v(78.44, 88.12, 59.37),  # USD
            FWD_COPPER: v(9120.5, 8455.0, 9899.75),  # GBP
            FWD_K_GOLD: v(2410.55, 2688.20, 2101.40),  # USD (strike ccy)
            FWD_K_SILVER: v(28.115, 31.44, 24.07),  # GBP (strike ccy)
        },
        discount={
            DF_USD_1Y: v(0.96150, 0.95022, 0.97311),
            DF_GBP_1Y: v(0.94270, 0.93115, 0.95402),
            DF_EUR_2Y: v(0.88120, 0.86033, 0.90277),
            DF_USD_18M: v(0.94010, 0.92588, 0.95466),
        },
        fx={
            ("USD", "EUR"): v(FX_USD_EUR, 0.90118, 0.92744),
            ("EUR", "GBP"): v(FX_EUR_GBP, 0.85901, 0.83402),  # REVERSE quote
        },
        vol={
            VOL_GOLD: v(0.2150, 0.2795, 0.1830),
            VOL_SILVER: v(0.3410, 0.4102, 0.2988),
        },
        forward_rate={},
    )


def economic_book() -> list:
    """The four trades, in the order their block rows will appear."""
    return [
        # row 0 -- SHORT Brent spot, quoted USD, settled EUR
        SpotTrade(
            trade_uid=9007, qty=-1500.0, spot_factor_id=SPOT_BRENT, quote_ccy="USD"
        ),
        # row 1 -- LONG WTI forward, asset USD, strike GBP: both legs differ
        ForwardTrade(
            trade_uid=41,
            qty=250.0,
            forward_factor_id=FWD_WTI,
            asset_ccy="USD",
            asset_discount_factor_id=DF_USD_1Y,
            strike=61.25,
            strike_ccy="GBP",
            strike_discount_factor_id=DF_GBP_1Y,
        ),
        # row 2 -- SHORT gold call, strike ccy USD, settled EUR
        OptionTrade(
            trade_uid=120003,
            qty=-80.0,
            is_call=True,
            strike=2500.0,
            strike_ccy="USD",
            tte_years=1.0,
            forward_strike_ccy_factor_id=FWD_K_GOLD,
            vol_factor_id=VOL_GOLD,
            settle_discount_factor_id=DF_USD_18M,
        ),
        # row 3 -- LONG silver put, strike ccy GBP (inverse-quoted FX)
        OptionTrade(
            trade_uid=5,
            qty=600.0,
            is_call=False,
            strike=30.00,
            strike_ccy="GBP",
            tte_years=0.5,
            forward_strike_ccy_factor_id=FWD_K_SILVER,
            vol_factor_id=VOL_SILVER,
            settle_discount_factor_id=DF_GBP_1Y,
        ),
    ]


# =========================================================================
# PRODUCTION SIDE -- test-owned stand-in for the unwritten block compilers
# =========================================================================

# Row layout of each factor universe. DELIBERATELY not sorted by id.
SPOT_ROWS = [SPOT_ALU, SPOT_TTF, SPOT_BRENT]  # Brent at row 2
FWD_ROWS = [FWD_COPPER, FWD_WTI]  # WTI at row 1
FWDK_ROWS = [FWD_K_SILVER, FWD_K_GOLD]  # gold at row 1
VOL_ROWS = [VOL_SILVER, VOL_GOLD]  # gold at row 1
DF_ROWS = [DF_EUR_2Y, DF_USD_18M, DF_GBP_1Y, DF_USD_1Y]  # USD 1y at row 3


def _row(universe: list[int], factor_id: int) -> int:
    return universe.index(factor_id)


def compile_state(m: EconomicMarket) -> ValuationState:
    """Fold the economic market into the pre-multiplied effective factors."""

    def fx(frm: str) -> np.ndarray:
        return m.fx_rate(frm, SETTLE_CCY)

    # effective_spot_prices[r] = S_native * FX(native -> settle)
    spot_ccy = {SPOT_BRENT: "USD", SPOT_ALU: "GBP", SPOT_TTF: "EUR"}
    eff_spot = np.ascontiguousarray(
        np.stack([m.spot_native[f] * fx(spot_ccy[f]) for f in SPOT_ROWS])
    )

    # effective_discount_factors[r] = DF_ccy * FX(ccy -> settle)
    df_ccy = {DF_USD_1Y: "USD", DF_GBP_1Y: "GBP", DF_EUR_2Y: "EUR", DF_USD_18M: "USD"}
    eff_df = np.ascontiguousarray(
        np.stack([m.discount[f] * fx(df_ccy[f]) for f in DF_ROWS])
    )

    # effective_forward_prices[r] = F_native * DF_native * FX(native -> settle)
    fwd_ccy = {FWD_WTI: "USD", FWD_COPPER: "GBP"}
    fwd_df = {FWD_WTI: DF_USD_1Y, FWD_COPPER: DF_GBP_1Y}
    eff_fwd = np.ascontiguousarray(
        np.stack([
            m.forward_native[f] * m.discount[fwd_df[f]] * fx(fwd_ccy[f])
            for f in FWD_ROWS
        ])
    )

    fwd_k = np.ascontiguousarray(np.stack([m.forward_native[f] for f in FWDK_ROWS]))
    log_fwd_k = np.ascontiguousarray(np.log(fwd_k))
    vols = np.ascontiguousarray(np.stack([m.vol[f] for f in VOL_ROWS]))

    return ValuationState(
        batch_size=m.n_scenarios,
        effective_spot_prices=eff_spot,
        effective_discount_factors=eff_df,
        effective_forward_prices=eff_fwd,
        forward_prices_strike_ccy=fwd_k,
        option_log_forward_prices_strike_ccy=log_fwd_k,
        vols=vols,
    )


def compile_book() -> CompiledPortfolio:
    """Compile the same four economic trades into SoA blocks."""
    spot = SpotBlock(
        trade_uids=np.array([9007], dtype=np.int64),
        qty=np.array([-1500.0], dtype=np.float64),
        effective_spot_price_uix=np.array(
            [_row(SPOT_ROWS, SPOT_BRENT)], dtype=np.int32
        ),
    )
    fwd = VanillaForwardBlock(
        trade_uids=np.array([41], dtype=np.int64),
        qty=np.array([250.0], dtype=np.float64),
        strike=np.array([61.25], dtype=np.float64),
        effective_forward_price_uix=np.array([_row(FWD_ROWS, FWD_WTI)], dtype=np.int32),
        effective_discount_factor_uix=np.array(
            [_row(DF_ROWS, DF_GBP_1Y)], dtype=np.int32
        ),
    )
    strikes = np.array([2500.0, 30.00], dtype=np.float64)
    tte = np.array([1.0, 0.5], dtype=np.float64)
    opt = VanillaEuropeanOptionBlock(
        trade_uids=np.array([120003, 5], dtype=np.int64),
        qty=np.array([-80.0, 600.0], dtype=np.float64),
        strike=strikes,
        log_strike=np.log(strikes),
        cp_mult=np.array([1, -1], dtype=np.int8),
        sqrt_tte=np.sqrt(tte),
        half_tte=0.5 * tte,
        forward_price_strike_ccy_uix=np.array(
            [_row(FWDK_ROWS, FWD_K_GOLD), _row(FWDK_ROWS, FWD_K_SILVER)], dtype=np.int32
        ),
        option_log_forward_price_strike_ccy_uix=np.array(
            [_row(FWDK_ROWS, FWD_K_GOLD), _row(FWDK_ROWS, FWD_K_SILVER)], dtype=np.int32
        ),
        vol_uix=np.array(
            [_row(VOL_ROWS, VOL_GOLD), _row(VOL_ROWS, VOL_SILVER)], dtype=np.int32
        ),
        effective_discount_factor_uix=np.array(
            [_row(DF_ROWS, DF_USD_18M), _row(DF_ROWS, DF_GBP_1Y)], dtype=np.int32
        ),
    )
    return CompiledPortfolio(
        unique_dates_ord32=np.array([], dtype=np.int32),
        spot=spot,
        vanilla_forward=fwd,
        vanilla_european_option=opt,
    )
