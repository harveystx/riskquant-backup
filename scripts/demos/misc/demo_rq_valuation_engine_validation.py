# scripts/demo_rq_valuation_engine_validation.py

import math
import time
from datetime import date

import numpy as np
from scipy.stats import norm

# Import Constants
from riskquant.constants import (
    CALL_OPTION_MULT,
    NO_DATE_UIX,
    NO_DELIVERY_DATE_ORD32,
    NO_DISCOUNT_CURVE_ID,
    NO_EXPIRY_DATE_ORD32,
    NO_FX_RF_REQUIRED,
    NO_SETTLEMENT_DATE_ORD32,
    NOT_OPTION_MULT,
    PUT_OPTION_MULT,
)
from riskquant.market_data.builder import build_market_data_matrix
from riskquant.market_data.curves.extrap_methods import EXTRAP_FLAT
from riskquant.market_data.curves.interp_methods import INTERP_LINEAR
from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_STRIKE,
)
from riskquant.portfolio.engine import compile_portfolio_from_ingest_batch

# Import Core Engine Components
from riskquant.portfolio.ingest.batch import IngestBatch

# Import Instrument IDs
from riskquant.portfolio.instruments import (
    SPOT_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_CURVE_ID,
    VANILLA_FORWARD_NODE_ID,
)
from riskquant.utils.dates import date_to_ord32
from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant.valuation.engine import run_valuation


def exact_bs_pv(fwd, strike, vol, tte, df, is_call, qty):
    """Calculates the exact expected Black-Scholes PV for validation."""
    cp = 1.0 if is_call else -1.0
    std_dev = vol * math.sqrt(tte)
    d1 = (math.log(fwd / strike) + 0.5 * (vol**2) * tte) / std_dev
    d2 = d1 - std_dev
    fv = cp * (fwd * norm.cdf(cp * d1) - strike * norm.cdf(cp * d2))
    return qty * fv * df


def run_validation_demo():
    print_calling_file()

    print("===================================================================")
    print(" 🧪 RISKQUANT ENGINE: DETERMINISTIC PV VALIDATION SUITE")
    print("===================================================================")

    t0 = time.perf_counter() if hasattr(time, "perf_counter") else time.time()

    # --- 1. SETUP DATES ---
    as_of = date(2026, 1, 1)
    as_of_ord = date_to_ord32(as_of)

    # Target exactly 1 Year (365.25 days) for clean math
    target_ord = as_of_ord + 365.25
    target_ord_int = int(target_ord)
    tte = (target_ord_int - as_of_ord) / 365.25  # Engine's exact TTE

    unique_dates = np.array([target_ord_int], dtype=np.int32)
    T_UIX = 0

    # --- 2. SETUP PORTFOLIO ---
    n_trades = 6
    instrument_ids = np.array(
        [
            SPOT_ID,
            SPOT_ID,
            VANILLA_FORWARD_NODE_ID,
            VANILLA_FORWARD_CURVE_ID,
            VANILLA_EUROPEAN_OPTION_NODE_ID,
            VANILLA_EUROPEAN_OPTION_CURVE_ID,
        ],
        dtype=np.int32,
    )

    uids = np.arange(101, 107, dtype=np.int64)

    # 1: Long Spot (No FX)
    # 2: Short Spot (Inverted FX)
    # 3: Long Fwd Node (Discounted)
    # 4: Short Fwd Curve (Discounted)
    # 5: Long Call Node (Discounted)
    # 6: Short Put Curve (Discounted)
    qty = np.array([100.0, -100.0, 10.0, -50.0, 1000.0, -500.0], dtype=np.float64)
    strike_price = np.array([0.0, 0.0, 90.0, 105.0, 100.0, 115.0], dtype=np.float64)
    cp_mult = np.array(
        [
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            CALL_OPTION_MULT,
            PUT_OPTION_MULT,
        ],
        dtype=np.int8,
    )

    settle_uix = np.array(
        [NO_DATE_UIX, NO_DATE_UIX, T_UIX, T_UIX, T_UIX, T_UIX], dtype=np.int32
    )
    delivery_uix = np.array(
        [NO_DATE_UIX, NO_DATE_UIX, NO_DATE_UIX, T_UIX, NO_DATE_UIX, T_UIX],
        dtype=np.int32,
    )
    expiry_uix = np.array(
        [NO_DATE_UIX, NO_DATE_UIX, NO_DATE_UIX, NO_DATE_UIX, T_UIX, T_UIX],
        dtype=np.int32,
    )

    # UIX Mappings
    # FX: 0 = Standard (1.0), 1 = Inverted Test (1.25)
    fx_uix = np.array(
        [
            NO_FX_RF_REQUIRED,
            1,
            NO_FX_RF_REQUIRED,
            NO_FX_RF_REQUIRED,
            NO_FX_RF_REQUIRED,
            NO_FX_RF_REQUIRED,
        ],
        dtype=np.int32,
    )
    fx_inv = np.array([0, 1, 0, 0, 0, 0], dtype=np.int32)

    # Curves/Nodes all point to index 0
    zero_uix = np.zeros(n_trades, dtype=np.int32)
    df_uix = np.array(
        [NO_DISCOUNT_CURVE_ID, NO_DISCOUNT_CURVE_ID, 0, 0, 0, 0], dtype=np.int32
    )

    y_axis_types = np.full(n_trades, SURFACE_Y_AXIS_TYPE_STRIKE, dtype=np.int8)
    zero_f64 = np.zeros(n_trades, dtype=np.float64)

    batch = IngestBatch(
        size=n_trades,
        instrument_ids=instrument_ids,
        uids=uids,
        qty=qty,
        strike_price=strike_price,
        cp_mult=cp_mult,
        settle_ord=np.full(n_trades, NO_SETTLEMENT_DATE_ORD32),
        delivery_ord=np.full(n_trades, NO_DELIVERY_DATE_ORD32),
        expiry_ord=np.full(n_trades, NO_EXPIRY_DATE_ORD32),
        unique_dates_ord=unique_dates,
        settle_date_uix=settle_uix,
        delivery_date_uix=delivery_uix,
        expiry_date_uix=expiry_uix,
        fx_under_to_settle_ids=fx_uix,
        fx_strike_to_settle_ids=np.full(n_trades, NO_FX_RF_REQUIRED, dtype=np.int32),
        fx_under_to_settle_invert=fx_inv,
        fx_strike_to_settle_invert=zero_uix,
        price_rf_ids=zero_uix,
        price_curve_ids=zero_uix,
        vol_rf_ids=zero_uix,
        vol_surface_rf_ids=zero_uix,
        under_discount_curve_ids=df_uix,
        strike_discount_curve_ids=df_uix,
        vol_surface_y_axis_type_ids=y_axis_types,
        ead=zero_f64,
        pd=zero_f64,
        lgd=zero_f64,
        beta=zero_f64,
        systemic_uix=zero_uix,
        idiosyncratic_uix=zero_uix,
        chi_squared_uix=zero_uix,
    )

    print("\n[1/3] Compiling Portfolio...")
    portfolio = compile_portfolio_from_ingest_batch(batch)

    # --- 3. SETUP EXACT MARKET DATA ---
    S = 1  # Deterministic single scenario

    # Scalars
    fx_spots = np.array(
        [[1.0], [1.25]], dtype=np.float64
    )  # Index 1 is 1.25 (inverted it will be 0.8)
    prices = np.array([[50.0]], dtype=np.float64)
    fwd_nodes = np.array([[100.0]], dtype=np.float64)
    vol_nodes = np.array([[0.20]], dtype=np.float64)

    # Discount Curve (DF = 0.95 exactly)
    curve_interp = np.array([INTERP_LINEAR], dtype=np.int8)
    curve_extrap = np.array([EXTRAP_FLAT], dtype=np.int8)
    disc_tenors = np.array([[target_ord_int]], dtype=np.float64)
    disc_ln_dfs = np.log(np.array([[[0.95]]], dtype=np.float64))

    # Price Curve (Fwd = 110.0 exactly)
    price_tenors = np.array([[target_ord_int]], dtype=np.float64)
    price_vals = np.array([[[110.0]]], dtype=np.float64)

    # Vol Surface (Vol = 0.25 exactly)
    surf_interp = np.array([INTERP_LINEAR], dtype=np.int8)
    surf_extrap = np.array([EXTRAP_FLAT], dtype=np.int8)
    surf_y_axis = np.array([SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8)
    surf_tenors = np.array([[target_ord_int]], dtype=np.float64)
    surf_strikes = np.array([[100.0, 120.0]], dtype=np.float64)
    surf_vals = np.array([[[[0.25], [0.25]]]], dtype=np.float64)

    print("[2/3] Building MarketDataMatrix...")
    mdm = build_market_data_matrix(
        as_of_date=as_of,
        portfolio=portfolio,
        num_scenarios=S,
        raw_fx_spots=fx_spots,
        raw_prices=prices,
        raw_forward_nodes=fwd_nodes,
        raw_vol_nodes=vol_nodes,
        curve_interp_methods=curve_interp,
        curve_extrap_methods=curve_extrap,
        raw_discount_tenors=disc_tenors,
        raw_discount_ln_dfs=disc_ln_dfs,
        raw_price_curve_tenors=price_tenors,
        raw_price_curve_values=price_vals,
        surface_interp_x_methods=surf_interp,
        surface_extrap_x_methods=surf_extrap,
        surface_y_axis_type_ids=surf_y_axis,
        raw_vol_surface_tenors=surf_tenors,
        raw_vol_surface_strikes=surf_strikes,
        raw_vol_surface_values=surf_vals,
    )

    # --- 4. PRE-CALCULATE EXPECTED EXACT PVS ---
    expected_pvs = {}
    # 101: Spot Long (Qty=100, P=50) -> 5000.0
    expected_pvs[101] = 100.0 * 50.0

    # 102: Spot Short Inverted FX (Qty=-100, P=50, FX=1/1.25=0.8) -> -4000.0
    expected_pvs[102] = -100.0 * 50.0 * (1.0 / 1.25)

    # 103: Fwd Node Long (Qty=10, F=100, K=90, DF=0.95) -> 10 * 10 * 0.95 = 95.0
    expected_pvs[103] = 10.0 * (100.0 - 90.0) * 0.95

    # 104: Fwd Curve Short (Qty=-50, F=110, K=105, DF=0.95) -> -50 * 5 * 0.95 = -237.5
    expected_pvs[104] = -50.0 * (110.0 - 105.0) * 0.95

    # 105: Opt Node Long Call (Qty=1000, F=100, K=100, V=0.20, DF=0.95)
    expected_pvs[105] = exact_bs_pv(
        fwd=100.0,
        strike=100.0,
        vol=0.20,
        tte=tte,
        df=0.95,
        is_call=True,
        qty=1000.0,
    )

    # 106: Opt Curve Short Put (Qty=-500, F=110, K=115, V=0.25, DF=0.95)
    expected_pvs[106] = exact_bs_pv(
        fwd=110.0,
        strike=115.0,
        vol=0.25,
        tte=tte,
        df=0.95,
        is_call=False,
        qty=-500.0,
    )

    trade_names = {
        101: "Spot (Standard)",
        102: "Spot (Inverted FX)",
        103: "Fwd Node (ITM)",
        104: "Fwd Curve (ITM)",
        105: "Opt Node (ATM Call)",
        106: "Opt Curve (ITM Put)",
    }

    print("[3/3] Executing Vectorized Pricer...")
    results = run_valuation(portfolio=portfolio, market_data_mtx=mdm)

    t1 = time.perf_counter() if hasattr(time, "perf_counter") else time.time()

    print("\n" + "=" * 90)
    print(
        f" {'UID':<4} | {'Instrument Detail':<25} | {'Expected PV':>15} | {'Engine PV':>15} | {'Error Diff':>12}"
    )
    print("-" * 90)

    compiled_uids = []
    for block in portfolio.iter_blocks():
        compiled_uids.extend(block.trade_uids.tolist())

    passed_all = True

    for i, pvs in enumerate(results):
        uid = compiled_uids[i]
        name = trade_names[uid]
        engine_pv = pvs[0]  # S=0
        exact_pv = expected_pvs[uid]

        diff = abs(engine_pv - exact_pv)

        # Tolerance check (floating point math creates tiny variances)
        is_match = diff < 1e-8
        if not is_match:
            passed_all = False

        marker = "✅" if is_match else "❌"

        print(
            f" {uid:<4} | {name:<25} | {exact_pv:>15,.4f} | {engine_pv:>15,.4f} | {diff:>10.2e} {marker}"
        )

    print("=" * 90)
    if passed_all:
        print(" 🏆 GOD-TIER VERIFICATION: PASSED (All precision errors < 1e-8)")
    else:
        print(" ⚠️ VERIFICATION FAILED: Found precision mismatch.")
    print("=" * 90)
    print(f"\nEngine executed successfully in {(t1 - t0) * 1000:.2f} ms")

    print_success_message()


if __name__ == "__main__":
    run_validation_demo()
