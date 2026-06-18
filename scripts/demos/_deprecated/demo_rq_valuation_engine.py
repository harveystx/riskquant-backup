# scripts/demo_rq_valuation_engine.py
import time
from datetime import date

import numpy as np

# Import Constants
from riskquant.constants import (
    CALL_OPTION_MULT,
    NO_DATE_UIX,
    NO_DELIVERY_DATE_ORD32,
    NO_EXPIRY_DATE_ORD32,
    NO_FX_RF_REQUIRED,
    NO_SETTLEMENT_DATE_ORD32,
    NOT_OPTION_MULT,
    PUT_OPTION_MULT,
)
from riskquant.market_data.builder import build_market_data_matrix
from riskquant.market_data.curves.extrap_methods import EXTRAP_FLAT

# Import interpolation/extrapolation method constants
from riskquant.market_data.curves.interp_methods import INTERP_LINEAR

# Import tuple of Y-Axis Types
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
    InstrumentType,
)
from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant.valuation.engine import run_valuation


def run_full_engine_demo():
    print_calling_file()

    print("===================================================================")
    print(" 🚀 RISKQUANT ENGINE DEMO: END-TO-END PRICING & VECTORIZED PNL    ")
    print("===================================================================")

    t0 = time.perf_counter() if hasattr(time, "perf_counter") else time.time()

    print("\n[1/4] Constructing SoA IngestBatch (14 Permutations)...")

    # Construct symmetric long/short pairs across all products
    n_trades = 14
    instrument_ids = np.array(
        [
            SPOT_ID,
            SPOT_ID,
            VANILLA_FORWARD_NODE_ID,
            VANILLA_FORWARD_NODE_ID,
            VANILLA_FORWARD_CURVE_ID,
            VANILLA_FORWARD_CURVE_ID,
            VANILLA_EUROPEAN_OPTION_NODE_ID,
            VANILLA_EUROPEAN_OPTION_NODE_ID,
            VANILLA_EUROPEAN_OPTION_NODE_ID,
            VANILLA_EUROPEAN_OPTION_NODE_ID,
            VANILLA_EUROPEAN_OPTION_CURVE_ID,
            VANILLA_EUROPEAN_OPTION_CURVE_ID,
            VANILLA_EUROPEAN_OPTION_CURVE_ID,
            VANILLA_EUROPEAN_OPTION_CURVE_ID,
        ],
        dtype=np.int32,
    )

    # Economics
    uids = np.arange(101, 115, dtype=np.int64)

    qty = np.array(
        [
            100.0,
            -100.0,  # Spot
            50.0,
            -50.0,  # Fwd Node
            10.0,
            -10.0,  # Fwd Curve
            100.0,
            -100.0,
            100.0,
            -100.0,  # Opt Node (Call L, Call S, Put L, Put S)
            50.0,
            -50.0,
            50.0,
            -50.0,  # Opt Curve (Call L, Call S, Put L, Put S)
        ],
        dtype=np.float64,
    )

    strike_price = np.array(
        [
            0.0,
            0.0,
            105.0,
            105.0,
            55.0,
            55.0,
            100.0,
            100.0,
            100.0,
            100.0,
            60.0,
            60.0,
            60.0,
            60.0,
        ],
        dtype=np.float64,
    )

    is_call = np.array(
        [
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            NOT_OPTION_MULT,
            1,  # IS CALL
            1,  # IS CALL
            0,  # IS PUT
            0,  # IS PUT
            1,  # IS CALL
            1,  # IS CALL
            0,  # IS PUT
            0,  # IS PUT
        ],
        dtype=np.int8,
    )

    # Date Grid Setup
    unique_dates = np.array([20600, 20650], dtype=np.int32)
    EXP_UIX, STL_UIX = 0, 1

    settle_uix = np.array([NO_DATE_UIX] * 2 + [STL_UIX] * 12, dtype=np.int32)
    delivery_uix = np.array(
        [NO_DATE_UIX] * 4 + [STL_UIX] * 2 + [NO_DATE_UIX] * 4 + [STL_UIX] * 4,
        dtype=np.int32,
    )
    expiry_uix = np.array([NO_DATE_UIX] * 6 + [EXP_UIX] * 8, dtype=np.int32)

    # Map all lookups to index 0 for simplicity, disable FX
    valid_uix = np.zeros(n_trades, dtype=np.int32)
    no_fx = np.full(n_trades, NO_FX_RF_REQUIRED, dtype=np.int32)
    false_flags = np.zeros(n_trades, dtype=np.int32)
    y_axis_types = np.full(n_trades, SURFACE_Y_AXIS_TYPE_STRIKE, dtype=np.int8)

    cp_mult = np.where(
        is_call == 1,
        CALL_OPTION_MULT,
        np.where(is_call == 0, PUT_OPTION_MULT, NOT_OPTION_MULT),
    )

    batch = IngestBatch(
        size=n_trades,
        instrument_ids=instrument_ids,
        uids=uids,
        qty=qty,
        strike_price=strike_price,
        cp_mult=cp_mult,
        ead=np.zeros(n_trades, dtype=np.float64),
        pd=np.zeros(n_trades, dtype=np.float64),
        lgd=np.zeros(n_trades, dtype=np.float64),
        beta=np.zeros(n_trades, dtype=np.float64),
        systemic_uix=valid_uix,
        idiosyncratic_uix=valid_uix,
        chi_squared_uix=valid_uix,
        settle_ord=np.full(n_trades, NO_SETTLEMENT_DATE_ORD32),
        delivery_ord=np.full(n_trades, NO_DELIVERY_DATE_ORD32),
        expiry_ord=np.full(n_trades, NO_EXPIRY_DATE_ORD32),
        unique_dates_ord=unique_dates,
        settle_date_uix=settle_uix,
        delivery_date_uix=delivery_uix,
        expiry_date_uix=expiry_uix,
        fx_under_to_settle_ids=valid_uix,
        fx_strike_to_settle_ids=no_fx,
        fx_under_to_settle_invert=false_flags,
        fx_strike_to_settle_invert=false_flags,
        price_rf_ids=valid_uix,
        price_curve_ids=valid_uix,
        vol_rf_ids=valid_uix,
        vol_surface_rf_ids=valid_uix,
        under_discount_curve_ids=valid_uix,
        strike_discount_curve_ids=valid_uix,
        vol_surface_y_axis_type_ids=y_axis_types,
    )

    # Create mapping of what we expect for the printout later
    trade_desc_map = {}
    for i in range(n_trades):
        uid = uids[i]
        inst_name = InstrumentType(instrument_ids[i]).name
        pos = "Long" if qty[i] > 0 else "Short"
        if is_call[i] == 1:
            pos += " Call"
        elif is_call[i] == 0:
            pos += " Put"
        trade_desc_map[uid] = (inst_name.replace("_", " ").title(), pos)

    print("\n[2/4] Compiling Portfolio Blocks...")
    portfolio = compile_portfolio_from_ingest_batch(batch)
    print(
        f"      Mapped "
        f"{portfolio.total_trades} trades across "
        f"{portfolio.total_blocks} memory-contiguous blocks."
    )

    print("\n[3/4] Building 3D/4D MarketDataMatrix...")
    S = 2  # 2 Scenarios: Base, Shocked
    fx_spots = np.array([[1.05, 1.10]], dtype=np.float64)
    prices = np.array([[100.0, 110.0]], dtype=np.float64)
    fwd_nodes = np.array([[102.0, 112.0]], dtype=np.float64)
    vol_nodes = np.array([[0.20, 0.25]], dtype=np.float64)

    curve_interp = np.array([INTERP_LINEAR], dtype=np.int8)
    curve_extrap = np.array([EXTRAP_FLAT], dtype=np.int8)

    disc_tenors = np.array([[20500, 20800]], dtype=np.float64)
    disc_ln_dfs = np.log(np.array([[[0.98, 0.97], [0.90, 0.88]]], dtype=np.float64))

    price_tenors = np.array([[20500, 20800]], dtype=np.float64)
    price_vals = np.array([[[50.0, 52.0], [55.0, 58.0]]], dtype=np.float64)

    surf_interp = np.array([INTERP_LINEAR], dtype=np.int8)
    surf_extrap = np.array([EXTRAP_FLAT], dtype=np.int8)
    surf_y_axis = np.array([SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8)

    surf_tenors = np.array([[20500, 20800]], dtype=np.float64)
    surf_strikes = np.array([[40.0, 50.0, 60.0]], dtype=np.float64)
    surf_vals = np.zeros((1, 2, 3, 2), dtype=np.float64)
    surf_vals[0, :, :, 0] = [[0.22, 0.20, 0.22], [0.24, 0.22, 0.24]]
    surf_vals[0, :, :, 1] = [[0.27, 0.25, 0.27], [0.29, 0.27, 0.29]]

    mdm = build_market_data_matrix(
        as_of_date=date(2026, 1, 1),
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

    print("\n[4/4] Executing Hot-Loop C-Kernel Pricing...")
    results = run_valuation(portfolio=portfolio, valuation_state=mdm)

    t1 = time.perf_counter() if hasattr(time, "perf_counter") else time.time()
    calc_time_ms = (t1 - t0) * 1000

    print("\n" + "=" * 107)
    print(
        f" {'UID':<4} |"
        f" {'Instrument':<40} |"
        f" {'Position':<10} |"
        f" {'Base PV':>12} |"
        f" {'Shocked PV':>12} |"
        f" {'PnL (Var)':>12}"
    )
    print("-" * 107)

    compiled_uids = []
    for block in portfolio.iter_blocks():
        compiled_uids.extend(block.trade_uids.tolist())

    tot_base = 0.0
    tot_shock = 0.0
    tot_pnl = 0.0

    for i, pvs in enumerate(results):
        uid = compiled_uids[i]
        inst_name, pos = trade_desc_map[uid]

        base_pv = pvs[0]
        shock_pv = pvs[1]
        pnl = shock_pv - base_pv

        tot_base += base_pv
        tot_shock += shock_pv
        tot_pnl += pnl

        # Format strings safely
        bpv_str = f"{base_pv:>12,.2f}"
        spv_str = f"{shock_pv:>12,.2f}"
        pnl_str = f"{pnl:>+12,.2f}"

        print(
            f" {uid:<4} |"
            f" {inst_name:<40} |"
            f" {pos:<10} |"
            f" {bpv_str} |"
            f" {spv_str} |"
            f" {pnl_str}"
        )

    print("=" * 107)
    print(
        f" {'TOTAL PORTFOLIO AGGREGATION':<60} |"
        f" {tot_base:>12,.2f} |"
        f" {tot_shock:>12,.2f} |"
        f" {tot_pnl:>+12,.2f}"
    )
    print("=" * 107)
    print(f"\nEngine executed successfully in {calc_time_ms:.2f} ms")

    print_success_message()


if __name__ == "__main__":
    run_full_engine_demo()
