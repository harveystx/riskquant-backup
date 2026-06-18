# scripts/demos/valuation/verify_pricing_kernels.py

import numpy as np

# Import the kernels directly
from riskquant.valuation.pricer_kernels.spot import _kernel_price_spot
from riskquant.valuation.pricer_kernels._deprecated.vanilla_european_option_curve_v1 import (
    _kernel_price_vanilla_european_option_curve,
)
from riskquant.valuation.pricer_kernels._deprecated.vanilla_european_option_node_v1 import (
    _kernel_price_vanilla_european_option_node,
)
from riskquant.valuation.pricer_kernels._deprecated.vanilla_forward_node_v1 import (
    _kernel_price_vanilla_forward_node,
)

# Surface type constants mapping to y_kind dispatch
SURFACE_Y_AXIS_TYPE_STRIKE = 0


def test_spot_kernel():
    print("--- Testing Spot Kernel ---")
    m_trades, s_scenarios = 7, 1
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    # 1D Market Data
    prices = np.array([[100.0]], dtype=np.float64)  # p_idx = 0
    fx_rates = np.array([[0.5]], dtype=np.float64)  # f_idx = 0

    # Trade Semantics
    # Trade 0: Qty  0, Px 100, No FX  -> Expected    0.0
    # Trade 1: Qty  1, Px 100, No FX  -> Expected  100.0
    # Trade 2: Qty  1, Px 100, FX 0.5 -> Expected   50.0
    # Trade 3: Qty  2, Px 100, FX 0.5 -> Expected  100.0
    # Trade 4: Qty -1, Px 100, No FX  -> Expected -100.0
    # Trade 5: Qty -1, Px 100, FX 0.5 -> Expected  -50.0
    # Trade 6: Qty -2, Px 100, FX 0.5 -> Expected -100.0
    qty = np.array([0.0, 1.0, 1.0, 2.0, -1.0, -1.0, -2.0], dtype=np.float64)
    price_uix = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    fx_uix = np.array([-1, -1, 0, 0, -1, 0, 0], dtype=np.int32)  # -1 skips FX branch

    _kernel_price_spot(out, prices, price_uix, qty, fx_rates, fx_uix)

    np.testing.assert_allclose(
        out[:, 0], [0.0, 100.0, 50.0, 100.0, -100.0, -50.0, -100.0]
    )
    print("Spot Kernel: PASS\n")


def test_forward_node_kernel():
    print("--- Testing Forward Node Kernel ---")
    m_trades, s_scenarios = 5, 1
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    forward_nodes = np.array([[120.0]], dtype=np.float64)
    fx_rates = np.array([[1.0]], dtype=np.float64)
    df_grid = np.ones((1, 1, 1), dtype=np.float64)
    df_grid[0, 0, 0] = 0.9  # 10% discount

    # Trade 0: ATM (Strike = Fwd) -> Expected 0.0
    # Trade 1: ITM Long (Strike 100, Fwd 120), No Discount -> Expected  20.0
    # Trade 2: ITM Long (Strike 100, Fwd 120), Discount 0.9 -> Expected 18.0
    # Trade 3: OTM Long (Strike 140, Fwd 120), No Discount -> Expected -20.0
    # Trade 3: OTM Long (Strike 140, Fwd 120), Discount 0.9 -> Expected -18.0
    qty = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    strike = np.array([120.0, 100.0, 100.0, 140.0, 140.0], dtype=np.float64)

    price_uix = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    settle_date_uix = np.array([0, 0, 0, 0, 0], dtype=np.int32)

    fx_u_eff_uix = np.array([-1, -1, -1, -1, -1], dtype=np.int32)
    df_u_uix = np.array([-1, -1, 0, -1, 0], dtype=np.int32)
    fx_k_eff_uix = np.array([-1, -1, -1, -1, -1], dtype=np.int32)
    df_k_uix = np.array([-1, -1, 0, -1, 0], dtype=np.int32)

    _kernel_price_vanilla_forward_node(
        out,
        forward_nodes,
        fx_rates,
        df_grid,
        qty,
        strike,
        price_uix,
        fx_u_eff_uix,
        df_u_uix,
        fx_k_eff_uix,
        df_k_uix,
        settle_date_uix,
    )

    np.testing.assert_allclose(out[:, 0], [0.0, 20.0, 18.0, -20.0, -18.0])
    print("Forward Node Kernel: PASS\n")


def test_european_option_node_kernel():
    print("--- Testing European Option Node Kernel ---")
    m_trades, s_scenarios = 4, 1
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    forward_nodes = np.array([[100.0]], dtype=np.float64)
    vol_nodes = np.array([[0.20]], dtype=np.float64)  # 20% vol
    fx_rates = np.array([[1.0]], dtype=np.float64)
    df_grid = np.ones((1, 1, 1), dtype=np.float64)

    sqrt_tte_grid = np.array([1.0], dtype=np.float64)  # 1 Year expiry
    half_tte_grid = np.array([0.5], dtype=np.float64)

    # Trade 0: Deep ITM Call (K=0.001) -> Approaches Fwd (100.0)
    # Trade 1: Deep OTM Call (K=1000) -> Approaches 0.0
    # Trade 2: Deep ITM Put (K=1000) -> Approaches K - Fwd (900.0)
    # Trade 3: Qty 0 ATM Call -> Exactly 0.0
    qty = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    strike = np.array([0.001, 1000.0, 1000.0, 100.0], dtype=np.float64)
    cp_mult = np.array([1, 1, -1, 1], dtype=np.int8)  # +1 Call, -1 Put

    uix_zeros = np.zeros(m_trades, dtype=np.int32)
    uix_neg1 = np.full(m_trades, -1, dtype=np.int32)

    _kernel_price_vanilla_european_option_node(
        out,
        forward_nodes,
        vol_nodes,
        fx_rates,
        df_grid,
        sqrt_tte_grid,
        half_tte_grid,
        qty,
        strike,
        cp_mult,
        uix_zeros,
        uix_zeros,
        uix_neg1,
        uix_neg1,
        uix_neg1,
        uix_neg1,
        uix_zeros,
        uix_zeros,
    )

    # Asserting boundaries
    np.testing.assert_allclose(out[0, 0], 100.0, rtol=1e-3)
    np.testing.assert_allclose(out[1, 0], 0.0, atol=1e-8)
    np.testing.assert_allclose(out[2, 0], 900.0, rtol=1e-3)
    np.testing.assert_allclose(out[3, 0], 0.0, atol=1e-12)
    print("European Option Node Kernel: PASS\n")


def test_curve_and_surface_variants():
    print("--- Testing Curve and Surface Variants ---")
    m_trades, s_scenarios = 1, 1

    # Testing Option Curve variant with 4D Vol surface
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    # 3D Curve grids: [curve_uix, delivery/settle_uix, s]
    price_curve_grid = np.array([[[100.0]]], dtype=np.float64)
    df_grid = np.ones((1, 1, 1), dtype=np.float64)
    fx_rates = np.ones((1, 1), dtype=np.float64)
    tte_grid = np.array([1.0], dtype=np.float64)

    # 4D Vol Surface: [surface, expiry, scenario, pillar]
    # We place the vol exactly on the strike to test binary search interp
    vol_surface_strikes = np.array([[100.0]], dtype=np.float64)
    vol_surfaces = np.array([[[[0.20]]]], dtype=np.float64)

    qty = np.array([1.0], dtype=np.float64)
    strike = np.array([0.001], dtype=np.float64)  # Deep ITM Call
    cp_mult = np.array([1], dtype=np.int8)
    y_axis_type_id = np.array([SURFACE_Y_AXIS_TYPE_STRIKE], dtype=np.int8)

    uix_zeros = np.zeros(m_trades, dtype=np.int32)
    uix_neg1 = np.full(m_trades, -1, dtype=np.int32)

    _kernel_price_vanilla_european_option_curve(
        out,
        price_curve_grid,
        vol_surface_strikes,
        vol_surfaces,
        fx_rates,
        df_grid,
        tte_grid,
        qty,
        strike,
        cp_mult,
        uix_zeros,
        uix_zeros,
        y_axis_type_id,
        uix_neg1,
        uix_neg1,
        uix_neg1,
        uix_neg1,
        uix_zeros,
        uix_zeros,
        uix_zeros,
    )

    # Deep ITM Call should resolve to ~Forward (100.0)
    np.testing.assert_allclose(out[0, 0], 100.0, rtol=1e-3)
    print("European Option Curve Kernel: PASS\n")


if __name__ == "__main__":
    print("===========================================")
    print("RiskQuant Pricing Kernel Verification Suite")
    print("===========================================\n")
    test_spot_kernel()
    test_forward_node_kernel()
    test_european_option_node_kernel()
    test_curve_and_surface_variants()
    print("ALL KERNELS PASSED.")
