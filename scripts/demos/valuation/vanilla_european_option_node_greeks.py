# scripts/demos/valuation/vanilla_european_option_node_greeks.py

import matplotlib.pyplot as plt
import numpy as np

from riskquant.valuation.kernels.vanilla_european_option_node import (
    _kernel_price_vanilla_european_option_node,
)


def plot_option_greeks():
    print("--- Generating Option Greeks Visualization ---")

    # ------------------------------------------------------------------
    # Engine Setup (Vectorizing over Scenarios for X-Axis Grid)
    # ------------------------------------------------------------------
    m_trades = 2  # Trade 0: Call, Trade 1: Put
    s_scenarios = 1_000  # 1000 points on the X-axis (Forward Price)

    # Pre-allocate SoA output buffers for Base, Delta (Up/Down), and Vega (Up)
    out_base = np.zeros((m_trades, s_scenarios), dtype=np.float64)
    out_up = np.zeros((m_trades, s_scenarios), dtype=np.float64)
    out_dn = np.zeros((m_trades, s_scenarios), dtype=np.float64)
    out_vol_up = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    # X-Axis Grid: Forward nodes from 50.0 to 150.0
    fwd_grid = np.linspace(50.0, 150.0, s_scenarios)

    # Finite Difference Bumps
    delta_underlying = 0.01  # 1 bp bump for Delta/Gamma
    delta_bump_vols = 0.0001  # 1 bp bump for Vega (0.01%)

    # Market Data Matrices
    fwd_nodes_base = fwd_grid.reshape(1, s_scenarios).astype(np.float64)
    fwd_nodes_up = (
        (fwd_grid + delta_underlying).reshape(1, s_scenarios).astype(np.float64)
    )
    fwd_nodes_dn = (
        (fwd_grid - delta_underlying).reshape(1, s_scenarios).astype(np.float64)
    )

    vol_nodes_base = np.full((1, s_scenarios), 0.20, dtype=np.float64)  # 20% flat vol
    vol_nodes_up = vol_nodes_base + delta_bump_vols

    # Dummy grids for optional legs (-1 bypasses these in the kernel)
    fx_rates = np.ones((1, 1), dtype=np.float64)  # No FX adjustment for simplicity
    df_grid = np.ones((1, 1, 1), dtype=np.float64)  # No discounting for simplicity

    # Time to Expiry: 1 Year
    sqrt_tte_grid = np.array([1.0], dtype=np.float64)
    half_tte_grid = np.array([0.5], dtype=np.float64)

    # ------------------------------------------------------------------
    # Trade Semantics
    # ------------------------------------------------------------------
    qty = np.array([1.0, 1.0], dtype=np.float64)
    strike = np.array([100.0, 100.0], dtype=np.float64)
    cp_mult = np.array([1, -1], dtype=np.int8)  # +1 Call, -1 Put

    uix_zeros = np.zeros(m_trades, dtype=np.int32)
    uix_neg1 = np.full(m_trades, -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Kernel Execution (Zero Allocations, C-Contiguous)
    # ------------------------------------------------------------------
    def run_pricer(out_buffer, fwd_data, vol_data):
        _kernel_price_vanilla_european_option_node(
            out_buffer,
            fwd_data,
            vol_data,
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

    run_pricer(out_base, fwd_nodes_base, vol_nodes_base)
    run_pricer(out_up, fwd_nodes_up, vol_nodes_base)
    run_pricer(out_dn, fwd_nodes_dn, vol_nodes_base)
    run_pricer(out_vol_up, fwd_nodes_base, vol_nodes_up)

    # ------------------------------------------------------------------
    # Finite Difference Math
    # ------------------------------------------------------------------
    delta = (out_up - out_dn) / (2 * delta_underlying)
    gamma = (out_up - 2 * out_base + out_dn) / (delta_underlying**2)
    vega = (out_vol_up - out_base) / delta_bump_vols

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Vanilla European Option Node - Mathematical Stability Verification",
        fontsize=16,
    )

    labels = ["Call (K=100)", "Put (K=100)"]
    colors = ["#1f77b4", "#ff7f0e"]

    for i in range(m_trades):
        # Present Value
        axes[0, 0].plot(fwd_grid, out_base[i], label=labels[i], color=colors[i], lw=2)
        axes[0, 0].set_title("Present Value (PV)")
        axes[0, 0].set_ylabel("Price")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(100.0, color="k", linestyle="--", alpha=0.5)

        # Delta
        axes[0, 1].plot(fwd_grid, delta[i], label=labels[i], color=colors[i], lw=2)
        axes[0, 1].set_title("Delta (∂V/∂S)")
        axes[0, 1].set_ylabel("Delta")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(100.0, color="k", linestyle="--", alpha=0.5)

        # Gamma
        axes[1, 0].plot(fwd_grid, gamma[i], label=labels[i], color=colors[i], lw=2)
        axes[1, 0].set_title("Gamma (∂²V/∂S²)")
        axes[1, 0].set_xlabel("Underlying Forward Price")
        axes[1, 0].set_ylabel("Gamma")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(100.0, color="k", linestyle="--", alpha=0.5)

        # Vega
        axes[1, 1].plot(fwd_grid, vega[i], label=labels[i], color=colors[i], lw=2)
        axes[1, 1].set_title("Vega (∂V/∂σ)")
        axes[1, 1].set_xlabel("Underlying Forward Price")
        axes[1, 1].set_ylabel("Vega (per 1bp Vol)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(100.0, color="k", linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_option_greeks()
