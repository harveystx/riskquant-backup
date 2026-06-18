# scripts/demos/valuation/vanilla_european_option_node_throughput.py

import time

import numpy as np

from riskquant.constants import RNG_SEED
from riskquant.utils.printer import print_calling_file, print_success_message

# ============================================================
# IMPORT PRODUCTION KERNEL
# ============================================================
from riskquant.valuation.pricer_kernels._deprecated.vanilla_european_option_node_v1 import (
    _kernel_price_vanilla_european_option_node,
)

# ============================================================
# BENCHMARK RUNNER
# ============================================================


def _rng():
    # Fixed seed for reproducibility
    return np.random.default_rng(seed=RNG_SEED)


def run_benchmark():
    print_calling_file()

    # --- 1. Scale Parameters ---
    m_trades = 100_000  # 100k Trades
    s_scenarios = 10_000  # 10k Scenarios

    # Market Data Dimensions
    p_prices = 1_000  # 1k Unique underlying forward nodes
    v_vols = 500  # 500 Unique volatility nodes
    f_fx_spots = 50  # 50 Unique FX pairs
    c_df_curves = 20  # 20 Unique discount curves
    t_tenors = 60  # 60 Standardized dates (for expiry, settle, and TTE)

    print("--- S-TIER VANILLA EUROPEAN OPTION BENCHMARK ---")
    print(f"Trades:    {m_trades:,}")
    print(f"Scenarios: {s_scenarios:,}")
    print(f"Total PVs: {m_trades * s_scenarios:,} (1 Billion valuations)")
    print("-" * 52)

    # --- 2. Allocate C-Contiguous Memory ---
    print("Allocating engine memory...")
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    rng = _rng()

    # Generate random market data
    forward_nodes = rng.uniform(10.0, 100.0, (p_prices, s_scenarios)).astype(np.float64)
    vol_nodes = rng.uniform(0.1, 0.5, (v_vols, s_scenarios)).astype(
        np.float64
    )  # 10% to 50% vol
    fx_rates = rng.uniform(0.5, 1.5, (f_fx_spots, s_scenarios)).astype(np.float64)
    df_grid = rng.uniform(0.8, 1.0, (c_df_curves, t_tenors, s_scenarios)).astype(
        np.float64
    )

    # Time to expiry grid (e.g., 0.1 to 10.0 years)
    tte_grid = np.linspace(0.1, 10.0, t_tenors).astype(np.float64)
    sqrt_tte_grid = np.sqrt(tte_grid)
    half_tte_grid = 0.5 * tte_grid

    # Generate random portfolio semantics
    qty = rng.uniform(-1000, 1000, m_trades).astype(np.float64)
    strike = rng.uniform(10.0, 100.0, m_trades).astype(np.float64)
    cp_mult = rng.choice([-1, 1], size=m_trades).astype(np.int8)  # Calls and Puts

    # Indices for lookups
    price_uix = rng.integers(0, p_prices, m_trades).astype(np.int32)
    vol_uix = rng.integers(0, v_vols, m_trades).astype(np.int32)
    expiry_date_uix = rng.integers(0, t_tenors, m_trades).astype(np.int32)
    settle_date_uix = rng.integers(0, t_tenors, m_trades).astype(np.int32)

    # Mix base currency (-1) and foreign currency trades
    fx_u_eff_uix = rng.integers(-1, f_fx_spots, m_trades).astype(np.int32)
    fx_k_eff_uix = rng.integers(-1, f_fx_spots, m_trades).astype(np.int32)

    # Mix un-discounted (-1) and discounted trades
    df_u_uix = rng.integers(-1, c_df_curves, m_trades).astype(np.int32)
    df_k_uix = rng.integers(-1, c_df_curves, m_trades).astype(np.int32)

    # Validate memory layout
    assert out.flags.c_contiguous
    assert forward_nodes.flags.c_contiguous
    assert vol_nodes.flags.c_contiguous
    assert fx_rates.flags.c_contiguous
    assert df_grid.flags.c_contiguous

    # Outer loop.
    for _ in range(3):
        # --- 3. JIT Warmup ---
        print("Warming up JIT compiler...")
        start_compile = time.perf_counter()

        # Call the imported kernel with sliced outer dimensions for compilation
        _kernel_price_vanilla_european_option_node(
            out[:10, :],
            forward_nodes,
            vol_nodes,
            fx_rates,
            df_grid,
            sqrt_tte_grid,
            half_tte_grid,
            qty[:10],
            strike[:10],
            cp_mult[:10],
            price_uix[:10],
            vol_uix[:10],
            fx_u_eff_uix[:10],
            df_u_uix[:10],
            fx_k_eff_uix[:10],
            df_k_uix[:10],
            expiry_date_uix[:10],
            settle_date_uix[:10],
        )

        print(
            f"Compilation/Cache-load finished in {time.perf_counter() - start_compile:.4f}s"
        )
        print("-" * 52)

        # --- 4. The Hot Run ---
        print("Executing 1 Billion valuations...")
        runs = 10
        total_time = 0.0

        for r in range(runs):
            start_time = time.perf_counter()

            # Call the imported kernel
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
                price_uix,
                vol_uix,
                fx_u_eff_uix,
                df_u_uix,
                fx_k_eff_uix,
                df_k_uix,
                expiry_date_uix,
                settle_date_uix,
            )

            run_time = time.perf_counter() - start_time
            total_time += run_time
            print(f"  Run {r + 1}: {run_time:.4f} seconds")

        # --- 5. Throughput Math ---
        avg_time = total_time / runs
        total_valuations = m_trades * s_scenarios
        throughput = total_valuations / avg_time

        # Memory Bandwidth Calc (Writing 8 bytes per PV)
        bytes_written = total_valuations * 8
        bandwidth_gb_s = (bytes_written / avg_time) / (1024**3)

        print("-" * 52)
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Throughput:   {throughput / 1_000_000_000:.2f} Billion PVs/sec")
        print(f"Write B/W:    {bandwidth_gb_s:.2f} GB/s")
        print("-" * 52)

    print_success_message()


if __name__ == "__main__":
    run_benchmark()
