# scripts/demos/valuation/vanilla_forward_curve_kernel_throughput.py

import time

import numpy as np

from riskquant.constants import RNG_SEED
from riskquant.utils.printer import print_calling_file, print_success_message

# ============================================================
# IMPORT PRODUCTION KERNEL
# ============================================================
from riskquant.valuation.kernels.vanilla_forward_curve import (
    _kernel_price_vanilla_forward_curve,
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
    c_price_curves = 500  # 500 Unique underlying forward curves
    t_delivery = 120  # 120 Standardized delivery dates (e.g., 10 yrs monthly)
    f_fx_spots = 50  # 50 Unique FX pairs
    c_df_curves = 20  # 20 Unique discount curves
    t_settle = 60  # 60 Standardized settlement dates

    print("--- S-TIER VANILLA FORWARD CURVE KERNEL BENCHMARK ---")
    print(f"Trades:    {m_trades:,}")
    print(f"Scenarios: {s_scenarios:,}")
    print(f"Total PVs: {m_trades * s_scenarios:,} (1 Billion valuations)")
    print("-" * 53)

    # --- 2. Allocate C-Contiguous Memory ---
    print("Allocating engine memory...")
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    rng = _rng()

    # Generate random market data (2D and 3D Grids)
    # The price curve is now a 3D tensor: [Curve, DeliveryDate, Scenario]
    price_curve_grid = rng.uniform(
        10.0, 100.0, (c_price_curves, t_delivery, s_scenarios)
    ).astype(np.float64)

    fx_rates = rng.uniform(0.5, 1.5, (f_fx_spots, s_scenarios)).astype(np.float64)
    df_grid = rng.uniform(0.8, 1.0, (c_df_curves, t_settle, s_scenarios)).astype(
        np.float64
    )

    # Generate random portfolio semantics
    qty = rng.uniform(-1000, 1000, m_trades).astype(np.float64)
    strike = rng.uniform(10.0, 100.0, m_trades).astype(np.float64)

    # Indices for lookups
    price_curve_uix = rng.integers(0, c_price_curves, m_trades).astype(np.int32)
    delivery_date_uix = rng.integers(0, t_delivery, m_trades).astype(np.int32)
    settle_date_uix = rng.integers(0, t_settle, m_trades).astype(np.int32)

    # Mix base currency (-1) and foreign currency trades
    fx_u_eff_uix = rng.integers(-1, f_fx_spots, m_trades).astype(np.int32)
    fx_k_eff_uix = rng.integers(-1, f_fx_spots, m_trades).astype(np.int32)

    # Mix un-discounted (-1) and discounted trades
    df_u_uix = rng.integers(-1, c_df_curves, m_trades).astype(np.int32)
    df_k_uix = rng.integers(-1, c_df_curves, m_trades).astype(np.int32)

    # Validate memory layout
    assert out.flags.c_contiguous
    assert price_curve_grid.flags.c_contiguous
    assert fx_rates.flags.c_contiguous
    assert df_grid.flags.c_contiguous

    # Outer loop.
    for _ in range(3):
        # --- 3. JIT Warmup ---
        print("Warming up JIT compiler...")
        start_compile = time.perf_counter()

        # Call the imported kernel.
        # Slicing the outer dimension of 'out' and the 1D trade arrays limits the loop
        # size for compilation, but preserves the strict C-contiguity flags.
        _kernel_price_vanilla_forward_curve(
            out[:10, :],
            price_curve_grid,
            fx_rates,
            df_grid,
            qty[:10],
            strike[:10],
            price_curve_uix[:10],
            delivery_date_uix[:10],
            fx_u_eff_uix[:10],
            df_u_uix[:10],
            fx_k_eff_uix[:10],
            df_k_uix[:10],
            settle_date_uix[:10],
        )

        print(
            f"Compilation/Cache-load finished in {time.perf_counter() - start_compile:.4f}s"
        )
        print("-" * 53)

        # --- 4. The Hot Run ---
        print("Executing 1 Billion valuations...")
        runs = 10
        total_time = 0.0

        for r in range(runs):
            start_time = time.perf_counter()

            # Call the imported kernel
            _kernel_price_vanilla_forward_curve(
                out,
                price_curve_grid,
                fx_rates,
                df_grid,
                qty,
                strike,
                price_curve_uix,
                delivery_date_uix,
                fx_u_eff_uix,
                df_u_uix,
                fx_k_eff_uix,
                df_k_uix,
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

        print("-" * 53)
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Throughput:   {throughput / 1_000_000_000:.2f} Billion PVs/sec")
        print(f"Write B/W:    {bandwidth_gb_s:.2f} GB/s")
        print("-" * 53)

    print_success_message()


if __name__ == "__main__":
    run_benchmark()
