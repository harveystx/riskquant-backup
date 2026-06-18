# scripts/demos/valuation/spot_kernel_throughput.py

import time

import numpy as np

from riskquant.constants import RNG_SEED
from riskquant.utils.printer import print_calling_file, print_success_message

# ============================================================
# IMPORT PRODUCTION KERNEL
# ============================================================
from riskquant.valuation.pricer_kernels.spot import _kernel_price_spot

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
    p_prices = 1_000  # 1k Unique price factors
    f_fx_spots = 50  # 50 Unique FX pairs

    print("--- S-TIER SPOT KERNEL BENCHMARK ---")
    print(f"Trades:    {m_trades:,}")
    print(f"Scenarios: {s_scenarios:,}")
    print(f"Total PVs: {m_trades * s_scenarios:,} (1 Billion valuations)")
    print("-" * 36)

    # --- 2. Allocate C-Contiguous Memory ---
    print("Allocating engine memory...")
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)

    rng = _rng()

    # Generate random market data
    prices = rng.uniform(10.0, 100.0, (p_prices, s_scenarios)).astype(np.float64)
    fx_rates = rng.uniform(0.5, 1.5, (f_fx_spots, s_scenarios)).astype(np.float64)

    # Generate random portfolio semantics
    qty = rng.uniform(-1000, 1000, m_trades).astype(np.float64)
    price_uix = rng.integers(0, p_prices, m_trades).astype(np.int32)

    # Mix base currency (-1) and foreign currency trades
    fx_uix = rng.integers(-1, f_fx_spots, m_trades).astype(np.int32)

    # Validate memory layout
    assert out.flags.c_contiguous
    assert prices.flags.c_contiguous

    # Outer loop.
    for _ in range(3):
        # --- 3. JIT Warmup ---
        print("Warming up JIT compiler...")
        start_compile = time.perf_counter()

        # Call the imported kernel
        _kernel_price_spot(
            out[:10, :],  # Sliced outer dimension only to preserve C-contiguity
            prices,
            price_uix[:10],
            qty[:10],
            fx_rates,
            fx_uix[:10],
        )

        print(
            f"Compilation/Cache-load finished in {time.perf_counter() - start_compile:.4f}s"
        )
        print("-" * 36)

        # --- 4. The Hot Run ---
        print("Executing 1 Billion valuations...")
        runs = 10
        total_time = 0.0

        for r in range(runs):
            start_time = time.perf_counter()

            # Call the imported kernel
            _kernel_price_spot(out, prices, price_uix, qty, fx_rates, fx_uix)

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

        print("-" * 36)
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Throughput:   {throughput / 1_000_000_000:.2f} Billion PVs/sec")
        print(f"Write B/W:    {bandwidth_gb_s:.2f} GB/s")
        print("-" * 36)

    print_success_message()


if __name__ == "__main__":
    run_benchmark()
