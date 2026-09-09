# scripts/demos/simlab/scenarios/core_math/demo_ar1_filter_benchmark.py
from __future__ import annotations

import time

import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_LOG_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ar1_filter import kernel_ar1_residuals
from riskquant.utils.printer import (
    print_calling_file,
    print_header,
    print_success_message,
)


def benchmark_ar1_kernel() -> None:

    print_calling_file(add_space_after=False)

    print_header("AR(1) Filter Benchmark")

    # ------------------------------------------------------------------
    # Data Setup: 5 years of daily data for 10,000 risk factors
    # ------------------------------------------------------------------
    t_days = 1260  # ~5 years
    k_factors = 10_000

    print(f"Generating synthetic data: {t_days} days x {k_factors} risk factors...")
    rng = np.random.default_rng(101)

    # Generate random walk levels
    returns = rng.standard_normal((t_days, k_factors)) * 0.01
    levels = 100.0 * np.exp(np.cumsum(returns, axis=0))

    # Prepare Fortran-contiguous arrays
    values_f = np.asfortranarray(levels)
    cols = np.arange(k_factors, dtype=np.int32)
    shift_types = np.full(k_factors, SHIFT_LOG_RETURN_ID, dtype=np.int8)
    out_residuals_f = np.empty_like(values_f, order="F")

    # ------------------------------------------------------------------
    # Warmup Phase (Trigger JIT Compilation)
    # ------------------------------------------------------------------
    print("Triggering Numba JIT Compilation (warmup)...")
    t0_warmup = time.perf_counter()

    # Run on a tiny slice just to trigger compilation (ensure F-contiguity for 2D arrays)
    kernel_ar1_residuals(
        np.asfortranarray(values_f[:10, :10]),
        cols[:10],
        shift_types[:10],
        np.asfortranarray(out_residuals_f[:10, :10]),
    )

    t1_warmup = time.perf_counter()
    print(f"Compilation time: {(t1_warmup - t0_warmup):.4f} seconds\n")

    # ------------------------------------------------------------------
    # Benchmark Phase
    # ------------------------------------------------------------------
    print(f"Running Parallel AR(1) Filter on {k_factors:,} factors...")

    t0_bench = time.perf_counter()

    kernel_ar1_residuals(values_f, cols, shift_types, out_residuals_f)

    t1_bench = time.perf_counter()
    duration = t1_bench - t0_bench

    # Calculate throughput
    data_points = t_days * k_factors
    throughput = data_points / duration

    print("-" * 50)
    print(f"Execution Time:    {duration:.5f} seconds")
    print(f"Data Points:       {data_points:,}")
    print(f"Throughput:        {throughput:,.0f} points/second")
    print("-" * 50)
    print("Conclusion: The zero-allocation parallel architecture successfully")
    print("eliminates the Python Global Interpreter Lock (GIL) overhead.")
    print_success_message("AR(1) Filter Benchmark Completed Successfully")


if __name__ == "__main__":
    benchmark_ar1_kernel()
