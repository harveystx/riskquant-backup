# scripts/demos/simlab/scenarios/core_math/demo_ewma_volatility_throughput.py
"""
benchmark_ewma.py - Throughput demo script for RiskQuant EWMA kernels.
"""

import time

import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.core_math.ewma_volatility import (
    kernel_ewma_vol_ratios,
    kernel_ewma_vol_ratios_from_residuals,
)


def run_throughput_demo():
    # Shape parameters matching a large-scale risk calculation
    # T = 1000 days of history, K = 5000 unique risk factors
    T = 1000
    K = 5000
    decay = 0.94
    max_ratio = 10.0

    print("=" * 70)
    # 1. Warm up Numba JIT compiler
    print("Initializing structures and warming up JIT compiler...")
    warmup_val = np.asfortranarray(np.random.randn(10, 2).astype(np.float64))
    warmup_out = np.asfortranarray(np.empty((10, 2), dtype=np.float64))
    kernel_ewma_vol_ratios_from_residuals(warmup_val, decay, max_ratio, warmup_out)

    # 2. Benchmark: kernel_ewma_vol_ratios_from_residuals
    # Generate random residuals data (already centered/filtered returns)
    print(
        f"Generating synthetic residuals matrix of size ({T} x {K}) [F-contiguous]..."
    )
    residuals = np.asfortranarray(
        np.random.normal(0, 0.02, size=(T, K)).astype(np.float64)
    )
    out_ratios_res = np.asfortranarray(np.empty((T, K), dtype=np.float64))

    print("\n--- Executing `kernel_ewma_vol_ratios_from_residuals` ---")
    start_time = time.perf_counter()
    # Execute the parallel kernel
    kernel_ewma_vol_ratios_from_residuals(residuals, decay, max_ratio, out_ratios_res)
    duration = time.perf_counter() - start_time

    # Calculate performance metadata
    total_elements = T * K
    gb_processed = (residuals.nbytes + out_ratios_res.nbytes) / (1024**3)
    throughput_vectors_sec = K / duration

    print(f"Execution Time     : {duration:.4f} seconds")
    print(f"Vector Throughput  : {throughput_vectors_sec:.2f} risk factors / sec")
    print(f"Memory Throughput  : {gb_processed / duration:.2f} GB/sec (Read + Write)")

    # 3. Benchmark: kernel_ewma_vol_ratios (Raw values requiring inline returns)
    print("\n--- Executing `kernel_ewma_vol_ratios` (Inline Returns Evaluation) ---")
    # Simulate equity prices/interest rates drifting upwards
    raw_values = np.asfortranarray(
        100.0
        * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, size=(T, K)), axis=0)).astype(
            np.float64
        )
    )
    cols = np.arange(K, dtype=np.int32)
    # Mix up shift types across columns to exercise branch hoisting
    shift_types = np.array(
        [SHIFT_LOG_RETURN_ID, SHIFT_SIMPLE_RETURN_ID, SHIFT_DIFFERENCE_ID]
        * (K // 3 + 1)
    )[:K].astype(np.int8)
    out_ratios_raw = np.asfortranarray(np.empty((T, K), dtype=np.float64))

    start_time = time.perf_counter()
    kernel_ewma_vol_ratios(
        raw_values, cols, shift_types, decay, out_ratios_raw, max_ratio
    )
    duration_raw = time.perf_counter() - start_time

    print(f"Execution Time     : {duration_raw:.4f} seconds")
    print(f"Vector Throughput  : {K / duration_raw:.2f} risk factors / sec")
    print(
        f"Memory Throughput  : {(raw_values.nbytes + out_ratios_raw.nbytes) / (1024**3) / duration_raw:.2f} GB/sec"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_throughput_demo()
