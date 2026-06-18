# scripts/demos/market_data/curves/demo_interp_methods_3d_throughput.py

from __future__ import annotations

import random
import statistics
import time

import numpy as np

from riskquant.market_data.curves.extrap_methods import (
    EXTRAP_FLAT,
    EXTRAP_LINEAR,
)
from riskquant.market_data.curves.interp_dispatch import CURVE_MATH_REGISTRY
from riskquant.market_data.curves.interp_methods import (
    INTERP_LINEAR,
    INTERP_LOCF,
    INTERP_LOG_LINEAR,
    INTERP_NOCB,
)


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def run_3d_throughput_benchmark() -> None:
    print("\n" + "=" * 60)
    print("   RISKQUANT NUMBA 3D KERNEL THROUGHPUT BENCHMARK")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    C_CURVES = 500
    M_PILLARS = 30
    D_TARGETS = 5000
    S_SCENARIOS = 500

    total_interps = C_CURVES * D_TARGETS * S_SCENARIOS

    print("Workload Dimensions:")
    print(f"  Curves:    {C_CURVES:,}")
    print(f"  Pillars:   {M_PILLARS:,}")
    print(f"  Targets:   {D_TARGETS:,}")
    print(f"  Scenarios: {S_SCENARIOS:,}")
    print(f"  Total Interpolations: {total_interps:,}")
    print()

    # ------------------------------------------------------------------
    # Data Generation
    # ------------------------------------------------------------------

    print("Generating synthetic market data...\n")

    rng = np.random.default_rng(42)

    tenors = np.sort(
        rng.uniform(0.0, 50.0, size=(C_CURVES, M_PILLARS)),
        axis=1,
    )

    values = rng.uniform(
        0.1,
        1.5,
        size=(C_CURVES, M_PILLARS, S_SCENARIOS),
    )

    log_values = np.log(values)

    targets = np.linspace(
        -5.0,
        55.0,
        D_TARGETS,
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Memory Report
    # ------------------------------------------------------------------

    output_bytes = C_CURVES * D_TARGETS * S_SCENARIOS * np.dtype(np.float64).itemsize

    print("Approximate Memory Footprint:")
    print(f"  Values:      {bytes_to_gib(values.nbytes):.2f} GiB")
    print(f"  Log Values:  {bytes_to_gib(log_values.nbytes):.2f} GiB")
    print(f"  Output:      {bytes_to_gib(output_bytes):.2f} GiB")
    print()

    # ------------------------------------------------------------------
    # Kernel Names
    # ------------------------------------------------------------------

    method_names = {
        (int(INTERP_LINEAR), int(EXTRAP_FLAT)): "Linear + Flat Extrap",
        (int(INTERP_LINEAR), int(EXTRAP_LINEAR)): "Linear + Linear Extrap",
        (int(INTERP_LOG_LINEAR), int(EXTRAP_FLAT)): "Log-Linear + Flat Extrap",
        (int(INTERP_LOG_LINEAR), int(EXTRAP_LINEAR)): "Log-Linear + Linear Extrap",
        (int(INTERP_LOCF), int(EXTRAP_FLAT)): "LOCF (Step Left) + Flat Extrap",
        (int(INTERP_LOCF), int(EXTRAP_LINEAR)): "LOCF (Step Left) + Linear Extrap",
        (int(INTERP_NOCB), int(EXTRAP_FLAT)): "NOCB (Step Right) + Flat Extrap",
        (int(INTERP_NOCB), int(EXTRAP_LINEAR)): "NOCB (Step Right) + Linear Extrap",
    }

    # ------------------------------------------------------------------
    # JIT Warmup
    # ------------------------------------------------------------------

    print("Triggering Numba JIT Compilation...\n")

    dummy_tenors = np.ascontiguousarray(tenors[:1, :2])

    dummy_targets = np.ascontiguousarray(targets[:1])

    dummy_out = np.empty((1, 1, 1))

    for key, kernel in CURVE_MATH_REGISTRY.items():
        interp_id = key[0]

        input_vals = log_values if interp_id == int(INTERP_LOG_LINEAR) else values

        dummy_vals = np.ascontiguousarray(input_vals[:1, :2, :1])

        kernel(
            dummy_tenors,
            dummy_vals,
            dummy_targets,
            dummy_out,
        )

    print("Compilation complete.")
    print()

    # ------------------------------------------------------------------
    # Benchmark Setup
    # ------------------------------------------------------------------

    iterations = 7

    kernel_items = list(CURVE_MATH_REGISTRY.items())

    results: dict[str, list[float]] = {method_names[key]: [] for key, _ in kernel_items}

    print(f"Running {iterations} randomized benchmark rounds...")
    print()

    # ------------------------------------------------------------------
    # Benchmark Loop
    # ------------------------------------------------------------------

    for round_idx in range(iterations):
        random.shuffle(kernel_items)

        for key, kernel in kernel_items:
            interp_id = key[0]

            input_vals = log_values if interp_id == int(INTERP_LOG_LINEAR) else values

            # Allocate fresh output every run.
            #
            # This removes a large amount of cache-state bias
            # between kernels.
            out = np.empty(
                (
                    C_CURVES,
                    D_TARGETS,
                    S_SCENARIOS,
                ),
                dtype=np.float64,
            )

            t0 = time.perf_counter()

            kernel(
                tenors,
                input_vals,
                targets,
                out,
            )

            t1 = time.perf_counter()

            elapsed = t1 - t0

            # Touch output so the compiler/runtime
            # cannot treat writes as dead.
            checksum = float(out[0, 0, 0])

            if np.isnan(checksum):
                raise RuntimeError("Unexpected NaN.")

            results[method_names[key]].append(elapsed)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print("-" * 120)

    print(
        f"{'Kernel Name':<35}"
        f" | {'Median (s)':>12}"
        f" | {'Mean (s)':>12}"
        f" | {'Min (s)':>12}"
        f" | {'Throughput (Interp/s)':>26}"
    )

    print("-" * 120)

    for kernel_name, samples in results.items():
        median_time = statistics.median(samples)
        mean_time = statistics.mean(samples)
        min_time = min(samples)

        throughput = total_interps / median_time

        print(
            f"{kernel_name:<35}"
            f" | {median_time:>12.4f}"
            f" | {mean_time:>12.4f}"
            f" | {min_time:>12.4f}"
            f" | {throughput:>26_.0f}"
        )

    print("-" * 120)

    print()
    print("Notes:")
    print("  • Throughput uses MEDIAN runtime.")
    print("  • Kernel order randomized every round.")
    print("  • Fresh output tensor allocated each run.")
    print("  • Results are less sensitive to cache warming.")
    print()

    print("=" * 60)
    print("   END OF BENCHMARK DEMO")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run_3d_throughput_benchmark()
