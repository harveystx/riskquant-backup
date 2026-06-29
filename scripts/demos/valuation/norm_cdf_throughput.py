# scripts/demos/valuation/norm_cdf_throughput.py

from __future__ import annotations

import time

import numba as nb

from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant.valuation.pricing_models.math import (
    numba_norm_cdf,
    numba_norm_cdf_abrom_steg,
)

# ------------------------------------------------------------
# JIT KERNELS
# ------------------------------------------------------------


@nb.njit(cache=True, fastmath=True)
def bench_exact(n: int) -> float:
    acc = 0.0

    for i in range(n):
        x = ((i % 10000) - 5000) * 0.001
        acc += numba_norm_cdf(x)

    return acc


@nb.njit(cache=True, fastmath=True)
def bench_abrom(n: int) -> float:
    acc = 0.0

    for i in range(n):
        x = ((i % 10000) - 5000) * 0.001
        acc += numba_norm_cdf_abrom_steg(x)

    return acc


# ------------------------------------------------------------
# WARMUP (CRITICAL)
# ------------------------------------------------------------


def warmup() -> None:
    """
    Fully warms Numba compilation + branch paths + CPU cache.
    Must be called BEFORE benchmarking.
    """
    print("Warming up JIT kernels...")

    # compile + exercise both kernels
    bench_exact(10_000)
    bench_abrom(10_000)

    # extra pass to stabilize instruction cache + branch predictor
    bench_exact(50_000)
    bench_abrom(50_000)

    print("Warmup complete.\n")


# ------------------------------------------------------------
# BENCHMARK DRIVER
# ------------------------------------------------------------


def benchmark(name: str, fn, n: int) -> float:
    print(f"{name}")
    print("-" * 60)

    # ensure CPU is not in first-call state (already compiled)
    _ = fn(1_000)

    t0 = time.perf_counter()
    result = fn(n)
    t1 = time.perf_counter()

    elapsed = t1 - t0

    print(f"Time:        {elapsed:.6f} sec")
    print(f"Evaluations: {n:,}")
    print(f"Throughput:  {n / elapsed / 1e9:.3f} Billion evals/sec")
    print(f"Checksum:    {result:.6f}")
    print()

    return elapsed


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print_calling_file()

    n = 1_000_000_000

    # CRITICAL: isolate compile/warmup phase
    warmup()

    exact = benchmark(
        "Exact ERF CDF (numba_norm_cdf)",
        bench_exact,
        n,
    )

    approx = benchmark(
        "Abramowitz-Stegun approximation",
        bench_abrom,
        n,
    )

    print("Summary")
    print("-" * 60)
    print(f"Speedup: {exact / approx:.3f}x")

    print_success_message("Benchmark completed successfully.")
