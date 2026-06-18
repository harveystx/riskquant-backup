# scripts/demos/valuation/norm_cdf_throughput.py

from __future__ import annotations

import time

import numba as nb

from riskquant.valuation.pricer_kernels._deprecated._math_v1 import (
    numba_norm_cdf,
    numba_norm_cdf_abrom_steg,
)


@nb.njit(cache=True)
def bench_exact(n: int) -> float:

    acc = 0.0

    for i in range(n):
        x = ((i % 10000) - 5000) * 0.001
        acc += numba_norm_cdf(x)

    return acc


@nb.njit(cache=True)
def bench_abrom(n: int) -> float:

    acc = 0.0

    for i in range(n):
        x = ((i % 10000) - 5000) * 0.001
        acc += numba_norm_cdf_abrom_steg(x)

    return acc


def benchmark(name, fn, n):

    print(f"\n{name}")
    print("-" * 60)

    fn(1)

    t0 = time.perf_counter()
    result = fn(n)
    t1 = time.perf_counter()

    elapsed = t1 - t0

    print(f"Time:        {elapsed:.4f} sec")
    print(f"Evaluations: {n:,}")
    print(f"Throughput:  {n / elapsed / 1e9:.3f} Billion evals/sec")
    print(f"Checksum:    {result:.6f}")

    return elapsed


if __name__ == "__main__":
    n = 1_000_000_000

    exact = benchmark(
        "Exact ERF CDF",
        bench_exact,
        n,
    )

    approx = benchmark(
        "Abramowitz-Stegun",
        bench_abrom,
        n,
    )

    print("\nSummary")
    print("-" * 60)
    print(f"Speedup: {exact / approx:.3f}x")
    print("-" * 60)
    print("-" * 60)
