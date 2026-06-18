# scripts/demos/market_data/curves/demo_interp_strike_linear_flat_extrap_benchmark.py

from __future__ import annotations

import statistics
import time

import numpy as np

from riskquant.market_data.curves.interp_defs import (
    interp_strike_linear_flat_extrap_numba_kernel,
)


def bytes_to_gib(n: int) -> float:
    return n / (1024**3)


def benchmark_case(
    *,
    n_trades: int,
    k_strikes: int,
    s_scenarios: int,
    runs: int,
) -> dict[str, float]:
    rng = np.random.default_rng(42)

    pillars = np.ascontiguousarray(
        np.sort(
            rng.uniform(
                50.0,
                150.0,
                size=(n_trades, k_strikes),
            ),
            axis=1,
        )
    )

    values = np.ascontiguousarray(
        rng.uniform(
            0.15,
            0.50,
            size=(n_trades, k_strikes, s_scenarios),
        )
    )

    targets = np.ascontiguousarray(
        rng.uniform(
            40.0,
            160.0,
            size=(n_trades, s_scenarios),
        )
    )

    out = np.empty(
        (n_trades, s_scenarios),
        dtype=np.float64,
    )

    timings: list[float] = []

    for _ in range(runs):
        # out.fill(0.0)

        t0 = time.perf_counter()

        interp_strike_linear_flat_extrap_numba_kernel(
            pillars,
            values,
            targets,
            out,
        )

        elapsed = time.perf_counter() - t0

        checksum = float(out[0, 0])

        if np.isnan(checksum):
            raise RuntimeError("Unexpected NaN.")

        timings.append(elapsed)

    median_time = statistics.median(timings)

    total_lookups = n_trades * s_scenarios

    throughput = total_lookups / median_time

    approx_bytes = values.nbytes + targets.nbytes + out.nbytes

    bandwidth_gib = approx_bytes / median_time / (1024**3)

    return {
        "trades": n_trades,
        "strikes": k_strikes,
        "scenarios": s_scenarios,
        "lookups": total_lookups,
        "median_time": median_time,
        "min_time": min(timings),
        "max_time": max(timings),
        "throughput": throughput,
        "bandwidth_gib": bandwidth_gib,
        "values_gib": bytes_to_gib(values.nbytes),
    }


def run_benchmark() -> None:
    print("=" * 90)
    print("RISKQUANT STRIKE INTERPOLATION PERFORMANCE SUITE")
    print("=" * 90)
    print()

    # ----------------------------------------------------------
    # JIT Warmup
    # ----------------------------------------------------------

    print("Compiling Numba kernel...")

    dummy_pillars = np.array([[90.0, 100.0]])
    dummy_values = np.array([[[0.20], [0.25]]])
    dummy_targets = np.array([[95.0]])
    dummy_out = np.empty((1, 1))

    interp_strike_linear_flat_extrap_numba_kernel(
        dummy_pillars,
        dummy_values,
        dummy_targets,
        dummy_out,
    )

    print("Compilation complete.")
    print()

    # ----------------------------------------------------------
    # Scaling Matrix
    # ----------------------------------------------------------

    strike_counts = [5, 15, 30, 60, 120]
    scenario_counts = [100, 1000]

    results = []

    for strikes in strike_counts:
        for scenarios in scenario_counts:
            result = benchmark_case(
                n_trades=10_000,
                k_strikes=strikes,
                s_scenarios=scenarios,
                runs=7,
            )

            results.append(result)

    print("-" * 140)

    print(
        f"{'Strikes':>8} | "
        f"{'Scenarios':>10} | "
        f"{'Values GiB':>10} | "
        f"{'Median(s)':>10} | "
        f"{'Throughput':>18} | "
        f"{'Bandwidth GiB/s':>18}"
    )

    print("-" * 140)

    for r in results:
        print(
            f"{r['strikes']:>8,.0f} | "
            f"{r['scenarios']:>10,.0f} | "
            f"{r['values_gib']:>10.2f} | "
            f"{r['median_time']:>10.4f} | "
            f"{r['throughput']:>18,.0f} | "
            f"{r['bandwidth_gib']:>18.2f}"
        )

    print("-" * 140)

    print()
    print("Interpretation:")
    print()
    print("• If throughput falls sharply as strike count increases,")
    print("  binary search is becoming dominant.")
    print()
    print("• If throughput stays relatively stable as strike count")
    print("  increases, memory traffic dominates.")
    print()
    print("• Bandwidth approaching system memory limits indicates")
    print("  further kernel optimization is unlikely to help.")
    print()


if __name__ == "__main__":
    run_benchmark()
