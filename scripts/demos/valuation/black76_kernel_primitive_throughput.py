# scripts/demos/valuation/yyy.py

import time

import numba as nb
import numpy as np

from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant.valuation.pricing_models.black76.model import (
    _black76_fv_kernel_primitive,
)

# --- THE BENCHMARK HARNESS ---
# We wrap the scalar primitive in a parallelized loop to simulate
# executing across millions of scenarios/trades.


@nb.njit(
    parallel=True,
    fastmath=True,
    boundscheck=False,
    nogil=True,
)
def _run_benchmark_loop(
    out: np.ndarray,
    fwd: np.ndarray,
    log_fk: np.ndarray,
    strike: np.ndarray,
    vol_sqrt_tte: np.ndarray,
    vol2_half_tte: np.ndarray,
    cp_mult: np.ndarray,
) -> None:
    n = out.shape[0]
    for i in nb.prange(n):
        out[i] = _black76_fv_kernel_primitive(
            fwd[i],
            log_fk[i],
            strike[i],
            vol_sqrt_tte[i],
            vol2_half_tte[i],
            cp_mult[i],
        )


# --- RUNNER ---


def run(num_runs=5):
    print_calling_file()

    n_evals = 25_000_000
    print(f"Allocating data for {n_evals:,} evaluations...")

    rng = np.random.default_rng(seed=42)

    # Allocate C-contiguous arrays
    fwd = rng.uniform(90.0, 110.0, size=n_evals).astype(np.float64)
    strike = rng.uniform(90.0, 110.0, size=n_evals).astype(np.float64)
    log_fk = np.log(fwd / strike).astype(np.float64)

    vol = rng.uniform(0.1, 0.4, size=n_evals).astype(np.float64)
    tte = rng.uniform(0.1, 5.0, size=n_evals).astype(np.float64)

    vol_sqrt_tte = (vol * np.sqrt(tte)).astype(np.float64)
    vol2_half_tte = (vol * vol * 0.5 * tte).astype(np.float64)

    cp_mult = rng.choice(np.array([1, -1], dtype=np.int8), size=n_evals)

    out = np.zeros(n_evals, dtype=np.float64)

    print("Warming up JIT compiler...")
    # Compile with tiny subset
    _run_benchmark_loop(
        out[:10],
        fwd[:10],
        log_fk[:10],
        strike[:10],
        vol_sqrt_tte[:10],
        vol2_half_tte[:10],
        cp_mult[:10],
    )

    print(f"\nRunning benchmark {num_runs} times...")

    times = np.zeros(num_runs, dtype=np.float64)

    for r in range(num_runs):
        t0 = time.perf_counter()
        _run_benchmark_loop(
            out,
            fwd,
            log_fk,
            strike,
            vol_sqrt_tte,
            vol2_half_tte,
            cp_mult,
        )
        t1 = time.perf_counter()

        elapsed = t1 - t0
        times[r] = elapsed
        print(f"  Run {r + 1:>2.0f}/{num_runs} elapsed: {elapsed:.4f} seconds")

    # Compute Statistics
    avg_time = np.mean(times)
    mdn_time = np.median(times)
    std_time = np.std(times)
    q75_time = np.percentile(times, 75)
    q90_time = np.percentile(times, 90)
    q95_time = np.percentile(times, 95)
    q99_time = np.percentile(times, 99)

    # Note: Minimum time yields Maximum throughput
    avg_tpt = n_evals / avg_time
    mdn_tpt = n_evals / mdn_time
    q75_tpt = n_evals / q75_time
    q90_tpt = n_evals / q90_time
    q95_tpt = n_evals / q95_time
    q99_tpt = n_evals / q99_time

    print("\n=========================================")
    print("  BLACK-76 KERNEL PRIMITIVE BENCHMARK")
    print("=========================================")
    print(f"  Evaluations per run : {n_evals:,}")
    print(f"  Total runs          : {num_runs}")
    print("-----------------------------------------")
    print("  Execution Time (Seconds):")
    print(f"    Avg : {avg_time:.4f}")
    print(f"    Mdn : {mdn_time:.4f}")
    print(f"    Std : {std_time:.4f}")
    print(f"    75% : {q75_time:.4f}")
    print(f"    90% : {q90_time:.4f}")
    print(f"    95% : {q95_time:.4f}")
    print(f"    99% : {q99_time:.4f}")
    print("-----------------------------------------")
    print("  Throughput (Million PVs/sec):")
    print(f"    Avg : {avg_tpt / 1e6:.2f}")
    print(f"    Mdn : {mdn_tpt / 1e6:.2f}")
    print(f"    75% : {q75_tpt / 1e6:.2f}")
    print(f"    90% : {q90_tpt / 1e6:.2f}")
    print(f"    95% : {q95_tpt / 1e6:.2f}")
    print(f"    99% : {q99_tpt / 1e6:.2f}")
    print("=========================================")

    print_success_message("Benchmark completed successfully.")


if __name__ == "__main__":
    run(num_runs=20)
