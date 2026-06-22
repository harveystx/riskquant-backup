import time

import numba as nb
import numpy as np

from riskquant.valuation.pricing_models.black76.model import (
    _black76_fv_kernel_primitive,
)

# --- THE BENCHMARK HARNESS ---
# We wrap the scalar primitive in a parallelized loop to simulate
# executing across millions of scenarios/trades.


@nb.njit(
    parallel=True,
    fastmath={"contract", "reassoc", "nsz"},
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
    N_EVALS = 250_000_000
    print(f"Allocating data for {N_EVALS:,} evaluations...")

    # Allocate C-contiguous arrays
    fwd = np.random.uniform(90.0, 110.0, size=N_EVALS).astype(np.float64)
    strike = np.random.uniform(90.0, 110.0, size=N_EVALS).astype(np.float64)
    log_fk = np.log(fwd / strike).astype(np.float64)

    vol = np.random.uniform(0.1, 0.4, size=N_EVALS).astype(np.float64)
    tte = np.random.uniform(0.1, 5.0, size=N_EVALS).astype(np.float64)

    vol_sqrt_tte = (vol * np.sqrt(tte)).astype(np.float64)
    vol2_half_tte = (vol * vol * 0.5 * tte).astype(np.float64)

    cp_mult = np.random.choice(np.array([1, -1], dtype=np.int8), size=N_EVALS)

    out = np.zeros(N_EVALS, dtype=np.float64)

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

    print(f"Running benchmark {num_runs} times...")

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
        print(f"  Run {r + 1}/{num_runs} elapsed: {elapsed:.4f} seconds")

    # Compute Statistics
    min_time = np.min(times)
    max_time = np.max(times)
    avg_time = np.mean(times)

    # Note: Minimum time yields Maximum throughput
    max_tpt = N_EVALS / min_time
    min_tpt = N_EVALS / max_time
    avg_tpt = N_EVALS / avg_time

    print("\n=========================================")
    print("  BLACK-76 PRIMITIVE BENCHMARK")
    print("=========================================")
    print(f"  Evaluations per run : {N_EVALS:,}")
    print(f"  Total runs          : {num_runs}")
    print("-----------------------------------------")
    print("  Execution Time (Seconds):")
    print(f"    Min : {min_time:.4f}")
    print(f"    Max : {max_time:.4f}")
    print(f"    Avg : {avg_time:.4f}")
    print("-----------------------------------------")
    print("  Throughput (Million PVs/sec):")
    print(f"    Max : {max_tpt / 1e6:.2f}")
    print(f"    Min : {min_tpt / 1e6:.2f}")
    print(f"    Avg : {avg_tpt / 1e6:.2f}")
    print("=========================================\n")


if __name__ == "__main__":
    run(num_runs=5)
