# scripts/demos/credit/exposure/evaluators/kernels/rc_kernel_throughput.py

import time

import numpy as np

from riskquant.constants import RNG_SEED

# ============================================================
# IMPORT PRODUCTION KERNEL
# ============================================================
from riskquant.credit.exposure.evaluators.kernels.replacement_cost import (
    _kernel_replacement_cost,
)
from riskquant.utils.printer import print_calling_file, print_success_message

# ============================================================
# BENCHMARK RUNNER
# ============================================================


def _rng():
    # Fixed seed for reproducibility
    return np.random.default_rng(seed=RNG_SEED)


def run_benchmark():
    print_calling_file()

    # --- 1. Scale Parameters ---
    m_trades = 10_000  # 100k Trades
    s_scenarios = 1_000  # 10k Scenarios
    n_netting_sets = 500  # 5k Counterparties (Avg 20 trades per Netting Set)

    print("--- S-TIER REPLACEMENT COST KERNEL BENCHMARK ---")
    print(f"Trades:       {m_trades:,}")
    print(f"Netting Sets: {n_netting_sets:,}")
    print(f"Scenarios:    {s_scenarios:,}")
    print(f"Total Aggrs:  {m_trades * s_scenarios:,} (1 Billion MtM additions)")
    print("-" * 46)

    # --- 2. Allocate C-Contiguous Memory ---
    print("Allocating engine memory...")

    # Pre-allocate output matrix
    out = np.empty((n_netting_sets, s_scenarios), dtype=np.float64)

    rng = _rng()

    # Generate random MtM matrix
    mtm = rng.uniform(-1000.0, 1000.0, (m_trades, s_scenarios)).astype(np.float64)

    # Generate random collateral per netting set
    collateral = rng.uniform(0.0, 5000.0, n_netting_sets).astype(np.float64)

    # Create CSR Netting Set Offsets (Assume evenly distributed 20 trades per set)
    trades_per_set = m_trades // n_netting_sets
    ns_offsets = np.arange(0, m_trades + 1, trades_per_set, dtype=np.int64)

    # Create and shuffle the permutation array to simulate realistic memory fragmentation.
    # The CPU cannot just stream the `mtm` matrix linearly; it has to gather.
    mtm_row_perm = np.arange(m_trades, dtype=np.int64)
    rng.shuffle(mtm_row_perm)

    # Validate memory layout
    assert out.flags.c_contiguous
    assert mtm.flags.c_contiguous

    # Outer loop for stability check
    for _ in range(3):
        # --- 3. JIT Warmup ---
        print("Warming up JIT compiler...")
        start_compile = time.perf_counter()

        # Call the imported kernel with a small slice (10 netting sets)
        # We pass the full mtm and perm arrays, but the kernel only reads up to ns_offsets[10]
        _kernel_replacement_cost(
            out[:10, :],
            mtm,
            mtm_row_perm,
            ns_offsets[:11],
            collateral[:10],
        )

        print(
            f"Compilation/Cache-load finished in {time.perf_counter() - start_compile:.4f}s"
        )
        print("-" * 46)

        # --- 4. The Hot Run ---
        print("Executing 1 Billion MtM Aggregations...")
        runs = 10
        total_time = 0.0

        for r in range(runs):
            start_time = time.perf_counter()

            # Call the imported kernel
            _kernel_replacement_cost(out, mtm, mtm_row_perm, ns_offsets, collateral)

            run_time = time.perf_counter() - start_time
            total_time += run_time
            print(f"  Run {r + 1}: {run_time:.4f} seconds")

        # --- 5. Throughput Math ---
        avg_time = total_time / runs
        total_aggregations = m_trades * s_scenarios
        throughput = total_aggregations / avg_time

        # Memory Traffic Calc (Read 8 bytes per MtM, Read/Write 8 bytes per output)
        bytes_read = total_aggregations * 8
        bytes_written = (n_netting_sets * s_scenarios) * 8
        bandwidth_gb_s = ((bytes_read + bytes_written) / avg_time) / (1024**3)

        print("-" * 46)
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Throughput:   {throughput / 1_000_000_000:.2f} Billion Aggrs/sec")
        print(f"Mem Traffic:  {bandwidth_gb_s:.2f} GB/s")
        print("-" * 46)

    print_success_message()


if __name__ == "__main__":
    run_benchmark()
