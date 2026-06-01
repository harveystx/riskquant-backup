# scripts/demo_rq_simlab_scenarios_gen_bootstrap_fhs.py
import time
from datetime import date

import numpy as np

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.bootstrap_fhs import (
    FilteredBootstrapGenerator,
)
from riskquant.utils.printer import print_calling_file, print_success_message


def run_generator_demo():

    print_calling_file()

    MASTER_SEED = 42
    rng = np.random.default_rng(MASTER_SEED)

    print("--- Mocking Historical Risk Factor Matrix ---")
    # 5001 days of history, 5000 unique risk factors
    T, K_total = 5_001, 5_000
    factor_ids = np.arange(1, K_total + 1, dtype=np.int32)
    values = rng.lognormal(mean=0.0001, sigma=0.015, size=(T, K_total))

    # Cumprod to create realistic price paths
    values = 100.0 * np.cumprod(values, axis=0)

    # HARDWARE RULE: Must be Fortran-contiguous for optimal column-wise time-series reads
    values = np.asfortranarray(values, dtype=np.float64)

    # Generate T consecutive ordinal dates (e.g., starting from 2010-01-01)
    base_ordinal = date(2010, 1, 1).toordinal()
    dates = np.arange(base_ordinal, base_ordinal + T, dtype=np.int32)

    # Initialize the matrix
    matrix = RiskFactorMatrix(
        factor_ids=factor_ids,
        dates=dates,
        values=values,
    )

    # Mock ProxyResolution outputs (What the portfolio needs)
    N_req = int(K_total * 0.5)  # 50% of the total factors are required
    req_source_ids = rng.choice(factor_ids, size=N_req, replace=True).astype(np.int32)

    # MIX THE SHIFT TYPES: Randomly assign Log, Simple, or Difference returns
    available_shifts = np.array(
        [SHIFT_LOG_RETURN_ID, SHIFT_SIMPLE_RETURN_ID, SHIFT_DIFFERENCE_ID],
        dtype=np.int8,
    )
    req_shift_types = rng.choice(available_shifts, size=N_req)

    shock_scales = np.ones(N_req, dtype=np.float32)

    # Print distribution diagnostic
    log_count = np.sum(req_shift_types == SHIFT_LOG_RETURN_ID)
    simple_count = np.sum(req_shift_types == SHIFT_SIMPLE_RETURN_ID)
    diff_count = np.sum(req_shift_types == SHIFT_DIFFERENCE_ID)

    print(f"Shift Type Distribution across N={N_req} required exposures:")
    print(f"  - Log Returns:    {log_count}")
    print(f"  - Simple Returns: {simple_count}")
    print(f"  - Differences:    {diff_count}")

    print("\n--- Initializing Filtered Bootstrap Generator ---\n")
    total_scenarios = 5_000

    # Init
    generator = FilteredBootstrapGenerator(
        matrix=matrix,
        total_scenarios=total_scenarios,
        decay_factor=0.94,
        horizon=10,
        seed=MASTER_SEED,
        shock_dtype=np.float32,
    )

    # Warm up Numba compilers before timing
    _warmup_batch = next(
        generator.iter_batches(
            required_shift_type_id=req_shift_types[:10],
            source_factor_ids=factor_ids[:10],
            shock_scale=shock_scales[:10],
            batch_size=2,
        )
    )

    # Init
    generator = FilteredBootstrapGenerator(
        matrix=matrix,
        total_scenarios=total_scenarios,
        decay_factor=0.94,
        horizon=10,
        seed=MASTER_SEED,
        shock_dtype=np.float32,
    )

    print("\n--- Streaming Scenario Batches ---\n")
    total_scenarios_processed = 0
    first_shock_value = None

    st = time.time()

    for i, batch in enumerate(
        generator.iter_batches(
            required_shift_type_id=req_shift_types,
            source_factor_ids=req_source_ids,
            shock_scale=shock_scales,
            batch_size=250,
        )
    ):
        B = batch.shocks.shape[0]
        K = batch.shocks.shape[1]
        total_scenarios_processed += B

        if i == 0:
            first_shock_value = batch.shocks[0, 0]

        print(
            f"    - "
            f"Yielded Batch: Offset={batch.scenario_offset:04d} | "
            f"Batch Shape (B,K)=({B}, {K}) | "
            f"Memory={batch.shocks.nbytes / 1024:.1f} KB"
        )

    et = time.time()
    elapsed_sec = et - st

    print(
        f"\nSuccessfully generated {total_scenarios_processed} scenarios in {elapsed_sec:.2f} seconds."
    )
    print(f"Deterministic Checkpoint (First Shock): {first_shock_value:.8f}")

    print_success_message()


if __name__ == "__main__":
    run_generator_demo()
