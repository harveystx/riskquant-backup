# scripts/demo_rq_simlab_scenarios_var_loop_bootstrap_sbb.py
import time
from datetime import date

import numpy as np

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.shocks.shock_kernel import (
    apply_shocks_to_base_levels,
)
from riskquant.simlab.scenarios.bootstrap_sbb import (
    StationaryBlockBootstrapGenerator,
)
from riskquant.utils.printer import print_calling_file, print_success_message


def run_var_loop_demo():
    print_calling_file()

    MASTER_SEED = 42
    rng = np.random.default_rng(MASTER_SEED)

    # 1. Setup Mock Dimensions
    T, N_req = 5_001, 10_000  # 5,001 days of history, 10,000 portfolio nodes
    horizon = 10
    expected_block_size = 10
    total_scenarios = 5_000  # Explicitly defined for MC Generators

    print(f"Initializing Engine: {N_req} exposures x {total_scenarios} scenarios...")

    # 2. Mock Market Data (Universe B)
    factor_ids = np.arange(1, N_req + 1, dtype=np.int32)

    # Generate realistic price paths (Cumprod of lognormal returns)
    # Uniform prices create massive, unrealistic phantom shocks in a bootstrap
    values_history = rng.lognormal(mean=0.0001, sigma=0.015, size=(T, N_req))
    values_history = 100.0 * np.cumprod(values_history, axis=0)

    # HARDWARE RULE: Must be Fortran-contiguous for optimal column-wise time-series reads
    values_history = np.asfortranarray(values_history, dtype=np.float64)

    base_ordinal = date(2010, 1, 1).toordinal()
    dates = np.arange(base_ordinal, base_ordinal + T, dtype=np.int32)

    matrix = RiskFactorMatrix(
        factor_ids=factor_ids,
        dates=dates,
        values=values_history,
    )

    # 3. Mock Portfolio Requirements (Universe A)
    # Mix shift types for the application kernel
    available_shifts = np.array(
        [SHIFT_LOG_RETURN_ID, SHIFT_SIMPLE_RETURN_ID, SHIFT_DIFFERENCE_ID],
        dtype=np.int8,
    )
    req_shift_types = rng.choice(available_shifts, size=N_req)

    shock_scales = np.ones(N_req, dtype=np.float32)

    # "Today's" prices from the very last row (T-1)
    t0_base_levels = values_history[-1, :]

    # 4. PRE-ALLOCATE THE MASSIVE OUTPUT ARRAY ONCE
    # Shape: (10000, 5000) float64 -> 400 MB of RAM
    print(f"Pre-allocating master pricing grid: (N={N_req}, S={total_scenarios})")
    future_prices_ns = np.empty((N_req, total_scenarios), dtype=np.float64, order="F")

    print("\n--- Initializing SBB Generator ---")

    # 5. Initialize Generator
    generator = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=total_scenarios,
        expected_block_size=expected_block_size,  # Preserves volatility clustering
        horizon=horizon,
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

    # Reset generator to clear the internal scenario_offset state
    generator = StationaryBlockBootstrapGenerator(
        matrix=matrix,
        total_scenarios=total_scenarios,
        expected_block_size=expected_block_size,
        horizon=horizon,
        seed=MASTER_SEED,
        shock_dtype=np.float32,
    )

    # 6. EXECUTE THE HOT LOOP
    print("\n--- Executing God-Tier Vectorized Loop ---\n")

    t0 = time.perf_counter()

    # The actual execution loop
    for batch in generator.iter_batches(
        required_shift_type_id=req_shift_types,
        source_factor_ids=factor_ids,
        shock_scale=shock_scales,
        batch_size=250,
    ):
        start_idx = batch.scenario_offset
        end_idx = start_idx + batch.shocks.shape[0]

        # PASS A SLICE of the pre-allocated array.
        # Numba will scatter-write directly into this memory.
        apply_shocks_to_base_levels(
            base_levels=t0_base_levels,
            batch=batch,
            out=future_prices_ns[:, start_idx:end_idx],
        )

    t1 = time.perf_counter()

    # 7. Audit Performance
    total_nodes_generated = N_req * total_scenarios
    time_ms = (t1 - t0) * 1_000
    throughput = total_nodes_generated / (t1 - t0)

    print(f"Generation Loop Completed in: {time_ms:.2f} ms")
    print(f"Total Pricing Nodes Created:  {total_nodes_generated:,.0f}")
    print(f"Throughput:                   {throughput:,.0f} nodes / second")

    print("\nSample Future Prices (First 5 Scenarios for Exposure 0):")
    print(future_prices_ns[0, :5])

    print_success_message()


if __name__ == "__main__":
    run_var_loop_demo()
