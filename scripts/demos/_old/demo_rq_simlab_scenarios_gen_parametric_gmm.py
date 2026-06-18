# scripts/demo_rq_simlab_scenarios_gen_parametric_gmm.py
import time

import numpy as np

from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.parametric import ParametricGenerator
from riskquant.utils.printer import print_calling_file, print_success_message


def run_generator_demo():
    print_calling_file()

    # ---------------------------------------------------------
    # 1. STRICT LOCAL DETERMINISM (NO GLOBAL SEEDING)
    # ---------------------------------------------------------
    MASTER_SEED = 42
    rng = np.random.default_rng(MASTER_SEED)

    print("--- Mocking Parametric Calibration Data ---")
    K_total = 5_000
    available_factor_ids = np.arange(1, K_total + 1, dtype=np.int32)

    cholesky_matrix = np.eye(K_total, dtype=np.float64)
    volatilities = np.full(K_total, 0.015, dtype=np.float64)

    # Use the local RNG to generate mock inputs
    heterogeneous_multipliers = rng.uniform(1.5, 4.0, size=K_total)

    print("\n--- Initializing Parametric MC Scenario Generator ---\n")
    total_scenarios = 5_000

    # Initialize the Regime-Switching GMM, passing the master seed
    generator = ParametricGenerator.regime_switching_gmm(
        prob_crisis=0.05,
        crisis_vol_multipliers=heterogeneous_multipliers,
        available_factor_ids=available_factor_ids,  # <-- REQUIRED: Maps the universe to the Cholesky matrix
        cholesky_matrix=cholesky_matrix,
        volatilities=volatilities,
        total_scenarios=total_scenarios,
        seed=MASTER_SEED,
        shock_dtype=np.float32,
    )

    # ---------------------------------------------------------
    # 2. MOCK PORTFOLIO REQUIREMENTS
    # ---------------------------------------------------------
    N_req = int(K_total * 0.5)

    # Use local RNG to pick required factors
    req_source_ids = rng.choice(available_factor_ids, size=N_req, replace=False).astype(
        np.int32
    )

    available_shifts = np.array(
        [SHIFT_LOG_RETURN_ID, SHIFT_SIMPLE_RETURN_ID, SHIFT_DIFFERENCE_ID],
        dtype=np.int8,
    )
    req_shift_types = rng.choice(available_shifts, size=N_req)
    shock_scales = np.ones(N_req, dtype=np.float32)

    log_count = np.sum(req_shift_types == SHIFT_LOG_RETURN_ID)
    simple_count = np.sum(req_shift_types == SHIFT_SIMPLE_RETURN_ID)
    diff_count = np.sum(req_shift_types == SHIFT_DIFFERENCE_ID)

    print(f"Shift Type Distribution across N={N_req} required exposures:")
    print(f"  - Log Returns:    {log_count}")
    print(f"  - Simple Returns: {simple_count}")
    print(f"  - Differences:    {diff_count}")

    # ---------------------------------------------------------
    # 3. STREAM AND VERIFY
    # ---------------------------------------------------------
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

        # Capture the very first shock generated to prove determinism
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
