# test_black_scholes_3d.py

from time import time

import numpy as np

from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant.valuation.models.black_scholes import (
    fv_black_scholes_vanilla_3d,
    pv_black_scholes_vanilla_3d,
)


def generate_test_inputs(N=5, S=1000, seed=42):
    rng = np.random.default_rng(seed)

    fwd = rng.lognormal(mean=4.5, sigma=0.2, size=(N, S))
    strike = rng.uniform(70, 120, size=(N, 1))
    vol = rng.uniform(0.1, 0.5, size=(N, S))
    tte = rng.uniform(0.1, 2.0, size=(N, 1))
    is_call = rng.integers(0, 2, size=(N, 1))
    df_settle = rng.uniform(0.90, 1.0, size=(N, S))

    return fwd, strike, vol, tte, is_call, df_settle


def print_run_time(start_time: float, end_time: float):
    elapsed_ss = end_time - start_time
    elapsed_hh_mm_ss = f"{int(elapsed_ss // 3600):02d}:{int((elapsed_ss % 3600) // 60):02d}:{elapsed_ss % 60:06.3f}"
    print(f"\nElapsed time: {elapsed_hh_mm_ss} ({elapsed_ss:.3f} seconds)\n")


def run_basic_tests():
    # Print the calling file info at the start of the test run
    print_calling_file()

    # start_time = time.perf_counter()
    start_time = time()

    # Parameters for test data generation
    N = 100
    S = 10_000

    fwd, strike, vol, tte, is_call, df_settle = generate_test_inputs(N, S)

    # ---- Run pricing functions ----
    undiscounted = fv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=strike,
        vol=vol,
        tte=tte,
        is_call=is_call,
    )

    pv = pv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=strike,
        vol=vol,
        tte=tte,
        is_call=is_call,
        df_settle=df_settle,
    )

    # ---- Shape checks ----
    assert undiscounted.shape == (N, S), "Undiscounted output shape incorrect"
    assert pv.shape == (N, S), "PV output shape incorrect"

    print("Shape test passed")

    # ---- Discounting check ----
    diff = np.abs(pv - undiscounted * df_settle).max()
    assert diff < 1e-10, "Discounting logic incorrect"

    print("Discounting test passed")

    # ---- Basic positivity checks ----
    assert np.all(
        undiscounted >= -1e-12
    ), "Option values should be non-negative"

    print("Non-negativity test passed")

    # ---- Print sample results ----
    print("\nSample outputs:")
    print("Undiscounted values (first option, first 10 scenarios):")
    print(undiscounted[0, :10])

    print("\nPV values (first option, first 10 scenarios):")
    print(pv[0, :10])

    end_time = time()

    print_run_time(start_time, end_time)

    print_success_message()


if __name__ == "__main__":
    run_basic_tests()
