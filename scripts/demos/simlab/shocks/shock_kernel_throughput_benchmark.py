# scripts/demos/simlab/shocks/shock_kernel_throughput_benchmark.py
import time

import numpy as np

from riskquant.constants import RNG_SEED
from riskquant.simlab.shocks.shock_kernel import _kernel_apply_shocks
from riskquant.utils.printer import (
    print_calling_file,
    print_header,
    print_success_message,
)


def main():
    print_calling_file()

    print_header(title="Shock kernel throughput benchmark")

    for b, s in [
        (2_000, 10_000),
        (1_000_000, 2),
        (100_000, 10),
        (10_000_000, 10),
        (1_000_000, 100),
        (100_000, 1000),
        (10_000, 10_000),
        (1_000, 100_000),
        (100, 1_000_000),
    ]:
        n_base_lvls = b
        s_scenarios = s

        rng = np.random.default_rng(seed=RNG_SEED)

        base_levels = rng.random(n_base_lvls)
        req_to_uix = rng.integers(0, 500, n_base_lvls, dtype=np.int32)
        shock_scale = np.ones(n_base_lvls)
        shift_types = np.ones(n_base_lvls, dtype=np.int8)

        shocks = np.asfortranarray(
            rng.normal(size=(s_scenarios, 500)).astype(np.float32)
        )

        out = np.empty((n_base_lvls, s_scenarios), dtype=np.float64)

        # warmup
        _kernel_apply_shocks(
            base_levels,
            shocks,
            req_to_uix,
            shock_scale,
            shift_types,
            out,
        )

        start = time.perf_counter()

        _kernel_apply_shocks(
            base_levels,
            shocks,
            req_to_uix,
            shock_scale,
            shift_types,
            out,
        )

        elapsed = time.perf_counter() - start

        ops = n_base_lvls * s_scenarios

        print("-" * 80)
        print(
            f"Applied shocks to {n_base_lvls:,} base levels across {s_scenarios:,} scenarios: \n"
        )
        print(f"- Elapsed: {elapsed:.4f}s")
        print(f"- Shock applications: {ops:,}")
        print(f"- Shock applications/sec: {ops / elapsed:,.0f}")
        print(f"- Million shock applications/sec: {(ops / elapsed) / 1e6:.2f}")
        print("-" * 80 + "\n")

    print_success_message(message="Shock kernel throughput benchmark complete.")


if __name__ == "__main__":
    main()
