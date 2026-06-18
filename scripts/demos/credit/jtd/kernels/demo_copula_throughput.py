# scripts/demos/credit/jtd/kernels/demo_copula_throughput.py

import time

import numpy as np

from riskquant.constants import RNG_SEED

# ============================================================
# IMPORT PRODUCTION KERNELS
# ============================================================
from riskquant.credit.jtd.kernels.sf_copula_gaussian import (
    kernel_sf_gaussian_copula_jtd,
)
from riskquant.credit.jtd.kernels.sf_copula_students_t import (
    kernel_sf_students_t_copula_jtd,
)
from riskquant.utils.printer import print_calling_file, print_success_message

# ============================================================
# BENCHMARK RUNNER
# ============================================================

M_OBLIGORS = 2_500
S_SCENARIOS = 200_000

NUM_SYS_FACTORS = 100
NUM_IDIO_FACTORS = M_OBLIGORS
NUM_CHI_FACTORS = M_OBLIGORS

F_FACTORS = NUM_SYS_FACTORS + NUM_IDIO_FACTORS + NUM_CHI_FACTORS

TOTAL_EVALS = M_OBLIGORS * S_SCENARIOS


def _rng():
    """Fixed seed for reproducibility."""
    return np.random.default_rng(seed=RNG_SEED)


def run_benchmark():
    print_calling_file()

    # ============================================================
    # SCALE PARAMETERS
    # ============================================================

    print("--- S-TIER COPULA JTD KERNELS BENCHMARK ---")
    print(f"Obligors:     {M_OBLIGORS:,}")
    print(f"Scenarios:    {S_SCENARIOS:,}")
    print(f"Latent Pool:  {F_FACTORS:,} factors total")
    print(
        f"Total Evals:  {TOTAL_EVALS:,} Default Simulations "
        f"({TOTAL_EVALS / 1_000_000_000:.2f} Billion Evals)"
    )
    print("-" * 46)

    # ============================================================
    # MEMORY ALLOCATION
    # ============================================================

    print("Allocating engine memory...")
    print("-" * 46)

    out = np.empty(
        (M_OBLIGORS, S_SCENARIOS),
        dtype=np.float64,
    )

    rng = _rng()

    # ============================================================
    # EXPOSURE PATHS
    # ============================================================

    ead = rng.uniform(
        1e6,
        10e6,
        (M_OBLIGORS, S_SCENARIOS),
    ).astype(np.float64)

    # ============================================================
    # LATENT FACTOR POOL
    # ============================================================
    #
    # Layout:
    #
    #   [ Systematic Gaussian Factors ]
    #   [ Idiosyncratic Gaussian Factors ]
    #   [ Student-T Scaling Factors ]
    #
    # IMPORTANT
    # ---------
    # The production Student-T kernel expects rows referenced
    # by chi_uix to contain:
    #
    #     inv_sqrt_w = 1 / sqrt(W)
    #
    # and NOT raw chi-square draws.
    #
    # This mirrors the production architecture where the
    # expensive square-root transformation is performed once
    # during factor generation rather than inside the kernel.
    #

    systematic_factors = rng.standard_normal((NUM_SYS_FACTORS, S_SCENARIOS)).astype(
        np.float64
    )

    idiosyncratic_factors = rng.standard_normal((NUM_IDIO_FACTORS, S_SCENARIOS)).astype(
        np.float64
    )

    # ------------------------------------------------------------
    # Student-T scaling factors
    # ------------------------------------------------------------
    #
    # For throughput benchmarking we do not require an exact
    # chi-square process. We only need positive values that
    # satisfy the production kernel contract:
    #
    #     inv_sqrt_w = 1 / sqrt(W)
    #
    # where W > 0.
    #

    chi_proxy = (
        np.abs(rng.standard_normal((NUM_CHI_FACTORS, S_SCENARIOS))).astype(np.float64)
        + 0.1
    )

    inv_sqrt_w = 1.0 / np.sqrt(chi_proxy)

    latent_factors = np.vstack(
        (
            systematic_factors,
            idiosyncratic_factors,
            inv_sqrt_w,
        )
    )

    # ============================================================
    # OBLIGOR PARAMETERS
    # ============================================================

    lgd = rng.uniform(
        0.2,
        0.8,
        M_OBLIGORS,
    ).astype(np.float64)

    factor_loading = rng.uniform(
        0.1,
        0.9,
        M_OBLIGORS,
    ).astype(np.float64)

    thresholds = rng.uniform(
        -3.0,
        -1.0,
        M_OBLIGORS,
    ).astype(np.float64)

    # Obligor-specific Student-T degrees of freedom.
    nu = rng.uniform(
        3.0,
        10.0,
        M_OBLIGORS,
    ).astype(np.float64)
    sqrt_nu = np.sqrt(nu)

    # ============================================================
    # FACTOR INDEX MAPPINGS
    # ============================================================

    sys_start = 0
    idio_start = NUM_SYS_FACTORS
    chi_start = NUM_SYS_FACTORS + NUM_IDIO_FACTORS

    sys_uix = rng.integers(
        sys_start,
        idio_start,
        M_OBLIGORS,
    ).astype(np.int32)

    idio_uix = rng.integers(
        idio_start,
        chi_start,
        M_OBLIGORS,
    ).astype(np.int32)

    chi_uix = rng.integers(
        chi_start,
        F_FACTORS,
        M_OBLIGORS,
    ).astype(np.int32)

    # ============================================================
    # VALIDATE MEMORY LAYOUT
    # ============================================================

    assert out.flags.c_contiguous
    assert ead.flags.c_contiguous
    assert latent_factors.flags.c_contiguous

    runs = 10

    # ============================================================
    # GAUSSIAN COPULA BENCHMARK
    # ============================================================

    print("\n[1] Warming up Gaussian Copula JIT...")

    start_compile = time.perf_counter()

    kernel_sf_gaussian_copula_jtd(
        out=out[:10, :],
        latent_factors=latent_factors,
        ead=ead[:10, :],
        lgd=lgd[:10],
        factor_loading=factor_loading[:10],
        thresholds=thresholds[:10],
        sys_uix=sys_uix[:10],
        idio_uix=idio_uix[:10],
    )

    print(f"Compilation finished in {time.perf_counter() - start_compile:.4f}s")
    print("-" * 46)

    print(f"Executing {TOTAL_EVALS:,} Gaussian JTD Simulations...")

    total_time_gaussian = 0.0

    for r in range(runs):
        start_time = time.perf_counter()

        kernel_sf_gaussian_copula_jtd(
            out=out,
            latent_factors=latent_factors,
            ead=ead,
            lgd=lgd,
            factor_loading=factor_loading,
            thresholds=thresholds,
            sys_uix=sys_uix,
            idio_uix=idio_uix,
        )

        run_time = time.perf_counter() - start_time
        total_time_gaussian += run_time

        print(f"  Run {r + 1}: {run_time:.4f} seconds")

    avg_time_gauss = total_time_gaussian / runs
    tput_gauss = TOTAL_EVALS / avg_time_gauss

    # Approximate bytes touched per evaluation.
    bw_gauss = (TOTAL_EVALS * 32 / avg_time_gauss) / (1024**3)

    print("-" * 46)
    print(f"Gaussian Avg Time:  {avg_time_gauss:.4f} seconds")
    print(f"Gaussian Throughput:{tput_gauss / 1_000_000_000:.2f} Billion Evals/sec")
    print(f"Gaussian Traffic:   {bw_gauss:.2f} GB/s")
    print("-" * 46)

    # ============================================================
    # STUDENT-T COPULA BENCHMARK
    # ============================================================

    # out.fill(0.0)

    print("\n[2] Warming up Student-T Copula JIT...")

    start_compile = time.perf_counter()

    kernel_sf_students_t_copula_jtd(
        out=out[:10, :],
        latent_factors=latent_factors,
        ead=ead[:10, :],
        lgd=lgd[:10],
        factor_loading=factor_loading[:10],
        thresholds=thresholds[:10],
        sqrt_nu=sqrt_nu[:10],
        sys_uix=sys_uix[:10],
        idio_uix=idio_uix[:10],
        chi_uix=chi_uix[:10],
    )

    print(f"Compilation finished in {time.perf_counter() - start_compile:.4f}s")
    print("-" * 46)

    print(f"Executing {TOTAL_EVALS:,} Student-T JTD Simulations...")

    total_time_t = 0.0

    for r in range(runs):
        start_time = time.perf_counter()

        kernel_sf_students_t_copula_jtd(
            out=out,
            latent_factors=latent_factors,
            ead=ead,
            lgd=lgd,
            factor_loading=factor_loading,
            thresholds=thresholds,
            sqrt_nu=sqrt_nu,
            sys_uix=sys_uix,
            idio_uix=idio_uix,
            chi_uix=chi_uix,
        )

        run_time = time.perf_counter() - start_time
        total_time_t += run_time

        print(f"  Run {r + 1}: {run_time:.4f} seconds")

    avg_time_t = total_time_t / runs
    tput_t = TOTAL_EVALS / avg_time_t

    # Approximate bytes touched per evaluation.
    bw_t = (TOTAL_EVALS * 40 / avg_time_t) / (1024**3)

    print("-" * 46)
    print(f"Student-T Avg Time:  {avg_time_t:.4f} seconds")
    print(f"Student-T Throughput:{tput_t / 1_000_000_000:.2f} Billion Evals/sec")
    print(f"Student-T Traffic:   {bw_t:.2f} GB/s")
    print("-" * 46)

    print_success_message()


if __name__ == "__main__":
    run_benchmark()
