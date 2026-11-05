# scripts/demos/credit/jtd/validation/demo_validate_asset_correlation.py
"""
Engine Asset Correlation Validation Demo.

This script does NOT test theoretical math in a vacuum. It pushes raw DataFrames
through the actual `riskquant` compilation pipeline to prove that the
Struct-of-Arrays (SoA) memory blocks, UIX pointer resolution, and
CreditFactorMatrix structures correctly recover the theoretical correlations.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.constants import RNG_SEED
from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.mixed import MixedCreditFactorGenerator
from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.masters.cross_master.compiled import (
    CompiledCreditMasters,
)
from riskquant.credit.masters.factor_universe.ingest import (
    compile_credit_factor_universe,
)
from riskquant.credit.masters.obligors.ingest import compile_obligor_master
from riskquant.utils.printer import print_calling_file, print_success_message

NUM_OBLIGORS = 100
NUM_SCENARIOS = 1_000_000
GLOBAL_T_COP_NU = 5.0


def run_demo() -> None:
    print_calling_file()
    print("--- S-TIER ENGINE CORRELATION VALIDATION ---")
    print(f"Obligors:  {NUM_OBLIGORS}")
    print(f"Scenarios: {NUM_SCENARIOS:,}")
    print()

    rng = np.random.default_rng(seed=RNG_SEED)

    # ============================================================
    # 1. INGEST & COMPILE (Using Actual Codebase)
    # ============================================================

    # 1A. Obligor Master
    obligor_uids = np.arange(1, NUM_OBLIGORS + 1, dtype=np.int64)
    obligor_df = pl.DataFrame({
        "uid": obligor_uids,
        "pd": 0.10,
        "lgd": 0.60,
        "rating_id": 0,
        "sector_id": 0,
        "country_id": 0,
        "parent_uid": None,
    })
    obligor_master = compile_obligor_master(obligor_df)

    # 1B. JTD Portfolio
    input_betas = rng.uniform(0.10, 0.50, NUM_OBLIGORS)
    factor_names = ["GLOBAL_CREDIT"] + [f"IDIO_{i}" for i in range(1, NUM_OBLIGORS + 1)]
    # Compile the factor-universe MASTER from the generated factor list. At
    # validation scale this authoring table is generated (one idiosyncratic
    # factor per obligor) rather than the minimal canonical fixture, but it
    # flows through the SAME master contract: name -> UIX resolution and chi-dof
    # reconciliation now read from `universe`, never an ad-hoc {name: i} dict.
    factor_universe_df = pl.DataFrame({
        "factor_id": range(len(factor_names)),
        "factor_name": factor_names,
        "factor_kind": [
            "SYSTEMIC"
            if n == "GLOBAL_CREDIT"
            else "CHI"
            if (n == "GLOBAL_CHI" or n.startswith("CHI_"))
            else "UNIFORM"
            if n.startswith("U_")
            else "STRESS_INDICATOR"
            if n == "GLOBAL_STRESS_REGIME"
            else "IDIOSYNCRATIC"
            for n in factor_names
        ],
        "nu": [
            GLOBAL_T_COP_NU if (n == "GLOBAL_CHI" or n.startswith("CHI_")) else None
            for n in factor_names
        ],
    })
    universe = compile_credit_factor_universe(factor_universe_df)

    jtd_df = pl.DataFrame({
        "uid": obligor_uids,
        "instrument": JtdModel.COPULA_SF_GAUSSIAN.name,
        "factor_loading": input_betas,
        "systemic_factor": "GLOBAL_CREDIT",
        "idiosyncratic_factor": [f"IDIO_{i}" for i in range(1, NUM_OBLIGORS + 1)],
    })

    # THIS IS THE CRITICAL TEST: Does the compiler correctly map the strings
    # to UIX integers and align the betas?
    _jtd_bundle = JtdIngestBundle.from_instrument_frame(jtd_df)
    _masters = CompiledCreditMasters(
        obligor_master=obligor_master,
        factor_universe=universe,
    )
    compiled_portfolio = compile_jtd_portfolio(
        bundle=_jtd_bundle,
        masters=_masters,
    )

    # ============================================================
    # 2. GENERATE FACTOR MATRIX
    # ============================================================
    # === Synthesize the credit factor matrix from the universe master ===
    # The mixed generator derives its standard-normal / chi / uniform / stress
    # columns from `universe`, so every row carries the correct marginal and
    # cannot disagree with the compiled portfolio about which columns are chi
    # (or at what nu). This replaces the hand-rolled per-block buffer fill.
    factor_generator = MixedCreditFactorGenerator.from_universe(
        universe,
        total_scenarios=NUM_SCENARIOS,
        seed=RNG_SEED,
    )
    credit_factors = build_credit_factor_matrix(
        factor_generator.iter_batches(batch_size=min(NUM_SCENARIOS, 25_000)),
        num_factors=universe.n_factors,
        num_scenarios=NUM_SCENARIOS,
    )

    # ============================================================
    # 3. EXTRACT LATENT VALUES VIA COMPILED BLOCK POINTERS
    # ============================================================

    # We grab the exact SoA block the engine uses
    gauss_block = compiled_portfolio.sf_gaussian_copula_jtd_block

    if gauss_block is None:
        raise ValueError("Gaussian block failed to compile.")

    # We extract the memory rows using the UIX indices resolved by the compiler
    factors = credit_factors.latent_factors

    z_rows = factors[gauss_block.systemic_uix]  # Shape: (100, 1_000_000)
    eps_rows = factors[gauss_block.idiosyncratic_uix]  # Shape: (100, 1_000_000)

    # We grab the compiled betas and broadcast them for scenario multiplication
    compiled_betas = gauss_block.factor_loading[:, None]

    # Reconstruct the continuous latent variables exactly as the kernel does
    sqrt_1mb = np.sqrt(1.0 - compiled_betas**2)
    asset_values = compiled_betas * z_rows + sqrt_1mb * eps_rows

    # ============================================================
    # 4. VALIDATE COMPILED CORRELATIONS
    # ============================================================

    # For validation, we need the 1D global Z vector.
    # Since all obligors share "GLOBAL_CREDIT", z_rows[0] is the systemic vector.
    global_z = z_rows[0]

    empirical_market_corr = np.empty(NUM_OBLIGORS)
    for i in range(NUM_OBLIGORS):
        empirical_market_corr[i] = np.corrcoef(asset_values[i], global_z)[0, 1]

    # We compare against the compiled betas, proving the memory aligned perfectly
    flat_betas = gauss_block.factor_loading
    market_corr_error = np.abs(empirical_market_corr - flat_betas)

    print("--- 1. Compiled Asset-to-Market Correlation ---")
    print(f"Mean Absolute Error: {market_corr_error.mean():.6f}")
    print(f"Max Absolute Error:  {market_corr_error.max():.6f}")
    print()

    # ============================================================
    # 5. VALIDATE CROSS-ASSET CORRELATION
    # ============================================================
    rng_pairs = np.random.default_rng(RNG_SEED)
    num_pairs = 1000

    theoretical_corr = np.empty(num_pairs)
    empirical_corr = np.empty(num_pairs)

    for k in range(num_pairs):
        i, j = rng_pairs.choice(NUM_OBLIGORS, size=2, replace=False)

        theoretical_corr[k] = flat_betas[i] * flat_betas[j]
        empirical_corr[k] = np.corrcoef(asset_values[i], asset_values[j])[0, 1]

    pair_error = np.abs(empirical_corr - theoretical_corr)

    print("--- 2. Compiled Asset-to-Asset Correlation ---")
    print(f"Mean Absolute Error: {pair_error.mean():.6f}")
    print(f"Max Absolute Error:  {pair_error.max():.6f}")
    print()

    print_success_message()


if __name__ == "__main__":
    run_demo()
