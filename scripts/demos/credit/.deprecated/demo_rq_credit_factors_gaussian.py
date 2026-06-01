# scripts/demos/demo_rq_credit_factors_gaussian.py
"""
Demo: Gaussian-copula latent-factor generation.

Walks the streaming generator through several batches, assembles the
`CreditFactorMatrix`, and verifies sample moments. All factors are
independent standard normals — dependence is applied downstream in the
JTD kernels via `beta`.
"""

from __future__ import annotations

from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.parametric_copula_gaussian import (
    GaussianCopulaGenerator,
)
from riskquant.ingest_fixtures.credit_factors.minimal_factor_universe import (
    CANONICAL_CREDIT_FACTOR_NAMES,
    CANONICAL_FACTOR_NAMES,
    CANONICAL_NUM_FACTORS,
)

NUM_SCENARIOS: int = 100_000
BATCH_SIZE: int = 16_384
SEED: int = 42


def main() -> None:
    print("=" * 70)
    print(" GAUSSIAN COPULA LATENT-FACTOR GENERATION ".center(70, "="))
    print("=" * 70)

    gen = GaussianCopulaGenerator(
        available_factor_ids=CANONICAL_CREDIT_FACTOR_NAMES,
        total_scenarios=NUM_SCENARIOS,
        seed=SEED,
    )

    matrix = build_credit_factor_matrix(
        gen.iter_batches(batch_size=BATCH_SIZE),
        num_factors=CANONICAL_NUM_FACTORS,
        num_scenarios=NUM_SCENARIOS,
    )

    print(f"Factors        : {CANONICAL_NUM_FACTORS}")
    print(f"Scenarios      : {NUM_SCENARIOS:,}")
    print(f"Batch size     : {BATCH_SIZE:,}")
    print(f"Matrix shape   : {matrix.latent_factors.shape}")
    print(f"Matrix dtype   : {matrix.latent_factors.dtype}")
    print(f"C-contiguous   : {matrix.latent_factors.flags.c_contiguous}")
    print()

    print(f"{'Factor':<16} | {'Mean':>10} | {'Std':>10} | {'Skew':>10}")
    print("-" * 56)
    for i, name in enumerate(CANONICAL_FACTOR_NAMES):
        row = matrix.latent_factors[i]
        m = row.mean()
        s = row.std()
        skew = ((row - m) ** 3).mean() / (s**3 + 1e-30)
        print(f"{name:<16} | {m:>10.4f} | {s:>10.4f} | {skew:>10.4f}")

    print("\nExpected (Gaussian): mean=0, std=1, skew=0")
    print("=" * 70)


if __name__ == "__main__":
    main()
