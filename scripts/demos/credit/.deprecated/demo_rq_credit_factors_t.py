# scripts/demos/demo_rq_credit_factors_t.py
"""
Demo: Student-t-copula latent-factor generation.

The generator emits RAW draws — standard normals for systemic /
idiosyncratic columns and chi-square draws for each chi-mixing column.
The `sqrt(nu / W)` transform is applied in the JTD kernel, not here.
This demo verifies the raw moments per column.
"""

from __future__ import annotations

from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.parametric_copula_t import TCopulaGenerator
from riskquant.ingest_fixtures.credit_factors.minimal_factor_universe import (
    CANONICAL_CREDIT_FACTOR_NAMES,
    CANONICAL_FACTOR_NAMES,
    CANONICAL_NUM_FACTORS,
    CANONICAL_T_COPULA_CHI_COLUMNS,
    CANONICAL_T_COPULA_NU_BY_CHI_FACTOR,
)

NUM_SCENARIOS: int = 100_000
BATCH_SIZE: int = 16_384
SEED: int = 1337


def main() -> None:
    print("=" * 78)
    print(" STUDENT-T COPULA LATENT-FACTOR GENERATION ".center(78, "="))
    print("=" * 78)

    gen = TCopulaGenerator(
        available_factor_ids=CANONICAL_CREDIT_FACTOR_NAMES,
        total_scenarios=NUM_SCENARIOS,
        chi_factor_columns=CANONICAL_T_COPULA_CHI_COLUMNS,
        nu_by_chi=CANONICAL_T_COPULA_NU_BY_CHI_FACTOR,
        seed=SEED,
    )

    matrix = build_credit_factor_matrix(
        gen.iter_batches(batch_size=BATCH_SIZE),
        num_factors=CANONICAL_NUM_FACTORS,
        num_scenarios=NUM_SCENARIOS,
    )

    nu_lookup: dict[int, float] = {
        int(c): float(nu)
        for c, nu in zip(
            CANONICAL_T_COPULA_CHI_COLUMNS,
            CANONICAL_T_COPULA_NU_BY_CHI_FACTOR,
            strict=True,
        )
    }

    print(f"Factors        : {CANONICAL_NUM_FACTORS}")
    print(f"Scenarios      : {NUM_SCENARIOS:,}")
    print(f"chi columns    : {CANONICAL_T_COPULA_CHI_COLUMNS.tolist()}")
    print(f"nu_by_chi      : {CANONICAL_T_COPULA_NU_BY_CHI_FACTOR.tolist()}")
    print()

    print(f"{'Factor':<16} | {'Type':<8} | {'Mean':>10} | {'Var':>10} | Expected")
    print("-" * 78)
    for i, name in enumerate(CANONICAL_FACTOR_NAMES):
        row = matrix.latent_factors[i]
        mean_ = float(row.mean())
        var_ = float(row.var())
        if i in nu_lookup:
            nu = nu_lookup[i]
            kind = "chi^2"
            expected = f"mean=nu={nu}, var=2*nu={2 * nu}"
        else:
            kind = "normal"
            expected = "mean=0, var=1"
        print(f"{name:<16} | {kind:<8} | {mean_:>10.4f} | {var_:>10.4f} | {expected}")
    print("=" * 78)


if __name__ == "__main__":
    main()
