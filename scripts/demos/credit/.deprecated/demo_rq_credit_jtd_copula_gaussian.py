# scripts/demos/demo_rq_credit_jtd_copula_gaussian.py
"""
Demo: single-factor Gaussian-copula JTD.

End-to-end of the credit-only path (obligor master -> factor registry
-> latent matrix -> compiled JTD -> losses). EAD is synthetic and
constant across scenarios so the empirical default rate per obligor
converges to PD — a clean correctness check on the threshold transform
and the kernel's branchless `(t > a)` indicator.
"""

from __future__ import annotations

import numpy as np

from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.parametric_copula_gaussian import (
    GaussianCopulaGenerator,
)
from riskquant.credit.jtd.engine import run_jtd
from riskquant.credit.jtd.ingest import compile_jtd_portfolio_from_df
from riskquant.credit.obligors.ingest import compile_obligor_master_from_df
from riskquant.ingest_fixtures.credit_factors.minimal_factor_universe import (
    CANONICAL_CREDIT_FACTOR_NAMES,
    CANONICAL_FACTOR_NAME_TO_UIX,
    CANONICAL_NUM_FACTORS,
)
from riskquant.ingest_fixtures.jtd.copula_jtd.sf_gaussian_minimal import (
    CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)

NUM_SCENARIOS: int = 500_000
BATCH_SIZE: int = 32_768
SEED: int = 7
SYNTHETIC_EAD_PER_OBLIGOR: float = 10_000_000.0


def main() -> None:
    print("=" * 80)
    print(" SINGLE-FACTOR GAUSSIAN COPULA JTD ".center(80, "="))
    print("=" * 80)

    obligor_master = compile_obligor_master_from_df(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)

    jtd_portfolio = compile_jtd_portfolio_from_df(
        credit_df=CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF,
        obligor_master=obligor_master,
        factor_name_to_uix=CANONICAL_FACTOR_NAME_TO_UIX,
    )

    gen = GaussianCopulaGenerator(
        available_factor_ids=CANONICAL_CREDIT_FACTOR_NAMES,
        total_scenarios=NUM_SCENARIOS,
        seed=SEED,
    )
    credit_factors = build_credit_factor_matrix(
        gen.iter_batches(batch_size=BATCH_SIZE),
        num_factors=CANONICAL_NUM_FACTORS,
        num_scenarios=NUM_SCENARIOS,
    )

    n_obligors = jtd_portfolio.total_obligors
    ead = np.full(
        (n_obligors, NUM_SCENARIOS), SYNTHETIC_EAD_PER_OBLIGOR, dtype=np.float64
    )

    result = run_jtd(
        jtd_portfolio=jtd_portfolio,
        ead_by_obligor=ead,
        credit_factors=credit_factors,
    )

    pd_lookup = dict(
        zip(obligor_master.uid.tolist(), obligor_master.pd.tolist(), strict=True)
    )
    lgd_lookup = dict(
        zip(obligor_master.uid.tolist(), obligor_master.lgd.tolist(), strict=True)
    )

    print(f"Scenarios       : {NUM_SCENARIOS:,}")
    print(f"Obligors        : {n_obligors}")
    print(f"EAD per obligor : {SYNTHETIC_EAD_PER_OBLIGOR:,.0f} (constant)")
    print()
    print(
        f"{'uid':>4} | {'PD':>8} | {'PD_hat':>8} | "
        f"{'rel_err':>8} | {'EL':>16} | {'EL_hat':>16}"
    )
    print("-" * 80)
    for i, uid in enumerate(result.uid):
        uid_i = int(uid)
        row = result.loss[i]
        pd_hat = float((row > 0.0).mean())
        pd_ = pd_lookup[uid_i]
        lgd_ = lgd_lookup[uid_i]
        el = pd_ * lgd_ * SYNTHETIC_EAD_PER_OBLIGOR
        el_hat = float(row.mean())
        rel_err = abs(pd_hat - pd_) / pd_
        print(
            f"{uid_i:>4} | {pd_:>8.4f} | {pd_hat:>8.4f} | "
            f"{rel_err:>7.2%} | {el:>16,.2f} | {el_hat:>16,.2f}"
        )

    portfolio_loss = result.loss.sum(axis=0)
    print()
    print("Portfolio loss distribution:")
    print(f"  mean        : {portfolio_loss.mean():>16,.2f}")
    print(f"  std         : {portfolio_loss.std():>16,.2f}")
    print(f"  99.0% VaR   : {np.quantile(portfolio_loss, 0.99):>16,.2f}")
    print(f"  99.9% VaR   : {np.quantile(portfolio_loss, 0.999):>16,.2f}")
    print(f"  max         : {portfolio_loss.max():>16,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
