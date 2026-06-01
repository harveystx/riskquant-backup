# scripts/demos/demo_rq_credit_jtd_deterministic.py
"""
Demo: deterministic JTD loss.

Scenario-independent: loss = ead * lgd, default assumed certain. Runs at
S=1 because broadcasting more scenarios produces identical rows. LGD is
joined off the canonical obligor master at compile time; EAD comes from
the canonical stress-overlay dict (production path would derive from the
exposure engine).
"""

from __future__ import annotations

import numpy as np

from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.engine import run_jtd
from riskquant.credit.jtd.ingest import compile_jtd_portfolio_from_df
from riskquant.credit.obligors.ingest import compile_obligor_master_from_df
from riskquant.ingest_fixtures.jtd.deterministic_jtd.minimal_deterministic_jtd import (
    CANONICAL_DETERMINISTIC_JTD_EAD_OVERLAY,
    CANONICAL_DETERMINISTIC_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)

SUB_IG_PD_THRESHOLD: float = 0.005


def main() -> None:
    print("=" * 88)
    print(" DETERMINISTIC JTD ".center(88, "="))
    print("=" * 88)

    obligor_master = compile_obligor_master_from_df(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)

    # Deterministic ignores latent factors -> empty factor_name_to_uix.
    jtd_portfolio = compile_jtd_portfolio_from_df(
        credit_df=CANONICAL_DETERMINISTIC_JTD_INGEST_DF,
        obligor_master=obligor_master,
        factor_name_to_uix={},
    )

    n_obligors = jtd_portfolio.total_obligors
    n_scenarios = 1

    # Align overlay EAD to compiled obligor_uids order.
    ead = np.empty((n_obligors, n_scenarios), dtype=np.float64, order="C")
    for i, uid in enumerate(jtd_portfolio.obligor_uids):
        ead[i, 0] = CANONICAL_DETERMINISTIC_JTD_EAD_OVERLAY[int(uid)]

    # Latent matrix is never read by the deterministic kernel.
    credit_factors = CreditFactorMatrix(
        latent_factors=np.empty((0, n_scenarios), dtype=np.float64, order="C"),
    )

    result = run_jtd(
        jtd_portfolio=jtd_portfolio,
        ead_by_obligor=ead,
        credit_factors=credit_factors,
    )

    # Join PD/LGD back off the master purely for display.
    master_lookup_lgd = dict(
        zip(obligor_master.uid.tolist(), obligor_master.lgd.tolist(), strict=True)
    )
    master_lookup_pd = dict(
        zip(obligor_master.uid.tolist(), obligor_master.pd.tolist(), strict=True)
    )

    print(
        f"{'uid':>4} | {'PD':>8} | {'LGD':>5} | {'EAD':>16} | "
        f"{'expected':>16} | {'actual':>16} | sub-IG"
    )
    print("-" * 88)
    total_subig = 0.0
    for i, uid in enumerate(result.uid):
        uid_i = int(uid)
        pd_ = master_lookup_pd[uid_i]
        lgd_ = master_lookup_lgd[uid_i]
        ead_i = float(ead[i, 0])
        actual = float(result.loss[i, 0])
        expected = ead_i * lgd_
        subig = pd_ > SUB_IG_PD_THRESHOLD
        if subig:
            total_subig += actual
        print(
            f"{uid_i:>4} | {pd_:>8.4f} | {lgd_:>5.2f} | "
            f"{ead_i:>16,.2f} | {expected:>16,.2f} | "
            f"{actual:>16,.2f} | {'Y' if subig else '.'}"
        )

    total_all = float(result.loss.sum())
    print("-" * 88)
    print(f"{'TOTAL (all)':>49} | {'':>16} | {total_all:>16,.2f}")
    print(f"{'TOTAL (sub-IG only)':>49} | {'':>16} | {total_subig:>16,.2f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
