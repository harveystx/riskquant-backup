# scripts/demos/demo_rq_credit_jtd_copula_t.py
"""
Demo: single-factor Student-t-copula JTD vs Gaussian reference.

The canonical t-copula case uses per-obligor chi factors with matching
nu (CHI_1=5, CHI_2=7, CHI_3=10) so each obligor's marginal default-rate
indicator is mathematically a true t. PD convergence holds (the t-copula
preserves marginals) and the portfolio loss tail is fatter than the
Gaussian case at the same correlation — visible side-by-side below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.parametric_copula_gaussian import (
    GaussianCopulaGenerator,
)
from riskquant.credit.factors.simulate.parametric_copula_t import TCopulaGenerator
from riskquant.credit.jtd.engine import run_jtd
from riskquant.credit.jtd.ingest import compile_jtd_portfolio_from_df
from riskquant.credit.jtd.models import JtdModel
from riskquant.credit.obligors.ingest import compile_obligor_master_from_df
from riskquant.ingest_fixtures.credit_factors.minimal_factor_universe import (
    CANONICAL_CREDIT_FACTOR_NAMES,
    CANONICAL_FACTOR_NAME_TO_UIX,
    CANONICAL_NUM_FACTORS,
    CANONICAL_T_COPULA_CHI_COLUMNS,
    CANONICAL_T_COPULA_NU_BY_CHI_FACTOR,
)
from riskquant.ingest_fixtures.jtd.copula_jtd.sf_t_minimal import (
    CANONICAL_T_COPULA_JTD_INGEST_DATA,
    CANONICAL_T_COPULA_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)

NUM_SCENARIOS: int = 500_000
BATCH_SIZE: int = 32_768
SEED: int = 9
SYNTHETIC_EAD_PER_OBLIGOR: float = 10_000_000.0


def _gaussian_reference_df() -> pd.DataFrame:
    """Same obligors, beta, factor wiring — routed through the Gaussian model."""
    df = pd.DataFrame(CANONICAL_T_COPULA_JTD_INGEST_DATA)
    df["instrument"] = JtdModel.COPULA_SF_GAUSSIAN.name
    return df.drop(columns=["chi_squared_factor", "nu"])


def main() -> None:
    print("=" * 80)
    print(" SINGLE-FACTOR STUDENT-T COPULA JTD ".center(80, "="))
    print("=" * 80)

    obligor_master = compile_obligor_master_from_df(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)

    ead = np.full((3, NUM_SCENARIOS), SYNTHETIC_EAD_PER_OBLIGOR, dtype=np.float64)

    # --- t-copula run ----------------------------------------------------
    t_portfolio = compile_jtd_portfolio_from_df(
        credit_df=CANONICAL_T_COPULA_JTD_INGEST_DF,
        obligor_master=obligor_master,
        factor_name_to_uix=CANONICAL_FACTOR_NAME_TO_UIX,
    )
    t_gen = TCopulaGenerator(
        available_factor_ids=CANONICAL_CREDIT_FACTOR_NAMES,
        total_scenarios=NUM_SCENARIOS,
        chi_factor_columns=CANONICAL_T_COPULA_CHI_COLUMNS,
        nu_by_chi=CANONICAL_T_COPULA_NU_BY_CHI_FACTOR,
        seed=SEED,
    )
    t_factors = build_credit_factor_matrix(
        t_gen.iter_batches(batch_size=BATCH_SIZE),
        num_factors=CANONICAL_NUM_FACTORS,
        num_scenarios=NUM_SCENARIOS,
    )
    t_result = run_jtd(
        jtd_portfolio=t_portfolio,
        ead_by_obligor=ead,
        credit_factors=t_factors,
    )

    # --- Gaussian reference on the same obligors -------------------------
    g_portfolio = compile_jtd_portfolio_from_df(
        credit_df=_gaussian_reference_df(),
        obligor_master=obligor_master,
        factor_name_to_uix=CANONICAL_FACTOR_NAME_TO_UIX,
    )
    g_gen = GaussianCopulaGenerator(
        available_factor_ids=CANONICAL_CREDIT_FACTOR_NAMES,
        total_scenarios=NUM_SCENARIOS,
        seed=SEED + 1,
    )
    g_factors = build_credit_factor_matrix(
        g_gen.iter_batches(batch_size=BATCH_SIZE),
        num_factors=CANONICAL_NUM_FACTORS,
        num_scenarios=NUM_SCENARIOS,
    )
    g_result = run_jtd(
        jtd_portfolio=g_portfolio,
        ead_by_obligor=ead,
        credit_factors=g_factors,
    )

    # --- Marginal sanity (t-copula) --------------------------------------
    pd_lookup = dict(
        zip(obligor_master.uid.tolist(), obligor_master.pd.tolist(), strict=True)
    )
    nu_lookup = dict(
        zip(
            CANONICAL_T_COPULA_JTD_INGEST_DATA["uid"],
            CANONICAL_T_COPULA_JTD_INGEST_DATA["nu"],
            strict=True,
        )
    )
    print(f"Scenarios     : {NUM_SCENARIOS:,}")
    print("Obligors      : 3 (uids 1, 2, 3) with per-obligor nu")
    print()
    print(f"{'uid':>4} | {'nu':>4} | {'PD':>8} | {'PD_hat (t)':>10} | {'rel_err':>8}")
    print("-" * 48)
    for i, uid in enumerate(t_result.uid):
        uid_i = int(uid)
        pd_hat = float((t_result.loss[i] > 0.0).mean())
        pd_ = pd_lookup[uid_i]
        rel = abs(pd_hat - pd_) / pd_
        print(
            f"{uid_i:>4} | {nu_lookup[uid_i]:>4.0f} | {pd_:>8.4f} | "
            f"{pd_hat:>10.4f} | {rel:>7.2%}"
        )

    # --- Tail comparison -------------------------------------------------
    t_port = t_result.loss.sum(axis=0)
    g_port = g_result.loss.sum(axis=0)
    print()
    print("Portfolio loss tail — t vs Gaussian (same obligors, beta, EAD):")
    print(f"{'quantile':<10} | {'Gaussian':>16} | {'t-copula':>16} | {'ratio':>8}")
    print("-" * 60)
    for q in (0.95, 0.99, 0.995, 0.999, 0.9999):
        g_q = float(np.quantile(g_port, q))
        t_q = float(np.quantile(t_port, q))
        ratio = (t_q / g_q) if g_q > 0 else float("nan")
        print(f"{q:<10.4f} | {g_q:>16,.0f} | {t_q:>16,.0f} | {ratio:>8.2f}x")

    print("=" * 80)
    print("Tail ratio > 1 at high quantiles is the chi-square mixing signature.")
    print("=" * 80)


if __name__ == "__main__":
    main()
