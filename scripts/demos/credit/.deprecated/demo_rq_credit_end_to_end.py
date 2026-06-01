# scripts/demos/demo_rq_credit_end_to_end.py
"""
Demo: end-to-end credit pipeline.

  Ingest (positions)
      -> Portfolio compile
      -> MtM            (synthetic — production: run_valuation)
      -> compile_exposure
      -> run_exposure                 (EAD per netting set)
      -> aggregate_ead_to_obligor     (EAD per obligor)
      -> compile_jtd_portfolio_from_df  (uses obligor master for PD/LGD)
      -> latent factors (Gaussian copula)
      -> run_jtd                      (loss per obligor)
      -> portfolio loss distribution

All three master ingest cases are exercised: obligor, netting-set, and
the credit-factor registry. The market-risk netting sets (101-103) map
to obligor uids 1-3, which are also the uids the Gaussian JTD case
references — so the bridge populates every JTD row with real EAD.
"""

from __future__ import annotations

import numpy as np

from riskquant.credit.exposure.compile import compile_exposure
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.parametric_copula_gaussian import (
    GaussianCopulaGenerator,
)
from riskquant.credit.jtd.bridge import aggregate_ead_to_obligor
from riskquant.credit.jtd.engine import run_jtd
from riskquant.credit.jtd.ingest import compile_jtd_portfolio_from_df
from riskquant.credit.netting_set.ingest import compile_netting_set_master_from_df
from riskquant.credit.obligors.ingest import compile_obligor_master_from_df
from riskquant.ingest_fixtures.credit_exposure.minimal_spot_with_netting_sets import (
    CANONICAL_CREDIT_EXPOSURE_INGEST_DF,
)
from riskquant.ingest_fixtures.credit_factors.minimal_factor_universe import (
    CANONICAL_CREDIT_FACTOR_NAMES,
    CANONICAL_FACTOR_NAME_TO_UIX,
    CANONICAL_NUM_FACTORS,
)
from riskquant.ingest_fixtures.jtd.copula_jtd.sf_gaussian_minimal import (
    CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF,
)
from riskquant.ingest_fixtures.netting_set.minimal_netting_set_master import (
    CANONICAL_NETTING_SET_MASTER_INGEST_DF,
)
from riskquant.ingest_fixtures.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)
from riskquant.portfolio.engine import compile_portfolio_from_ingest_batch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

RF_NAME_TO_RF_ID: dict[str, int] = {"TEST_BRENT_CRUDE_202701": 0}
RF_ID_TO_CCY: dict[int, str] = {0: "USD"}

NUM_SCENARIOS: int = 200_000
BATCH_SIZE: int = 16_384
MTM_SEED: int = 11
FACTOR_SEED: int = 23


def main() -> None:
    print("=" * 80)
    print(" END-TO-END CREDIT PIPELINE (RC -> Gaussian-copula JTD) ".center(80, "="))
    print("=" * 80)

    # ---------- MASTERS ----------
    obligor_master = compile_obligor_master_from_df(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)
    ns_master = compile_netting_set_master_from_df(
        CANONICAL_NETTING_SET_MASTER_INGEST_DF
    )

    # ---------- MARKET INGEST / PORTFOLIO ----------
    batch = compile_raw_df_into_ingest_batch(
        raw_df=CANONICAL_CREDIT_EXPOSURE_INGEST_DF,
        rf_name_to_rf_id=RF_NAME_TO_RF_ID,
        rf_id_to_ccy=RF_ID_TO_CCY,
    )
    portfolio = compile_portfolio_from_ingest_batch(batch)

    rng = np.random.default_rng(MTM_SEED)
    mtm = rng.normal(0.0, 1.0e5, size=(portfolio.total_trades, NUM_SCENARIOS)).astype(
        np.float64, copy=False
    )

    # ---------- EXPOSURE ----------
    exposure = compile_exposure(
        portfolio=portfolio, batch=batch, netting_set_master=ns_master
    )
    ead_result = run_exposure(exposure=exposure, mtm=mtm)

    # ---------- JTD COMPILE (defines obligor row order) ----------
    jtd_portfolio = compile_jtd_portfolio_from_df(
        credit_df=CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DF,
        obligor_master=obligor_master,
        factor_name_to_uix=CANONICAL_FACTOR_NAME_TO_UIX,
    )

    # ---------- BRIDGE TO OBLIGOR ----------
    ead_by_obligor = aggregate_ead_to_obligor(
        ead_result=ead_result,
        netting_set_master=ns_master,
        obligor_uids_target=jtd_portfolio.obligor_uids,
    )

    # ---------- LATENT FACTORS ----------
    gen = GaussianCopulaGenerator(
        available_factor_ids=CANONICAL_CREDIT_FACTOR_NAMES,
        total_scenarios=NUM_SCENARIOS,
        seed=FACTOR_SEED,
    )
    credit_factors = build_credit_factor_matrix(
        gen.iter_batches(batch_size=BATCH_SIZE),
        num_factors=CANONICAL_NUM_FACTORS,
        num_scenarios=NUM_SCENARIOS,
    )

    # ---------- JTD ----------
    jtd_result = run_jtd(
        jtd_portfolio=jtd_portfolio,
        ead_by_obligor=ead_by_obligor,
        credit_factors=credit_factors,
    )

    # ---------- REPORT ----------
    print(f"Scenarios               : {NUM_SCENARIOS:,}")
    print(f"Trades (RC)             : {portfolio.total_trades}")
    print(f"Netting sets (RC)       : {exposure.total_netting_sets}")
    print(f"Obligors (JTD)          : {jtd_portfolio.total_obligors}")
    print()

    print("EAD per obligor (mean across scenarios):")
    for i, obl in enumerate(jtd_portfolio.obligor_uids):
        print(f"  obligor {int(obl):>3} : {ead_by_obligor[i].mean():>14,.2f}")

    print()
    print("Default rate per obligor (empirical vs input PD from obligor master):")
    pd_lookup = dict(
        zip(obligor_master.uid.tolist(), obligor_master.pd.tolist(), strict=True)
    )
    for i, obl in enumerate(jtd_result.uid):
        pd_hat = float((jtd_result.loss[i] > 0.0).mean())
        pd_ = pd_lookup[int(obl)]
        print(f"  obligor {int(obl):>3} : PD={pd_:.4f}   PD_hat={pd_hat:.4f}")

    print()
    portfolio_loss = jtd_result.loss.sum(axis=0)
    print("Portfolio JTD loss distribution:")
    print(f"  mean      : {portfolio_loss.mean():>16,.2f}")
    print(f"  std       : {portfolio_loss.std():>16,.2f}")
    print(f"  99.0% VaR : {np.quantile(portfolio_loss, 0.99):>16,.2f}")
    print(f"  99.9% VaR : {np.quantile(portfolio_loss, 0.999):>16,.2f}")
    print(f"  max       : {portfolio_loss.max():>16,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
