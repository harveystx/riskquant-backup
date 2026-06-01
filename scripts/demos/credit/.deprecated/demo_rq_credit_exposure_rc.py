# scripts/demos/demo_rq_credit_exposure_rc.py
"""
Demo: Replacement Cost exposure pipeline.

Builds the canonical credit-exposure ingest case (SPOT trades + netting
set linkages) and the canonical netting-set master into compiled
artifacts, synthesizes an MtM matrix, then runs:

  compile_exposure -> run_exposure -> aggregate_ead_to_obligor

MtM is synthesized rather than market-valued because this demo's focus
is the CSR band layout, the branchless RC kernel, and the bridge into
obligor space — not market pricing. The end-to-end demo wires real
valuation in.
"""

from __future__ import annotations

import numpy as np

from riskquant.credit.exposure.compile import compile_exposure
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.jtd.bridge import aggregate_ead_to_obligor
from riskquant.credit.netting_set.ingest import compile_netting_set_master_from_df
from riskquant.ingest_fixtures.credit_exposure.minimal_spot_with_netting_sets import (
    CANONICAL_CREDIT_EXPOSURE_INGEST_DF,
)
from riskquant.ingest_fixtures.netting_set.minimal_netting_set_master import (
    CANONICAL_NETTING_SET_MASTER_INGEST_DF,
)
from riskquant.portfolio.engine import compile_portfolio_from_ingest_batch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

RF_NAME_TO_RF_ID: dict[str, int] = {"TEST_BRENT_CRUDE_202701": 0}
RF_ID_TO_CCY: dict[int, str] = {0: "USD"}

NUM_SCENARIOS: int = 4
SEED: int = 17


def main() -> None:
    print("=" * 80)
    print(" REPLACEMENT COST EXPOSURE PIPELINE ".center(80, "="))
    print("=" * 80)

    # 1. Ingest.
    batch = compile_raw_df_into_ingest_batch(
        raw_df=CANONICAL_CREDIT_EXPOSURE_INGEST_DF,
        rf_name_to_rf_id=RF_NAME_TO_RF_ID,
        rf_id_to_ccy=RF_ID_TO_CCY,
    )
    print(f"Ingest batch       : {batch.uids.shape[0]} trades")

    # 2. Compile portfolio.
    portfolio = compile_portfolio_from_ingest_batch(batch)
    print(
        f"Compiled portfolio : {portfolio.total_trades} trades in "
        f"{portfolio.total_blocks} block(s)"
    )

    # 3. Synthetic MtM in portfolio-row order. Production: run_valuation(...).
    rng = np.random.default_rng(SEED)
    mtm = rng.normal(0.0, 1.0e5, size=(portfolio.total_trades, NUM_SCENARIOS)).astype(
        np.float64, copy=False
    )
    print(f"MtM matrix         : {mtm.shape} (synthetic)")

    # 4. Compile netting-set master from the canonical ingest case.
    netting_set_master = compile_netting_set_master_from_df(
        CANONICAL_NETTING_SET_MASTER_INGEST_DF
    )
    print(f"Netting-set master : {netting_set_master.uid.shape[0]} set(s)")

    # 5. Compile exposure.
    exposure = compile_exposure(
        portfolio=portfolio,
        batch=batch,
        netting_set_master=netting_set_master,
    )
    print(
        f"Compiled exposure  : {exposure.total_netting_sets} netting set(s) "
        f"in {exposure.total_views} view(s)"
    )
    print(f"  netting_set_uids : {exposure.netting_set_uids.tolist()}")

    # 6. Run exposure (RC kernel).
    ead_result = run_exposure(exposure=exposure, mtm=mtm)
    coll_lookup = dict(
        zip(
            netting_set_master.uid.tolist(),
            netting_set_master.net_collateral.tolist(),
            strict=True,
        )
    )
    print()
    print("EAD per netting set (across scenarios):")
    print(
        f"  {'ns_uid':>6} | {'collateral':>12} | "
        f"{'min':>14} | {'mean':>14} | {'max':>14}"
    )
    print("-" * 76)
    for i, ns in enumerate(ead_result.uid):
        row = ead_result.ead[i]
        print(
            f"  {int(ns):>6} | {coll_lookup[int(ns)]:>12,.0f} | "
            f"{row.min():>14,.2f} | {row.mean():>14,.2f} | {row.max():>14,.2f}"
        )

    # 7. Bridge: NS -> obligor.
    obligor_uids_target = np.sort(netting_set_master.obligor_uid)
    ead_by_obligor = aggregate_ead_to_obligor(
        ead_result=ead_result,
        netting_set_master=netting_set_master,
        obligor_uids_target=obligor_uids_target,
    )
    print()
    print("EAD per obligor (after CSR aggregation):")
    print(f"  {'obligor':>8} | {'min':>14} | {'mean':>14} | {'max':>14}")
    print("-" * 60)
    for i, obl in enumerate(obligor_uids_target):
        row = ead_by_obligor[i]
        print(
            f"  {int(obl):>8} | {row.min():>14,.2f} | "
            f"{row.mean():>14,.2f} | {row.max():>14,.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
