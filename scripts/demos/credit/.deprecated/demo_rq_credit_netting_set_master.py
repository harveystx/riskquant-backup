# scripts/demos/demo_rq_credit_netting_set_master.py
"""
Demo: compile the canonical netting-set master.

Runs `compile_netting_set_master_from_df` on the canonical three-set
master and prints the resulting SoA layout.
"""

from __future__ import annotations

import numpy as np

from riskquant.credit.netting_set.ingest import compile_netting_set_master_from_df
from riskquant.ingest_fixtures.netting_set.minimal_netting_set_master import (
    CANONICAL_NETTING_SET_MASTER_INGEST_DF,
)


def main() -> None:
    print("=" * 78)
    print(" NETTING-SET MASTER COMPILE ".center(78, "="))
    print("=" * 78)

    master = compile_netting_set_master_from_df(CANONICAL_NETTING_SET_MASTER_INGEST_DF)

    print(f"Netting sets        : {master.uid.shape[0]}")
    print(f"uid dtype           : {master.uid.dtype}")
    print(f"uid ascending       : {bool((np.diff(master.uid) > 0).all())}")
    print(f"obligor_uid dtype   : {master.obligor_uid.dtype}")
    print(f"net_collateral dtype: {master.net_collateral.dtype}")
    print()

    print(f"{'ns_uid':>6} | {'obligor_uid':>11} | {'net_collateral':>16}")
    print("-" * 40)
    for i in range(master.uid.shape[0]):
        print(
            f"{int(master.uid[i]):>6} | "
            f"{int(master.obligor_uid[i]):>11} | "
            f"{master.net_collateral[i]:>16,.2f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
