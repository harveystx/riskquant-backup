# scripts/demos/demo_rq_credit_obligor_master.py
"""
Demo: compile the canonical obligor master.

Runs `compile_obligor_master_from_df` on the canonical 12-obligor master
and prints the resulting SoA layout plus the sub-IG filter at the
default 0.005 threshold.
"""

from __future__ import annotations

import numpy as np

from riskquant.credit.obligors.ingest import compile_obligor_master_from_df
from riskquant.ingest_fixtures.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)

SUB_IG_PD_THRESHOLD: float = 0.005


def main() -> None:
    print("=" * 78)
    print(" OBLIGOR MASTER COMPILE ".center(78, "="))
    print("=" * 78)

    master = compile_obligor_master_from_df(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)

    n = master.uid.shape[0]
    print(f"Obligors        : {n}")
    print(f"uid dtype       : {master.uid.dtype}")
    print(f"uid ascending   : {bool((np.diff(master.uid) > 0).all())}")
    print(f"pd dtype        : {master.pd.dtype}")
    print(f"lgd dtype       : {master.lgd.dtype}")
    print(f"rating_id dtype : {master.rating_id.dtype}")
    print(f"sector_id dtype : {master.sector_id.dtype}")
    print(f"country_id dtype: {master.country_id.dtype}")
    print(f"parent_uid dtype: {master.parent_uid.dtype}")
    print()

    print(
        f"{'uid':>4} | {'PD':>8} | {'LGD':>5} | "
        f"{'rating_id':>9} | {'parent_uid':>10} | sub_IG"
    )
    print("-" * 60)
    sub_ig_mask = master.pd > SUB_IG_PD_THRESHOLD
    for i in range(n):
        flag = "Y" if bool(sub_ig_mask[i]) else "."
        print(
            f"{int(master.uid[i]):>4} | {master.pd[i]:>8.4f} | "
            f"{master.lgd[i]:>5.2f} | {int(master.rating_id[i]):>9} | "
            f"{int(master.parent_uid[i]):>10} | {flag}"
        )

    print("-" * 60)
    print(f"sub-IG obligors at PD > {SUB_IG_PD_THRESHOLD}: {int(sub_ig_mask.sum())}/{n}")
    print("=" * 78)


if __name__ == "__main__":
    main()
