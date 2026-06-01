# scripts/demos/demo_rq_credit_cross_master_validate.py
"""
Demo: layered cross-master validation.

Compiles the canonical obligor master and netting-set master, then runs
`validate_cross_masters` to confirm the netting-set master's
`obligor_uid` is a subset of the obligor master's `uid`. A second pass
mutates the netting-set master to point at a non-existent obligor and
demonstrates the failure mode.
"""

from __future__ import annotations

from riskquant.credit.cross_master.validate import validate_cross_masters
from riskquant.credit.netting_set.ingest import compile_netting_set_master_from_df
from riskquant.credit.obligors.ingest import compile_obligor_master_from_df
from riskquant.ingest_fixtures.netting_set.minimal_netting_set_master import (
    CANONICAL_NETTING_SET_MASTER_INGEST_DF,
)
from riskquant.ingest_fixtures.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)


def main() -> None:
    print("=" * 78)
    print(" CROSS-MASTER VALIDATION ".center(78, "="))
    print("=" * 78)

    obligor_master = compile_obligor_master_from_df(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)
    ns_master = compile_netting_set_master_from_df(
        CANONICAL_NETTING_SET_MASTER_INGEST_DF
    )

    print("Obligor master uids       :", obligor_master.uid.tolist())
    print("Netting-set obligor_uids  :", ns_master.obligor_uid.tolist())
    print()

    # --- HAPPY PATH ---
    print("Validating canonical masters ...")
    validate_cross_masters(obligor_master=obligor_master, netting_set_master=ns_master)
    print("OK -- every netting set's obligor_uid is on the obligor master.")
    print()

    # --- SAD PATH ---
    print("Mutating netting-set master to point at a non-existent obligor (9999) ...")
    bad_df = CANONICAL_NETTING_SET_MASTER_INGEST_DF.copy()
    bad_df.loc[0, "obligor_uid"] = 9999
    bad_ns_master = compile_netting_set_master_from_df(bad_df)

    try:
        validate_cross_masters(
            obligor_master=obligor_master, netting_set_master=bad_ns_master
        )
    except ValueError as e:
        print("Caught expected ValueError:")
        print(f"  {e}")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
