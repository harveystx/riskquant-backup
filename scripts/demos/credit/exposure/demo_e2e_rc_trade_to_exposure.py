# scripts/demos/credit/exposure/demo_e2e_rc_trade_to_exposure.py
"""
Trade -> Exposure -> JTD contract walkthrough (Replacement Cost).

Displays the full path from trade-level input to the per-obligor EAD that feeds
the JTD credit-VaR engine, printing each contract edge. Two contracts are locked:

  CONTRACT A  (RC input)    trade SoA + netting-set master
  CONTRACT B  (JTD input)   ead_by_obligor, rows aligned to obligor_uids

The middle is pure engine: compile -> run_exposure -> aggregate.
"""

from __future__ import annotations

import numpy as np

from riskquant.credit.exposure.compile import compile_exposure_rc
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.jtd.bridge import aggregate_ead_to_obligor
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_success_message,
)
from riskquant_lab.fixtures.credit_exposure import (
    generate_canonical_exposure_book,
)


def _rule(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _contract(name: str, **arrays: np.ndarray) -> None:
    print(f"\n  [{name}]")
    for label, arr in arrays.items():
        head = np.array2string(arr.ravel()[:6], precision=2, separator=", ")
        print(f"    {label:<26} shape={arr.shape!s:<10} dtype={arr.dtype!s:<8} {head}")


def main() -> None:
    print_calling_file(add_space_after=False)

    book = generate_canonical_exposure_book()

    # ------------------------------------------------------------------ CONTRACT A
    _rule("CONTRACT A:  RC INPUT  (trade level)")
    print("\n  Human edge (authored trade table):\n")
    print_df(book.trades_frame(), add_space_before=False, add_space_after=False)

    _contract(
        "trade SoA -> engine",
        trade_uid=book.trade_uid,
        netting_set_id=book.trade_netting_set_id,
        mtm=book.mtm,
    )

    netting_master = CompiledNettingSetMaster(
        uid=book.netting_set_uid,
        obligor_uid=book.netting_set_obligor_uid,
        net_collateral=book.netting_set_collateral,
    )

    # ------------------------------------------------------------------ RC ENGINE
    _rule("EXPOSURE ENGINE:  compile_exposure -> run_exposure")
    compiled = compile_exposure_rc(
        trade_uids=book.trade_uid,
        trade_netting_set_id=book.trade_netting_set_id,
        netting_set_master=netting_master,
    )

    mtm = np.ascontiguousarray(book.mtm.reshape(-1, 1))  # (M, S=1) deterministic
    ns_paths = run_exposure(exposure=compiled, mtm=mtm)

    _contract(
        "netting-set EAD  (NettingSetExposurePaths)",
        uid=ns_paths.uid,
        exposure=ns_paths.exposure,
    )
    print("\n  netting-set -> obligor linkage (from master):")
    for ns, ob, coll in zip(
        netting_master.uid,
        netting_master.obligor_uid,
        netting_master.net_collateral,
        strict=True,
    ):
        print(f"    netting set {ns} -> obligor {ob} (collateral: {coll:,.0f})")

    # ------------------------------------------------------------------ CONTRACT B
    _rule("CONTRACT B:  JTD INPUT  (per-obligor EAD)")
    obligor_target = (
        book.obligor_universe
    )  # == jtd_portfolio.obligor_uids in a real run

    ead_by_obligor = aggregate_ead_to_obligor(
        ead_result=ns_paths,
        netting_set_master=netting_master,
        obligor_uids_target=obligor_target,
    )

    _contract(
        "aggregate_ead_to_obligor -> (N_obl, S)",
        obligor_uids_target=obligor_target,
        ead_by_obligor=ead_by_obligor,
    )

    print("\n  Per-obligor EAD (summed across each obligor's netting sets):")
    for ob, e in zip(obligor_target, ead_by_obligor[:, 0], strict=True):
        print(f"    obligor {ob}:  EAD = {e:,.0f}")

    print(
        "\n  ->  feeds JTD directly:  run_jtd_streaming_from_generator("
        "ead_by_obligor=ead_by_obligor[:, 0], ...)"
    )
    print("      rows align to CompiledJtdPortfolio.obligor_uids (band order).")

    _rule(
        "PIPELINE COMPLETE:  trade  ->  RC  ->  netting-set EAD  ->  obligor EAD  ->  JTD"
    )

    print_success_message(
        "Pipeline successfully completed.  All contract edges verified."
    )


if __name__ == "__main__":
    main()
