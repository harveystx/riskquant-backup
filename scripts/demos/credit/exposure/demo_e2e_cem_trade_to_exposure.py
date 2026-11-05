# scripts/demos/credit/exposure/demo_e2e_cem_trade_to_exposure.py
"""
Trade -> Exposure -> JTD contract walkthrough (CEM).

Displays the full path from trade-level input to the per-obligor EAD that feeds
the JTD credit-VaR engine, printing each contract edge. Two contracts are locked:

  CONTRACT A  (CEM input)   trade SoA + netting-set master + add-on table
  CONTRACT B  (JTD input)   ead_by_obligor, rows aligned to obligor_uids

The middle is pure engine: resolve -> compile -> run_exposure -> aggregate.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.credit.exposure.add_on.ingest import compile_add_on_factors
from riskquant.credit.exposure.add_on.resolve import (
    resolve_quantity_price_effective_notional,
    resolve_supervisory_factors,
)
from riskquant.credit.exposure.compile import compile_exposure_cem
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
    _rule("CONTRACT A:  CEM INPUT  (trade level)")
    print("\n  Human edge (authored trade table):\n")
    print_df(book.trades_frame(), add_space_before=False, add_space_after=False)
    print("\n  Supervisory add-on table (HOUSE-DEFINED, not SA-CCR):")
    for name, uix, f in zip(
        book.subclass_name,
        book.subclass_uix,
        book.subclass_factor,
        strict=True,
    ):
        print(f"    {name:<18} uix={uix:<3} factor={f:.2%}")
    _contract(
        "trade SoA -> engine",
        trade_uid=book.trade_uid,
        netting_set_id=book.trade_netting_set_id,
        asset_subclass_uix=book.asset_subclass_uix,
        volume=book.volume,
        current_price=book.current_price,
        mtm=book.mtm,
    )

    # ------------------------------------------------------------------ INGEST/RESOLVE
    _rule("RESOLVE  (edge):  effective_notional  &  supervisory_factor")
    add_on = compile_add_on_factors(
        pl.DataFrame({
            "subclass_uix": book.subclass_uix,
            "factor": book.subclass_factor,
        })
    )
    netting_master = CompiledNettingSetMaster(
        uid=book.netting_set_uid,
        obligor_uid=book.netting_set_obligor_uid,
        net_collateral=book.netting_set_collateral,
    )
    effective_notional = resolve_quantity_price_effective_notional(
        volume=book.volume,
        current_price=book.current_price,
    )
    supervisory_factor = resolve_supervisory_factors(book.asset_subclass_uix, add_on)
    _contract(
        "resolved per-trade",
        effective_notional=effective_notional,
        supervisory_factor=supervisory_factor,
        gross_add_on=effective_notional * supervisory_factor,
    )

    # ------------------------------------------------------------------ CEM ENGINE
    _rule("EXPOSURE ENGINE:  compile_exposure_cem -> run_exposure")
    compiled = compile_exposure_cem(
        trade_uids=book.trade_uid,
        trade_netting_set_id=book.trade_netting_set_id,
        trade_effective_notional=effective_notional,
        trade_supervisory_factor=supervisory_factor,
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
    for ns, ob in zip(netting_master.uid, netting_master.obligor_uid, strict=True):
        print(f"    netting set {ns} -> obligor {ob}")

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
        "PIPELINE COMPLETE:  trade  ->  CEM  ->  netting-set EAD  ->  obligor EAD  ->  JTD"
    )

    print_success_message(
        "Pipeline successfully completed.  All contract edges verified."
    )


if __name__ == "__main__":
    main()
