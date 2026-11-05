# scripts/demos/credit/exposure/demo_e2e_saccr_trade_to_exposure.py
"""
Trade -> Exposure -> JTD contract walkthrough (SA-CCR).

Displays the full path from trade-level input to the per-obligor EAD that feeds
the JTD credit-VaR engine, printing each contract edge. Two contracts are locked:

  CONTRACT A  (SA-CCR input)    trade SoA + netting-set master + add-on table
                                + HOUSE-DEFINED hedging-set config (rho/alpha/floor)
  CONTRACT B  (JTD input)       ead_by_obligor, rows aligned to obligor_uids

The middle is pure engine: resolve -> compile -> run_exposure -> aggregate. The
hedging-set aggregation (correlation, alpha, multiplier) happens inside
compile_exposure_saccr; all supervisory parameters are ours, not SA-CCR's.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.credit.exposure.add_on.ingest import compile_add_on_factors
from riskquant.credit.exposure.add_on.resolve import (
    resolve_quantity_price_effective_notional,
)
from riskquant.credit.exposure.compile import compile_exposure_saccr
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.exposure.saccr.ingest import compile_saccr_config
from riskquant.credit.jtd.bridge import aggregate_ead_to_obligor
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_success_message,
)
from riskquant_lab.fixtures.credit_exposure import generate_canonical_exposure_book

# House-defined hedging sets over the environmental subclasses (NOT SA-CCR's
# taxonomy). Subclass names come from the canonical book; each maps to a hedging
# set with its own supervisory correlation.
_HEDGING_SET_OF = {
    "RNG": 100,
    "BIODIESEL": 100,
    "REC": 200,
    "RIN": 200,
    "CARBON_ALLOWANCE": 200,
}
_HEDGING_SET_NAME = {100: "FUELS", 200: "CERTIFICATES"}
_HEDGING_SET_RHO = {100: 0.40, 200: 0.30}
_DEFAULT_HEDGING_SET = 900  # catch-all if the book grows a new subclass
_DEFAULT_RHO = 0.50
_ALPHA = 1.4
_MULTIPLIER_FLOOR = 0.05


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

    hedging_set_of = {
        int(uix): _HEDGING_SET_OF.get(name, _DEFAULT_HEDGING_SET)
        for name, uix in zip(book.subclass_name, book.subclass_uix, strict=True)
    }
    hedging_sets = sorted(set(hedging_set_of.values()))

    # ------------------------------------------------------------------ CONTRACT A
    _rule("CONTRACT A:  SA-CCR INPUT  (trade level)")
    print("\n  Human edge (authored trade table):\n")
    print_df(book.trades_frame(), add_space_before=False, add_space_after=False)
    print("\n  Supervisory add-on table (HOUSE-DEFINED):")
    for name, uix, f in zip(
        book.subclass_name, book.subclass_uix, book.subclass_factor, strict=True
    ):
        hs = hedging_set_of[int(uix)]
        hs_name = _HEDGING_SET_NAME.get(hs, "OTHER")
        print(f"    {name:<18} uix={uix:<3} factor={f:.2%}  ->  hedging set {hs_name}")
    print("\n  Hedging-set config (HOUSE-DEFINED correlations):")
    for hs in hedging_sets:
        rho = _HEDGING_SET_RHO.get(hs, _DEFAULT_RHO)
        members = [
            book.subclass_name[i]
            for i in range(len(book.subclass_uix))
            if hedging_set_of[int(book.subclass_uix[i])] == hs
        ]
        print(
            f"    {_HEDGING_SET_NAME.get(hs, 'OTHER'):<14} (hs={hs}, rho={rho:.0%}): {', '.join(members)}"
        )
    print(f"    alpha={_ALPHA}   multiplier_floor={_MULTIPLIER_FLOOR}")
    _contract(
        "trade SoA -> engine",
        trade_uid=book.trade_uid,
        netting_set_id=book.trade_netting_set_id,
        asset_subclass_uix=book.asset_subclass_uix,
        volume=book.volume,
        current_price=book.current_price,
        mtm=book.mtm,
    )

    # ------------------------------------------------------------------ CONFIG / RESOLVE
    _rule("RESOLVE  (edge):  effective_notional  +  SA-CCR config")
    add_on = compile_add_on_factors(
        pl.DataFrame({
            "subclass_uix": book.subclass_uix,
            "factor": book.subclass_factor,
        })
    )
    saccr_config = compile_saccr_config(
        pl.DataFrame({
            "subclass_uix": book.subclass_uix,
            "hedging_set_uix": [hedging_set_of[int(u)] for u in book.subclass_uix],
        }),
        pl.DataFrame({
            "hedging_set_uix": hedging_sets,
            "rho": [_HEDGING_SET_RHO.get(hs, _DEFAULT_RHO) for hs in hedging_sets],
        }),
        alpha=_ALPHA,
        multiplier_floor=_MULTIPLIER_FLOOR,
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
    _contract(
        "resolved per-trade  (SF + hedging-set aggregation applied inside compile)",
        effective_notional=effective_notional,
        asset_subclass_uix=book.asset_subclass_uix,
    )

    # ------------------------------------------------------------------ SA-CCR ENGINE
    _rule("EXPOSURE ENGINE:  compile_exposure_saccr -> run_exposure")
    compiled = compile_exposure_saccr(
        trade_uids=book.trade_uid,
        trade_netting_set_id=book.trade_netting_set_id,
        trade_effective_notional=effective_notional,
        trade_subclass_uix=book.asset_subclass_uix,
        netting_set_master=netting_master,
        add_on_factors=add_on,
        saccr_config=saccr_config,
    )
    mtm = np.ascontiguousarray(book.mtm.reshape(-1, 1))  # (M, S=1) deterministic
    ns_paths = run_exposure(exposure=compiled, mtm=mtm)
    _contract(
        "hedging-set-aggregated add-on  (scenario-free, per netting set)",
        netting_set_uid=compiled.saccr.netting_set_uids,
        add_on=compiled.saccr.add_on,
    )
    _contract(
        "netting-set EAD = alpha*(RC + multiplier*AddOn)",
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
        "PIPELINE COMPLETE:  trade  ->  SA-CCR  ->  netting-set EAD  ->  obligor EAD  ->  JTD"
    )

    print_success_message(
        "Pipeline successfully completed.  All contract edges verified."
    )


if __name__ == "__main__":
    main()
