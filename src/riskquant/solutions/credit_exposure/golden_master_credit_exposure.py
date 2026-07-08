# src/riskquant/solutions/credit_exposure/golden_master_credit_exposure.py
"""
Contract lock for ``solutions.credit_exposure`` — two anti-fragile layers.

LAYER 1  Structural invariants that hold for ANY valid book. They assert
         *properties*, not numbers, so they ride whatever the canonical fixture
         becomes: editing the demo book never re-blesses them, only a real bug
         (a broken aggregation, a sign error, a mis-scaled NGR) trips them.

LAYER 2  A tiny, test-OWNED book with EAD computed on paper and pinned. It never
         imports the evolving fixture, so it drifts only on an intentional
         model-math change — exactly when a gate *should* fire.

Together they lock the composed trade -> exposure contract without coupling to a
moving input. Run directly (``python golden_master_credit_exposure.py``) or under
pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.constants import CEM_PFE_COEFFICIENT_DEFAULT, CEM_PFE_FLOOR_DEFAULT
from riskquant.solutions.credit_exposure.api import (
    CreditExposureRequest,
    run_credit_exposure,
)

_TOL = 1e-6


# ============================================================ LAYER 1: invariants
def _run(model, trades, netting_sets, add_on=None):
    return run_credit_exposure(
        CreditExposureRequest.from_dataframe(
            trades, netting_sets, model=model, add_on_factors=add_on
        )
    )


def _gross_add_on_by_netting_set(trades, add_on):
    """Per-netting-set gross add-on: sum of ``volume * price * factor`` — the
    quantity the NGR multiplier compresses. Recomputed independently of the engine
    so the invariant can bound the engine's output."""
    factor = dict(
        zip(
            add_on["subclass_uix"].astype(int),
            add_on["factor"].astype(float),
            strict=True,
        )
    )
    gross = (
        trades["volume"]
        * trades["current_price"]
        * trades["asset_subclass_uix"].map(factor)
    )
    return trades.assign(_g=gross).groupby("netting_set_id")["_g"].sum().to_dict()


def assert_contract_invariants(trades, netting_sets, add_on):
    """Every property that must hold for *any* book. Anti-fragile: fixture-agnostic."""
    cem = _run("current_exposure_method", trades, netting_sets, add_on)
    rc = _run("replacement_cost", trades, netting_sets)

    cem_obl = {o.uid: o.ead for o in cem.obligor_ead}
    rc_obl = {o.uid: o.ead for o in rc.obligor_ead}
    cem_ns = {n.netting_set_uid: n.ead for n in cem.netting_set_ead}
    rc_ns = {n.netting_set_uid: n.ead for n in rc.netting_set_ead}

    # (1) Non-negativity.
    for uid, ead in cem_obl.items():
        assert ead >= -_TOL, f"negative obligor EAD {uid}: {ead}"

    # (2) Aggregation conserves exposure (default target == full obligor universe).
    assert abs(sum(cem_obl.values()) - sum(cem_ns.values())) <= _TOL, (
        "aggregation does not conserve: "
        f"Sigma obligor {sum(cem_obl.values())} != Sigma netting-set {sum(cem_ns.values())}"
    )

    # (3) Many-to-one is a pure sum: each obligor == sum of its netting sets.
    per_obligor = {}
    for n in cem.netting_set_ead:
        per_obligor[n.obligor_uid] = per_obligor.get(n.obligor_uid, 0.0) + n.ead
    for uid, total in per_obligor.items():
        assert abs(total - cem_obl[uid]) <= _TOL, (
            f"obligor {uid}: netting-set sum {total} != rolled-up EAD {cem_obl[uid]}"
        )

    # (4) CEM dominates RC (the PFE add-on is non-negative): CEM >= RC everywhere.
    for uid in cem_obl:
        assert cem_obl[uid] >= rc_obl[uid] - _TOL, (
            f"obligor {uid}: CEM {cem_obl[uid]} < RC {rc_obl[uid]}"
        )

    # (5) Zero add-on collapses CEM onto RC exactly (isolates the NRC path).
    cem_zero = _run(
        "current_exposure_method", trades, netting_sets, add_on.assign(factor=0.0)
    )
    cem_zero_obl = {o.uid: o.ead for o in cem_zero.obligor_ead}
    for uid in cem_zero_obl:
        assert abs(cem_zero_obl[uid] - rc_obl[uid]) <= _TOL, (
            f"obligor {uid}: zero-add-on CEM {cem_zero_obl[uid]} != RC {rc_obl[uid]}"
        )

    # (6) NGR compression bounds: per netting set the net PFE (CEM - RC, same NRC)
    #     lies in [floor * gross, gross] — locks 0.4 + 0.6*NGR without recomputing NGR.
    gross = _gross_add_on_by_netting_set(trades, add_on)
    for ns_uid, g in gross.items():
        net_pfe = cem_ns[ns_uid] - rc_ns[ns_uid]
        assert CEM_PFE_FLOOR_DEFAULT * g - _TOL <= net_pfe <= g + _TOL, (
            f"netting set {ns_uid}: net PFE {net_pfe} outside "
            f"[{CEM_PFE_FLOOR_DEFAULT * g}, {g}] (gross={g})"
        )
    assert (
        abs((CEM_PFE_FLOOR_DEFAULT + CEM_PFE_COEFFICIENT_DEFAULT) - 1.0) < 1e-12
    )  # multiplier caps at 1


# ============================================================= LAYER 2: frozen case
def _frozen_book():
    """Tiny hand-computed book (subclass factor 0.10, unit price so notional=volume).

    Covers every CEM branch by construction:
      ns1 (obligor 1)  +100 / -60          netting benefit, NGR=0.4
      ns2 (obligor 2)  +80, collateral 30  collateralised, NGR=0.625
      ns3 (obligor 3)  -40                 fully OTM, GRC=0 -> PFE floor
      ns4 (obligor 3)  +20                 NGR=1 (no compression) + aggregation
    """
    trades = pd.DataFrame(
        {
            "uid": [1, 2, 3, 4, 5],
            "netting_set_id": [1, 1, 2, 3, 4],
            "mtm": [100.0, -60.0, 80.0, -40.0, 20.0],
            "volume": [1000.0, 500.0, 200.0, 300.0, 100.0],
            "current_price": [1.0, 1.0, 1.0, 1.0, 1.0],
            "asset_subclass_uix": [1, 1, 1, 1, 1],
        }
    )
    netting_sets = pd.DataFrame(
        {
            "netting_set_uid": [1, 2, 3, 4],
            "obligor_uid": [1, 2, 3, 3],
            "net_collateral": [0.0, 30.0, 0.0, 0.0],
        }
    )
    add_on = pd.DataFrame({"subclass_uix": [1], "factor": [0.10]})
    return trades, netting_sets, add_on


# EAD computed by hand (see _frozen_book); regenerate only on a reviewed model change.
#   ns1: V=40, GRC=100, NRC=40, NGR=0.4,   gross=150, PFE=150*0.64=96   -> 136.0
#   ns2: V=80, GRC=80,  NRC=50, NGR=0.625, gross=20,  PFE=20*0.775=15.5 ->  65.5
#   ns3: V=-40,GRC=0,   NRC=0,  NGR=0,     gross=30,  PFE=30*0.4=12     ->  12.0
#   ns4: V=20, GRC=20,  NRC=20, NGR=1.0,   gross=10,  PFE=10*1.0=10     ->  30.0
_FROZEN_NETTING_SET_EAD = {1: 136.0, 2: 65.5, 3: 12.0, 4: 30.0}
_FROZEN_OBLIGOR_EAD = {1: 136.0, 2: 65.5, 3: 42.0}  # obligor 3 = ns3 + ns4


def test_frozen_case():
    trades, netting_sets, add_on = _frozen_book()
    result = _run("current_exposure_method", trades, netting_sets, add_on)

    got_ns = {n.netting_set_uid: n.ead for n in result.netting_set_ead}
    for ns_uid, expected in _FROZEN_NETTING_SET_EAD.items():
        assert np.isclose(got_ns[ns_uid], expected, atol=_TOL), (
            f"frozen netting set {ns_uid}: {got_ns[ns_uid]} != {expected}"
        )
    got_obl = {o.uid: o.ead for o in result.obligor_ead}
    assert set(got_obl) == set(_FROZEN_OBLIGOR_EAD)
    for uid, expected in _FROZEN_OBLIGOR_EAD.items():
        assert np.isclose(got_obl[uid], expected, atol=_TOL), (
            f"frozen obligor {uid}: {got_obl[uid]} != {expected}"
        )

    assert_contract_invariants(trades, netting_sets, add_on)


def test_invariants_on_canonical_book():
    """Invariants ride the evolving fixture — properties, not pinned numbers."""
    from riskquant_lab.fixtures.credit_exposure import (
        generate_canonical_exposure_book,
    )

    book = generate_canonical_exposure_book()
    trades = pd.DataFrame(
        {
            "uid": book.trade_uid,
            "netting_set_id": book.trade_netting_set_id,
            "mtm": book.mtm,
            "volume": book.volume,
            "current_price": book.current_price,
            "asset_subclass_uix": book.asset_subclass_uix,
        }
    )
    netting_sets = pd.DataFrame(
        {
            "netting_set_uid": book.netting_set_uid,
            "obligor_uid": book.netting_set_obligor_uid,
            "net_collateral": book.netting_set_collateral,
        }
    )
    add_on = pd.DataFrame(
        {"subclass_uix": book.subclass_uix, "factor": book.subclass_factor}
    )
    assert_contract_invariants(trades, netting_sets, add_on)


def main():
    test_frozen_case()
    test_invariants_on_canonical_book()
    print("CONTRACT LOCK PASSED")
    print(f"  frozen obligor EAD (hand-computed): {_FROZEN_OBLIGOR_EAD}")
    print("  structural invariants: conservation, many-to-one sum, CEM>=RC,")
    print("  zero-add-on==RC, NGR compression bounds  (frozen + canonical books)")


if __name__ == "__main__":
    main()
