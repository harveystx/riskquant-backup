# scripts/demos/e2e/10_forward_node_full_reval.py
"""
RiskQuant VANILLA_FORWARD_NODE -- chapter 10: full revaluation.

A forward obliges the holder to exchange the underlying for the strike K on a
future date. RiskQuant values it as

    V = Q * (EFP - K * EDF)
    EFP = F * DF_a * X_a      (asset leg: forward level, discount, FX)
    EDF = DF_k * X_k          (strike leg: discount, FX)

In DiscountingMode.NONE every discount factor is 1, so V = Q * (F*X_a - K*X_k).
This chapter builds a five-position book through the production path, checks
every base value against that hand formula, shows how the NODE identity is
resolved, converts the values into a reporting currency, and runs FULL_REVAL
VaR / ES with the one production entry point `run_scenario_var_es`.

    STEP 1  the market world        (price curves, FX orientation)
    STEP 2  the book                (five rows, human-readable)
    STEP 3  ingest and NODE identity (exact vs relative NODE, delivery)
    STEP 4  value the base          (production V0 vs hand formula)
    STEP 5  reporting currency      (D081: W = V * r)
    STEP 6  FULL_REVAL risk         (VaR / ES, per UID and portfolio)

Every printed claim is a check; a failed check stops the script (FAIL FAST).
The primary correctness evidence is STEP 4. VaR / ES are downstream results.

Run:  python scripts/demos/e2e/10_forward_node_full_reval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _forward_world as fw
import _spot_world as sw

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import MethodologyDeclaration, ReportingSpec
from riskquant.solutions.scenario_models.api import parse_scenario_model_config

D_FEB, D_MAR = fw.D_FEB, fw.D_MAR
ROWS = fw.MAIN_BOOK
LABEL = fw.LABELS


def run(world: fw.ForwardWorld) -> object:
    return run_scenario_var_es(
        scenario_config=parse_scenario_model_config({
            "model": "historical",
            "horizon": 1,
        }),
        data=world.data,
        required_factor_ids=world.axis.factor_ids.copy(),
        required_shift_type_id=world.axis.shift_type_ids.copy(),
        proxy_terms=(),
        base_levels=world.base.copy(),
        methodology=MethodologyDeclaration(
            production_position_uids=world.uids,
            full_reval_uids=world.uids,
            taylor_supplied_uids=np.empty(0, dtype=np.int64),
            taylor_derived_uids=np.empty(0, dtype=np.int64),
        ),
        valuation_context=world.context,
        reporting=ReportingSpec(reporting_ccy=world.reporting_ccy),
        confidence_level=0.95,
        batch_size=32,
    )


def main() -> None:
    checks = sw.Checks()
    sw.title("VANILLA_FORWARD_NODE -- chapter 10: full revaluation")

    # ------------------------------------------------------------------ STEP 1
    sw.step("STEP 1", "THE MARKET WORLD")
    sw.para(
        "A forward NODE is a forward whose underlying is ONE scalar risk factor: "
        "a pillar of a PRICE curve. Two kinds of pillar exist. An EXACT node is a "
        "fixed LABEL pillar such as DEMO_GAS_FWD[1M]: it is the same risk factor "
        "on every run date. A RELATIVE node such as BRENT_CRUDE_FWD[M+0] names a "
        "position on a calendar curve: the risk factor is fixed by the name, and "
        "the contract it refers to is resolved at the as-of date."
    )
    rsc = fw.static_config()
    sw.kv([
        ("as-of date", fw.AS_OF.isoformat()),
        ("exact curve", "DEMO_GAS_FWD: PRICE, LABEL, pillars [1M], [3M] (demo-owned)"),
        (
            "relative curves",
            "BRENT_CRUDE_FWD, WTI_CRUDE_FWD: PRICE, OFFSET_M, CALENDAR_PERIOD",
        ),
        ("FX node", "USDEUR = EUR per 1 USD"),
        ("discounting", "DiscountingMode.NONE: every discount factor DF = 1"),
    ])
    sw.para(
        "FX orientation matters. USD -> EUR multiplies by USDEUR (direct, "
        "invert = 0). EUR -> USD divides by USDEUR (inverted, invert = 1). Every "
        "FX leg of this book is direct or absent; chapter 11 shows why that "
        "matters for TAYLOR/DERIVED."
    )

    # ------------------------------------------------------------------ STEP 2
    sw.step("STEP 2", "THE BOOK")
    table_rows = []
    for r in ROWS:
        if r["instrument"] == "SPOT":
            table_rows.append([
                LABEL[r["uid"]],
                r["uid"],
                "SPOT",
                r["underlying"],
                sw.num(r["quantity"], 0),
                "-",
                "-",
                r["settlement_ccy"],
            ])
            continue
        table_rows.append([
            LABEL[r["uid"]],
            r["uid"],
            "FWD NODE",
            r["underlying"],
            sw.num(r["quantity"], 0),
            sw.num(r["strike_price"], 2) + " " + r["strike_ccy"],
            r["delivery_date"].isoformat() if r["delivery_date"] else "(none)",
            r["settlement_ccy"],
        ])
    sw.table(
        ["", "uid", "type", "underlying", "Q", "strike", "delivery", "settle"],
        table_rows,
    )
    sw.para(
        "F1 and F2 reference the SAME exact node DEMO_GAS_FWD[1M]: F1 carries a "
        "delivery date, F2 does not. F3 is a relative node. F4 settles in EUR, so "
        "both of its legs (asset USD, strike USD) carry the FX node. S1 is an "
        "ordinary SPOT position beside the forwards."
    )
    fx_rows = []
    for r in ROWS:
        if r["instrument"] != "VANILLA_FORWARD_NODE":
            continue
        und_ccy = fw.UNDERLYING_CCY[fw._family(r["underlying"])]
        a = fw.fx_leg(und_ccy, r["settlement_ccy"])
        k = fw.fx_leg(r["strike_ccy"], r["settlement_ccy"])
        fx_rows.append([
            LABEL[r["uid"]],
            f"{und_ccy}->{r['settlement_ccy']}",
            a[1],
            f"{r['strike_ccy']}->{r['settlement_ccy']}",
            k[1],
        ])
        checks.check(
            f"{LABEL[r['uid']]}: no inverted FX leg in the main book",
            "inverted" not in (a[0], k[0]),
        )
    sw.table(["", "asset leg", "X_a", "strike leg", "X_k"], fx_rows)

    # ------------------------------------------------------------------ STEP 3
    sw.step("STEP 3", "INGEST AND NODE IDENTITY")
    world = fw.build_world(ROWS, reporting_ccy="EUR", rsc=rsc)
    batch = world.context.ingest_batch
    rf_of = {
        int(u): int(r) for u, r in zip(batch.uids, batch.underlying_ids, strict=True)
    }
    sw.para(
        "Production ingest turns every name into an integer risk-factor id (rf "
        "id). The table shows the rf id each forward row was bound to."
    )
    sw.table(
        ["", "underlying", "rf id"],
        [
            [LABEL[r["uid"]], r["underlying"], rf_of[r["uid"]]]
            for r in ROWS
            if r["instrument"] == "VANILLA_FORWARD_NODE"
        ],
    )
    checks.check(
        "exact NODE: F1 (delivery) and F2 (no delivery) bind the SAME rf id",
        rf_of[7_301] == rf_of[12] == rsc.rel_name_to_rf_id["DEMO_GAS_FWD[1M]"],
    )
    sw.para(
        "The delivery date does NOT choose the exact node. It only says when the "
        "asset leg is delivered; without it the asset leg uses the settlement "
        "date (D101). Both rows are the same risk factor."
    )
    contract = fw.resolved_contract(rsc, "BRENT_CRUDE_FWD[M+0]")
    sw.kv([
        ("BRENT_CRUDE_FWD[M+0] at " + fw.AS_OF.isoformat(), contract),
        ("F3 delivery date", D_FEB.isoformat()),
    ])
    checks.check(
        "relative NODE: F3 binds the rf id of its NAME (fixed for the run)",
        rf_of[90_001] == rsc.rel_name_to_rf_id["BRENT_CRUDE_FWD[M+0]"],
    )
    checks.check(
        "relative NODE: the delivery date lies in the resolved contract month",
        contract.endswith(f"{D_FEB.year}{D_FEB.month:02d}"),
    )
    sw.para(
        "A relative node REQUIRES a delivery date, and ingest refuses a date "
        "outside the resolved contract period (D065). The rf id never changes "
        "with the as-of date; the contract behind it does."
    )

    # ------------------------------------------------------------------ STEP 4
    sw.step("STEP 4", "VALUE THE BASE: PRODUCTION V0 VERSUS THE HAND FORMULA")
    sw.formula(
        "V = Q * (F * X_a - K * X_k)        (forward, NONE)",
        "V = Q * S * X                      (SPOT)",
    )
    v0 = fw.base_value_by_uid(world)
    rows4, scale = [], {}
    for r in ROWS:
        hand = fw.hand_value(r, fw.LEVELS)
        prod = v0[r["uid"]]
        if r["instrument"] == "SPOT":
            scale[r["uid"]] = abs(hand)
        else:
            und_ccy = fw.UNDERLYING_CCY[fw._family(r["underlying"])]
            x_a = fw.fx_value(
                fw.fx_leg(und_ccy, r["settlement_ccy"])[0], fw.LEVELS["USDEUR"]
            )
            x_k = fw.fx_value(
                fw.fx_leg(r["strike_ccy"], r["settlement_ccy"])[0], fw.LEVELS["USDEUR"]
            )
            scale[r["uid"]] = abs(r["quantity"]) * (
                abs(fw.LEVELS[r["underlying"]] * x_a) + abs(r["strike_price"] * x_k)
            )
        tol = 2.0**-48 * scale[r["uid"]]
        rows4.append([
            LABEL[r["uid"]],
            sw.num(hand, 6),
            sw.num(prod, 6),
            f"{abs(prod - hand):.2e}",
            f"{tol:.1e}",
        ])
        checks.check(
            f"{LABEL[r['uid']]}: production V0 == hand formula within 2^-48 * T",
            abs(prod - hand) <= tol,
        )
    sw.table(["", "hand V0", "production V0", "|diff|", "tolerance"], rows4)
    sw.para(
        "The tolerance is 2^-48 times the absolute leg scale T = |Q|(|F X_a| + "
        "|K X_k|): both sides round each multiplication once in float64, so they "
        "can differ in the last bits but never by more than that."
    )
    checks.check(
        "F1 and F2 (same node, opposite sides) value independently",
        v0[7_301] > 0.0 > v0[12],
    )

    # ------------------------------------------------------------------ STEP 5
    sw.step("STEP 5", "REPORTING CURRENCY (D081)")
    sw.para(
        "Each position is valued in its SETTLEMENT currency. The run converts "
        "values into ONE reporting currency with the rate r of that day: W = V * "
        "r, and the reported scenario P&L is dW = V_s * r_s - V_0 * r_0 (D081). "
        "The reporting FX is never a sensitivity factor."
    )
    rows5 = []
    for r in ROWS:
        orient, text = fw.fx_leg(r["settlement_ccy"], "EUR")
        rate = fw.fx_value(orient, fw.LEVELS["USDEUR"])
        rows5.append([
            LABEL[r["uid"]],
            r["settlement_ccy"],
            sw.num(v0[r["uid"]], 4),
            text,
            sw.num(v0[r["uid"]] * rate, 4),
        ])
    sw.table(["", "settle", "V0 (settle ccy)", "r0 into EUR", "W0 (EUR)"], rows5)

    # ------------------------------------------------------------------ STEP 6
    sw.step("STEP 6", "FULL_REVAL RISK: VaR AND ES")
    sw.para(
        "One call of run_scenario_var_es: historical scenarios (1-day returns of "
        f"{world.data.values.shape[0]} daily observations), every position "
        "FULL_REVAL, confidence 0.95, reporting in EUR and then in USD."
    )
    results = {}
    for ccy in ("EUR", "USD"):
        w = (
            world
            if ccy == "EUR"
            else fw.build_world(ROWS, reporting_ccy="USD", rsc=rsc)
        )
        result = run(w)
        results[ccy] = result
        checks.check(
            f"{ccy} run: one VaR and ES per UID plus the portfolio",
            result.var.shape == (world.uids.size + 1,)
            and result.position_uids.tolist() == world.uids.tolist(),
        )
        checks.check(
            f"{ccy} run: every VaR and ES is finite",
            bool(np.isfinite(result.var).all() and np.isfinite(result.es).all()),
        )
        checks.check(
            f"{ccy} run: ES is not above VaR (lower tail)",
            bool(np.all(result.es <= result.var + 1e-9)),
        )
    rows6 = []
    for i, uid in enumerate(world.uids.tolist()):
        rows6.append([
            LABEL[uid],
            uid,
            sw.num(results["EUR"].var[i], 4),
            sw.num(results["EUR"].es[i], 4),
            sw.num(results["USD"].var[i], 4),
        ])
    rows6.append([
        "book",
        "-",
        sw.num(results["EUR"].portfolio_var, 4),
        sw.num(results["EUR"].portfolio_es, 4),
        sw.num(results["USD"].portfolio_var, 4),
    ])
    sw.table(["", "uid", "VaR (EUR)", "ES (EUR)", "VaR (USD)"], rows6)
    sw.para(
        "VaR is the 5% lower-tail scenario P&L (a loss is negative). The EUR and "
        "USD columns differ because the reporting rate itself moves in every "
        "scenario (D081)."
    )
    sw.write_evidence(
        "10_forward_node_full_reval",
        {
            "as_of": fw.AS_OF.isoformat(),
            "v0_by_uid": {str(u): float(v).hex() for u, v in sorted(v0.items())},
            "var_eur": [float(v).hex() for v in results["EUR"].var],
            "identity_eur": results["EUR"].identity,
        },
    )
    checks.done()


if __name__ == "__main__":
    main()
