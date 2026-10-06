# scripts/demos/e2e/11_forward_node_taylor_derived.py
"""
RiskQuant VANILLA_FORWARD_NODE -- chapter 11: TAYLOR/DERIVED risk.

The SAME book as chapter 10, now priced from sensitivities:

    production sensitizer (run_sensitivities)  -> closed-form partials at the base
    build_sensitivity_book_spec                -> the book, on the run's valuation axis
    derive_taylor_declaration                  -> order + factor set F(p) + base binding
    SensitivityPnlCalculator                   -> compiled, certified book
    TaylorBlock -> run_scenario_var_es          (TAYLOR/DERIVED partition)

The coordinate is the LEVEL of each factor. With V = Q*(F*DF_a*X_a - K*DF_k*X_k)
and DF = 1 (DiscountingMode.NONE):

    dV/dF   = Q * X_a         dV/dX_a = Q * F         dV/dX_k = -Q * K
    d2V/(dF dX_a) = Q         every other second derivative is 0

V is bilinear in (F, X_a) and linear in X_k, so the SECOND_ORDER_FULL Taylor
polynomial reproduces full revaluation on every scenario in this NONE
contract. The chapter shows that scenario by scenario (the primary check),
then compares VaR / ES (downstream), then shows the admission boundaries:

    STEP 1  the book and the base           (chapter 10's world)
    STEP 2  production sensitivities        (checked against the formulas)
    STEP 3  certify the Taylor book         (base binding, SECOND_ORDER_FULL)
    STEP 4  scenario P&L: FULL vs TAYLOR    (every scenario, every UID)
    STEP 5  VaR / ES: FULL vs TAYLOR        (downstream confirmation)
    STEP 6  selected admission boundaries   (D103.3, D103.4(b), P-1 executed)

Run:  python scripts/demos/e2e/11_forward_node_taylor_derived.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _forward_world as fw
import _spot_world as sw

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    ReportingSpec,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config

U32 = 2.0**-24  # float32 unit roundoff: the P&L grid stores float32
LABEL = fw.LABELS
EMPTY = np.empty(0, dtype=np.int64)


def run(world, *, full=(), derived=(), supplied=(), block=None, context=None):
    """One production call; the methodology partition is explicit."""
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
            full_reval_uids=np.array(sorted(full), dtype=np.int64),
            taylor_supplied_uids=np.array(sorted(supplied), dtype=np.int64),
            taylor_derived_uids=np.array(sorted(derived), dtype=np.int64),
        ),
        valuation_context=world.context if context is None else context,
        reporting=ReportingSpec(reporting_ccy=world.reporting_ccy),
        confidence_level=0.95,
        batch_size=32,
        taylor_derived=block,
    )


def refusal(checks, label, call, *, expected_type, prefix, why, decision):
    """Run a deliberately invalid request and show CASE / EXPECTED / ACTUAL / WHY."""
    error = checks.expect_rejection(
        label, expected_type, call, reason=why, stage=decision
    )
    message = str(error)
    sw.kv([
        ("CASE", label),
        ("EXPECTED", f"{expected_type.__name__} starting with {prefix!r}"),
        ("ACTUAL", f"{type(error).__name__}: {message[:120]}"),
        ("WHY", why),
        ("OWNER", decision),
    ])
    checks.check(f"{label}: message starts with {prefix!r}", message.startswith(prefix))


def main() -> None:
    checks = sw.Checks()
    sw.title("VANILLA_FORWARD_NODE -- chapter 11: TAYLOR/DERIVED")

    # ------------------------------------------------------------------ STEP 1
    sw.step("STEP 1", "THE BOOK AND THE BASE (CHAPTER 10'S WORLD)")
    rsc = fw.static_config()
    world = fw.build_world(fw.MAIN_BOOK, reporting_ccy="EUR", rsc=rsc)
    v0 = fw.base_value_by_uid(world)
    sw.table(
        ["", "uid", "underlying", "Q", "settle", "V0 (settle ccy)"],
        [
            [
                LABEL[r["uid"]],
                r["uid"],
                r["underlying"],
                sw.num(r["quantity"], 0),
                r["settlement_ccy"],
                sw.num(v0[r["uid"]], 4),
            ]
            for r in fw.MAIN_BOOK
        ],
    )
    sw.para(
        "Every FX leg of this book is direct (USD -> EUR) or absent, and the run "
        "uses DiscountingMode.NONE: both are conditions of TAYLOR/DERIVED "
        "admission for a forward (D103.3, D103.4(b)). STEP 6 shows what happens "
        "when either condition fails."
    )

    # ------------------------------------------------------------------ STEP 2
    sw.step("STEP 2", "PRODUCTION SENSITIVITIES")
    base_cols = {n: np.array([world.level(n)]) for n in world.names}
    sens = run_sensitivities(
        portfolio=world.context.portfolio,
        uix_mapper=world.context.uix_mapper,
        market_data=fw.market_at(world, base_cols),
    )
    name_of = {v: k for k, v in rsc.rel_name_to_rf_id.items()}
    sw.formula(
        "dV/dF   = Q * X_a       dV/dX_a = Q * F       dV/dX_k = -Q * K",
        "d2V/(dF dX_a) = Q       (X_k never enters a second derivative)",
    )
    rows2 = []
    for r in fw.MAIN_BOOK:
        uid = r["uid"]
        m = sens.position_uids == uid
        expected = _expected_first_order(r)
        for rf, d1 in zip(
            sens.factor_rf_ids[m].tolist(), sens.first_order[m].tolist(), strict=True
        ):
            name = name_of[rf]
            exp = expected[name]
            rows2.append([LABEL[uid], name, sw.num(d1, 6), sw.num(exp, 6)])
            checks.check(
                f"{LABEL[uid]} dV/d{name}: production == formula",
                abs(d1 - exp) <= 2.0**-48 * (abs(exp) + 1.0),
            )
    sw.table(["", "factor", "production", "formula"], rows2)
    c = sens.cross_position_uids == 4_444
    sw.kv([
        (
            "F4 cross term d2V/(dF dX)",
            f"{float(sens.cross_second_order[c][0]):.6f}"
            f"  (formula Q = {fw.MAIN_BOOK[3]['quantity']:.1f})",
        )
    ])
    checks.check("F4 cross term == Q", float(sens.cross_second_order[c][0]) == -90.0)
    sw.para(
        "F4's asset leg and strike leg use the SAME FX node, so their partials "
        "merge into one USDEUR row: Q * (F - K). The sensitizer works in the "
        "LEVEL coordinate of each factor; no proxy loading ever enters it."
    )

    # ------------------------------------------------------------------ STEP 3
    sw.step("STEP 3", "CERTIFY THE TAYLOR BOOK")
    sw.para(
        "Certification binds the book to the run's exact base (levels, static "
        "configuration, valuation contract, V0) and declares its order. "
        "SECOND_ORDER_FULL requires every factor pair of each position, with an "
        "explicit zero where the pair is mathematically zero (D086)."
    )
    block = fw.certified_taylor_block(world, sens, v0)
    sw.kv([
        ("positions in the book", block.calculator.num_positions),
        ("first-order entries", block.calculator.num_first_order_entries),
        ("cross entries", block.calculator.num_cross_entries),
    ])

    # ------------------------------------------------------------------ STEP 4
    sw.step("STEP 4", "SCENARIO P&L: FULL REVALUATION VERSUS TAYLOR")
    sw.para(
        "Both paths are fed the SAME production scenarios. They come from the "
        "production scenario path, in the order run_scenario_var_es composes it: "
        "proxy model over the run axis (no proxies here), axis operator, world "
        "binding, the configured scenario generator (historical, 1-day horizon), "
        "and the inverse chart that turns each shock into a shocked LEVEL. The "
        "demo does not compute a return or a shock itself."
    )
    scenario_config = parse_scenario_model_config({"model": "historical", "horizon": 1})
    shocked = fw.production_shocked_levels(
        world, scenario_config=scenario_config, batch_size=32
    )
    sw.kv([
        ("scenario model", "historical, horizon 1 (production generator)"),
        (
            "shocked levels",
            f"{shocked.shape[0]} factors x {shocked.shape[1]} scenarios",
        ),
    ])
    full_pnl, taylor_pnl = fw.scenario_pnl_full_and_taylor(world, block, shocked)
    by_name = {n: shocked[i] for i, n in enumerate(world.names)}
    rows4 = []
    for r in fw.MAIN_BOOK:
        uid = r["uid"]
        diff = np.abs(full_pnl[uid] - taylor_pnl[uid])
        scale = np.abs(full_pnl[uid]) + fw.leg_scale(r, by_name)
        tol = 8.0 * U32 * scale
        big = np.abs(full_pnl[uid]) > 1e-6 * scale
        rel = (
            float((diff[big] / np.abs(full_pnl[uid][big])).max()) if big.any() else 0.0
        )
        ok = bool(np.all(diff <= tol))
        rows4.append([
            LABEL[uid],
            diff.size,
            sw.num(np.abs(full_pnl[uid]).max(), 3),
            f"{diff.max():.2e}",
            f"{rel:.1e}",
            f"{(diff / tol).max():.3f}",
            "PASS" if ok else "FAIL",
        ])
        checks.check(
            f"{LABEL[uid]}: TAYLOR == FULL on every scenario within 8u * scale", ok
        )
    sw.table(
        [
            "",
            "scenarios",
            "max |dV| FULL",
            "max |diff|",
            "max rel",
            "diff/tol",
            "result",
        ],
        rows4,
    )
    sw.para(
        "The tolerance is 8 x the float32 unit roundoff (u = 2^-24) times the "
        "position's scale (|dV| plus its leg magnitudes): Taylor P&L is stored in "
        "float32, full revaluation is float64. Nothing else separates them: in "
        "the TAYLOR/DERIVED, DiscountingMode.NONE contract for VANILLA_FORWARD_NODE "
        "the second-order polynomial IS the value change. This is not a claim "
        "about any other mode or instrument. 'max rel' skips scenarios whose "
        "|dV| is below 1e-6 of the scale."
    )

    # ------------------------------------------------------------------ STEP 5
    sw.step("STEP 5", "VaR AND ES: FULL VERSUS TAYLOR (DOWNSTREAM)")
    uids = world.uids.tolist()
    r_full = run(world, full=uids)
    r_taylor = run(world, derived=uids, block=block)
    checks.check(
        "TAYLOR run carries Taylor provenance", r_taylor.pnl_provenance is not None
    )
    rows5 = []
    for i, uid in enumerate([*uids, "book"]):
        dvar = abs(r_full.var[i] - r_taylor.var[i])
        des = abs(r_full.es[i] - r_taylor.es[i])
        tol = 8.0 * U32 * (abs(r_full.var[i]) + abs(r_full.es[i]) + 1.0)
        rows5.append([
            LABEL.get(uid, uid),
            sw.num(r_full.var[i], 4),
            sw.num(r_taylor.var[i], 4),
            sw.num(r_full.es[i], 4),
            sw.num(r_taylor.es[i], 4),
        ])
        checks.check(
            f"{LABEL.get(uid, uid)}: VaR and ES agree within 8u * scale",
            dvar <= tol and des <= tol,
        )
    sw.table(["", "VaR FULL", "VaR TAYLOR", "ES FULL", "ES TAYLOR"], rows5)
    sw.para(
        "VaR and ES are order statistics of the scenario P&L, so they agree "
        "because STEP 4 agrees. They confirm the result; STEP 4 proves it."
    )

    # ------------------------------------------------------------------ STEP 6
    sw.step("STEP 6", "SELECTED ADMISSION-BOUNDARY DEMONSTRATIONS")
    sw.para(
        "The table lists the forward-relevant admission gates. This chapter "
        "EXECUTES only a selection: D103.3, D103.4(b), P-1 and the isolation of "
        "FULL_REVAL (an inverted forward stays admitted there). D102.3, "
        "D103.4(a) and D084 are listed for orientation only: their composition "
        "cases (repeated UIDs, mixed source rows) are exercised by the "
        "certification test suites, not by this guide."
    )
    sw.table(
        ["gate", "what it refuses", "executed here"],
        [
            [
                "D102.3",
                "a forward-family source row that is not NODE",
                "no (tests)",
            ],
            ["D103.3", "DERIVED forward with an inverted FX leg", "yes"],
            [
                "D103.4(a)",
                "DERIVED forward UID with a non-SPOT other row",
                "no (tests)",
            ],
            ["D103.4(b)", "DERIVED forward when the mode is not NONE", "yes"],
            ["D084", "TAYLOR SPOT with an inverted FX leg", "no (tests)"],
            ["P-1", "every TAYLOR/SUPPLIED position", "yes"],
        ],
        align="lll",
    )
    sw.para(
        "Inverted forward. F6 is a BRENT_CRUDE_FWD[M+1] forward with a EUR "
        "strike, settled in USD. Its strike leg converts EUR -> USD, which is "
        "1 / USDEUR: invert = 1. FULL_REVAL values it like any other forward."
    )
    f6 = fw.forward(
        815,
        "BRENT_CRUDE_FWD[M+1]",
        50.0,
        74.0,
        strike_ccy="EUR",
        settle_ccy="USD",
        delivery=dt.date(2026, 3, 16),
        settlement=dt.date(2026, 3, 18),
    )
    inverted = fw.build_world([f6], reporting_ccy="EUR", rsc=rsc)
    sw.kv([
        ("asset leg", "USD -> USD: " + fw.fx_leg("USD", "USD")[1]),
        ("strike leg", "EUR -> USD: " + fw.fx_leg("EUR", "USD")[1]),
    ])
    r_inv = run(inverted, full=[815])
    checks.check(
        "F6 in FULL_REVAL: admitted and valued (finite VaR / ES)",
        bool(np.isfinite(r_inv.var).all() and np.isfinite(r_inv.es).all()),
    )
    refusal(
        checks,
        "F6 sensitivities (producer side)",
        lambda: run_sensitivities(
            portfolio=inverted.context.portfolio,
            uix_mapper=inverted.context.uix_mapper,
            market_data=fw.market_at(
                inverted, {n: np.array([inverted.level(n)]) for n in inverted.names}
            ),
        ),
        expected_type=NotImplementedError,
        prefix="D103.3",
        why="the partials of an inverted leg are outside the admission contract",
        decision="D103.3 (sensitizer refuses before any market level is read)",
    )
    refusal(
        checks,
        "F6 declared TAYLOR/DERIVED (run boundary)",
        lambda: run(inverted, derived=[815], block=block),
        expected_type=CompositionContractError,
        prefix="D103.3",
        why="an inverted forward FX leg is admitted as FULL_REVAL only",
        decision="D103.3 (run admission, before any downstream work)",
    )
    curve_context = ValuationContext(
        static_config=world.rsc,
        ingest_batch=world.context.ingest_batch,
        uix_mapper=world.context.uix_mapper,
        discounting_mode=DiscountingMode.CURVE,
    )
    refusal(
        checks,
        "F1 TAYLOR/DERIVED with DiscountingMode.CURVE",
        lambda: run(
            world,
            full=[u for u in uids if u != 7_301],
            derived=[7_301],
            block=block,
            context=curve_context,
        ),
        expected_type=CompositionContractError,
        prefix="D103.4",
        why="TAYLOR/DERIVED forwards are admitted in DiscountingMode.NONE only",
        decision="D103.4(b) (CURVE TAYLOR is deferred, D103.6)",
    )
    refusal(
        checks,
        "F1 declared TAYLOR/SUPPLIED",
        lambda: run(world, full=[u for u in uids if u != 7_301], supplied=[7_301]),
        expected_type=CompositionContractError,
        prefix="TAYLOR/SUPPLIED is not",
        why="caller-supplied Taylor books are not admitted (owner policy P-1, D103.5)",
        decision="P-1 (refused before any forward-specific gate)",
    )
    sw.para(
        "None of these refusals switches methodology or drops a position: the "
        "run stops, names the decision, and nothing downstream is computed."
    )
    sw.write_evidence(
        "11_forward_node_taylor_derived",
        {
            "as_of": fw.AS_OF.isoformat(),
            "var_full": [float(v).hex() for v in r_full.var],
            "var_taylor": [float(v).hex() for v in r_taylor.var],
        },
    )
    checks.done()


def _expected_first_order(row) -> dict[str, float]:
    """The section-4 partials, written out from the row and today's levels."""
    lv, q = fw.LEVELS, row["quantity"]
    if row["instrument"] == "SPOT":
        orient = fw.fx_leg("USD", row["settlement_ccy"])[0]
        out = {row["underlying"]: q * fw.fx_value(orient, lv["USDEUR"])}
        if orient != "identity":
            out["USDEUR"] = q * lv[row["underlying"]]
        return out
    und_ccy = fw.UNDERLYING_CCY[fw._family(row["underlying"])]
    a = fw.fx_leg(und_ccy, row["settlement_ccy"])[0]
    k = fw.fx_leg(row["strike_ccy"], row["settlement_ccy"])[0]
    out = {row["underlying"]: q * fw.fx_value(a, lv["USDEUR"])}
    if a != "identity":
        out["USDEUR"] = out.get("USDEUR", 0.0) + q * lv[row["underlying"]]
    if k != "identity":
        out["USDEUR"] = out.get("USDEUR", 0.0) - q * row["strike_price"]
    return out


if __name__ == "__main__":
    main()
