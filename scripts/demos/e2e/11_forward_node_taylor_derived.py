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
    STEP 6  admission boundaries            (D103.3, D103.4(b), P-1)

Run:  python scripts/demos/e2e/11_forward_node_taylor_derived.py
"""

from __future__ import annotations

import dataclasses
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
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.api import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.book_adapter import (
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorOrder,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
)
from riskquant.utils.dates import date_to_ord32

U32 = 2.0**-24  # float32 unit roundoff: the P&L grid stores float32
LABEL = fw.LABELS
EMPTY = np.empty(0, dtype=np.int64)


def certified_block(world: fw.ForwardWorld, sens, v0: dict[int, float]) -> TaylorBlock:
    """Certify `sens` at the run's base and compile it on the valuation prefix."""
    uids = np.unique(np.asarray(sens.position_uids, dtype=np.int64))
    binding = TaylorBaseBinding(
        as_of_ord32=int(date_to_ord32(fw.AS_OF)),
        base_levels_digest=base_levels_digest(
            factor_ids=world.axis.factor_ids, base_levels=world.base
        ),
        discounting_mode=world.context.discounting_mode.value,
        static_config_digest=static_config_content_digest(world.rsc),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=uids, values=np.array([v0[int(u)] for u in uids])
        ),
    )
    spec = build_sensitivity_book_spec(
        sensitivities=sens, required_factor_ids=world.valuation_factor_ids
    )
    spec = dataclasses.replace(
        spec,
        taylor_declaration=derive_taylor_declaration(
            sensitivities=sens,
            order=TaylorOrder.SECOND_ORDER_FULL,
            base_binding=binding,
        ),
    )
    return TaylorBlock(calculator=SensitivityPnlCalculator.compile(spec))


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
        market_data=_market_at(world, base_cols),
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
    block = certified_block(world, sens, v0)
    sw.kv([
        ("positions in the book", block.calculator.num_positions),
        ("first-order entries", block.calculator.num_first_order_entries),
        ("cross entries", block.calculator.num_cross_entries),
    ])

    # ------------------------------------------------------------------ STEP 4
    sw.step("STEP 4", "SCENARIO P&L: FULL REVALUATION VERSUS TAYLOR")
    sw.para(
        "The SAME scenarios feed both paths: the historical 1-day log returns of "
        "the world's history, applied to today's levels. FULL revalues every "
        "position with the production pricer; TAYLOR evaluates the certified "
        "book with the production calculator."
    )
    hist = world.data.values
    ids = list(world.data.factor_ids)
    returns = np.log(hist[1:] / hist[:-1])  # (S, K) in rf-id order
    shocked = {
        name_of[int(f)]: world.level(name_of[int(f)]) * np.exp(returns[:, ids.index(f)])
        for f in world.axis.factor_ids
    }
    full_pnl = {u: v - v0[u] for u, v in fw.value_by_uid(world, shocked).items()}
    prefix = [name_of[int(f)] for f in world.valuation_factor_ids]
    market = block.calculator.axis.levels(
        base_levels=np.array([world.level(n) for n in prefix]),
        shocked_levels=np.ascontiguousarray(np.stack([shocked[n] for n in prefix])),
    )
    result = block.calculator.run(market)
    taylor_uids = result.pnl_matrix.trade_uids.tolist()
    taylor = np.asarray(result.pnl_matrix.pnl, dtype=np.float64)
    rows4 = []
    for r in fw.MAIN_BOOK:
        uid = r["uid"]
        t_pnl = taylor[taylor_uids.index(uid)]
        diff = np.abs(full_pnl[uid] - t_pnl)
        scale = np.abs(full_pnl[uid]) + _leg_scale(r, shocked)
        tol = 8.0 * U32 * scale
        rows4.append([
            LABEL[uid],
            t_pnl.size,
            sw.num(np.abs(full_pnl[uid]).max(), 3),
            f"{diff.max():.2e}",
            f"{(diff / tol).max():.3f}",
        ])
        checks.check(
            f"{LABEL[uid]}: TAYLOR == FULL on every scenario within 8u * scale",
            bool(np.all(diff <= tol)),
        )
    sw.table(["", "scenarios", "max |dV| FULL", "max |diff|", "max diff/tol"], rows4)
    sw.para(
        "The tolerance is 8 x the float32 unit roundoff (u = 2^-24) times the "
        "position's scale (|dV| plus its leg magnitudes): Taylor P&L is stored in "
        "float32, full revaluation is float64. Nothing else separates them: in "
        "this NONE contract the second-order polynomial IS the forward's value "
        "change. This is not a claim about other discounting modes."
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
    sw.step("STEP 6", "ADMISSION BOUNDARIES")
    sw.table(
        ["gate", "what it refuses"],
        [
            [
                "D102.3",
                "a UID whose forward source rows are not all VANILLA_FORWARD_NODE",
            ],
            ["D103.3", "TAYLOR/DERIVED forward with an inverted FX leg"],
            ["D103.4(a)", "TAYLOR/DERIVED forward UID with a non-SPOT non-forward row"],
            ["D103.4(b)", "TAYLOR/DERIVED forward unless DiscountingMode.NONE"],
            ["D084", "TAYLOR SPOT position with an inverted FX leg"],
            ["P-1", "every TAYLOR/SUPPLIED position"],
        ],
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
            market_data=_market_at(
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


def _market_at(world: fw.ForwardWorld, levels_by_name):
    """Production market grids at the given levels (used by the sensitizer)."""
    from riskquant.market_data.assemble import (
        build_market_data_matrix_from_flat_levels,
        build_universe_mapping,
    )

    layout = build_universe_mapping(
        static_config=world.rsc,
        uix_mapper=world.context.uix_mapper,
        discounting_mode=world.context.discounting_mode,
    )
    name_of = {v: k for k, v in world.rsc.rel_name_to_rf_id.items()}
    flat = np.ascontiguousarray(
        np.stack([levels_by_name[name_of[int(f)]] for f in layout.required_factor_ids])
    )
    return build_market_data_matrix_from_flat_levels(
        discounting_mode=world.context.discounting_mode,
        as_of_date=fw.AS_OF,
        uix_mapper=world.context.uix_mapper,
        universe_map=layout,
        flat_levels=flat,
        num_scenarios=flat.shape[1],
    )


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


def _leg_scale(row, shocked) -> np.ndarray:
    """|Q| (|F_s X_a,s| + |K X_k,s|) per scenario (|Q S_s X_s| for a SPOT)."""
    x = shocked["USDEUR"]
    if row["instrument"] == "SPOT":
        return np.abs(row["quantity"] * shocked[row["underlying"]])
    xa = x if row["settlement_ccy"] == "EUR" else 1.0
    xk = x if row["settlement_ccy"] == "EUR" else 1.0
    return abs(row["quantity"]) * (
        np.abs(shocked[row["underlying"]] * xa) + abs(row["strike_price"]) * np.abs(xk)
    )


if __name__ == "__main__":
    main()
