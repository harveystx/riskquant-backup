# scripts/demos/e2e/03_spot_fx_and_reporting_currency.py
"""
RiskQuant SPOT -- chapter 03: FX legs and the reporting currency.

Besides being an underlying itself, an FX rate enters a SPOT run in two
roles that never mix:

    VALUATION FX  the pricer reads a leg that converts the underlying's quote
                  currency into the settlement currency. It is a risk factor
                  of the position: it appears in its sensitivities.
    REPORTING FX  the run converts settlement-currency P&L into the one
                  reporting currency. It is applied after P&L; it is never a
                  Taylor factor.

Admitted and refused FX configurations (Part D shows the valuation legs):

    no FX leg (quote ccy == settlement ccy)        FULL_REVAL, TAYLOR/DERIVED
    direct valuation leg (quanto, q P X)           FULL_REVAL, TAYLOR/DERIVED
    inverted valuation leg (q P / X, D084)         FULL_REVAL only
    reporting route: direct or reciprocal pair     admitted (D085)
    reporting route: no configured pair            refused, no triangulation

Scenario P&L is computed per position in its OWN settlement currency. The
reporting layer (`riskquant.reporting.fx_conversion`) then converts every row
into ONE reporting currency and only then sums the portfolio:

    q    = settlement -> reporting rate, read from the scenario's own FX level
    dW   = V_s q_s - V_0 q_0 = dV q_s + V_0 (q_s - q_0)          (D081)

The caller chooses ONLY the reporting currency (D085). The run derives each
position's settlement currency from its compiled state, and its route from
the configured FX pairs: "EURUSD" multiplies EUR -> USD, so converting USD
into EUR uses 1 / EURUSD (invert = 1). A route that no configured pair
supports is refused before identity and numerics; no rate is ever defaulted.

Reporting FX is NOT a Taylor factor. The sensitivity book of a position
depends on what its PRICER reads, never on the reporting currency: the same
certified book serves every reporting currency, and the manifest proves it.
A reporting FX factor that no pricer reads still enters the run: it is a
reporting-only row after the valuation prefix of the run axis (D082, D085).

Part A: tiny exact world, reported in USD and in EUR.
Part B: realistic world with USD, EUR and GBP settlement.
Part C: a reporting-only FX factor, exact; why V_0 enters the P&L.
Part D: valuation FX legs, direct and inverted, in one run.

Run:  python scripts/demos/e2e/03_spot_fx_and_reporting_currency.py
"""

from __future__ import annotations

import _spot_world as sw
import numpy as np
import pandas as pd

from riskquant.reporting.fx_conversion import build_fx_conversion_plan
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    ReportingSpec,
    RiskRun,
)
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig

SCENARIOS = HistoricalGeneratorConfig(model="historical", horizon=1)
EPS = float(np.finfo(np.float64).eps)


def _run(world: sw.SpotWorld, data, reporting: ReportingSpec, *, derived: bool,
         confidence: float, batch_size: int) -> tuple[RiskRun, sw.ScenarioAnatomy]:
    uids = world.uids.tolist()
    method = sw.methodology(derived=uids) if derived else sw.methodology(full=uids)
    block = sw.certified_taylor_block(world, uids) if derived else None
    run = run_scenario_var_es(
        scenario_config=SCENARIOS,
        data=data,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        proxy_terms=world.proxy_terms,
        base_levels=world.base_levels,
        methodology=method,
        valuation_context=sw.valuation_context(world),
        taylor_derived=block,
        reporting=reporting,
        confidence_level=confidence,
        batch_size=batch_size,
    )
    anatomy = sw.scenario_pnl_anatomy(world, scenario_config=SCENARIOS, data=data, method=method,
                                      taylor_derived=block, reporting=reporting,
                                      confidence_level=confidence, batch_size=batch_size)
    return run, anatomy


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 03 FX legs and the reporting currency")

    # ================================================================ A
    sw.step("A1", "The conversion plan: USD vs EUR reporting")
    sw.para(
        """
        The run derives this plan itself (D085). It is built here by hand,
        with the public reporting building block and the settlement
        currencies of the trade record, to make the routes visible.
        """
    )
    tiny = sw.tiny_fixture()
    world = tiny.world
    settle = tuple(str(world.position(u)["settlement_ccy"]) for u in world.uids.tolist())
    rows = []
    for ccy in ("USD", "EUR"):
        plan = build_fx_conversion_plan(settlement_ccys=list(settle), reporting_ccy=ccy, fx_factor_names=["EURUSD"])
        for u, c, r, inv in zip(world.uids.tolist(), settle, plan.fx_row_index.tolist(), plan.invert.tolist(), strict=True):
            route = "identity (q = 1)" if r < 0 else ("1 / EURUSD" if inv else "EURUSD")
            rows.append((ccy, u, c, route))
    sw.table(["reporting", "uid", "settles", "rate q"], rows, align="lrll")
    usd_plan = build_fx_conversion_plan(settlement_ccys=list(settle), reporting_ccy="USD", fx_factor_names=["EURUSD"])
    eur_plan = build_fx_conversion_plan(settlement_ccys=list(settle), reporting_ccy="EUR", fx_factor_names=["EURUSD"])
    ok.check("USD reporting: only the EUR-settled 102 converts, directly",
             usd_plan.invert.tolist() == [0, 0, 0] and usd_plan.is_identity.tolist() == [True, False, True])
    ok.check("EUR reporting: the USD-settled 101 and 103 convert via 1 / EURUSD",
             eur_plan.invert.tolist() == [1, 0, 1] and eur_plan.is_identity.tolist() == [False, True, False])

    sw.step("A2", "Same book, two reporting currencies")
    runs = {}
    for ccy in ("USD", "EUR"):
        rep = sw.reporting_spec(world, reporting_ccy=ccy)
        runs[ccy] = _run(world, tiny.history, rep, derived=True, confidence=0.75, batch_size=4)
        ok.check(f"{ccy}: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(runs[ccy][1], runs[ccy][0]))
    r_usd, a_usd = runs["USD"]
    r_eur, a_eur = runs["EUR"]
    x0 = float(world.base_levels[world.row("EURUSD")])
    xs = a_usd.shocked_levels[world.row("EURUSD")]
    sw.table(
        ["uid", "ccy", "dV (settlement) s=0..3", "dW EUR s=0..3"],
        [(u, settle[i], " ".join(sw.num(x, 2) for x in a_eur.settlement_pnl[i]),
          " ".join(sw.num(x, 2) for x in a_eur.reporting.converted_pnl[i])) for i, u in enumerate(world.uids.tolist())],
        align="rlll",
    )
    # independent oracle: definition form, EUR value = settlement value * q
    q0 = {"USD": 1.0 / x0, "EUR": 1.0}
    qs = {"USD": 1.0 / xs, "EUR": np.ones_like(xs)}
    oracle = np.vstack([
        (world.v0_by_uid[u] + a_eur.settlement_pnl[i]) * qs[settle[i]] - world.v0_by_uid[u] * q0[settle[i]]
        for i, u in enumerate(world.uids.tolist())
    ])
    bound = 8.0 * EPS * np.vstack([
        np.abs(world.v0_by_uid[u] + a_eur.settlement_pnl[i]) * np.abs(qs[settle[i]]) + abs(world.v0_by_uid[u] * q0[settle[i]])
        for i, u in enumerate(world.uids.tolist())
    ])
    sw.para(
        """
        Rule: 1 / EURUSD is not exact in binary, so the hand definition
        V_s q_s - V_0 q_0 and the production form dV q_s + V_0 (q_s - q_0)
        round differently. Each uses at most four float64 operations, so they
        agree within 8 eps times the magnitude of the terms, per element.
        """
    )
    ok.check("EUR dW equals V_s q_s - V_0 q_0 within the stated float64 bound",
             bool((np.abs(a_eur.reporting.converted_pnl - oracle) <= bound).all()))
    ok.check("settlement-currency P&L is identical under both reporting currencies",
             np.array_equal(a_usd.settlement_pnl, a_eur.settlement_pnl))
    w_usd_s = sum((world.v0_by_uid[u] + a_usd.settlement_pnl[i]) * (xs if settle[i] == "EUR" else 1.0)
                  for i, u in enumerate(world.uids.tolist()))
    w_usd_0 = sum(world.v0_by_uid[u] * (x0 if settle[i] == "EUR" else 1.0) for i, u in enumerate(world.uids.tolist()))
    cross = w_usd_s / xs - w_usd_0 / x0
    ok.check("consistency: EUR portfolio dW = USD portfolio value / EURUSD, re-based",
             bool((np.abs(a_eur.reporting.portfolio_pnl - cross)
                   <= 16.0 * EPS * (np.abs(w_usd_s / xs) + abs(w_usd_0 / x0))).all()))
    sw.table(["reporting", "portfolio VaR", "portfolio ES"],
             [("USD", sw.num(r_usd.portfolio_var, 4), sw.num(r_usd.portfolio_es, 4)),
              ("EUR", sw.num(r_eur.portfolio_var, 4), sw.num(r_eur.portfolio_es, 4))], align="lrr")

    sw.step("A3", "Reporting FX is not a Taylor factor")
    sw.para(
        """
        Both runs above used a TAYLOR/DERIVED book. Its content digest is
        recorded in the manifest. If reporting FX were a Taylor factor, the
        book would have to change with the reporting currency. It does not:
        only the reporting section of the identity changes.
        """
    )
    ok.check("sensitivities_digest is identical for USD and EUR reporting",
             r_usd.manifest.pnl.sensitivities_digest == r_eur.manifest.pnl.sensitivities_digest)
    ok.check("the reporting section (currency, FX plan digest) differs, so the identity differs",
             r_usd.manifest.reporting != r_eur.manifest.reporting and r_usd.identity != r_eur.identity)

    # ================================================================ B
    sw.step("B1", "Three settlement currencies, USD reporting")
    real = sw.realistic_fixture()
    w = real.world
    rep = sw.reporting_spec(w, reporting_ccy="USD")
    r_b, a_b = _run(w, real.history, rep, derived=False, confidence=0.99, batch_size=256)
    ok.check("anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(a_b, r_b))
    counts = w.positions["settlement_ccy"].value_counts().to_dict()
    settle_b = [str(w.position(u)["settlement_ccy"]) for u in w.uids.tolist()]
    trans = a_b.reporting.converted_pnl - a_b.settlement_pnl * np.where(
        np.array([c == "USD" for c in settle_b])[:, None], 1.0,
        a_b.shocked_levels[[w.row("EURUSD") if c == "EUR" else w.row("GBPUSD") if c == "GBP" else 0
                            for c in settle_b]])
    sw.kv([
        ("settlement currencies", counts),
        ("FX factors on the run axis", [n for n in w.axis_names() if n.endswith("USD") and len(n) == 6]),
        ("portfolio VaR / ES 99%", f"{sw.num(r_b.portfolio_var, 2)} / {sw.num(r_b.portfolio_es, 2)} USD"),
        ("max |translation term V0 (q_s - q0)|", sw.num(float(np.abs(trans).max()), 2)),
    ])
    ok.check("USD-settled rows carry no translation term (q = 1 exactly)",
             not trans[np.array([c == "USD" for c in settle_b])].any())

    sw.step("B2", "An unsupported route fails fast")
    sw.para(
        """
        EUR reporting would need GBP -> EUR for the GBP-settled positions.
        Only EURUSD and GBPUSD are configured, and RiskQuant does not
        triangulate silently. The run derives the route during composition,
        so it refuses before the identity is built and before any valuation.
        """
    )
    ok.expect_rejection(
        "EUR reporting with only EURUSD and GBPUSD configured",
        CompositionContractError,
        lambda: run_scenario_var_es(
            scenario_config=SCENARIOS,
            data=real.history,
            required_factor_ids=w.required_factor_ids,
            required_shift_type_id=w.required_shift_type_id,
            proxy_terms=w.proxy_terms,
            base_levels=w.base_levels,
            methodology=sw.methodology(full=w.uids.tolist()),
            valuation_context=sw.valuation_context(w),
            reporting=ReportingSpec(reporting_ccy="EUR"),
            confidence_level=0.99,
            batch_size=256,
        ),
        reason="no rate is ever defaulted, filled, triangulated or assumed to be 1.0",
        stage="composition (D085 route derivation), before identity and numerics",
    )

    # ================================================================ C
    sw.step("C1", "A reporting-only FX factor")
    book = pd.DataFrame({
        "uid": [102],
        "instrument": ["SPOT"],
        "quantity": [-4.0],
        "underlying": ["SAP_EQ"],
        "settlement_ccy": ["EUR"],
    })
    c_usd = sw.tiny_fixture(book, reporting_ccy="USD")
    c_eur = sw.tiny_fixture(book, reporting_ccy="EUR")
    wc = c_usd.world
    sw.para(
        """
        Short 4 SAP, settled in EUR, and nothing else. No pricer reads
        EURUSD, so the valuation axis is SAP_EQ alone. USD reporting still
        needs EURUSD: W = V(EUR) x EURUSD. The run axis therefore carries
        EURUSD as a reporting-only row after the valuation prefix. EUR
        reporting needs no FX at all, and its run axis has no suffix.
        """
    )
    sw.table(
        ["reporting", "valuation prefix", "reporting-only suffix"],
        [("USD", [wc.name_of(int(r)) for r in wc.valuation_factor_ids],
          [wc.name_of(int(r)) for r in wc.reporting_factor_ids]),
         ("EUR", [c_eur.world.name_of(int(r)) for r in c_eur.world.valuation_factor_ids],
          [c_eur.world.name_of(int(r)) for r in c_eur.world.reporting_factor_ids])],
        align="lll",
    )
    ok.check("USD reporting: run axis = [SAP_EQ | EURUSD]",
             wc.axis_names() == ["SAP_EQ", "EURUSD"] and wc.num_valuation_factors == 1)
    ok.check("EUR reporting: run axis = [SAP_EQ], no suffix",
             c_eur.world.axis_names() == ["SAP_EQ"] and c_eur.world.reporting_factor_ids.size == 0)

    sw.step("C2", "Reporting-only FX risk, FULL_REVAL and TAYLOR/DERIVED")
    rep_c = sw.reporting_spec(wc)
    r_full, a_full = _run(wc, c_usd.history, rep_c, derived=False, confidence=0.75, batch_size=4)
    r_der, a_der = _run(wc, c_usd.history, rep_c, derived=True, confidence=0.75, batch_size=4)
    ok.check("FULL_REVAL: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(a_full, r_full))
    ok.check("DERIVED: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(a_der, r_der))
    sap, fx = a_full.shocked_levels[wc.row("SAP_EQ")], a_full.shocked_levels[wc.row("EURUSD")]
    s0, f0 = sw.TINY_LEVELS["SAP_EQ"], sw.TINY_LEVELS["EURUSD"]
    hand = -4.0 * sap * fx - (-4.0 * s0) * f0  # V_s q_s - V_0 q_0, dyadic: exact
    sw.table(
        ["scenario", "SAP_s", "EURUSD_s", "dV (EUR)", "dW hand (USD)", "dW RiskQuant"],
        [(s_, sw.num(sap[s_]), sw.num(fx[s_]), sw.num(a_full.settlement_pnl[0, s_], 2),
          sw.num(hand[s_], 2), sw.num(a_full.reporting.converted_pnl[0, s_], 2))
         for s_ in range(a_full.num_scenarios)],
        align="rrrrrr",
    )
    ok.check("dW equals the hand D081 value V_s q_s - V_0 q_0 exactly, every scenario",
             np.array_equal(a_full.reporting.converted_pnl[0], hand))
    sw.para(
        """
        Scenario 2 shows why V_0 enters reporting P&L. SAP rises 4, so short
        4 SAP loses 16 EUR. EURUSD falls 0.0625, so the base liability of 480
        EUR is worth 30 USD less. In USD the position GAINS 11. The term
        dV * q_s alone reports -19: the wrong sign.
        """
    )
    ok.check("scenario 2: dV = -16 EUR, dW = +11 USD; dV * q_s alone would give -19",
             float(a_full.settlement_pnl[0, 2]) == -16.0
             and float(a_full.reporting.converted_pnl[0, 2]) == 11.0
             and float(a_full.settlement_pnl[0, 2] * fx[2]) == -19.0)
    ok.check("DERIVED equals FULL_REVAL bitwise (linear payoff, exact inputs)",
             np.array_equal(r_der.var, r_full.var) and np.array_equal(r_der.es, r_full.es))
    ok.check("the Taylor book never sees EURUSD: its axis is the valuation prefix",
             sw.certified_taylor_block(wc, [102]).calculator.axis.keys.to_list() == [wc.rf("SAP_EQ")])

    sw.step("C3", "The caller cannot drop the reporting-only row")
    ew = c_eur.world
    ok.expect_rejection(
        "USD reporting with scenario data bound to the valuation-only axis",
        CompositionContractError,
        lambda: run_scenario_var_es(
            scenario_config=SCENARIOS,
            data=c_usd.history,
            required_factor_ids=ew.required_factor_ids,
            required_shift_type_id=ew.required_shift_type_id,
            proxy_terms=ew.proxy_terms,
            base_levels=ew.base_levels,
            methodology=sw.methodology(full=[102]),
            valuation_context=sw.valuation_context(ew),
            reporting=ReportingSpec(reporting_ccy="USD"),
            confidence_level=0.75,
            batch_size=4,
        ),
        reason="the run owns the axis: scenario data must be bound to "
        "derive_run_factor_axis(...), reporting-only rows included",
        stage="composition (canonical run-axis gate), before identity and numerics",
    )

    # ================================================================ D
    sw.step("D1", "Valuation FX legs: direct and inverted")
    legs_book = pd.concat(
        [
            sw.TINY_POSITIONS,
            pd.DataFrame({
                "uid": [104],
                "instrument": ["SPOT"],
                "quantity": [8.0],
                "underlying": ["AAPL_EQ"],
                "settlement_ccy": ["EUR"],
            }),
        ],
        ignore_index=True,
    )
    legs = sw.tiny_fixture(legs_book)
    wd = legs.world
    batch = wd.ingest_batch
    sw.para(
        """
        103 holds 8 SAP (EUR asset) settled in USD: its pricer reads EURUSD
        directly, V = 8 x SAP x EURUSD. 104 holds 8 AAPL (USD asset) settled
        in EUR. Only EURUSD is configured, so 104's leg is INVERTED:
        V = 8 x AAPL / EURUSD. Ingest records each leg and its orientation.
        """
    )
    leg_rows = []
    for u, fx_id, inv in zip(batch.uids.tolist(), batch.fx_u2s_ids.tolist(), batch.fx_u2s_inv.tolist(),
                             strict=True):
        leg = "-" if fx_id < 0 else ("1 / " if inv else "") + wd.name_of(fx_id)
        leg_rows.append((u, str(wd.position(u)["underlying"]), str(wd.position(u)["settlement_ccy"]), leg))
    sw.table(["uid", "underlying", "settles", "valuation FX leg"], leg_rows, align="rlll")
    leg_of = {u: (fx_id, inv) for u, fx_id, inv in zip(batch.uids.tolist(), batch.fx_u2s_ids.tolist(),
                                                      batch.fx_u2s_inv.tolist(), strict=True)}
    ok.check("103 reads EURUSD directly; 104 reads it inverted; 101 and 102 read no FX leg",
             leg_of[103] == (wd.rf("EURUSD"), 0) and leg_of[104] == (wd.rf("EURUSD"), 1)
             and leg_of[101][0] < 0 and leg_of[102][0] < 0)
    ok.check("V0 of 104 = 8 x 200 / 1.25 = 1280 EUR", wd.v0_by_uid[104] == 1280.0)
    ok.expect_rejection(
        "TAYLOR/DERIVED derivation for the inverted leg 104",
        NotImplementedError,
        lambda: sw.certified_taylor_block(wd, [104]),
        reason="V = q S / X is not bilinear in the quoted factor X; the inverted "
        "branch of the closed-form sensitizer is not authored, so it is refused "
        "rather than approximated",
        stage="run_sensitivities (book construction), before any run",
    )
    sw.para(
        """
        The run holds the same line on its own (D084): a position with an
        inverted leg declared TAYLOR/DERIVED is refused from the run's
        compiled state, before identity and numerics, whatever book is
        supplied (tests/solutions/risk_run/test_inverted_fx_taylor_admission.py).
        """
    )

    sw.step("D2", "An inverted FULL_REVAL leg beside TAYLOR/DERIVED legs")
    method_d = sw.methodology(full=[104], derived=[101, 102, 103])
    block_d = sw.certified_taylor_block(wd, [101, 102, 103])
    rep_d = sw.reporting_spec(wd, reporting_ccy="USD")
    run_d = run_scenario_var_es(
        scenario_config=SCENARIOS,
        data=legs.history,
        required_factor_ids=wd.required_factor_ids,
        required_shift_type_id=wd.required_shift_type_id,
        proxy_terms=wd.proxy_terms,
        base_levels=wd.base_levels,
        methodology=method_d,
        valuation_context=sw.valuation_context(wd),
        taylor_derived=block_d,
        reporting=rep_d,
        confidence_level=0.75,
        batch_size=4,
    )
    a_d = sw.scenario_pnl_anatomy(wd, scenario_config=SCENARIOS, data=legs.history, method=method_d,
                                  taylor_derived=block_d, reporting=rep_d, confidence_level=0.75, batch_size=4)
    ok.check("the mixed run completes; anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(a_d, run_d))
    sw.para(
        """
        The Taylor book is sensitised on U_D only, on its own compiled
        sub-portfolio, so 104 never reaches the sensitizer. One inverted leg
        does not block Taylor for the rest of the book.
        """
    )
    i104 = a_d.row(104)
    fx_s = a_d.shocked_levels[wd.row("EURUSD")]
    aapl_row = wd.row("AAPL_EQ")
    hand_104 = 8.0 * (a_d.shocked_levels[aapl_row] - float(wd.base_levels[aapl_row]))
    dv_104 = a_d.settlement_pnl[i104]
    dw_104 = a_d.reporting.converted_pnl[i104]
    v0_104 = wd.v0_by_uid[104]
    q0 = float(wd.base_levels[wd.row("EURUSD")])
    sw.para(
        """
        Economic oracle: in USD, 104 is worth V x EURUSD = 8 x AAPL. The
        inverted valuation leg and the direct reporting rate cancel, so
        dW = 8 x dAAPL. Rule: the grid rounds dV to float32 once (<= half a
        float32 ulp) and D081 multiplies it by q_s. So |dW - 8 dAAPL| <=
        q_s x one float32 ulp of dV, plus 16 float64 eps of the terms.
        """
    )
    bound_104 = (np.abs(fx_s) * np.spacing(np.abs(dv_104).astype(np.float32)).astype(np.float64)
                 + 16.0 * EPS * (np.abs(v0_104 + dv_104) * np.abs(fx_s) + abs(v0_104 * q0)))
    sw.table(
        ["scenario", "AAPL_s", "EURUSD_s", "dV (EUR)", "dW (USD)", "8 dAAPL"],
        [(s_, sw.num(a_d.shocked_levels[aapl_row, s_], 1), sw.num(fx_s[s_]), sw.num(dv_104[s_], 4),
          sw.num(dw_104[s_], 6), sw.num(hand_104[s_], 1)) for s_ in range(a_d.num_scenarios)],
        align="rrrrrr",
    )
    ok.check("104: dW = 8 x dAAPL within the stated bound, every scenario",
             bool((np.abs(dw_104 - hand_104) <= bound_104).all()))
    ok.done()


if __name__ == "__main__":
    main()
