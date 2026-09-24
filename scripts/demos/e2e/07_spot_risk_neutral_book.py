# scripts/demos/e2e/07_spot_risk_neutral_book.py
"""
RiskQuant SPOT -- chapter 07: neutralised (hedged) SPOT books.

RiskQuant has no "risk-neutral mode". A neutralised book is simply a book
whose exposures offset economically; the ordinary production run measures
what remains. Three books, from exact to approximate cancellation:

    A  same underlying, same currency, opposite quantity
       -> portfolio P&L is exactly zero in every scenario;
    B  the same asset held twice: once as a USD-settled quanto, once short
       and EUR-settled, reported in USD
       -> exactly zero again, BECAUSE reporting follows D081 (V_s q_s - V_0 q_0);
    C  WTI hedged with Brent, where WTI has no history of its own and is
       proxied to Brent with scale 0.95
       -> neutral to FIRST order only; the residual is a convexity loss.

For each book: gross exposure sum|W0|, net exposure sum W0 (USD, by hand),
the scenario P&L distribution, VaR and ES.

Run:  python scripts/demos/e2e/07_spot_risk_neutral_book.py
"""

from __future__ import annotations

import _spot_world as sw
import numpy as np
import pandas as pd

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import RiskRun
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig

SCENARIOS = HistoricalGeneratorConfig(model="historical", horizon=1)


def _book(rows: list[tuple[int, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame({
        "uid": [r[0] for r in rows],
        "instrument": ["SPOT"] * len(rows),
        "quantity": [r[1] for r in rows],
        "underlying": [r[2] for r in rows],
        "settlement_ccy": [r[3] for r in rows],
    })


def _measure(world: sw.SpotWorld, history, confidence: float,
             batch_size: int, ok: sw.Checks) -> tuple[RiskRun, sw.ScenarioAnatomy]:
    method = sw.methodology(full=world.uids.tolist())
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    run = run_scenario_var_es(
        scenario_config=SCENARIOS,
        data=history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        source_factor_ids=world.source_factor_ids,
        shock_scale=world.shock_scale,
        base_levels=world.base_levels,
        methodology=method,
        valuation_context=sw.valuation_context(world),
        reporting=reporting,
        confidence_level=confidence,
        batch_size=batch_size,
    )
    anatomy = sw.scenario_pnl_anatomy(world, scenario_config=SCENARIOS, data=history, method=method,
                                      taylor_derived=None, reporting=reporting,
                                      confidence_level=confidence, batch_size=batch_size)
    ok.check("anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(anatomy, run))
    return run, anatomy


def _exposure(world: sw.SpotWorld, fx_usd: dict[str, float]) -> tuple[float, float]:
    w0 = [world.v0_by_uid[u] * fx_usd[str(world.position(u)["settlement_ccy"])] for u in world.uids.tolist()]
    return float(np.sum(np.abs(w0))), float(np.sum(w0))


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 07 neutralised (hedged) books")

    # ---------------------------------------------------------------- A
    sw.step("A", "Exact offset: same underlying, same currency")
    fa = sw.tiny_fixture(_book([(201, 10.0, "AAPL_EQ", "USD"), (202, -10.0, "AAPL_EQ", "USD")]))
    run_a, an_a = _measure(fa.world, fa.history, 0.75, 4, ok)
    gross, net = _exposure(fa.world, {"USD": 1.0})
    sw.table(["", "value"], [("gross exposure (USD)", sw.num(gross, 2)), ("net exposure (USD)", sw.num(net, 2)),
                             ("position VaR 201 / 202", f"{sw.num(run_a.var[0], 2)} / {sw.num(run_a.var[1], 2)}"),
                             ("portfolio P&L s=0..3", " ".join(sw.num(x, 2) for x in an_a.reporting.portfolio_pnl)),
                             ("portfolio VaR / ES", f"{sw.num(run_a.portfolio_var, 2)} / {sw.num(run_a.portfolio_es, 2)}")],
             align="lr")
    ok.check("portfolio P&L is exactly 0.0 in every scenario", not an_a.reporting.portfolio_pnl.any())
    ok.check("portfolio VaR and ES are exactly 0.0; each leg alone carries risk",
             run_a.portfolio_var == 0.0 and run_a.portfolio_es == 0.0 and run_a.var[0] < 0 and run_a.var[1] < 0)

    # ---------------------------------------------------------------- B
    sw.step("B", "Exact offset across settlement currencies")
    fb = sw.tiny_fixture(_book([(211, 8.0, "SAP_EQ", "USD"), (212, -8.0, "SAP_EQ", "EUR")]))
    run_b, an_b = _measure(fb.world, fb.history, 0.75, 4, ok)
    gross, net = _exposure(fb.world, {"USD": 1.0, "EUR": 1.25})
    sw.para(
        """
        211 is a USD-settled quanto: V = 8 SAP EURUSD (USD). 212 is short 8
        SAP settled in EUR: V = -8 SAP (EUR), reported as W = V EURUSD. In
        USD the two are exact negatives, so ordinary-MTM reporting must
        cancel them exactly. The settlement-currency P&L does NOT cancel:
        its currencies differ and it is never summed.
        """
    )
    sw.table(["uid", "ccy", "dV (settlement) s=0..3", "dW (USD) s=0..3"],
             [(u, str(fb.world.position(u)["settlement_ccy"]),
               " ".join(sw.num(x, 2) for x in an_b.settlement_pnl[an_b.row(u)]),
               " ".join(sw.num(x, 2) for x in an_b.reporting.converted_pnl[an_b.row(u)])) for u in (211, 212)],
             align="rlll")
    sw.kv([("gross / net exposure (USD)", f"{sw.num(gross, 2)} / {sw.num(net, 2)}"),
           ("portfolio VaR / ES", f"{sw.num(run_b.portfolio_var, 2)} / {sw.num(run_b.portfolio_es, 2)}")])
    ok.check("reporting P&L of 211 and 212 are exact negatives in every scenario",
             np.array_equal(an_b.reporting.converted_pnl[0], -an_b.reporting.converted_pnl[1]))
    ok.check("portfolio VaR and ES are exactly 0.0", run_b.portfolio_var == 0.0 and run_b.portfolio_es == 0.0)
    d014 = an_b.settlement_pnl[an_b.row(212)] * an_b.shocked_levels[fb.world.row("EURUSD")] \
        + an_b.reporting.converted_pnl[an_b.row(211)]
    ok.check("the retired D014 form (dV * q_s) would report a phantom residual here",
             bool(d014.any()), f"D014 residual s=0..3: {' '.join(sw.num(x, 2) for x in d014)}")

    # ---------------------------------------------------------------- C
    sw.step("C", "Approximate neutrality: a proxy hedge")
    probe = sw.realistic_fixture(positions=_book([(301, 100.0, "WTI_CRUDE", "USD")]))
    w0_wti = probe.world.levels["WTI_CRUDE"]
    b0 = probe.world.levels["BRENT_CRUDE"]
    scale = float(np.float32(0.95))  # shock_scale is float32 by contract
    q_brent = -round(100.0 * w0_wti * scale / b0, 6)
    fc = sw.realistic_fixture(positions=_book([(301, 100.0, "WTI_CRUDE", "USD"), (302, q_brent, "BRENT_CRUDE", "USD")]))
    wc = fc.world
    sw.para(
        f"""
        WTI has no history; the proxy topology drives it with Brent's log
        return r, scaled: WTI_s = WTI_0 exp(0.95 r). Long 100 WTI is
        hedged with q_B = -100 WTI_0 0.95 / B_0 = {q_brent:,.6f} Brent, which
        makes the book neutral to FIRST order in r. The residual is
        """
    )
    sw.formula("dW(r) = 100 WTI_0 [ (e^(0.95 r) - 1) - 0.95 (e^r - 1) ]",
               "      ~ 100 WTI_0 (0.95^2 - 0.95) r^2 / 2  <= 0   for every r")
    run_c, an_c = _measure(wc, fc.history, 0.99, 128, ok)
    ok.check("proxy resolution: WTI is sourced from BRENT with scale float32(0.95)",
             int(wc.source_factor_ids[wc.row("WTI_CRUDE")]) == wc.rf("BRENT_CRUDE")
             and float(wc.shock_scale[wc.row("WTI_CRUDE")]) == scale)
    # independent oracle from the Brent history (float32 shocks by contract)
    brent = np.asarray(fc.history.values[:, list(fc.history.factor_ids).index(wc.rf("BRENT_CRUDE"))])
    r = np.diff(np.log(brent)).astype(np.float32).astype(np.float64)
    oracle = 100.0 * w0_wti * (np.exp(scale * r) - 1.0) + q_brent * b0 * (np.exp(r) - 1.0)
    port = an_c.reporting.portfolio_pnl
    legs = an_c.settlement_pnl
    bound = 0.5 * np.spacing(np.abs(legs).astype(np.float32)).astype(np.float64).sum(axis=0) \
        + 8.0 * np.finfo(np.float64).eps * (abs(100.0 * w0_wti) + abs(q_brent * b0)) * np.exp(np.abs(r))
    gross, net = _exposure(wc, {"USD": 1.0})
    sw.table(["", "value"], [
        ("gross exposure (USD)", sw.num(gross, 2)),
        ("net exposure (USD)", sw.num(net, 2)),
        ("leg VaR 99%: WTI / Brent", f"{sw.num(run_c.var[0], 2)} / {sw.num(run_c.var[1], 2)}"),
        ("portfolio P&L: min / max", f"{sw.num(port.min(), 4)} / {sw.num(port.max(), 4)}"),
        ("portfolio VaR / ES 99%", f"{sw.num(run_c.portfolio_var, 4)} / {sw.num(run_c.portfolio_es, 4)}"),
    ], align="lr")
    sw.para(
        """
        Note: the net VALUE is not zero (printed above). Neutrality is a
        statement about sensitivity to the risk factor, not about value.
        """
    )
    sw.para(
        """
        Comparison rule: each leg's dV is rounded once into the float32 grid
        (<= half a float32 ulp each); the float64 exponentials add at most a
        few float64 eps of the leg values. The bound is computed per scenario.
        """
    )
    ok.check("portfolio P&L matches the analytic residual within the stated bound",
             bool((np.abs(port - oracle) <= bound).all()))
    ok.check("residual is a loss (<= 0 up to the bound) in every scenario: short convexity",
             bool((port <= bound).all()))
    ok.check("the hedge removes > 99% of the WTI leg's VaR, but not all of it",
             0.0 > run_c.portfolio_var > 0.01 * run_c.var[0])
    sw.para(
        """
        Exact cancellation (A, B) is a property of the positions: identical
        exposures with opposite sign. Approximate cancellation (C) is a
        property of the hedge ratio: it removes the first-order term only,
        and what remains depends on the scenario distribution.
        """
    )
    ok.done()


if __name__ == "__main__":
    main()
